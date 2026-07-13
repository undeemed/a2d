"""Annealed causal->bidirectional attention for the RoPE/GQA family (Decision 2).

Llama / Qwen2 / Gemma (``transformers==4.51.3``) do NOT bake causality into a
per-layer ``self.bias`` buffer the way GPT-2 does. Instead the decoder ``*Model``
builds one 4D additive causal mask per forward in ``_update_causal_mask`` and hands
it down to every layer; eager attention just does ``scores + causal_mask``. Causality
therefore flows entirely through that mask, so annealing it (not a ``self.bias`` seam)
is what opens attention here. This is the ONE seam shared verbatim by all three
families, so a single handler covers Gemma (MQA), Qwen2/Llama (GQA), and even
full-attention RoPE models (Llama-2-7B): the mask is family-independent, and the GQA
group expansion (``repeat_kv``), RoPE, RMSNorm, and Gemma's sqrt(hidden) embedding
scaling all stay untouched in HF's own forward.

The patch wraps the decoder's bound ``_update_causal_mask``: it calls the original
to get the exact base mask, then re-reveals every cell that mask masked for a real
(non-padded) key under one shared ``AnnealState``. For the full-attention RoPE
family those cells are exactly the strictly-future set; Mistral v0.1 and Qwen2 with
an active ``use_sliding_window`` fold their sliding window into this SAME model-level
mask (they have no per-layer window), so their far-past out-of-window cells reopen
through the identical anneal and ``alpha=1`` is fully non-causal AND unwindowed:

* At ``alpha=0`` the penalty is exactly ``finfo(dtype).min`` - the same value the
  base mask already carries at masked cells - so ``torch.where(reveal, penalty,
  base)`` returns a tensor bit-identical to base, and patched@0 logits equal base to
  the bit (the D13 identity gate).
* At ``alpha=1`` the penalty is ``log(1)=0`` so masked cells become attendable and
  attention is fully bidirectional; intermediate ``alpha`` applies ``log(alpha)`` (a
  smooth monotone reveal), mirroring the GPT-2 seam's semantics.

Deriving the result FROM the base mask (rather than rebuilding it) is what makes the
``alpha=0`` no-op bit-identical by construction, independent of dtype, cache offset,
or padding. A genuine 2D padding mask is preserved: a masked cell is only revealed
when its key is a real (non-padded) token, so padding stays masked at every alpha.

The override is installed on the model INSTANCE (it shadows the class method via the
instance ``__dict__``), so a sibling un-patched base model in the same process - the
identity gate's reference copy - keeps its original causal attention.
"""

from __future__ import annotations

import math
import types
from typing import Any

import torch

from a2d_core.transform.attention import AnnealState

# Sentinel stashing the original bound ``_update_causal_mask`` on the decoder module
# once patched; also guards against double-wrapping (which would capture our own
# wrapper as the "original" and anneal twice).
_ORIG_ATTR = "_a2d_gqa_orig_update_causal_mask"

# The live ``AnnealState`` on the decoder module, read by the wrapper on every call
# and re-assigned by every install (parity with the GPT-2 seam's ``_a2d_anneal`` tag),
# so re-installing with a fresh state swaps it in instead of being silently ignored.
_STATE_ATTR = "_a2d_anneal"


def _reveal_penalty(alpha: float, dtype: torch.dtype) -> float:
    """Additive pre-softmax penalty for a revealed masked cell at this ``alpha``.

    ``finfo(dtype).min`` at ``alpha<=0`` (matches the base causal mask exactly, so the
    identity gate is bit-identical); ``log(alpha)`` clamped to ``finfo.min`` otherwise;
    ``0`` at ``alpha=1`` (fully bidirectional).
    """
    finfo_min = torch.finfo(dtype).min
    if alpha <= 0.0:
        return finfo_min
    return max(math.log(alpha), finfo_min)


def _find_causal_mask_owner(model: Any) -> Any | None:
    """The decoder submodule (the ``*Model``) that defines ``_update_causal_mask``.

    For ``*ForCausalLM`` this is ``model.model``; found structurally (by the class
    that actually defines the method) so no family module names are hardcoded.
    """
    for module in model.modules():
        if type(module).__dict__.get("_update_causal_mask") is not None:
            return module
    return None


def install_gqa_anneal_patch(model: Any, state: AnnealState) -> None:
    """Route ``model``'s causal-mask construction through the annealed reveal.

    Requires ``attn_implementation="eager"`` (Decision 2, parity with the GPT-2 seam);
    forces ``use_cache=False`` (diffusion decodes the full canvas, ARCHITECTURE.md §7);
    wraps the decoder's ``_update_causal_mask`` and tags the owner with ``state``,
    re-assigned on every install so a later install swaps in its fresh state.
    """
    if model.config._attn_implementation != "eager":
        raise ValueError(
            "anneal patch requires attn_implementation='eager', got "
            f"{model.config._attn_implementation!r} (Decision 2)"
        )
    owner = _find_causal_mask_owner(model)
    if owner is None:
        raise ValueError(
            "GQA anneal patch found no _update_causal_mask seam (not a RoPE-family causal model?)"
        )
    model.config.use_cache = False
    setattr(owner, _STATE_ATTR, state)

    if hasattr(owner, _ORIG_ATTR):
        return  # already wrapped; wrapping again would double-anneal
    orig = owner._update_causal_mask  # bound original, captured before we shadow it
    setattr(owner, _ORIG_ATTR, orig)

    def _annealed_update_causal_mask(
        self: Any,
        attention_mask: Any,
        input_tensor: Any,
        cache_position: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        base_mask = orig(attention_mask, input_tensor, cache_position, *args, **kwargs)
        # Eager always returns a 4D additive mask; None only appears on sdpa/flash
        # fast paths we never take. Nothing to anneal if it is absent.
        if base_mask is None:
            return None
        dtype = base_mask.dtype
        kv_len = base_mask.shape[-1]
        # Every cell the base mask masked (exactly finfo.min): the strictly-future
        # causal cells, plus - for single-mask windowed families like Mistral v0.1 or
        # Qwen2 with an active sliding window - the far-past out-of-window cells.
        reveal = base_mask == torch.finfo(dtype).min
        # Preserve genuine padding: only reveal a masked cell whose key is real, so a
        # padded key stays finfo.min at every alpha (parity with the GPT-2 seam).
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            real_key = (attention_mask != 0)[:, None, None, :kv_len]
            reveal = reveal & real_key
        live_state: AnnealState = getattr(self, _STATE_ATTR)
        penalty = torch.full(
            (), _reveal_penalty(live_state.alpha, dtype), dtype=dtype, device=base_mask.device
        )
        return torch.where(reveal, penalty, base_mask)

    owner._update_causal_mask = types.MethodType(_annealed_update_causal_mask, owner)
