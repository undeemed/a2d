"""Annealed causal->bidirectional attention for the RoPE/GQA family (Decision 2).

Llama / Qwen2 / Gemma (``transformers==4.48.3``) do NOT bake causality into a
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
to get the exact base mask, then re-reveals the strictly-future cells under one
shared ``AnnealState``:

* At ``alpha=0`` the future penalty is exactly ``finfo(dtype).min`` - the same value
  the base mask already carries at future cells - so ``torch.where(future, penalty,
  base)`` returns a tensor bit-identical to base, and patched@0 logits equal base to
  the bit (the D13 identity gate).
* At ``alpha=1`` the penalty is ``log(1)=0`` so future cells become attendable and
  attention is fully bidirectional; intermediate ``alpha`` applies ``log(alpha)`` (a
  smooth monotone reveal), mirroring the GPT-2 seam's semantics.

Deriving the result FROM the base mask (rather than rebuilding it) is what makes the
``alpha=0`` no-op bit-identical by construction, independent of dtype, cache offset,
or padding. A genuine 2D padding mask is preserved: a future cell is only revealed
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


def _future_penalty(alpha: float, dtype: torch.dtype) -> float:
    """Additive pre-softmax penalty for a strictly-future cell at this ``alpha``.

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
    wraps the decoder's ``_update_causal_mask`` and tags it with the shared ``state``.
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

    if hasattr(owner, _ORIG_ATTR):
        return  # already patched; re-installing would double-anneal
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
        # future[q, kv]: key strictly after the query's cache position - exactly the
        # cells the base mask masked for causality (mirrors base's own arange > pos).
        future = torch.arange(kv_len, device=base_mask.device) > cache_position.reshape(-1, 1)
        reveal = future[None, None, :, :]
        # Preserve genuine padding: only reveal a future cell whose key is real, so a
        # padded key stays finfo.min at every alpha (parity with the GPT-2 seam).
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            real_key = (attention_mask != 0)[:, None, None, :kv_len]
            reveal = reveal & real_key
        penalty = torch.full(
            (), _future_penalty(state.alpha, dtype), dtype=dtype, device=base_mask.device
        )
        return torch.where(reveal, penalty, base_mask)

    owner._update_causal_mask = types.MethodType(_annealed_update_causal_mask, owner)
