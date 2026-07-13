"""Annealed causal->bidirectional attention for the sliding-window Gemma family
(Gemma 2/3).

Gemma 2 and Gemma 3 (``transformers==4.51.3``) are the RoPE/GQA seam PLUS a per-layer
sliding window. Their decoder ``*Model`` builds ONE full 4D additive causal mask in
``_update_causal_mask`` - exactly the Gemma 1 seam - and hands it to every layer.
Then each *sliding* decoder layer's ``forward`` further masks the strictly-
far-past (keys more than ``sliding_window`` behind the query) to ``finfo(dtype).min``
with identical logic in both families; the remaining layers are *global* and skip
that step (Gemma 3 windows all but every ``sliding_window_pattern``-th layer, Gemma 2
every other layer). Eager attention just does ``scores + attention_mask``, so BOTH
causality (future) AND the window (far-past) flow entirely through the additive mask,
layer by layer. (Mistral-style models instead fold their window into the model-level
mask and have no sliding layers; they are the ``attn.gqa`` seam, not this one.)

Bidirectionalizing therefore needs TWO coordinated anneals over one shared
``AnnealState``:

* the ``_update_causal_mask`` wrap (reused verbatim from the ``attn.gqa`` seam via
  ``install_gqa_anneal_patch``) reveals strictly-future cells. This opens every
  layer's future AND feeds the sliding layers their future-revealed base mask.
* a per-sliding-layer ``forward`` wrap reveals the strictly-far-past cells the layer
  would otherwise window out.

Both use the same penalty ramp as the GPT-2/GQA seams: ``finfo(dtype).min`` at
``alpha=0`` (so each reveal is bit-identical to base -> the D13 identity gate reads
``max_abs_diff == 0.0``) and ``clamp(log(alpha), finfo.min)``, reaching ``0`` at
``alpha=1`` (fully non-causal AND unwindowed). Global layers are untouched by the
second wrap; they open through the first alone. RoPE (Gemma 3's per-layer local vs
global theta), query-key norm, the GQA/MQA layout, RMSNorm, logit softcapping
(Gemma 2), and Gemma's sqrt(hidden) embedding scaling all stay in HF's own forward -
only the additive mask is patched, exactly as the Gemma 1 seam does.

The far-past wrap temporarily flips the decoder layer's ``is_sliding`` to ``False``
so HF's own (hard ``finfo.min``) window re-mask is skipped for that one call, and
supplies its annealed replacement instead. It does NOT touch ``self_attn.is_sliding``,
which selects the local RoPE embedding, so positions stay exactly as base computes
them. Overrides live on module INSTANCES (shadowing the class methods), so a sibling
un-patched base model in the same process - the identity gate's reference copy -
keeps its original causal+windowed attention.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Iterator
from typing import Any

import torch

from a2d_core.transform.attention import AnnealState
from a2d_core.transform.gqa_attention import _reveal_penalty, install_gqa_anneal_patch

# Stash of the original bound decoder-layer ``forward`` (guards double-wrapping) and
# the live ``AnnealState`` tag read by the wrapper, re-assigned on every install so a
# fresh state swaps in (parity with the gqa/GPT-2 seams' per-install re-tag).
_ORIG_ATTR = "_a2d_swa_orig_forward"
_STATE_ATTR = "_a2d_anneal"


def _iter_sliding_decoder_layers(model: Any) -> Iterator[Any]:
    """Yield the decoder-layer modules that window their attention.

    Gemma 2/3 tag BOTH the decoder layer and its attention submodule with
    ``is_sliding``; only the decoder layer owns the per-layer window re-mask in its
    ``forward``, and it is the one that also holds a ``self_attn`` submodule - so we
    key off that pair. Found structurally, no family module names hardcoded.
    """
    for module in model.modules():
        if getattr(module, "is_sliding", False) and hasattr(module, "self_attn"):
            yield module


def has_sliding_window_seam(model: Any) -> bool:
    """True iff the model has at least one sliding-window (local) decoder layer.

    This is the ``attn.swa`` structural signal ``resolve_capabilities`` dispatches on,
    mirroring how ``_find_causal_mask_owner`` signals ``attn.gqa``.
    """
    return next(_iter_sliding_decoder_layers(model), None) is not None


def _anneal_window(
    layer: Any, attention_mask: Any, cache_position: Any, last_cache_position: int
) -> Any:
    """Mirror the Gemma 2/3 decoder layer's sliding re-mask but reveal the far-past
    per alpha.

    Reproduces HF's exact eager (4D) window logic (identical in both families) - the
    ``tril(diagonal=-window)`` far-past selection and the
    ``[offset : offset + effective_seq_len]`` slice - only swapping the hard
    ``finfo(dtype).min`` fill for the annealed penalty.

    The reveal is restricted to far-past cells that are currently *attendable* (mask
    ``== 0``), i.e. real past keys. Cells already at ``finfo.min`` are keys the base
    mask masked for another reason - padding folded in by ``_update_causal_mask``, or
    the future-reveal patch leaving a padded key masked - and must STAY masked at every
    alpha (parity with the GPT-2/GQA seams' padding preservation). Since the far-past
    region lives strictly below the diagonal it never overlaps the future cells the
    first patch anneals, so ``== 0`` cleanly separates real past keys from padded ones.

    At ``alpha=0`` the penalty IS ``finfo.min``: real far-past cells go to ``finfo.min``
    (windowed) and padded ones are untouched (already ``finfo.min``), which is exactly
    HF's ``where(sliding_window_mask, min_dtype, mask)`` - bit-identical to base. At
    ``alpha=1`` the penalty is ``0``, opening the window for real keys while padding
    stays masked.
    """
    if attention_mask is None:
        return attention_mask
    dtype = attention_mask.dtype
    effective_seq_len = max(cache_position.shape[0], layer.sliding_window)
    far_past = torch.tril(
        torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-layer.sliding_window
    )
    reveal = far_past & (attention_mask == 0)  # real (attendable) far-past keys only
    state: AnnealState = getattr(layer, _STATE_ATTR)
    penalty = torch.full(
        (), _reveal_penalty(state.alpha, dtype), dtype=dtype, device=attention_mask.device
    )
    attention_mask = torch.where(reveal, penalty, attention_mask)
    offset = max(0, last_cache_position - effective_seq_len)
    return attention_mask[:, :, :, offset : offset + effective_seq_len]


def _make_wrapped_forward(orig: Any) -> Any:
    """Build the decoder-layer ``forward`` replacement bound around ``orig``.

    Signature-agnostic: the call is re-bound against ``orig``'s own signature (Gemma 2
    takes one ``position_embeddings`` pair where Gemma 3 takes a global/local pair),
    so it works whether the layer is called with keywords (the model's own forward) or
    positionally (gradient checkpointing). Only ``attention_mask`` is replaced - with
    the annealed window mask - and every other argument passes through unchanged.
    ``orig`` then runs with ``is_sliding`` temporarily off so HF does not re-apply its
    own hard window mask.
    """
    signature = inspect.signature(orig)

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bound.arguments["attention_mask"] = _anneal_window(
            self,
            bound.arguments.get("attention_mask"),
            bound.arguments.get("cache_position"),
            int(bound.arguments.get("last_cache_position") or 0),
        )
        was_sliding = self.is_sliding
        self.is_sliding = False  # skip HF's hard finfo.min window re-mask for this call
        try:
            return orig(*bound.args, **bound.kwargs)
        finally:
            self.is_sliding = was_sliding

    return _wrapped


def install_swa_anneal_patch(model: Any, state: AnnealState) -> None:
    """Route ``model``'s causal+sliding-window masks through the annealed reveal.

    Two coordinated patches on one shared ``state``: the RoPE/GQA future-reveal on the
    single full causal mask (reused from ``install_gqa_anneal_patch`` - this also
    enforces ``attn_implementation='eager'`` and ``use_cache=False`` and tags the mask
    owner), plus a far-past reveal wrapped onto every sliding decoder layer's forward.
    The seam is validated up front so a failed install leaves the model untouched.
    Re-installing swaps in the fresh state without double-wrapping.
    """
    if not has_sliding_window_seam(model):
        raise ValueError(
            "SWA anneal patch found no sliding-window decoder layers "
            f"on {type(model).__name__!r} (not a Gemma 2/3 sliding-window model?)"
        )

    # Patch 1: reveal strictly-future cells on the model-level causal mask (global
    # layers + the base mask sliding layers receive).
    install_gqa_anneal_patch(model, state)

    # Patch 2: reveal the strictly-far-past on each sliding (local) decoder layer.
    for layer in _iter_sliding_decoder_layers(model):
        setattr(layer, _STATE_ATTR, state)  # live state; re-install swaps it in
        if hasattr(layer, _ORIG_ATTR):
            continue  # already wrapped; wrapping again would double-anneal
        orig = layer.forward  # bound original, captured before we shadow it
        setattr(layer, _ORIG_ATTR, orig)
        layer.forward = types.MethodType(_make_wrapped_forward(orig), layer)
