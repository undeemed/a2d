"""Annealed causal->bidirectional attention patch for GPT-2's eager seam (Decision 2).

``transformers==4.48.3`` GPT-2 bakes causality INSIDE the attention op: eager
``GPT2Attention`` masks scores with a lower-triangular ``self.bias`` buffer via
``torch.where(causal, scores, finfo(float32).min)``; the passed 4D
``attention_mask`` is padding-only. A model-level additive mask therefore does
NOT control causality, so this module patches the seam that does.

The patch neutralizes ``self.bias`` (registers it all-True so ``torch.where``
never masks) and re-supplies a single annealed additive mask driven by one shared
``AnnealState``: on/below the diagonal it is ``0``; strictly-future (``j>i``) it is
``clamp(log(alpha), finfo(float32).min)``.

* At ``alpha=0`` a future entry is ``finfo.min``, and a finite pre-softmax score is
  negligible against it in float32 (``score + finfo.min == finfo.min`` to the bit),
  so patched-at-0 scores/logits are bit-identical to base.
* At ``alpha=1`` the penalty is ``log(1)=0`` and, with ``self.bias`` neutralized,
  attention is fully bidirectional; intermediate ``alpha`` scales each future
  position's pre-softmax mass by ``alpha`` (a smooth monotone reveal).

This deliberately replaces the ML-recon's ``(1-alpha)*(-inf) + alpha*0`` blend,
which is ``-inf`` for every ``alpha<1`` and thus not an anneal at all (Risk 4).

The eager path is patched at the module-global ``eager_attention_forward`` that
``GPT2Attention.forward`` resolves at call time. The replacement is inert for any
attention module without a ``_a2d_anneal`` tag, so an unpatched base model (the
identity gate's reference copy) keeps its original causal attention even though
the global is process-wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AnnealState:
    """Mutable ``alpha in [0,1]`` shared by every patched attention layer.

    ``alpha=0`` is causal (bit-identical to base); ``alpha=1`` is bidirectional.
    Driven statelessly from the global step by ``AnnealCallback`` (Decision 4), so
    resume needs no persisted anneal state.
    """

    alpha: float = 0.0


def schedule(step: int, anneal_steps: int, kind: str = "linear") -> float:
    """Return ``alpha`` for ``step``: ``0`` at step 0, ``1`` at ``step>=anneal_steps``,
    monotone non-decreasing. Only ``"linear"`` exists in P2 (cosine is a drop-in)."""
    if kind != "linear":
        raise ValueError(f"unknown anneal schedule {kind!r}")
    if anneal_steps <= 0:
        return 1.0
    return max(0.0, min(step / anneal_steps, 1.0))


def annealed_additive_mask(
    q_len: int, k_len: int, alpha: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build the ``[q_len, k_len]`` additive pre-softmax mask for one attention call.

    ``0`` on/below the causal diagonal, ``clamp(log(alpha), finfo.min)`` strictly
    above it. At ``alpha=0`` the future penalty is exactly ``finfo(dtype).min`` so
    ``score + penalty`` reproduces base's ``torch.where``-to-``finfo.min`` to the bit.
    """
    finfo_min = torch.finfo(dtype).min
    # Match base's slice (module.bias[:, :, k-q:k, :k]) so the offset/cached q<k case
    # stays aligned with GPT-2 causality.
    causal = torch.tril(torch.ones(k_len, k_len, dtype=torch.bool, device=device))[
        k_len - q_len : k_len, :k_len
    ]
    penalty = finfo_min if alpha <= 0.0 else max(math.log(alpha), finfo_min)
    zero = torch.zeros((), dtype=dtype, device=device)
    pen = torch.full((), penalty, dtype=dtype, device=device)
    # Rebuilt per call, mirroring base's per-call causal slice.
    return torch.where(causal, zero, pen)


# The true library ``eager_attention_forward``, captured once when the global patch
# is installed so the replacement can delegate to it (reusing base's exact scaling,
# softmax and value matmul keeps the alpha=0 path bit-identical by construction).
_original_eager: Any = None


def _patched_eager(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    head_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    """Drop-in for ``eager_attention_forward``: inject the annealed additive mask.

    Only acts on modules tagged with ``_a2d_anneal`` (whose ``self.bias`` we have
    neutralized to all-True); every other module delegates unchanged, so a base
    reference model sharing this process keeps genuine causal attention.
    """
    state: AnnealState | None = getattr(module, "_a2d_anneal", None)
    if state is None:
        return _original_eager(
            module, query, key, value, attention_mask, head_mask=head_mask, **kwargs
        )
    add = annealed_additive_mask(
        query.size(-2), key.size(-2), state.alpha, query.dtype, query.device
    )
    if attention_mask is not None:
        # Fold any real (padding) mask in; base adds it after its where, we add once.
        add = add + attention_mask
    return _original_eager(module, query, key, value, add, head_mask=head_mask, **kwargs)


def _ensure_global_patch() -> None:
    """Install the process-global eager-attention replacement exactly once."""
    global _original_eager
    if _original_eager is not None:
        return
    import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2

    _original_eager = modeling_gpt2.eager_attention_forward
    modeling_gpt2.eager_attention_forward = _patched_eager


def install_anneal_patch(model: Any, state: AnnealState) -> None:
    """Route ``model``'s eager attention through the annealed additive mask.

    Requires ``attn_implementation="eager"`` (Decision 2); forces ``use_cache=False``,
    neutralizes each ``GPT2Attention``'s causal ``self.bias`` (all-True) so only the
    additive mask governs causality, and tags each attention module with the shared
    ``state``.
    """
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    if model.config._attn_implementation != "eager":
        raise ValueError(
            "anneal patch requires attn_implementation='eager', got "
            f"{model.config._attn_implementation!r} (Decision 2)"
        )
    model.config.use_cache = False
    patched_any = False
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            n = module.bias.shape[-1]
            module.bias = torch.ones((1, 1, n, n), dtype=torch.bool, device=module.bias.device)
            module._a2d_anneal = state  # dynamic tag read back in _patched_eager
            patched_any = True
    if not patched_any:
        raise ValueError("anneal patch found no GPT2Attention modules (not a GPT-2 eager model?)")
    _ensure_global_patch()
