"""Identity gate (Decision 2 / D13): base vs patched@alpha=0 on clean input.

At ``alpha=0`` the annealed patch reproduces base causality to the bit, so a correct
patch yields ``max_abs_diff == 0.0`` on CPU float32. The gate ALWAYS runs float32 on
CPU regardless of ``--dtype`` (Risk 2). Grow adds a logit COLUMN, so patched logits
are sliced to ``base_vocab`` before comparing (Decision 7 / Risk 3).

This gate CANNOT prove the patch reaches GPT-2's causality - a no-op seam that leaves
the model fully causal passes it too - so bidirectionality is proven separately by
``test_bidir`` (Decision 2).
"""

from __future__ import annotations

from typing import Any

from a2d_contracts.models.manifest_schema import IdentityResult

from a2d_core.transform.attention import AnnealState

IDENTITY_TOLERANCE = 1e-6


def check_identity(
    base: Any,
    patched: Any,
    state: AnnealState,
    probe: Any,
    base_vocab: int,
    tolerance: float = IDENTITY_TOLERANCE,
) -> IdentityResult:
    """Compare an unpatched ``base`` to a patched model at ``alpha=0`` on ``probe``.

    ``base`` must be un-patched and un-grown; ``patched`` is the patched (possibly
    grown) model whose ``_a2d_anneal`` is ``state``. Both are forced to CPU/eval so
    the gate is deterministic float32 (Risk 2).
    """
    import torch

    state.alpha = 0.0
    base = base.to("cpu").eval()
    patched = patched.to("cpu").eval()
    probe = probe.to("cpu")
    with torch.no_grad():
        base_logits = base(probe).logits.float()
        patched_logits = patched(probe).logits[:, :, :base_vocab].float()
    max_abs_diff = float((base_logits - patched_logits).abs().max().item())
    return IdentityResult(
        passed=max_abs_diff <= tolerance,
        max_abs_diff=max_abs_diff,
        tolerance=tolerance,
    )
