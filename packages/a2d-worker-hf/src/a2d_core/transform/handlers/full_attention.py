"""``attn.full`` transform: install the annealed bidirectional attention patch.

GPT-2 dense causal attention -> annealed causal->bidirectional via the eager-seam
patch (Decision 2). The ``alpha=0`` identity gate and ``test_bidir`` together prove
the patch is both bit-identical to base and genuinely reaches GPT-2's causality.
"""

from __future__ import annotations

from typing import Any

from a2d_core.transform.attention import AnnealState, install_anneal_patch
from a2d_core.transform.handlers import register


@register("attn.full")
def install(model: Any, state: AnnealState) -> None:
    install_anneal_patch(model, state)
