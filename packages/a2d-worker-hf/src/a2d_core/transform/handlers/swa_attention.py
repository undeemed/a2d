"""``attn.swa`` transform: annealed bidirectional attention for the sliding-window
Gemma family (Gemma 3).

Gemma 3 routes causality through the same model-level 4D mask as the RoPE/GQA family
(``_update_causal_mask``) but ADDS a per-layer sliding window on its local layers.
This handler installs the two-part SWA anneal: the shared future-reveal on the full
causal mask plus a far-past reveal on every sliding decoder layer, so at ``alpha=0``
the model is bit-identical to base (identity gate) and at ``alpha=1`` attention is
fully non-causal AND unwindowed. RoPE, qk-norm, the KV-group layout, RMSNorm, and
embedding scaling are left to HF's own forward.
"""

from __future__ import annotations

from typing import Any

from a2d_core.transform.attention import AnnealState
from a2d_core.transform.handlers import register
from a2d_core.transform.swa_attention import install_swa_anneal_patch


@register("attn.swa")
def install(model: Any, state: AnnealState) -> None:
    install_swa_anneal_patch(model, state)
