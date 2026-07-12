"""``attn.gqa`` transform: annealed bidirectional attention for the RoPE/GQA family.

Llama / Qwen2 / Gemma route causality through the 4D mask that ``_update_causal_mask``
builds (not GPT-2's per-layer ``self.bias``), so this handler installs the mask-seam
anneal patch. It covers GQA, MQA (Gemma's ``num_key_value_heads=1``), and full-attention
RoPE models alike, leaving RoPE, the KV-group expansion, RMSNorm, and Gemma's embedding
scaling to HF's own forward. The ``alpha=0`` identity gate and the ``alpha=1`` bidir test
together prove the patch is bit-identical to base yet genuinely reaches future tokens.
"""

from __future__ import annotations

from typing import Any

from a2d_core.transform.attention import AnnealState
from a2d_core.transform.gqa_attention import install_gqa_anneal_patch
from a2d_core.transform.handlers import register


@register("attn.gqa")
def install(model: Any, state: AnnealState) -> None:
    install_gqa_anneal_patch(model, state)
