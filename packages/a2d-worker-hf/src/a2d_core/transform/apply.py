"""Load a causal LM, resolve/grow its mask token, and apply capability transforms.

The source dir is only READ (Decision 6): ``from_pretrained`` loads into memory and
``save_pretrained`` later writes ``run_dir/model/`` fresh, so the source is never
mutated. Torch/transformers imports stay lazy so the worker's contract-violation
exit-2 path never pulls them in.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from a2d_core.transform.attention import AnnealState
from a2d_core.transform.handlers import TRANSFORM

# Grown mask token: distinct from GPT-2's eos (50256) so packing separators are
# never confused with a to-be-predicted mask (Decision 7).
MASK_TOKEN = "<|mdlm_mask|>"


def load_model(model_dir: str | Path, dtype: str = "float32") -> tuple[Any, Any]:
    """Load model (eager attention, given dtype) + tokenizer from a local dir."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from a2d_core.device import select_dtype

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), attn_implementation="eager", torch_dtype=select_dtype(dtype)
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tokenizer


def grow_embeddings(model: Any, new_num_tokens: int) -> None:
    """Resize to ``new_num_tokens`` and init each appended row = mean of existing rows.

    GPT-2 ties ``wte``<->``lm_head`` so one resize covers both (Decision 7). We
    disable transformers' default random mean-resizing and set the deterministic
    mean the plan specifies.
    """
    import torch

    old = int(model.get_input_embeddings().weight.shape[0])
    if new_num_tokens <= old:
        return
    model.resize_token_embeddings(new_num_tokens, mean_resizing=False)
    with torch.no_grad():
        weight = model.get_input_embeddings().weight
        weight[old:] = weight[:old].mean(dim=0, keepdim=True)


def resolve_mask_token(model: Any, tokenizer: Any, strategy: str = "grow") -> int:
    """Resolve the MDLM mask token id, growing the vocab by one row for ``"grow"``."""
    if strategy == "reuse":
        # ponytail: eos-as-mask conflates the doc separator with a mask, so it is only
        # safe for non-packed data; hence opt-in with this known ceiling (Decision 7).
        tokenizer.mask_token = tokenizer.eos_token
        return int(tokenizer.eos_token_id)
    if strategy != "grow":
        raise ValueError(f"unknown mask-token strategy {strategy!r}")
    if tokenizer.add_special_tokens({"mask_token": MASK_TOKEN}):
        grow_embeddings(model, len(tokenizer))
    return int(tokenizer.mask_token_id)


def apply_transforms(model: Any, capabilities: Iterable[str], state: AnnealState) -> None:
    """Install every registered transform whose capability the model carries.

    Capabilities without a handler (e.g. ``pos.learned``, ``ffn.dense``) are inherent
    no-ops; the Phase-1 gate already rejected the unconvertible set.
    """
    registered = set(TRANSFORM.names())
    for capability in capabilities:
        if capability in registered:
            TRANSFORM.get(capability)(model, state)
