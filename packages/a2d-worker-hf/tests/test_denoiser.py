"""Step 10: the MDLM denoiser fills every mask, keeps the prompt, correct length."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from a2d_core.sample import SAMPLERS
from a2d_core.transform.apply import grow_embeddings


def _denoise_and_check(model: Any) -> None:
    base_vocab = int(model.config.vocab_size)  # 64
    mask_token_id = base_vocab
    grow_embeddings(model, base_vocab + 1)  # +1 row so the mask id has an embedding

    prompt_ids = [1, 2, 3, 4]
    canvas_len = 10
    result = SAMPLERS.get("mdlm")(
        model,
        prompt_ids=prompt_ids,
        mask_token_id=mask_token_id,
        canvas_len=canvas_len,
        num_steps=5,
        temperature=1.0,
        device="cpu",
    )

    assert len(result) == canvas_len  # correct length
    assert result[: len(prompt_ids)] == prompt_ids  # prompt prefix untouched
    assert mask_token_id not in result  # no mask id remains


def test_denoise_fills_masks_keeps_prompt(tiny_gpt2: Callable[..., Any]) -> None:
    _denoise_and_check(tiny_gpt2(0))


@pytest.mark.parametrize("family", ["gemma", "llama", "qwen2"])
def test_denoise_works_on_rope_family(tiny_gqa: Callable[..., Any], family: str) -> None:
    """The sampler consumes RoPE-family run dirs too: capability dispatch resolves the
    attn.gqa seam instead of hard-failing on the GPT-2-only patch."""
    _denoise_and_check(tiny_gqa(family, 0))
