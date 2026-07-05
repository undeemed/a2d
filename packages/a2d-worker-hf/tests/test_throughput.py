"""Step 4: diffusion vs AR decode throughput."""

from __future__ import annotations

import hashlib
from typing import Any

from a2d_core.eval.throughput import measure_throughput
from conftest import ConvertSetup


def _converted(setup: ConvertSetup) -> tuple[Any, int]:
    from a2d_core.transform.apply import load_model, resolve_mask_token
    from a2d_core.transform.attention import AnnealState, install_anneal_patch

    model, tokenizer = load_model(str(setup.model_src))
    mask_id = resolve_mask_token(model, tokenizer, "grow")
    install_anneal_patch(model, AnnealState(alpha=1.0))
    return model, mask_id


def test_throughput_both_finite_with_source(convert_setup: ConvertSetup) -> None:
    model, mask_id = _converted(convert_setup)
    digest = hashlib.sha256(
        (convert_setup.model_src / "model.safetensors").read_bytes()
    ).hexdigest()
    result = measure_throughput(
        model,
        mask_id,
        str(convert_setup.model_src),
        digest,
        prompt_ids=[1, 2, 3],
        canvas_len=10,
        num_steps=4,
        device="cpu",
    )
    assert result.diffusion_tokens_per_sec > 0
    assert result.ar_tokens_per_sec is not None and result.ar_tokens_per_sec > 0
    assert result.speedup is not None and result.speedup > 0
    assert result.num_steps == 4


def test_throughput_ar_none_when_source_absent(convert_setup: ConvertSetup) -> None:
    model, mask_id = _converted(convert_setup)
    result = measure_throughput(
        model,
        mask_id,
        None,
        None,
        prompt_ids=[1, 2, 3],
        canvas_len=10,
        num_steps=4,
        device="cpu",
    )
    assert result.diffusion_tokens_per_sec > 0
    assert result.ar_tokens_per_sec is None and result.speedup is None
