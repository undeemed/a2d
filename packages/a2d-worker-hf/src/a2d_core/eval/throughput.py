"""Decode throughput: diffusion denoiser vs source-AR greedy (Decision 4).

Diffusion tokens/sec reuses ``sample.SAMPLERS["mdlm"]`` (revealed canvas tokens / wall
seconds at ``num_steps``); AR tokens/sec greedy-decodes the same budget on the source
model. MDLM runs ``num_steps`` full forward passes, so ``speedup`` is honestly below 1 -
the block-parallel payoff is BD3LM (P5). Each measurement warms once before timing.
AR is best-effort: None when the source is absent or its hash drifts (shared policy).
"""

from __future__ import annotations

import time
from typing import Any

import torch
from a2d_contracts import ThroughputResult

from a2d_core.eval.likelihood import source_reason
from a2d_core.sample import SAMPLERS


def _diffusion_tokens_per_sec(
    model: Any,
    mask_token_id: int,
    *,
    prompt_ids: list[int],
    canvas_len: int,
    num_steps: int,
    device: str,
) -> float:
    sampler = SAMPLERS.get("mdlm")
    kwargs = dict(
        prompt_ids=prompt_ids,
        mask_token_id=mask_token_id,
        canvas_len=canvas_len,
        num_steps=num_steps,
        temperature=1.0,
        device=device,
    )
    sampler(model, **kwargs)  # warm
    n_new = canvas_len - len(prompt_ids)
    start = time.perf_counter()
    sampler(model, **kwargs)
    elapsed = time.perf_counter() - start
    return n_new / elapsed if elapsed > 0 else 0.0


def _ar_tokens_per_sec(
    source_model: str | None,
    source_hash: str | None,
    *,
    prompt_ids: list[int],
    n_new: int,
    device: str,
) -> float | None:
    if source_reason(source_model, source_hash) is not None:
        return None
    from transformers import AutoModelForCausalLM

    dev = torch.device(device)
    model = (
        AutoModelForCausalLM.from_pretrained(source_model, attn_implementation="eager")
        .to(dev)
        .eval()
    )
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    pad = model.config.eos_token_id  # GPT-2 has no pad; silence generate's warning
    gen = dict(max_new_tokens=n_new, do_sample=False, pad_token_id=pad, use_cache=True)
    with torch.no_grad():
        model.generate(inp, **gen)  # warm
        start = time.perf_counter()
        model.generate(inp, **gen)
        elapsed = time.perf_counter() - start
    return n_new / elapsed if elapsed > 0 else 0.0


def measure_throughput(
    model: Any,
    mask_token_id: int,
    source_model: str | None,
    source_hash: str | None,
    *,
    prompt_ids: list[int],
    canvas_len: int,
    num_steps: int,
    device: str,
) -> ThroughputResult:
    """Diffusion vs AR decode throughput over an equal new-token budget."""
    n_new = canvas_len - len(prompt_ids)
    diffusion = _diffusion_tokens_per_sec(
        model,
        mask_token_id,
        prompt_ids=prompt_ids,
        canvas_len=canvas_len,
        num_steps=num_steps,
        device=device,
    )
    ar = _ar_tokens_per_sec(
        source_model, source_hash, prompt_ids=prompt_ids, n_new=n_new, device=device
    )
    speedup = diffusion / ar if ar else None  # ar is None or 0.0 => no speedup
    return ThroughputResult(
        diffusion_tokens_per_sec=diffusion,
        ar_tokens_per_sec=ar,
        speedup=speedup,
        num_steps=num_steps,
    )
