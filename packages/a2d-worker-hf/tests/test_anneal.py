from __future__ import annotations

import math

import pytest
import torch
from a2d_core.transform.attention import AnnealState, annealed_additive_mask, schedule


def test_schedule_endpoints_and_monotone() -> None:
    n = 10
    assert schedule(0, n) == 0.0
    assert schedule(n, n) == 1.0
    assert schedule(3 * n, n) == 1.0  # clamps past the anneal window
    prev = -1.0
    for step in range(n + 1):
        alpha = schedule(step, n)
        assert 0.0 <= alpha <= 1.0
        assert alpha >= prev  # monotone non-decreasing
        prev = alpha


def test_schedule_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        schedule(1, 10, "cosine")


def test_state_defaults_to_causal() -> None:
    assert AnnealState().alpha == 0.0


def test_mask_at_alpha0_matches_causal_finfo_min() -> None:
    finfo_min = torch.finfo(torch.float32).min
    q = k = 5
    mask = annealed_additive_mask(q, k, 0.0, torch.float32, torch.device("cpu"))
    assert mask.shape == (q, k)
    causal = torch.tril(torch.ones(q, k, dtype=torch.bool))
    # on/below diagonal == 0; strictly-future == finfo.min (the base causal pattern)
    assert bool(torch.all(mask[causal] == 0.0))
    assert bool(torch.all(mask[~causal] == finfo_min))


def test_mask_at_alpha1_is_fully_bidirectional() -> None:
    mask = annealed_additive_mask(4, 4, 1.0, torch.float32, torch.device("cpu"))
    assert bool(torch.all(mask == 0.0))


def test_mask_intermediate_alpha_is_log_penalty() -> None:
    alpha = 0.3
    mask = annealed_additive_mask(3, 3, alpha, torch.float32, torch.device("cpu"))
    causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    assert bool(torch.all(mask[causal] == 0.0))
    assert torch.allclose(mask[~causal], torch.tensor(math.log(alpha), dtype=torch.float32))
