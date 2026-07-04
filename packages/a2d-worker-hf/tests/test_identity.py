from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from a2d_core.transform.apply import grow_embeddings
from a2d_core.transform.attention import AnnealState, install_anneal_patch
from a2d_core.transform.identity import IDENTITY_TOLERANCE, check_identity


def test_patched_at_alpha0_is_bit_identical_to_base(tiny_gpt2: Callable[..., Any]) -> None:
    """HEADLINE gate: patched@alpha=0 logits (sliced to base_vocab) equal base to 0.0."""
    base = tiny_gpt2(0)
    patched = tiny_gpt2(0)
    patched.load_state_dict(base.state_dict())  # guarantee identical weights
    base_vocab = int(base.config.vocab_size)
    grow_embeddings(patched, base_vocab + 1)  # exercise the grown-column slice (Risk 3)
    grown = patched.get_input_embeddings().weight
    assert torch.allclose(grown[base_vocab], grown[:base_vocab].mean(dim=0))  # Decision 7 mean-init
    state = AnnealState()
    install_anneal_patch(patched, state)

    probe = torch.randint(0, base_vocab, (2, 8))
    result = check_identity(base, patched, state, probe, base_vocab)

    assert result.passed
    assert result.tolerance == IDENTITY_TOLERANCE
    assert result.max_abs_diff <= IDENTITY_TOLERANCE
    assert result.max_abs_diff == 0.0  # eager + fp32 is exact, not merely within tolerance
