from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from a2d_core.transform.attention import AnnealState, install_anneal_patch


def test_future_token_reaches_earlier_positions_only_when_bidirectional(
    tiny_gpt2: Callable[..., Any],
) -> None:
    """GUARD (Decision 2): the only proof the patch actually reaches GPT-2 causality.

    Perturbing a strictly-future token must MOVE an earlier position's logits at
    alpha=1 (bidirectional) and must NOT move them at alpha=0 (causal). The alpha=0
    identity gate passes even for a no-op seam, so this is the real regression guard.
    """
    model = tiny_gpt2(0)
    state = AnnealState()
    install_anneal_patch(model, state)

    ids = torch.randint(0, 64, (1, 8))
    perturbed = ids.clone()
    future_pos, earlier_pos = 6, 2
    perturbed[:, future_pos] = (perturbed[:, future_pos] + 1) % 64

    # alpha=0: causal, so an earlier position cannot see the future token.
    state.alpha = 0.0
    with torch.no_grad():
        causal_earlier = model(ids).logits[:, earlier_pos, :]
        causal_earlier_perturbed = model(perturbed).logits[:, earlier_pos, :]
    assert torch.equal(causal_earlier, causal_earlier_perturbed)

    # alpha=1: bidirectional, so an earlier position attends the perturbed future token.
    state.alpha = 1.0
    with torch.no_grad():
        bidir_earlier = model(ids).logits[:, earlier_pos, :]
        bidir_earlier_perturbed = model(perturbed).logits[:, earlier_pos, :]
    assert not torch.equal(bidir_earlier, bidir_earlier_perturbed)
    assert float((bidir_earlier - bidir_earlier_perturbed).abs().max().item()) > 1e-6
