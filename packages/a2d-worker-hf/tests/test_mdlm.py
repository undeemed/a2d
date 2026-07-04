from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from a2d_core.objectives import OBJECTIVES
from a2d_core.objectives.mdlm import MDLM


def test_registered_under_mdlm() -> None:
    assert OBJECTIVES.get("mdlm") is MDLM


def test_t1_all_masked_reduces_to_plain_ce() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    clean = torch.randint(0, 7, (2, 5))
    mask = torch.ones(2, 5)  # weight 1 == t=1: Bernoulli(1) masks every position
    got = MDLM.loss(logits, clean, mask)
    expected = F.cross_entropy(logits.reshape(-1, 7), clean.reshape(-1))
    assert torch.allclose(got, expected, atol=1e-6)


def test_unmasked_and_ignore_index_excluded() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 5)
    clean = torch.tensor([[1, 2, 3, 0]])
    # pos 0 unmasked (weight 0); pos 3 masked but target -100 -> ignored; only 1,2 count.
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    clean_ignored = clean.clone()
    clean_ignored[0, 3] = -100
    got = MDLM.loss(logits, clean_ignored, mask)
    expected = F.cross_entropy(logits[0, 1:3], clean[0, 1:3])
    assert torch.allclose(got, expected, atol=1e-6)


def test_hand_checked_scalar() -> None:
    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])  # [1, 2, 3]
    clean = torch.tensor([[0, 2]])
    mask = torch.tensor([[2.0, 2.0]])  # t=0.5 -> weight 1/t = 2
    # nll0 = -log(softmax([2,1,0])[0]) = 0.407605; nll1 = -log(1/3) = 1.098612
    # loss = (2*0.407605 + 2*1.098612) / 2 = 1.506217
    got = MDLM.loss(logits, clean, mask)
    assert math.isclose(float(got), 1.506217, abs_tol=1e-5)


def test_corrupt_masks_and_preserves_clean() -> None:
    obj = MDLM(mask_token_id=99, seed=0)
    batch = [{"input_ids": torch.arange(6)}, {"input_ids": torch.arange(6, 12)}]
    out = obj.corrupt(batch)
    assert out["input_ids"].shape == (2, 6)
    assert torch.equal(out["clean"], torch.stack([torch.arange(6), torch.arange(6, 12)]))
    masked = out["mask"] > 0
    assert torch.all(out["input_ids"][masked] == 99)  # masked -> mask token
    assert torch.all(out["input_ids"][~masked] == out["clean"][~masked])  # rest preserved
    assert torch.all(out["mask"][~masked] == 0.0)  # unmasked weight 0
    assert torch.all(out["mask"][masked] >= 1.0)  # masked weight 1/t, t in (0,1]
