"""MDLM objective: masked-diffusion corrupt + t-reweighted CE loss (Decision 1).

``corrupt`` samples a per-sequence diffusion time ``t ~ U(0,1]`` (the only mask
schedule MDLM needs, Decision 4), masks each token with probability ``t``, and
stashes the clean ids plus a per-position weight (``1/t`` at masked positions, ``0``
elsewhere) so ``loss`` can score masked positions only without a separate ``t``.
``loss`` is plain cross-entropy on the raw logits at the SAME index (no AR shift,
Decision 3), summed over masked positions and weighted by ``1/t``.

# ponytail: MDLM as weighted-MLM ~200 lines; swap for dllm.objectives.MDLM behind
# this same corrupt/loss protocol if P5 BD3LM justifies the dep
"""

from __future__ import annotations

from typing import Any

import torch

from a2d_core.objectives import register

# HF/torch convention: cross-entropy ignore index (e.g. padding), never a real id.
IGNORE_INDEX = -100


class MDLM:
    def __init__(self, mask_token_id: int, seed: int = 0) -> None:
        self.mask_token_id = mask_token_id
        # CPU generator: the collator runs before Trainer moves the batch to device.
        self.generator = torch.Generator().manual_seed(seed)

    def corrupt(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack(
            [torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        )
        # t in (0,1] per sequence (1 - U[0,1) so t is never 0, so 1/t is finite).
        t = 1.0 - torch.rand(input_ids.size(0), 1, generator=self.generator)
        masked = torch.rand(input_ids.shape, generator=self.generator) < t  # Bernoulli(t)
        noisy = input_ids.masked_fill(masked, self.mask_token_id)
        weight = torch.where(masked, 1.0 / t, torch.zeros(())).to(torch.float32)
        return {"input_ids": noisy, "clean": input_ids, "mask": weight}

    @staticmethod
    def loss(logits: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits.float(), dim=-1)
        # clamp -100 to a valid index for the gather; those positions are dropped below.
        nll = -logp.gather(-1, clean.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        weight = mask.to(nll.dtype)
        valid = (weight > 0) & (clean != IGNORE_INDEX)
        contrib = torch.where(valid, weight * nll, torch.zeros((), dtype=nll.dtype))
        return contrib.sum() / valid.sum().clamp_min(1)


register("mdlm")(MDLM)  # statement form: decorator would retype MDLM to type[Objective]
