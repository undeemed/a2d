"""The ``Objective`` protocol: a ``corrupt``/``loss`` pair (SPEC-HANDOFF 3.3).

An objective owns exactly two things: how a batch of clean examples is noised
into a model-ready batch (``corrupt``, used as the ``Trainer`` data collator), and
how the loss is scored from the model's raw logits (``loss``, called by the
``compute_loss`` override with NO label shift). Everything else - optimizer, LR
schedule, checkpointing - is HF ``Trainer``'s job (Decision 1/8).
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


class Objective(Protocol):
    def corrupt(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate + noise a list of examples into a model-ready batch, stashing the
        clean ids under ``"clean"`` and the masked-position weight under ``"mask"``."""
        ...

    def loss(self, logits: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scalar loss from raw (unshifted) ``logits`` against ``clean`` targets,
        scored over masked positions only (``mask``); unmasked positions ignored."""
        ...
