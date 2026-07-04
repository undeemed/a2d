"""``AnnealCallback``: drive ``alpha`` per step + emit TrainStep/Checkpoint events.

The callback is STATELESS about the anneal (Decision 4): on every ``on_step_begin``
it recomputes ``AnnealState.alpha`` from the current global step, so ``resume`` needs
no persisted anneal state - a restored step yields the same alpha. It also mirrors
the ``Trainer`` loop onto the worker event stream: a ``TrainStep`` per logged step
(``logging_steps=1``) and a ``Checkpoint`` per ``Trainer`` save.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from transformers import TrainerCallback

from a2d_core.transform.attention import AnnealState, schedule

# The worker passes an emit that wraps each event dict into an EventEnvelope line;
# tests pass a list.append. Kept as a plain dict so callbacks never import worker.py.
Emit = Callable[[dict[str, Any]], None]


class AnnealCallback(TrainerCallback):  # type: ignore[misc]
    def __init__(
        self,
        state: AnnealState,
        anneal_steps: int,
        anneal_schedule: str,
        tokens_per_step: int,
        emit: Emit,
    ) -> None:
        self.state = state
        self.anneal_steps = anneal_steps
        self.anneal_schedule = anneal_schedule
        self.tokens_per_step = tokens_per_step
        self.emit = emit

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.state.alpha = schedule(state.global_step, self.anneal_steps, self.anneal_schedule)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or "loss" not in logs:  # skip non-training logs (e.g. eval/final)
            return
        self.emit(
            {
                "type": "train_step",
                "step": state.global_step,
                "loss": float(logs["loss"]),
                "anneal": self.state.alpha,
                "lr": float(logs.get("learning_rate", 0.0)),
                "tokens": state.global_step * self.tokens_per_step,
            }
        )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.emit({"type": "checkpoint", "step": state.global_step, "path": path})
