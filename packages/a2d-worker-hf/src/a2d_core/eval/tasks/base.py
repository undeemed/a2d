"""Eval-task extension point: the ``Task`` shape + the ``EVAL_TASKS`` registry.

A task takes a ``TaskContext`` (the loaded converted model + tokenizer, already
bidirectional at ``alpha=1``) and returns a ``TaskScore``. New task => one file in
this package that self-registers, and it joins the harness matrix automatically
(SPEC-HANDOFF 3.3). Each task self-locates its bundled default fixture under
``_fixtures/`` unless ``TaskContext.data_overrides`` supplies a path, so the worker
is self-contained (no repo-relative path assumptions).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "_fixtures"


@dataclass
class TaskContext:
    """Everything a task needs; the model is already patched bidirectional (alpha=1)."""

    model: Any
    tokenizer: Any
    mask_token_id: int
    device: str
    seed: int
    max_examples: int
    # task name -> corpus path; missing => the task's bundled _fixtures default.
    data_overrides: dict[str, str]


@dataclass
class TaskScore:
    name: str
    metric: str
    value: float
    n: int


# A task is a callable TaskContext -> TaskScore, keyed by its name in the registry
# (defined in a2d_core.eval.tasks to match the DATA/OBJECTIVES/SAMPLERS convention).
Task = Callable[[TaskContext], TaskScore]


def load_jsonl(path: str, limit: int) -> list[dict[str, Any]]:
    """Read up to ``limit`` JSON objects, one per non-blank line."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def task_data_path(ctx: TaskContext, name: str, default_file: str) -> str:
    """The override path for ``name`` if given, else the bundled ``_fixtures`` default."""
    return ctx.data_overrides.get(name) or str(FIXTURES / default_file)
