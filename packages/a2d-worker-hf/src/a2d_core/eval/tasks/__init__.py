"""Extension point: eval tasks (SPEC-HANDOFF 3.3).

New eval task => one task module here. Tasks self-register in ``EVAL_TASKS`` and the
harness is parameterized over the registry, so a new task joins the matrix
automatically. Same registries-not-switches shape as ``data``/``objectives``/``sample``.
"""

from __future__ import annotations

from a2d_core.eval.tasks.base import Task
from a2d_core.registry import Registry

EVAL_TASKS: Registry[Task] = Registry("eval_task")
register = EVAL_TASKS.register

# Import submodules for their self-registration side effect (registry-table edit).
from a2d_core.eval.tasks import cloze_likelihood as cloze_likelihood  # noqa: E402
from a2d_core.eval.tasks import infill_accuracy as infill_accuracy  # noqa: E402
