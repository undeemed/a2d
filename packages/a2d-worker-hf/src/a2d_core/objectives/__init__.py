"""Extension point: training objectives (SPEC-HANDOFF 3.3).

New objective (post-MDLM / BD3LM research) => one module here implementing
``corrupt`` and ``loss``. Objectives self-register via a ``Registry``; there is
no central ``match`` to edit.
"""

from __future__ import annotations

from a2d_core.objectives.base import Objective
from a2d_core.registry import Registry

OBJECTIVES: Registry[type[Objective]] = Registry("objective")
register = OBJECTIVES.register

# Import submodules for their self-registration side effect (registry-table edit).
from a2d_core.objectives import mdlm as mdlm  # noqa: E402
