"""Extension point: transform handlers (SPEC-HANDOFF 3.3).

New attention variant => one handler module here, plus a capability tag and a
conformance test. Handlers self-register via a ``Registry`` and must leave
``anneal=0`` behavior identical to the base model (identity / golden-logits test).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from a2d_core.registry import Registry
from a2d_core.transform.attention import AnnealState

# A transform handler installs one in-place model modification for a capability,
# wiring it to the shared AnnealState (Any = an untyped transformers PreTrainedModel).
TransformHandler = Callable[[Any, AnnealState], None]

TRANSFORM: Registry[TransformHandler] = Registry("transform")
register = TRANSFORM.register

# Import submodules for their self-registration side effect (registry-table edit).
from a2d_core.transform.handlers import full_attention as full_attention  # noqa: E402
from a2d_core.transform.handlers import gqa_attention as gqa_attention  # noqa: E402
from a2d_core.transform.handlers import swa_attention as swa_attention  # noqa: E402
