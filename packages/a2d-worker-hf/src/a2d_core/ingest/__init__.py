"""Extension point: weight-format normalizers (SPEC-HANDOFF 3.3).

New weights format => drop one normalizer module here. Normalizers map an
on-disk weights layout into the canonical form the pipeline expects and
self-register via a ``Registry``; there is no central switch to edit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from a2d_core.registry import Registry

# One normalizer per weights format: maps a model dir to a canonical model dir.
Normalizer = Callable[[Path], Path]

INGEST: Registry[Normalizer] = Registry("ingest")
register = INGEST.register

# Import submodules for their self-registration side effect (registry-table edit).
from a2d_core.ingest import safetensors as safetensors  # noqa: E402
