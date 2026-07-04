"""Extension point: samplers / decoders (SPEC-HANDOFF 3.3).

Sampling strategies drop in here as self-registering modules, following the
registries-not-switches convention. P2 ships the MDLM denoiser; P5's BD3LM block
sampler is another module with zero edits here.
"""

from __future__ import annotations

from collections.abc import Callable

from a2d_core.registry import Registry

SAMPLERS: Registry[Callable[..., list[int]]] = Registry("sampler")
register = SAMPLERS.register

# Import submodules for their self-registration side effect (registry-table edit).
from a2d_core.sample import denoiser as denoiser  # noqa: E402
