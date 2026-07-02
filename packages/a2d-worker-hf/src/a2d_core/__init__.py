"""a2d-worker-hf reference conversion worker (module ``a2d_core``)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("a2d-worker-hf")
except PackageNotFoundError:  # running from a raw checkout, not installed
    __version__ = "0.1.0"
