"""Device and dtype selection for the worker.

torch is imported lazily inside each function so the contract-violation exit-2
path (which never selects a device) stays torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def select_device(override: str = "auto") -> str:
    """Resolve a device string. "auto" picks cuda > mps > cpu; anything else
    is returned as-is (the caller validated it against the wire enum)."""
    if override != "auto":
        return override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(dtype: str = "float32") -> torch.dtype:
    """Map the wire dtype string to a torch dtype. float32 is the default;
    bfloat16 is opt-in (MPS float16 has precision quirks - see Risk 11)."""
    import torch

    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    return dtypes[dtype]
