"""Local plain-text reader: the whole file is one document."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from a2d_core.data import concat_and_chunk, register


@register("txt")
def read_txt(path: str, tokenizer: Any, seq_len: int) -> Dataset[dict[str, torch.Tensor]]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return concat_and_chunk([text], tokenizer, seq_len)
