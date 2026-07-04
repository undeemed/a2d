"""Local JSONL reader: one JSON object per line, tokenize its ``"text"`` field."""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset

from a2d_core.data import concat_and_chunk, register


@register("jsonl")
def read_jsonl(path: str, tokenizer: Any, seq_len: int) -> Dataset[dict[str, torch.Tensor]]:
    texts: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
    return concat_and_chunk(texts, tokenizer, seq_len)
