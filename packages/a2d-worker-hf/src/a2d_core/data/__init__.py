"""Extension point: data pipelines (SPEC-HANDOFF 3.3).

Local-file readers self-register in ``DATA`` (registries-not-switches). Each reader
takes ``(path, tokenizer, seq_len)`` and returns a torch ``Dataset`` of fixed-length
token chunks: tokenize every document, concatenate with an eos separator (GPT-2's
packing convention), and slice into ``seq_len`` blocks (drop the ragged tail). No
``datasets`` dependency (Decision 1) - a ~40-line ``Dataset`` feeds ``Trainer`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

from a2d_core.registry import Registry

# reader: (path, tokenizer, seq_len) -> Dataset of {"input_ids": LongTensor[seq_len]}
DataReader = Callable[[str, Any, int], "Dataset[dict[str, torch.Tensor]]"]

DATA: Registry[DataReader] = Registry("data")
register = DATA.register


class ChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length token chunks; each item is ``{"input_ids": LongTensor[seq_len]}``."""

    def __init__(self, chunks: torch.Tensor) -> None:
        self.chunks = chunks

    def __len__(self) -> int:
        return int(self.chunks.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.chunks[index]}


def concat_and_chunk(texts: list[str], tokenizer: Any, seq_len: int) -> ChunkDataset:
    """Tokenize + concat (eos-separated) + slice into ``seq_len`` blocks."""
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer(text)["input_ids"])
        if eos is not None:
            ids.append(int(eos))  # document separator (GPT-2 packing convention)
    usable = (len(ids) // seq_len) * seq_len
    if usable == 0:
        raise ValueError(f"corpus too small: {len(ids)} tokens < seq_len {seq_len}")
    chunks = torch.tensor(ids[:usable], dtype=torch.long).view(-1, seq_len)
    return ChunkDataset(chunks)


# Import submodules for their self-registration side effect (registry-table edit).
from a2d_core.data import jsonl as jsonl  # noqa: E402
from a2d_core.data import txt as txt  # noqa: E402
