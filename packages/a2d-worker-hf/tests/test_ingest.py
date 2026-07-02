from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from a2d_core.ingest import INGEST
from a2d_core.ingest.safetensors import normalize


def _write_safetensors(path: Path, header: bytes) -> None:
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_discoverable_via_registry() -> None:
    assert "safetensors" in INGEST.names()
    assert INGEST.get("safetensors") is normalize


def test_valid_header_passes_through(tmp_path: Path) -> None:
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    _write_safetensors(tmp_path / "model.safetensors", header)
    assert normalize(tmp_path) is tmp_path


def test_invalid_header_rejected(tmp_path: Path) -> None:
    _write_safetensors(tmp_path / "model.safetensors", b"not json{")
    with pytest.raises(ValueError):
        normalize(tmp_path)


def test_empty_dir_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        normalize(tmp_path)
