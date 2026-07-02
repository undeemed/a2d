"""safetensors ingest normalizer: stdlib-only, dependency-free pass-through.

safetensors is already the canonical on-disk weights format, so this normalizer
does not rewrite anything: it validates each ``*.safetensors`` file's header and
returns the directory unchanged. The header check is the documented safetensors
framing - an 8-byte little-endian ``u64`` length prefix followed by that many
bytes of a UTF-8 JSON object - so no torch, numpy, or safetensors dependency is
needed to reject a truncated or non-safetensors file.

PHASE 2: the ``pickle -> safetensors`` normalizer (a named SPEC 6 Phase-1 ingest
format) is deferred - it needs torch to load pickle tensors and is a code-exec
surface, both banned until Phase 2 (see PLAN-PHASE1 Decision 6).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from a2d_core.ingest import register

_LEN_PREFIX_BYTES = 8
# Real safetensors headers are a few KB; cap at 100 MB so a corrupt length prefix
# is rejected instead of triggering a huge read.
# ponytail: fixed ceiling; raise it only if a real model ships a larger header.
_MAX_HEADER_BYTES = 100_000_000


def validate_header(path: Path) -> None:
    """Raise ``ValueError`` unless ``path`` starts with a well-formed safetensors header."""
    with path.open("rb") as fh:
        prefix = fh.read(_LEN_PREFIX_BYTES)
        if len(prefix) != _LEN_PREFIX_BYTES:
            raise ValueError(f"{path}: truncated safetensors (missing 8-byte header length)")
        header_len = int(struct.unpack("<Q", prefix)[0])
        if not 0 < header_len <= _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible safetensors header length {header_len}")
        raw = fh.read(header_len)
    if len(raw) != header_len:
        raise ValueError(f"{path}: safetensors header truncated ({len(raw)}/{header_len} bytes)")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not a JSON object")


@register("safetensors")
def normalize(src: Path) -> Path:
    """Validate every ``*.safetensors`` file in ``src`` and pass the directory through."""
    files = sorted(src.glob("*.safetensors"))
    if not files:
        raise ValueError(f"ingest.safetensors: no *.safetensors files in {src}")
    for f in files:
        validate_header(f)
    return src
