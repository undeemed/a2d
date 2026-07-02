"""Worker side of the a2d CLI<->worker process protocol (SPEC-HANDOFF).

Read ONE ``ConversionJob`` JSON document from stdin, validate it, then emit
``EventEnvelope`` objects to stdout as compact JSONL (one per line, flushed per
line): ``job_started`` -> ``log`` -> ``job_completed``. STDOUT carries envelopes
only; all diagnostics go to STDERR.

Exit codes: 0 = success (job_completed emitted); 1 = job failed during execution
(job_failed emitted first); 2 = contract violation (unparseable/invalid job).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.metadata import version
from typing import TextIO

from a2d_contracts import ConversionJob, EventEnvelope
from pydantic import ValidationError

from . import __version__

# schema_version tracks the CONTRACT (a2d-contracts), not this worker package, so
# it mirrors the Rust side's a2d_contracts::SCHEMA_VERSION (= a2d-contracts crate
# version). Sourcing it from __version__ would drift the moment a2d-worker-hf is
# bumped independently.
SCHEMA_VERSION = version("a2d-contracts")
WORKER = f"a2d-worker-hf {__version__}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


# Event builders: the single place the wire-shape names for each event kind live.
def job_started() -> dict[str, str]:
    return {"type": "job_started", "worker": WORKER}


def log(message: str, level: str = "info") -> dict[str, str]:
    return {"type": "log", "level": level, "message": message}


def job_completed() -> dict[str, str]:
    return {"type": "job_completed"}


def job_failed(message: str) -> dict[str, str]:
    return {"type": "job_failed", "message": message}


def _emit(job_id: str, seq: int, event: dict[str, str], out: TextIO) -> None:
    envelope = EventEnvelope.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "seq": seq,
            "ts": _now(),
            "event": event,
        }
    )
    out.write(envelope.model_dump_json() + "\n")
    out.flush()


def main() -> int:
    raw = sys.stdin.read()
    try:
        job = ConversionJob.model_validate_json(raw)
    except ValidationError as exc:
        print(f"a2d-worker: invalid ConversionJob: {exc}", file=sys.stderr)
        return 2

    if job.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        print(
            "a2d-worker: schema_version major mismatch: "
            f"job={job.schema_version!r} worker={SCHEMA_VERSION!r}",
            file=sys.stderr,
        )
        return 2

    out = sys.stdout
    events = [
        job_started(),
        log(f"no-op conversion of {job.model_path}"),
        job_completed(),
    ]
    seq = 0
    try:
        for event in events:
            _emit(job.job_id, seq, event, out)
            seq += 1
    except Exception as exc:  # noqa: BLE001 - report as job_failed then exit 1
        print(f"a2d-worker: conversion failed: {exc}", file=sys.stderr)
        try:
            _emit(job.job_id, seq, job_failed(str(exc)), out)
        except Exception:  # noqa: BLE001 - stdout already broken; stderr has it
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
