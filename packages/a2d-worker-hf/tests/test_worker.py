"""Worker protocol tests: the convert pipeline streams the SPEC-4.2 event sequence,
and the torch-free contract-violation paths still exit 2 before any processing."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from importlib.metadata import version
from typing import Any


def _run(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "a2d_core.worker"],
        input=stdin,
        capture_output=True,
        text=True,
    )


def _bad_path_job() -> dict[str, Any]:
    """A well-formed job pointing at a nonexistent model - enough for the exit-2
    (contract) paths, which reject before the pipeline ever reads the path."""
    return {
        "schema_version": version("a2d-contracts"),
        "job_id": str(uuid.uuid4()),
        "model_path": "/tmp/does-not-exist",
        "run_dir": "runs/demo",
        "conversion_config": {
            "objective": "mdlm",
            "data": "data.jsonl",
            "anneal_steps": 0,
            "anneal_schedule": "linear",
            "seq_len": 512,
            "per_device_batch_size": 8,
            "grad_accum": 1,
            "lr": 1e-4,
            "max_steps": 1,
            "mask_token": "grow",
            "keep_last": 3,
            "seed": 0,
            "device": "cpu",
            "dtype": "float32",
        },
    }


def test_valid_job_streams_pipeline_events(convert_setup: Any) -> None:
    """The real pipeline emits job_started -> progress -> identity_gate -> train_step
    -> checkpoint -> job_completed (the no-op ``log`` is gone)."""
    result = _run(convert_setup.build_job())
    assert result.returncode == 0, result.stderr

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "worker emitted no envelopes"

    types: list[str] = []
    for i, line in enumerate(lines):
        env = json.loads(line)
        assert env["schema_version"]
        assert env["job_id"]
        assert env["ts"]
        assert env["seq"] == i  # monotonic from 0
        types.append(env["event"]["type"])

    assert types[0] == "job_started"
    assert types[-1] == "job_completed"
    for kind in ("progress", "identity_gate", "train_step", "checkpoint"):
        assert kind in types, f"pipeline never emitted {kind}: {types}"
    assert "log" not in types  # the old no-op event is gone

    gate = next(json.loads(line)["event"] for line in lines if '"identity_gate"' in line)
    assert gate["passed"] is True


def test_garbage_stdin_is_contract_violation() -> None:
    result = _run("this is not json")
    assert result.returncode == 2
    assert result.stdout.strip() == ""  # no envelopes before validation fails


def test_incompatible_major_is_contract_violation() -> None:
    job = _bad_path_job()
    job["schema_version"] = "1.0.0"  # incompatible major vs the 0.x contract
    result = _run(json.dumps(job))
    assert result.returncode == 2
    assert result.stdout.strip() == ""  # version gate rejects before emitting envelopes
