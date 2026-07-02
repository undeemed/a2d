from __future__ import annotations

import json
import subprocess
import sys
import uuid


def _valid_job() -> str:
    return json.dumps(
        {
            "schema_version": "0.1.0",
            "job_id": str(uuid.uuid4()),
            "model_path": "/tmp/fake-model",
            "run_dir": "runs/demo",
        }
    )


def _run(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "a2d_core.worker"],
        input=stdin,
        capture_output=True,
        text=True,
    )


def test_valid_job_streams_envelopes() -> None:
    result = _run(_valid_job())
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


def test_garbage_stdin_is_contract_violation() -> None:
    result = _run("this is not json")
    assert result.returncode == 2
    assert result.stdout.strip() == ""  # no envelopes before validation fails


def test_incompatible_major_is_contract_violation() -> None:
    job = json.loads(_valid_job())
    job["schema_version"] = "1.0.0"  # incompatible major vs the 0.x contract
    result = _run(json.dumps(job))
    assert result.returncode == 2
    assert result.stdout.strip() == ""  # version gate rejects before emitting envelopes
