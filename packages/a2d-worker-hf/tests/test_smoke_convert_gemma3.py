"""End-to-end worker convert on a seeded tiny Gemma 3 (``gemma3_text``, CPU, no
download): the same SPEC-4.3 pipeline as ``test_smoke_convert_gemma`` (Gemma 1), but
proving the sliding-window (``attn.swa``) path.

This exercises what the Gemma 1 smoke test cannot: the worker's structural handler
dispatch resolving a disk-loaded Gemma 3 to the ``attn.swa`` seam (checked before the
shared ``attn.gqa`` mask seam it also owns), the identity gate passing bit-identically
with BOTH a local (sliding) and a global (full) layer in the stack, and grow/save
working through Gemma 3's tied embeddings, independent head_dim, and query-key norm.
The converted checkpoint is then driven through ``a2d-sample`` and ``a2d-eval`` to
prove the unified capability dispatch patches the sliding-window model bidirectional at
inference. Runs the worker/sample as subprocesses, exactly as the CLI would.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from a2d_core import eval_main


def _convert(setup: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "a2d_core.worker"],
        input=setup.build_job(),
        capture_output=True,
        text=True,
    )


def test_smoke_convert_gemma3_produces_spec_run_dir(gemma3_convert_setup: Any) -> None:
    result = _convert(gemma3_convert_setup)
    assert result.returncode == 0, result.stderr

    run_dir: Path = gemma3_convert_setup.run_dir
    model_out = run_dir / "model"

    # model/ HF triple (SPEC 4.3 deliverable).
    assert (model_out / "config.json").is_file()
    assert (model_out / "model.safetensors").is_file()
    assert (model_out / "tokenizer.json").is_file()

    # a2d config block spliced into config.json; model stays a Gemma 3 checkpoint.
    config = json.loads((model_out / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "gemma3_text"
    a2d = config["a2d"]
    assert a2d["objective"] == "mdlm"
    assert a2d["mask_token_id"] == 64  # grown row on the 64-token base vocab (Decision 7)
    assert isinstance(a2d["final_alpha"], float)
    assert set(a2d["sampler"]) == {"canvas_len", "num_steps", "temperature"}

    # The grown model gained exactly one vocab row (tied embeddings resized together).
    assert config["vocab_size"] == 65

    # checkpoints/ present (resume material).
    assert sorted((run_dir / "checkpoints").glob("checkpoint-*")), "no checkpoint written"

    # The event stream carries the gate + training events, and the gate passed exactly.
    events = [json.loads(line)["event"] for line in result.stdout.splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert "identity_gate" in types
    assert "train_step" in types
    assert "checkpoint" in types

    gate = next(e for e in events if e["type"] == "identity_gate")
    assert gate["passed"] is True
    assert gate["max_abs_diff"] == 0.0  # sliding-window family, eager + fp32 is exact


def test_sample_and_eval_smoke_on_converted_gemma3(
    gemma3_convert_setup: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Sample + eval share the unified dispatch: both load the converted Gemma 3 and
    patch it bidirectional (alpha=1) via ``resolve_capabilities`` -> ``attn.swa``."""
    assert _convert(gemma3_convert_setup).returncode == 0
    model_dir = gemma3_convert_setup.run_dir / "model"

    # a2d-sample denoises the sliding-window model and prints text.
    sample_req = json.dumps(
        {
            "schema_version": version("a2d-contracts"),
            "model_dir": str(model_dir),
            "prompt": "w1 w2 w3",
            "canvas_len": 8,
            "num_steps": 4,
            "temperature": 1.0,
            "seed": 0,
            "device": "cpu",
        }
    )
    sample = subprocess.run(
        [sys.executable, "-m", "a2d_core.sample_main"],
        input=sample_req,
        capture_output=True,
        text=True,
    )
    assert sample.returncode == 0, sample.stderr
    assert sample.stdout.strip(), "a2d-sample printed no text"

    # a2d-eval scores the sliding-window model and writes a report.
    out_dir = tmp_path / "eval"
    digest = hashlib.sha256(
        (gemma3_convert_setup.model_src / "model.safetensors").read_bytes()
    ).hexdigest()
    eval_req = json.dumps(
        {
            "schema_version": version("a2d-contracts"),
            "model_dir": str(model_dir),
            "source_model": str(gemma3_convert_setup.model_src),
            "source_hash": digest,
            "data": str(gemma3_convert_setup.corpus),
            "tasks": [],
            "seq_len": 8,
            "mc_samples": 2,
            "max_eval_tokens": 64,
            "eval_batch_size": 4,
            "num_steps": 4,
            "seed": 0,
            "device": "cpu",
            "out_dir": str(out_dir),
            "html": False,
        }
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(eval_req))
    assert eval_main.main() == 0
    report = json.loads((out_dir / "report.json").read_text())
    assert report["likelihood"]["nats_per_token"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
