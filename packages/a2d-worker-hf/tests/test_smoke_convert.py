"""Step 11: the worker convert pipeline end to end on a seeded tiny GPT-2 (CPU, no
download), producing a SPEC-4.3 run dir, plus the identity-gate hard stop.

The happy path runs the worker as a subprocess (as the CLI would) and asserts the
run dir: ``model/`` HF triple + the ``a2d`` config block, ``checkpoints/``, and an
event stream carrying IdentityGate + TrainStep + Checkpoint. The broken-patch case
neutralizes the anneal mask so the identity gate fails, and asserts the run aborts
BEFORE any training (no TrainStep, no checkpoints) with exit 1.

``manifest.json`` is written by a2d-run (Rust), not the worker (which owns only the
model/ + events stream), so it is intentionally not asserted here.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def test_smoke_convert_produces_spec_run_dir(convert_setup: Any) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "a2d_core.worker"],
        input=convert_setup.build_job(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    run_dir: Path = convert_setup.run_dir
    model_out = run_dir / "model"

    # model/ HF triple (SPEC 4.3 deliverable).
    assert (model_out / "config.json").is_file()
    assert (model_out / "model.safetensors").is_file()
    assert (model_out / "tokenizer.json").is_file()

    # a2d config block spliced into config.json.
    config = json.loads((model_out / "config.json").read_text(encoding="utf-8"))
    a2d = config["a2d"]
    assert a2d["objective"] == "mdlm"
    assert isinstance(a2d["mask_token_id"], int)
    assert a2d["mask_token_id"] == 64  # grown row on the 64-token base vocab (Decision 7)
    assert isinstance(a2d["final_alpha"], float)
    assert set(a2d["sampler"]) == {"canvas_len", "num_steps", "temperature"}

    # The grown model gained exactly one vocab row.
    assert config["vocab_size"] == 65

    # checkpoints/ present (resume material).
    assert sorted((run_dir / "checkpoints").glob("checkpoint-*")), "no checkpoint written"

    # The event stream a2d-run mirrors to events.jsonl carries the gate + training events.
    events_jsonl = run_dir / "events.jsonl"
    events_jsonl.write_text(result.stdout, encoding="utf-8")  # a2d-run tees stdout here
    events = [json.loads(line)["event"] for line in result.stdout.splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert "identity_gate" in types
    assert "train_step" in types
    assert "checkpoint" in types

    gate = next(e for e in events if e["type"] == "identity_gate")
    assert gate["passed"] is True
    assert gate["max_abs_diff"] == 0.0  # eager + fp32 is exact


def test_broken_patch_aborts_before_training(
    convert_setup: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import a2d_core.transform.attention as attn
    import a2d_core.worker as worker

    # A zero additive mask leaves the model fully bidirectional even at alpha=0, so
    # patched@0 != base and the identity gate MUST reject the patch (Decision 2).
    def broken_mask(q_len: int, k_len: int, alpha: float, dtype: Any, device: Any) -> Any:
        import torch

        return torch.zeros(q_len, k_len, dtype=dtype, device=device)

    monkeypatch.setattr(attn, "annealed_additive_mask", broken_mask)
    monkeypatch.setattr(sys, "stdin", io.StringIO(convert_setup.build_job()))

    rc = worker.main()
    assert rc == 1

    out = capsys.readouterr().out
    events = [json.loads(line)["event"] for line in out.splitlines() if line.strip()]
    types = [e["type"] for e in events]

    gate = [e for e in events if e["type"] == "identity_gate"]
    assert gate and gate[0]["passed"] is False
    assert gate[0]["max_abs_diff"] > gate[0]["tolerance"]
    assert "train_step" not in types  # HARD stop: nothing trained
    assert "checkpoint" not in types
    assert types[-1] == "job_failed"

    # Nothing was written to the run dir: no checkpoints, no model/.
    assert not (convert_setup.run_dir / "checkpoints").exists()
    assert not (convert_setup.run_dir / "model").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
