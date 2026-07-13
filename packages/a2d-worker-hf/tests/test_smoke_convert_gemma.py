"""End-to-end worker convert on a seeded tiny Gemma 1 (CPU, no download): the same
SPEC-4.3 pipeline as ``test_smoke_convert`` (GPT-2), but proving the RoPE/GQA path.

This exercises what the GPT-2 smoke test cannot: the worker's structural handler
dispatch resolving a disk-loaded Gemma to the ``attn.gqa`` mask seam, the identity
gate passing bit-identically for the RoPE family, and grow/save working through
Gemma's tied embeddings and independent head_dim. Runs the worker as a subprocess,
exactly as the CLI would.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def test_smoke_convert_gemma_produces_spec_run_dir(gemma_convert_setup: Any) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "a2d_core.worker"],
        input=gemma_convert_setup.build_job(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    run_dir: Path = gemma_convert_setup.run_dir
    model_out = run_dir / "model"

    # model/ HF triple (SPEC 4.3 deliverable).
    assert (model_out / "config.json").is_file()
    assert (model_out / "model.safetensors").is_file()
    assert (model_out / "tokenizer.json").is_file()

    # a2d config block spliced into config.json; model stays a Gemma checkpoint.
    config = json.loads((model_out / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "gemma"
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
    assert gate["max_abs_diff"] == 0.0  # RoPE family, eager + fp32 is exact


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
