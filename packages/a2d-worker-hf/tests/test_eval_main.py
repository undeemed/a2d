"""Step 5: the a2d-eval entry validates, evaluates, and writes the report."""

from __future__ import annotations

import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from a2d_core import eval_main
from conftest import ConvertSetup


def _converted_model_dir(setup: ConvertSetup, dst: Path) -> Path:
    from a2d_core.transform.apply import load_model, resolve_mask_token

    model, tokenizer = load_model(str(setup.model_src))
    resolve_mask_token(model, tokenizer, "grow")
    model.save_pretrained(str(dst))
    tokenizer.save_pretrained(str(dst))
    return dst


def _request_json(setup: ConvertSetup, model_dir: Path, out_dir: Path) -> str:
    digest = hashlib.sha256((setup.model_src / "model.safetensors").read_bytes()).hexdigest()
    return json.dumps(
        {
            "schema_version": version("a2d-contracts"),
            "model_dir": str(model_dir),
            "source_model": str(setup.model_src),
            "source_hash": digest,
            "data": str(setup.corpus),
            "tasks": [],
            "seq_len": 8,
            "mc_samples": 2,
            "max_eval_tokens": 64,
            "num_steps": 4,
            "seed": 0,
            "device": "cpu",
            "out_dir": str(out_dir),
            "html": True,
        }
    )


def test_eval_main_writes_report(
    convert_setup: ConvertSetup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    model_dir = _converted_model_dir(convert_setup, tmp_path / "model")
    out_dir = tmp_path / "eval"
    monkeypatch.setattr("sys.stdin", io.StringIO(_request_json(convert_setup, model_dir, out_dir)))
    code = eval_main.main()
    assert code == 0
    report = json.loads((out_dir / "report.json").read_text())
    assert (out_dir / "report.html").exists()
    assert report["likelihood"]["nats_per_token"] > 0
    assert len(report["tasks"]) == 2
    assert "likelihood:" in capsys.readouterr().out


def test_eval_main_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert eval_main.main() == 2
