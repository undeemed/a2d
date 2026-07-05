"""Step 5: the harness assembles a contract-shaped, reproducible EvalReport."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from a2d_contracts import EvalReport, EvalRequest
from conftest import ConvertSetup


def _converted_model_dir(setup: ConvertSetup, dst: Path) -> Path:
    """A structurally-converted model/ (grown mask token + tokenizer) for eval."""
    from a2d_core.transform.apply import load_model, resolve_mask_token

    model, tokenizer = load_model(str(setup.model_src))
    resolve_mask_token(model, tokenizer, "grow")
    model.save_pretrained(str(dst))
    tokenizer.save_pretrained(str(dst))
    return dst


def _request(setup: ConvertSetup, model_dir: Path, out_dir: Path, html: bool) -> EvalRequest:
    digest = hashlib.sha256((setup.model_src / "model.safetensors").read_bytes()).hexdigest()
    return EvalRequest(
        schema_version=version("a2d-contracts"),
        model_dir=str(model_dir),
        source_model=str(setup.model_src),
        source_hash=digest,
        data=str(setup.corpus),
        tasks=[],
        seq_len=8,
        mc_samples=2,
        max_eval_tokens=64,
        num_steps=4,
        seed=0,
        device="cpu",
        out_dir=str(out_dir),
        html=html,
    )


def test_report_shape_and_files(convert_setup: ConvertSetup, tmp_path: Path) -> None:
    from a2d_core.eval.harness import run_eval, write_report

    model_dir = _converted_model_dir(convert_setup, tmp_path / "model")
    req = _request(convert_setup, model_dir, tmp_path / "eval", html=True)
    report = run_eval(req, a2d_version="0.1.0", schema_version=version("a2d-contracts"))

    assert isinstance(report, EvalReport)
    assert report.likelihood.nats_per_token > 0
    assert len(report.tasks) == 2  # both registered tasks ran
    assert report.ar_baseline is not None and report.ar_baseline.available
    assert report.throughput.diffusion_tokens_per_sec > 0

    json_path = write_report(report, req.out_dir, req.html)
    assert json_path.exists()
    assert (tmp_path / "eval" / "report.html").exists()
    # report.json round-trips back through the contract.
    reloaded = EvalReport.model_validate_json(json_path.read_text())
    assert reloaded.likelihood == report.likelihood


def test_measurement_content_reproducible(convert_setup: ConvertSetup, tmp_path: Path) -> None:
    """Likelihood + tasks are seed-deterministic; throughput (wall-clock) and created_at
    are the inherently non-deterministic fields, excluded from the guarantee."""
    from a2d_core.eval.harness import run_eval

    model_dir = _converted_model_dir(convert_setup, tmp_path / "model")
    req = _request(convert_setup, model_dir, tmp_path / "eval", html=False)
    a = run_eval(req, a2d_version="0.1.0", schema_version=version("a2d-contracts"))
    b = run_eval(req, a2d_version="0.1.0", schema_version=version("a2d-contracts"))

    assert a.likelihood == b.likelihood
    assert a.tasks == b.tasks
    assert a.ar_baseline == b.ar_baseline
    assert (a.model_dir, a.source_hash, a.seed, a.data_source) == (
        b.model_dir,
        b.source_hash,
        b.seed,
        b.data_source,
    )


def test_report_json_is_valid_json(convert_setup: ConvertSetup, tmp_path: Path) -> None:
    from a2d_core.eval.harness import run_eval, write_report

    model_dir = _converted_model_dir(convert_setup, tmp_path / "model")
    req = _request(convert_setup, model_dir, tmp_path / "eval", html=False)
    report = run_eval(req, a2d_version="0.1.0", schema_version=version("a2d-contracts"))
    path = write_report(report, req.out_dir, req.html)
    json.loads(path.read_text())  # parses
