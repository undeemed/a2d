"""Worker side of the a2d CLI<->worker process protocol (SPEC-HANDOFF).

Read ONE ``ConversionJob`` JSON document from stdin, validate it, then run the
conversion pipeline (SPEC 4.2), emitting ``EventEnvelope`` objects to stdout as
compact JSONL (one per line, flushed per line): ``job_started`` -> ``progress`` per
stage -> ``identity_gate`` -> ``train_step``/``checkpoint`` -> ``job_completed``.
STDOUT carries envelopes only; all diagnostics go to STDERR.

The pipeline (SPEC 4.2 / Decisions 2, 6, 7, 8): validate the safetensors triple ->
load the model (READ-only source, Decision 6) -> grow the mask token (Decision 7) ->
install the anneal patch at ``alpha=0`` -> run the identity gate as a HARD pre-training
stop (fail => nothing trained, exit 1) -> continual-pretrain via HF ``Trainer`` ->
``save_pretrained`` ``model/`` plus the ``a2d`` config block (SPEC 4.3). torch and the
heavy transform/train imports stay lazy so the exit-2 contract path never pulls them in.

Exit codes: 0 = success (job_completed emitted); 1 = job failed during execution
(job_failed emitted first, e.g. the identity gate rejected the patch); 2 = contract
violation (unparseable/invalid job).
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

from a2d_contracts import ConversionJob, EventEnvelope
from pydantic import ValidationError

from . import __version__

# schema_version tracks the CONTRACT (a2d-contracts), not this worker package, so
# it mirrors the Rust side's a2d_contracts::SCHEMA_VERSION (= a2d-contracts crate
# version). Sourcing it from __version__ would drift the moment a2d-worker-hf is
# bumped independently.
SCHEMA_VERSION = version("a2d-contracts")
WORKER = f"a2d-worker-hf {__version__}"

# The convert stages, in order; each emits one Progress before it runs.
_STAGES = ("ingest", "materialize", "grow", "patch", "identity", "train", "save")

# Checkpoint cadence when config carries no explicit one: save at most every 500
# steps but never less than once (so short runs still leave a resumable checkpoint).
# ponytail: fixed cadence; expose --save-steps in ConversionConfig if runs need finer control.
_SAVE_CADENCE = 500

# Recorded sampler defaults for the a2d config block (SPEC 4.3); a2d sample uses
# these absent user overrides. num_steps caps at 128 so the default decode is cheap.
_SAMPLER_STEP_CAP = 128


def _now() -> str:
    return datetime.now(UTC).isoformat()


# Event builders: the single place the wire-shape names for each event kind live.
def job_started() -> dict[str, Any]:
    return {"type": "job_started", "worker": WORKER}


def log(message: str, level: str = "info") -> dict[str, Any]:
    return {"type": "log", "level": level, "message": message}


def progress(stage: str, step: int, total: int) -> dict[str, Any]:
    return {"type": "progress", "stage": stage, "step": step, "total": total}


def identity_gate(result: Any) -> dict[str, Any]:
    return {
        "type": "identity_gate",
        "passed": bool(result.passed),
        "max_abs_diff": float(result.max_abs_diff),
        "tolerance": float(result.tolerance),
    }


def job_completed() -> dict[str, Any]:
    return {"type": "job_completed"}


def job_failed(message: str) -> dict[str, Any]:
    return {"type": "job_failed", "message": message}


def _emit(job_id: str, seq: int, event: dict[str, Any], out: TextIO) -> None:
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


def _a2d_block(cfg: Any, mask_token_id: int, final_alpha: float) -> dict[str, Any]:
    """The ``a2d`` config block a sampler needs to run the checkpoint (SPEC 4.3)."""
    seq_len = int(cfg.seq_len)
    return {
        "objective": cfg.objective,
        "mask_token_id": int(mask_token_id),
        "final_alpha": float(final_alpha),  # the alpha the anneal actually reached (Decision 4)
        "sampler": {
            "canvas_len": seq_len,
            "num_steps": min(seq_len, _SAMPLER_STEP_CAP),
            "temperature": 1.0,
        },
    }


def _write_a2d_block(model_out: Path, block: dict[str, Any]) -> None:
    """Splice the ``a2d`` block into the freshly-saved ``model/config.json``."""
    config_path = model_out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["a2d"] = block
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _convert(job: ConversionJob, emit: Callable[[dict[str, Any]], None]) -> int:
    """Run the convert pipeline; return 0 (completed) or 1 (identity gate rejected).

    Heavy imports (torch + the transform/data/train layers) happen here, AFTER the
    contract validation in ``main``, so the exit-2 path stays torch-free.
    """
    cfg = job.conversion_config
    run_dir = Path(job.run_dir)
    total = len(_STAGES)

    # 1. ingest: validate the safetensors triple; the source dir is only READ (Decision 6).
    emit(progress("ingest", 0, total))
    from a2d_core.ingest import INGEST

    src = INGEST.get("safetensors")(Path(job.model_path))

    import torch

    from a2d_core.data import DATA
    from a2d_core.train import continual
    from a2d_core.transform.apply import (
        apply_transforms,
        load_model,
        resolve_capabilities,
        resolve_mask_token,
    )
    from a2d_core.transform.attention import AnnealState
    from a2d_core.transform.identity import check_identity

    if cfg.max_steps is None:
        raise ValueError(
            "max_steps must be resolved before conversion (config-build maps max_tokens)"
        )

    # 2. materialize: load the model to convert plus an un-patched, un-grown base copy
    #    for the identity gate (Decision 6); both eager float32 (the gate needs float32).
    emit(progress("materialize", 1, total))
    model, tokenizer = load_model(src, dtype="float32")
    base, _ = load_model(src, dtype="float32")
    base_vocab = int(base.config.vocab_size)

    # 3. grow the mask token (Decision 7 default grow: +1 embedding row on the model only).
    emit(progress("grow", 2, total))
    mask_token_id = resolve_mask_token(model, tokenizer, cfg.mask_token)

    # 4. patch: install the annealed attention seam at alpha=0 (Decision 2). The seam
    #    is resolved from the model's own eager causal structure: GPT-2's self.bias
    #    (attn.full) vs the RoPE family's _update_causal_mask mask (attn.gqa - Gemma/
    #    Qwen2/Llama). The ConversionJob carries no capability set, so the worker picks
    #    the handler honestly from the model itself.
    # ponytail: attention seam only; feed the manifest's model_spec.capabilities
    #   through the job when P6 adds SWA/sink handlers keyed off config-only fields.
    emit(progress("patch", 3, total))
    state = AnnealState(alpha=0.0)
    apply_transforms(model, resolve_capabilities(model), state)

    # 5. identity gate (D13, HARD stop): base vs patched@0 on a clean probe. Fail =>
    #    emit the gate result + job_failed and return with NOTHING trained.
    emit(progress("identity", 4, total))
    max_pos = getattr(base.config, "n_positions", 0) or getattr(
        base.config, "max_position_embeddings", 512
    )
    probe_len = min(16, int(max_pos))
    probe = torch.randint(
        0, base_vocab, (2, probe_len), generator=torch.Generator().manual_seed(int(cfg.seed))
    )
    result = check_identity(base, model, state, probe, base_vocab)
    emit(identity_gate(result))
    if not result.passed:
        emit(job_failed(f"identity gate failed: max_abs_diff={result.max_abs_diff}"))
        return 1
    del base  # free the reference copy before training

    # 6. train: continual-pretrain via HF Trainer (Decision 8). An existing
    #    checkpoints/checkpoint-* signals resume; continual.train restores it.
    emit(progress("train", 5, total))
    fmt = Path(cfg.data).suffix.lower().lstrip(".")
    dataset = DATA.get(fmt)(cfg.data, tokenizer, int(cfg.seq_len))
    save_steps = max(1, min(_SAVE_CADENCE, int(cfg.max_steps)))
    continual.train(
        model=model,
        dataset=dataset,
        cfg=cfg,
        mask_token_id=mask_token_id,
        state=state,
        emit=emit,
        output_dir=run_dir / "checkpoints",
        save_steps=save_steps,
    )

    # 7. save the deliverable HF triple + the a2d config block (SPEC 4.3).
    emit(progress("save", 6, total))
    model_out = run_dir / "model"
    model.save_pretrained(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    _write_a2d_block(model_out, _a2d_block(cfg, mask_token_id, state.alpha))

    emit(job_completed())
    return 0


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

    # STDOUT is the sacred event channel: emit writes envelopes to the real stdout
    # captured here, while the pipeline body runs with sys.stdout redirected to stderr
    # so library noise (HF Trainer's log printer, from_pretrained warnings) can never
    # corrupt the JSONL stream.
    out = sys.stdout
    seq = 0

    def emit(event: dict[str, Any]) -> None:
        nonlocal seq
        _emit(job.job_id, seq, event, out)
        seq += 1

    try:
        emit(job_started())
        with contextlib.redirect_stdout(sys.stderr):
            return _convert(job, emit)
    except Exception as exc:  # noqa: BLE001 - report as job_failed then exit 1
        print(f"a2d-worker: conversion failed: {exc}", file=sys.stderr)
        try:
            emit(job_failed(str(exc)))
        except Exception:  # noqa: BLE001 - stdout already broken; stderr has it
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
