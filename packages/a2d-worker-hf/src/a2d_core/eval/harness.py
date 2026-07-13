"""Assemble the EvalReport: likelihood bound + downstream tasks + throughput.

Loads the converted ``model/`` once, patches it bidirectional (alpha=1), runs the three
measurement families, and writes the self-contained ``report.json`` (+ ``report.html``
under ``html``). No manifest mutation and no event stream (Decision 1, 7).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from a2d_contracts import EvalReport, EvalRequest, TaskResult

# Fixed example cap for downstream tasks (ponytail: constant; expose in EvalRequest if a
# task needs a different budget than the likelihood corpus).
_TASK_MAX_EXAMPLES = 256


def run_eval(req: EvalRequest, a2d_version: str, schema_version: str) -> EvalReport:
    from a2d_core.data import DATA
    from a2d_core.eval.likelihood import ar_baseline, mdlm_bound
    from a2d_core.eval.tasks import EVAL_TASKS
    from a2d_core.eval.tasks.base import TaskContext
    from a2d_core.eval.throughput import measure_throughput
    from a2d_core.transform.apply import apply_transforms, load_model, resolve_capabilities
    from a2d_core.transform.attention import AnnealState

    model, tokenizer = load_model(req.model_dir)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(f"{req.model_dir} tokenizer has no mask token (not an a2d checkpoint?)")
    mask_id = int(mask_id)
    # Eval the trained bidirectional model, via the same capability dispatch as the worker.
    apply_transforms(model, resolve_capabilities(model), AnnealState(alpha=1.0))

    likelihood = mdlm_bound(
        model,
        tokenizer,
        mask_id,
        data_path=req.data,
        seq_len=int(req.seq_len),
        mc_samples=int(req.mc_samples),
        max_eval_tokens=int(req.max_eval_tokens),
        seed=int(req.seed),
        device=req.device,
        eval_batch_size=int(req.eval_batch_size),
    )
    baseline = ar_baseline(
        req.source_model,
        req.source_hash,
        data_path=req.data,
        seq_len=int(req.seq_len),
        max_eval_tokens=int(req.max_eval_tokens),
        device=req.device,
    )

    # Downstream tasks (default: all registered) on the bidirectional model.
    ctx = TaskContext(
        model=model,
        tokenizer=tokenizer,
        mask_token_id=mask_id,
        device=req.device,
        seed=int(req.seed),
        max_examples=_TASK_MAX_EXAMPLES,
        data_overrides={},
    )
    names = list(req.tasks) if req.tasks else EVAL_TASKS.names()
    tasks = [
        TaskResult(name=s.name, metric=s.metric, value=s.value, n=s.n)
        for s in (EVAL_TASKS.get(name)(ctx) for name in names)
    ]

    # Throughput: a short prompt from the first eval chunk, canvas = seq_len (Decision 4).
    fmt = Path(req.data).suffix.lower().lstrip(".")
    first_chunk = DATA.get(fmt)(req.data, tokenizer, int(req.seq_len))[0]["input_ids"]
    prompt_ids = first_chunk[: min(4, int(req.seq_len) // 2)].tolist()
    throughput = measure_throughput(
        model,
        mask_id,
        req.source_model,
        req.source_hash,
        prompt_ids=prompt_ids,
        canvas_len=int(req.seq_len),
        num_steps=int(req.num_steps),
        device=req.device,
    )

    return EvalReport(
        schema_version=schema_version,
        a2d_version=a2d_version,
        created_at=datetime.now(UTC).isoformat(),
        model_dir=req.model_dir,
        source_model=req.source_model,
        source_hash=req.source_hash,
        data_source=req.data,
        seed=int(req.seed),
        likelihood=likelihood,
        ar_baseline=baseline,
        throughput=throughput,
        tasks=tasks,
    )


def write_report(report: EvalReport, out_dir: str, html: bool) -> Path:
    """Write ``report.json`` (+ ``report.html`` under ``html``); return the json path."""
    from a2d_core.eval.report_html import render

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    if html:
        (out / "report.html").write_text(render(report), encoding="utf-8")
    return json_path
