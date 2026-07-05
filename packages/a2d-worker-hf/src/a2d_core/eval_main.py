"""``a2d-eval`` worker: evaluate a converted ``model/`` and write ``eval/report.json``.

Reads ONE ``EvalRequest`` JSON document from stdin, validates the generated pydantic
(exit 2 on contract violation), runs the harness, writes ``report.json`` (+ ``report.html``
under ``html``), and prints a human summary to stdout. Exit 0 (ok), 1 (eval failed),
2 (contract violation). Like ``a2d-sample`` this touches neither the manifest nor the
event stream (Decision 1); it is a read-only analysis on the converted checkpoint.
"""

from __future__ import annotations

import sys
from importlib.metadata import version

from a2d_contracts import EvalRequest
from pydantic import ValidationError

from a2d_core import __version__
from a2d_core.worker import SCHEMA_VERSION


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = EvalRequest.model_validate_json(raw)
    except ValidationError as exc:
        print(f"a2d-eval: invalid EvalRequest: {exc}", file=sys.stderr)
        return 2

    if req.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        print(
            "a2d-eval: schema_version major mismatch: "
            f"request={req.schema_version!r} worker={SCHEMA_VERSION!r}",
            file=sys.stderr,
        )
        return 2

    from a2d_core.eval.harness import run_eval, write_report

    try:
        report = run_eval(req, a2d_version=__version__, schema_version=version("a2d-contracts"))
        json_path = write_report(report, req.out_dir, req.html)
    except Exception as exc:  # noqa: BLE001 - report failure on stderr, exit 1
        print(f"a2d-eval: evaluation failed: {exc}", file=sys.stderr)
        return 1

    lk = report.likelihood
    ar = report.ar_baseline
    tp = report.throughput
    print(f"wrote {json_path}")
    print(
        f"likelihood: {lk.nats_per_token:.4g} nats/token "
        f"({lk.bits_per_token:.4g} bits, +/-{lk.std_error:.2g})"
    )
    for task in report.tasks:
        print(f"task {task.name}: {task.value:.4g} ({task.metric}, n={task.n})")
    ar_txt = (
        f"{ar.perplexity:.4g} ppl"
        if ar and ar.available and ar.perplexity is not None
        else "unavailable"
    )
    print(f"AR baseline: {ar_txt}")
    speed = f"{tp.speedup:.3g}x vs AR" if tp.speedup is not None else "AR unavailable"
    print(f"throughput: {tp.diffusion_tokens_per_sec:.4g} tok/s diffusion ({speed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
