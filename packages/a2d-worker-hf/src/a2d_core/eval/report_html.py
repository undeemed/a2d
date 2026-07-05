"""Minimal self-contained HTML render of an EvalReport (Decision 8).

Static, no JavaScript, no framework, inline CSS - openable directly in a browser. Leads
with the MDLM likelihood bound + tasks + throughput and renders AR perplexity in a
separate block with the incomparability caveat (Decision 5), never as a single
"diffusion beats AR" line.
# ponytail: static template; swap for a richer view if the platform track (SPEC 8) revives.
"""

from __future__ import annotations

from html import escape

from a2d_contracts import EvalReport

_CAVEAT = (
    "Diffusion NLL is a masked-diffusion likelihood BOUND and is not directly "
    "comparable to autoregressive perplexity (ARCHITECTURE.md 9). Compare within "
    "a column, not across."
)

_SPEED_NOTE = (
    "MDLM runs num_steps full forward passes; speedup below 1 is expected - "
    "the block-parallel payoff is BD3LM (P5)."
)

_CSS = """
 body { font: 15px/1.5 system-ui, sans-serif; max-width: 760px;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
 h1 { font-size: 1.4rem; }
 h2 { font-size: 1.05rem; margin-top: 1.6rem;
      border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
 table { border-collapse: collapse; width: 100%; margin-top: .5rem; }
 td, th { text-align: left; padding: .3rem .6rem; border-bottom: 1px solid #eee; }
 .meta { color: #666; font-size: .85rem; }
 .caveat { background: #fff8e1; border-left: 3px solid #e0b000;
           padding: .5rem .8rem; font-size: .88rem; margin-top: .5rem; }
"""


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.4g}"


def _ar_rows(report: EvalReport) -> str:
    ar = report.ar_baseline
    if ar and ar.available:
        return (
            f"<tr><td>perplexity</td><td>{_fmt(ar.perplexity)}</td></tr>"
            f"<tr><td>nats/token</td><td>{_fmt(ar.nats_per_token)}</td></tr>"
        )
    reason = ar.reason if ar and ar.reason else "no source"
    return f"<tr><td>unavailable</td><td>{escape(reason)}</td></tr>"


def render(report: EvalReport) -> str:
    task_rows = "".join(
        f"<tr><td>{escape(t.name)}</td><td>{escape(t.metric)}</td>"
        f"<td>{_fmt(t.value)}</td><td>{t.n}</td></tr>"
        for t in report.tasks
    )
    lk = report.likelihood
    tp = report.throughput
    meta = (
        f"model: {escape(report.model_dir)}<br>"
        f"source: {escape(report.source_model or '-')} &middot; "
        f"data: {escape(report.data_source)} &middot; seed: {report.seed} &middot; "
        f"a2d {escape(report.a2d_version)} &middot; {escape(report.created_at)}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>a2d eval report</title>
<style>{_CSS}</style></head><body>
<h1>a2d evaluation report</h1>
<p class="meta">{meta}</p>

<h2>MDLM likelihood bound (headline)</h2>
<table>
 <tr><td>nats/token</td><td>{_fmt(lk.nats_per_token)}</td></tr>
 <tr><td>bits/token</td><td>{_fmt(lk.bits_per_token)}</td></tr>
 <tr><td>std error</td><td>{_fmt(lk.std_error)}</td></tr>
 <tr><td>mc samples / tokens</td><td>{lk.mc_samples} / {lk.n_tokens}</td></tr>
</table>

<h2>Downstream tasks</h2>
<table><tr><th>task</th><th>metric</th><th>value</th><th>n</th></tr>{task_rows}</table>

<h2>Throughput (decode)</h2>
<table>
 <tr><td>diffusion tokens/sec</td><td>{_fmt(tp.diffusion_tokens_per_sec)}</td></tr>
 <tr><td>AR tokens/sec</td><td>{_fmt(tp.ar_tokens_per_sec)}</td></tr>
 <tr><td>speedup (diffusion/AR)</td><td>{_fmt(tp.speedup)}</td></tr>
 <tr><td>num steps</td><td>{tp.num_steps}</td></tr>
</table>
<p class="meta">{escape(_SPEED_NOTE)}</p>

<h2>AR baseline (source model)</h2>
<table>{_ar_rows(report)}</table>
<p class="caveat">{escape(_CAVEAT)}</p>
</body></html>
"""
