# a2d Phase 3 Implementation Plan - Evaluation

**Status:** draft plan, pre-implementation.
**Source spec:** [`SPEC-HANDOFF.md`](SPEC-HANDOFF.md) §4.3 (the `eval/` run-dir artifact), §6 (Phase 3 scope and exit criteria), §10 (eval-comparability risk); [`ARCHITECTURE.md`](ARCHITECTURE.md) §1 M1 (NLL bound + 2 downstream tasks + tokens/sec) and §9 (diffusion NLL is not AR perplexity).

## Context

Phase 2 shipped the first real conversion: `a2d convert` turns GPT-2 into an MDLM-diffusion checkpoint, the identity gate proves the attention surgery changed nothing before training, and `a2d sample` denoises the result.
The deliverable is a run dir whose `model/` holds a standard HF triple plus an `a2d` config block (`objective`, `mask_token_id`, `final_alpha`, sampler defaults), and whose `manifest.json` records provenance (`model_path`, `source_hash`, `conversion_config`, `identity`, `token_count`).
What is missing is any way to answer "is the converted model actually good, and what did conversion cost versus the AR original".
Phase 3 adds that: `a2d eval <run-dir>` reads the converted `model/`, measures it, and writes a reproducible `eval/report.json` (plus an optional `eval/report.html`) with three families of number - the MDLM likelihood bound, at least two downstream tasks, and decode throughput against the source AR model.

The governing philosophy stays narrow and additive, exactly as the extension playbook (SPEC §3.3) promises for "new eval task => one file".
Eval is a read-only spot analysis on `model/`, so its closest sibling is `a2d-sample`, not the training worker: a separate `a2d-eval` process reads one request on stdin, computes, writes the report, and exits.
It reuses the machinery Phase 2 already owns - `objectives.mdlm.MDLM.corrupt`/`loss` for the likelihood bound, `sample.SAMPLERS["mdlm"]` for diffusion throughput, `transform.apply.load_model` for loading, the `DATA` registry for the eval corpus, and `worker_cmd::resolve` for command discovery - so Phase 3 introduces no new ML and no new transport, only measurement and reporting.
Everything is proven GPU-free on a seeded tiny GPT-2 over tiny local fixtures on CPU float32, so correctness never depends on a GPU or a network download; only the single "real numbers on the real GPT-2 diffusion run" acceptance step is compute-bound and human-observed.

**Exit criterion:** `a2d eval <run-dir> --data <held-out corpus>` writes a reproducible `eval/report.json` (MDLM likelihood bound, at least two downstream tasks, diffusion-vs-AR tokens/sec) and, under `--html`, an `eval/report.html` that renders the AR-vs-diffusion comparison; the same inputs produce a byte-identical report (modulo the recorded timestamp); and when the source AR weights are present the report carries the AR baseline, when they are absent it degrades honestly with a reason rather than failing.

## Layout to create

Two new contract roots land in `a2d-contracts` (mirroring how `SampleRequest` rode into Phase 2); the CLI grows a real `eval` subcommand modeled on `sample`; the worker fills the `eval/` extension point stubbed since Phase 0.
No `a2d-run` change (eval never creates a run dir or mutates the manifest), no new `Event` variant (`Progress`/`Metric` already exist), and no ML port.

```
crates/
  a2d-contracts/
    src/lib.rs                 # EDIT: add EvalRequest + EvalReport (with nested LikelihoodBound,
                               #       ArBaseline, ThroughputResult, TaskResult); roundtrip tests
    src/bin/export-schema.rs   # EDIT: +2 lines (eval_request.schema.json, eval_report.schema.json)
  a2d-cli/
    src/main.rs                # EDIT: Command::Eval becomes eval::EvalArgs; drop stub("eval")
    src/eval.rs                # NEW: read manifest -> build EvalRequest -> spawn a2d-eval -> print summary
                               #       (mirrors sample.rs: stdin JSON, inherited stderr, propagate exit)
    tests/e2e.rs               # EDIT: torch-free `a2d eval` via a FAKE a2d-eval worker (writes a stub
                               #       report.json, exit 0/2) asserting eval/ shape + exit codes
schema/*.schema.json           # REGEN + eval_request.schema.json + eval_report.schema.json
packages/a2d-contracts/
  src/a2d_contracts/__init__.py  # EDIT: __all__ + TYPE_CHECKING + _MODULES += eval_request_schema, eval_report_schema
  src/a2d_contracts/models/      # REGEN via scripts/codegen.sh
packages/a2d-worker-hf/
  pyproject.toml               # EDIT: + a2d-eval script ([project.scripts])
  src/a2d_core/
    eval_main.py               # NEW: a2d-eval entry (validate EvalRequest, run harness, exit 0/1/2)
    eval/
      harness.py               # NEW: orchestrate bound + tasks + throughput -> EvalReport -> write report.json (+html)
      likelihood.py            # NEW: MDLM NLL upper bound (MC over t, reuses objectives.mdlm) + AR-baseline perplexity
      throughput.py            # NEW: diffusion tokens/sec (reuses SAMPLERS) vs AR greedy tokens/sec
      report_html.py           # NEW: minimal self-contained static HTML render (no JS framework, ponytail)
      tasks/
        __init__.py            # EDIT: EVAL_TASKS registry + import infill_accuracy, cloze_likelihood
        base.py                # NEW: Task protocol (run(model, tokenizer, data, ctx) -> TaskResult)
        infill_accuracy.py     # NEW: @register("infill_accuracy") exact-match masked-token recovery
        cloze_likelihood.py    # NEW: @register("cloze_likelihood") multiple-choice scored by the MDLM bound
  tests/
    test_likelihood.py test_tasks.py test_throughput.py test_eval_report.py test_eval_main.py  # NEW (all CPU)
fixtures/
  eval/
    infill.jsonl               # NEW: tiny local corpus for infill_accuracy (CI determinism)
    cloze.jsonl                # NEW: tiny {context, choices, answer} set for cloze_likelihood
# SKIPPED (YAGNI, flagged): eval event streaming + eval/events.jsonl (sample pattern has none; the Metric/Progress
#   events already exist for when eval gets slow or the TUI wants live progress); a Manifest `eval` field (report.json
#   is the SPEC-4.3 artifact and self-contained); LAMBADA/HellaSwag/hf:-streamed tasks (one-file adds behind the
#   registry); richer HTML (platform-track); MoE/BD3LM eval specifics (P4/P5).
```

## Contract changes

All new types live in `crates/a2d-contracts/src/lib.rs` (the source of truth); codegen regenerates the pydantic mirrors.
`EvalRequest` and `EvalReport` are the two new export-schema roots, added exactly the way `SampleRequest` was in Phase 2 (derive `Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq`, one `export-schema.rs` line each, one `_MODULES` entry each).
The nested result structs ride as `$defs` of `EvalReport`, so they need no extra export line.

```rust
// NEW: the a2d-eval worker request (export-schema root, sibling of SampleRequest).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct EvalRequest {
    pub schema_version: String,
    pub model_dir: String,          // <run-dir>/model (the converted checkpoint)
    pub source_model: Option<String>, // manifest.model_path (the AR base for comparison; None => skip AR baseline)
    pub source_hash: Option<String>,  // manifest.source_hash (verified before trusting source_model)
    pub data: String,               // held-out corpus for the likelihood bound + AR perplexity (local jsonl/txt)
    pub tasks: Vec<String>,         // downstream tasks to run (default: all registered)
    pub seq_len: u64,               // eval context length
    pub mc_samples: u64,            // Monte-Carlo t draws per chunk for the MDLM bound (variance reduction)
    pub max_eval_tokens: u64,       // cap on scored tokens (keeps eval bounded)
    pub num_steps: u64,             // denoiser steps for the diffusion throughput measurement
    pub seed: u64,
    pub device: String,             // "auto"|"cpu"|"mps"|"cuda"
    pub out_dir: String,            // <run-dir>/eval (where report.json / report.html land)
    pub html: bool,                 // also write report.html
}

// NEW: the reproducible report written to <run-dir>/eval/report.json (export-schema root).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct EvalReport {
    pub schema_version: String,
    pub a2d_version: String,
    pub created_at: String,         // RFC3339; the ONE field excluded from the determinism check
    pub model_dir: String,
    pub source_model: Option<String>,
    pub source_hash: Option<String>,
    pub data_source: String,
    pub seed: u64,
    pub likelihood: LikelihoodBound,
    pub ar_baseline: Option<ArBaseline>,  // None + reason when source weights are absent/mismatched
    pub throughput: ThroughputResult,
    pub tasks: Vec<TaskResult>,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct LikelihoodBound {
    pub nats_per_token: f64,        // MDLM NLL upper bound (the apples-to-apples number, ARCHITECTURE §9)
    pub bits_per_token: f64,        // nats_per_token / ln(2)
    pub std_error: f64,             // MC standard error over mc_samples
    pub mc_samples: u64,
    pub n_tokens: u64,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ArBaseline {
    pub available: bool,
    pub reason: Option<String>,     // why unavailable (source dir gone, source_hash mismatch); None when available
    pub perplexity: Option<f64>,    // exp(mean teacher-forced NLL) of the SOURCE AR model on `data`
    pub nats_per_token: Option<f64>,// the AR NLL, NOT directly comparable to the diffusion bound (rendered with a caveat)
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ThroughputResult {
    pub diffusion_tokens_per_sec: f64, // revealed canvas tokens / wall seconds, at num_steps
    pub ar_tokens_per_sec: Option<f64>,// source AR greedy decode tokens/sec (None when source absent)
    pub speedup: Option<f64>,          // diffusion / ar; honestly <1 for MDLM (BD3LM is the payoff, P5)
    pub num_steps: u64,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct TaskResult {
    pub name: String,
    pub metric: String,             // e.g. "accuracy"
    pub value: f64,
    pub n: u64,                     // number of scored examples
}
```

**Codegen impact.**
`export-schema.rs` gains two lines (`eval_request.schema.json`, `eval_report.schema.json`); the nested structs ride as `$defs` of `EvalReport`.
The Python `__init__._MODULES` tuple gains `eval_request_schema` and `eval_report_schema`, and `__all__` + the `TYPE_CHECKING` block gain `EvalRequest` + `EvalReport`.
The CI guard is unchanged: `scripts/codegen.sh && git diff --exit-code schema/ packages/a2d-contracts/` plus `cargo test`, applied atomically.

## Key decisions

- **Decision 1 - eval is a separate `a2d-eval` worker on the `a2d-sample` pattern, not the training worker.**
  Eval reads the converted `model/` and never trains, creates a run dir, or mutates the manifest, so its shape is exactly `a2d-sample`: read one JSON request on stdin, validate the generated pydantic (exit 2 on contract violation), compute, write the artifact, exit 0 (ok) or 1 (eval failed).
  It does NOT stream `EventEnvelope`s, because eval of a GPT-2-scale model over a tiny held-out corpus is fast and the exit criterion is a reproducible report, not live progress.
  The `Progress`/`Metric` events already exist (Phase 2 Risk 13 added `Metric` for exactly this horizon), so event-streamed eval is a zero-contract-change upgrade the moment eval gets slow or the TUI wants a live bar - it is deliberately deferred, not designed out.
  Consequence: Phase 3 touches `a2d-contracts`, `a2d-cli`, and the worker only; `a2d-run` is untouched, keeping the blast radius minimal and the extension additive (SPEC §3.3).

- **Decision 2 - the likelihood bound reuses `objectives.mdlm`; it is never reimplemented.**
  MDLM's training loss `(1/t) * CE` at masked positions IS the Monte-Carlo estimator of the model's NLL upper bound (the ELBO term), so `eval/likelihood.py` accumulates the SAME weighted NLL that `objectives.mdlm.MDLM.loss` computes, summed over `mc_samples` fresh `t` draws per chunk across the corpus, divided by the token count, to get `nats_per_token` with a Monte-Carlo `std_error`.
  It imports `objectives.mdlm.MDLM.corrupt`/`loss` directly rather than owning a second copy of the objective, so the bound and the training loss can never drift.
  `bits_per_token = nats_per_token / ln(2)` is reported alongside, because bits/token is the field-conventional intrinsic number.
  This is the "MDLM likelihood bound" of SPEC §6 and ARCHITECTURE M1, and it is the headline number precisely because it is the one AR-vs-diffusion quantity with a defensible interpretation (see Decision 5).

- **Decision 3 - two downstream tasks chosen for cheap, local, deterministic, and native to MDLM; more tasks are one file each.**
  `infill_accuracy`: mask a fixed, seed-determined fraction of each held-out sequence, denoise, and score exact-match token recovery - the model's core competency, deterministic given the seed, and needing no labels.
  `cloze_likelihood`: for each `{context, choices, answer}` example, score every choice by the same MDLM per-token bound and count `argmin(bound) == answer` - a multiple-choice accuracy that reuses Decision 2's machinery and mirrors how AR models are scored by perplexity.
  Both are honest intrinsic probes at GPT-2 scale, not SOTA benchmark claims; both run on tiny bundled fixtures for CI determinism and take a `--data`-style override for real sets.
  They self-register in an `EVAL_TASKS` registry, so LAMBADA, HellaSwag, or any `hf:`-streamed set is a single new file in `eval/tasks/` with zero edits elsewhere (SPEC §3.3), which is the entire point of the extension point.

- **Decision 4 - throughput is measured against the source AR model, honestly, and MDLM is expected to lose.**
  Diffusion tokens/sec is `(revealed canvas tokens) / (wall seconds)` running `sample.SAMPLERS["mdlm"]` at `num_steps`; AR tokens/sec is the source model's greedy decode over the same token budget; both use `time.perf_counter` and warm the model once before timing.
  The report states `speedup = diffusion / ar` and the render leads with the honest caveat that MDLM's fixed `num_steps` forward passes make it slower than AR - the decode-speed payoff is BD3LM's block-parallel sampler (Phase 5), and Phase 3 measures the baseline it will be compared against, not a win.
  In CI the budgets are tiny (a few tokens, a handful of steps) so the test only asserts both rates are finite and positive; the real numbers come from the human MPS acceptance step.

- **Decision 5 - AR perplexity is reported but never presented as directly comparable to the diffusion bound.**
  The source AR model's teacher-forced perplexity (`exp(mean next-token NLL)` over the same `data`) is the number users expect, so the report carries it, but ARCHITECTURE §9 and SPEC §10 are explicit that diffusion NLL is not AR perplexity, so `report.html` renders the two in separate blocks with an inline note and leads the comparison with the MDLM bound + downstream tasks + throughput, never with a single "diffusion beats AR by X perplexity" line.
  This is public-facing honesty policy, not just internal caution (SPEC §11): the report must not manufacture a false apples-to-apples.

- **Decision 6 - the AR baseline is best-effort; a missing source never fails eval.**
  `source_model` and `source_hash` come from the run's `manifest.json`; before loading the AR base, eval re-hashes the primary source safetensors and compares to `source_hash`.
  If the source dir is gone or the hash mismatches, `ar_baseline` is `{available:false, reason:...}` and `throughput.ar_tokens_per_sec`/`speedup` are `None`, but the diffusion bound, downstream tasks, and diffusion throughput still produce a complete report.
  Eval measures the converted checkpoint, which is always present in `model/`; the AR comparison is an enrichment, so its absence degrades the report, it does not abort it.

- **Decision 7 - `report.json` is self-contained and reproducible; the manifest is not touched.**
  The report embeds `source_hash`, `seed`, `a2d_version`, `data_source`, `mc_samples`, `num_steps`, and the task set, so the same inputs deterministically reproduce it, and it is the single SPEC §4.3 artifact (`eval/report.json`) with no need for a manifest breadcrumb.
  All RNG is seeded (a `torch.Generator` for corruption sampling, seeded task masking), so the ONLY non-deterministic field is `created_at`, which the determinism test excludes.
  Not adding a `Manifest.eval` field is a deliberate DRY choice (report.json is the artifact; a manifest copy would be a second source of truth that drifts), flagged in Risks.

- **Decision 8 - `report.html` is a minimal self-contained static template written by the worker under `--html`.**
  The worker owns the data, so it also owns the render: `eval/report_html.py` emits one static HTML file (inline CSS, no JavaScript, no framework, no CLI-side rendering), carrying the same numbers plus the Decision-5 caveat, openable directly in a browser.
  `# ponytail: static template; swap for a richer view (charts, run-to-run diff) if the platform track (SPEC §8) revives and needs a served report`.

- **Decision 9 - the eval corpus is required (`--data`); downstream tasks default to bundled fixtures.**
  The likelihood bound and AR perplexity need a held-out corpus, and per SPEC §9's lean toward explicit data, `a2d eval` requires `--data <local jsonl/txt>` read through the existing `DATA` registry (no new reader).
  The two downstream tasks default to the tiny `fixtures/eval/*.jsonl` sets so a bare `a2d eval <run-dir> --data ...` runs end to end, with a per-task data override documented for real benchmark sets; real streamed sets (`hf:`) are the one-file registry add later, keeping Phase 3's network surface at zero.

## Implementation order

Each step ends verifiable; every check except step 7 runs GPU-free on this arm64/macOS Mac.

1. **Contracts + codegen.**
   Add `EvalRequest`, `EvalReport`, and the nested `LikelihoodBound`/`ArBaseline`/`ThroughputResult`/`TaskResult` structs; add the two `export-schema.rs` lines; extend the Python `__init__` `_MODULES` + `__all__` + `TYPE_CHECKING`; add roundtrip tests (both roots, and an `ar_baseline: None` report).
   CHECK (GPU-free): `cargo test -p a2d-contracts`; `bash scripts/codegen.sh` run twice then `git diff --exit-code schema/ packages/a2d-contracts/src/a2d_contracts/models` empty (deterministic, per the Phase-2 idempotency convention).

2. **Eval tasks: protocol + registry + the two tasks.**
   `eval/tasks/base.py` (`Task` protocol + `EVAL_TASKS` registry); `infill_accuracy.py` and `cloze_likelihood.py` self-registering; the tiny `fixtures/eval/infill.jsonl` + `fixtures/eval/cloze.jsonl`.
   CHECK (GPU-free): `pytest test_tasks.py` (each task returns a `TaskResult` with `value in [0,1]`, deterministic given the seed; `cloze_likelihood` picks the argmin-bound choice on a hand-built example; the registry lists both).

3. **Likelihood bound + AR baseline.**
   `eval/likelihood.py`: the MC MDLM bound (reusing `objectives.mdlm`) returning `LikelihoodBound`; the best-effort AR teacher-forced perplexity returning `ArBaseline` (hash-verify `source_hash`, degrade with a reason on mismatch/absence).
   CHECK (GPU-free): `pytest test_likelihood.py` (bound is finite and strictly below a random-logits control; deterministic given the seed; more `mc_samples` shrinks `std_error`; `ArBaseline.available=false` with a reason when `source_model=None`).

4. **Throughput.**
   `eval/throughput.py`: diffusion tokens/sec via `SAMPLERS["mdlm"]` and AR greedy tokens/sec via the source model, both warmed then `perf_counter`-timed, returning `ThroughputResult`.
   CHECK (GPU-free): `pytest test_throughput.py` (both rates finite and > 0 on the tiny model; `ar_tokens_per_sec`/`speedup` are `None` when the source is absent).

5. **Harness + `a2d-eval` entry + report render.**
   `eval/harness.py` assembles `EvalReport` from steps 2-4 and writes `report.json` (+ `report.html` under `html`); `eval_main.py` is the `a2d-eval` entry (validate `EvalRequest`, run harness, exit 0/1/2); add the `a2d-eval` `[project.scripts]` line; `eval/report_html.py`.
   CHECK (GPU-free): `pytest test_eval_report.py` (report matches the pydantic; `report.json` written under `out_dir`; `--html` writes `report.html`; two runs with the same seed are byte-identical except `created_at`); `pytest test_eval_main.py` (`a2d-eval` on a tiny converted `model/` fixture writes the report; garbage stdin exits 2).

6. **CLI `eval` wiring.**
   `a2d-cli/src/eval.rs` (read `<run-dir>/manifest.json` for `model_path`/`source_hash` + assert `model/` exists, build `EvalRequest`, resolve the worker via `worker_cmd::resolve(.., "A2D_EVAL_CMD", "a2d-eval")`, spawn, feed stdin, inherit stderr, pass through the stdout summary, propagate exit) mirroring `sample.rs`; `main.rs` swaps `stub("eval")` for `eval::run`; `tests/e2e.rs` adds a torch-free `a2d eval` against a FAKE `a2d-eval` (writes a stub `report.json`, exits 0/2) asserting `eval/report.json` appears and exit codes propagate, plus a missing-`model/` abort.
   CHECK (GPU-free): `cargo test --workspace`; `cargo fmt --all --check`; `cargo clippy --workspace --all-targets -- -D warnings`.

7. **Real-GPT-2 acceptance (the ONE compute-bound step, on this Mac's MPS/CPU).**
   `a2d eval runs/gpt2-diffusion --data ./fixtures/eval/held_out.jsonl --html`, then open `runs/gpt2-diffusion/eval/report.html`; confirm the bound, the two task numbers, and the AR-vs-diffusion throughput render, and that `eval/report.json` matches SPEC §4.3.
   This is the inherently-long, human-observed exit item; CPU CI proves the machinery, this proves the numbers render on a real conversion.

8. **Full gate sweep.**
   CHECK: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`; `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`; `bash scripts/codegen.sh && git diff --exit-code`.

## Verification (exit criterion)

Every command below runs on this arm64/macOS Mac with CUDA absent and no network downloads, EXCEPT the single acceptance run flagged inline as compute-bound and shown small.

```sh
uv sync && cargo build --workspace

# --- Contracts round-trip and codegen drift (GPU-free) ---
cargo test -p a2d-contracts   # EvalRequest/EvalReport roundtrip; ar_baseline None round-trips
bash scripts/codegen.sh && git diff --exit-code schema/ packages/a2d-contracts/

# --- The eval core, proven GPU-free on a seeded tiny GPT-2 (no download) ---
uv run pytest packages/a2d-worker-hf/tests/test_likelihood.py   # MDLM bound finite < random control; std_error shrinks with mc_samples
uv run pytest packages/a2d-worker-hf/tests/test_tasks.py        # both tasks in [0,1], deterministic; cloze picks argmin-bound
uv run pytest packages/a2d-worker-hf/tests/test_throughput.py   # diffusion + AR tokens/sec finite > 0; None when source absent
uv run pytest packages/a2d-worker-hf/tests/test_eval_report.py  # report matches contract; report.json (+html) written; seed-reproducible
uv run pytest packages/a2d-worker-hf/tests/test_eval_main.py    # a2d-eval on a tiny model/ writes the report; garbage stdin exits 2

# --- Torch-free Rust e2e (fake a2d-eval): eval/ shape + exit codes 0/2 ---
cargo test -p a2d-cli --test e2e

# --- All gates green ---
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest

# --- THE ONE compute-bound step (MPS/CPU, shown small): real report on the P2 diffusion run ---
cargo run -p a2d-cli -- eval runs/gpt2-diffusion --data ./fixtures/eval/held_out.jsonl --html
cat runs/gpt2-diffusion/eval/report.json        # likelihood bound + 2 tasks + throughput + (best-effort) AR baseline
open runs/gpt2-diffusion/eval/report.html       # AR-vs-diffusion comparison renders
```

## Risks and ambiguities resolved

1. **Eval-comparability: diffusion NLL is not AR perplexity.**
   TOP RISK, and a public-honesty obligation (SPEC §10, §11).
   Resolved by Decision 5: lead every comparison with the MDLM likelihood bound + downstream tasks + throughput, carry AR perplexity in a separate block with an inline incomparability note, and never emit a single "diffusion beats AR" perplexity line.

2. **Downstream task choice is defensible-but-modest at GPT-2 scale.**
   Two intrinsic probes (`infill_accuracy`, `cloze_likelihood`) are honest measurements, not SOTA benchmark claims; both are cheap, local, deterministic, and native to MDLM, and the `EVAL_TASKS` registry makes LAMBADA/HellaSwag/`hf:` sets one-file adds, so the initial pair is a floor, not a ceiling (Decision 3).

3. **The source AR model may be absent or changed when eval runs later.**
   Resolved by Decision 6: hash-verify `source_hash` before trusting `source_model`, and degrade `ar_baseline`/`ar_tokens_per_sec` to `None` with a reason rather than failing, since eval's subject (`model/`) is always present.

4. **MDLM bound is a Monte-Carlo estimate, so it has variance.**
   Resolved by Decision 2: draw `mc_samples` fresh `t` per chunk, report the MC `std_error`, and seed the generator so the estimate is reproducible; the test asserts `std_error` shrinks as `mc_samples` grows.

5. **MDLM throughput will look bad versus AR.**
   Accepted and framed honestly (Decision 4): MDLM's fixed `num_steps` forward passes are slower than AR decode, the decode payoff is BD3LM (Phase 5), and Phase 3 measures the baseline for that future comparison, reporting `speedup` truthfully even when it is below 1.

6. **`report.html` scope creep.**
   Bounded by Decision 8: one static, framework-free, JS-free template written by the worker; richer views wait for the platform track (SPEC §8), ponytail-noted.

7. **Reproducibility versus the timestamp.**
   Resolved by Decision 7: everything is seeded and `report.json` is self-contained, so the determinism test compares two same-seed reports with `created_at` excluded.

8. **Two new export-schema roots widen the contract.**
   Low risk: it is the exact mechanism `SampleRequest` used in Phase 2 (two `export-schema.rs` lines, two `_MODULES` entries), the codegen drift check enforces Rust-Python agreement, and no existing type changes, so Phase-0/1/2 artifacts are untouched.

9. **No `a2d-run`/manifest change is a deliberate deviation from a "put eval in the manifest" reading of SPEC §4.3.**
   Defended by Decision 7: `eval/report.json` IS the SPEC §4.3 artifact and is self-contained; a manifest copy would be a second, drift-prone source of truth, so it is intentionally omitted, flagged here rather than dropped silently.

10. **Scope discipline: MoE-router eval telemetry (P4), BD3LM block-sampling throughput and the decode-speed payoff (P5), SWA/sink model eval (P6), event-streamed eval + `eval/events.jsonl`, and `hf:`-streamed benchmark tasks are explicitly NOT built in P3.**
    Flagged so they are not silently smuggled in; Phase 3 is the dense GPT-2 report only, and each deferred item maps to a later phase or a one-file registry add.
```
