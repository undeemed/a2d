# a2d Phase 0 Implementation Plan - Walking Skeleton

**Status:** approved plan, pre-implementation.
**Source spec:** [`SPEC-HANDOFF.md`](SPEC-HANDOFF.md) §5 (repo layout) and §6 (Phase 0 scope and exit criteria).

## Context

The repo currently holds only design docs (README, ARCHITECTURE, SPEC-HANDOFF, LANDSCAPE); no code, no git history.
This plan scaffolds the full Phase 0 walking skeleton: every layer exists and talks.
Decisions settled: full Phase 0 scope (not structure-only), MIT license (originally Apache-2.0, changed post-scaffold), OSS-quality tooling from day one.

**Exit criterion:** a no-op `a2d convert` spawns the Python worker, streams events to the terminal, and leaves a well-formed run dir with `manifest.json` + `events.jsonl`.

## Layout to create

Follows the folder-nesting conventions adopted into SPEC-HANDOFF §5 after review:
all Rust crates flat under `crates/` (root = virtual manifest, dir name = crate name, per [Large Rust Workspaces](https://matklad.github.io/2021/08/22/large-rust-workspaces.html));
all Python packages flat under `packages/` with src layout (per [uv workspace docs](https://docs.astral.sh/uv/concepts/projects/workspaces/));
no single-child directories.

```
a2d/
  README.md                     # stays; License section TBD -> MIT
  LICENSE                       # MIT
  CONTRIBUTING.md               # build/test/codegen commands; points at docs/SPEC-HANDOFF.md §3.3
  .gitignore  .editorconfig  rust-toolchain.toml
  Cargo.toml                    # cargo workspace: crates/*; workspace lints
  pyproject.toml                # uv virtual workspace: packages/*; ruff/mypy config
  scripts/codegen.sh            # cargo run export-schema + datamodel-codegen; single regen entrypoint
  .github/workflows/ci.yml     # rust / python / codegen-drift jobs
  docs/                         # SPEC-HANDOFF.md, ARCHITECTURE.md, LANDSCAPE.md move here
  crates/                       # ALL Rust, flat
    a2d-contracts/              # boundary types, SOURCE OF TRUTH (serde + schemars)
      src/lib.rs                # ConversionJob, EventEnvelope, Event, Manifest, SCHEMA_VERSION
      src/bin/export-schema.rs  # writes schema/*.schema.json deterministically
    a2d-cli/                    # the `a2d` binary (clap derive)
      src/main.rs               # convert + runs live; detect/eval/sample/resume stubbed (exit 2)
      tests/e2e.rs              # spawns real binary + real uv worker = exit-criterion test
    a2d-detect/                 # skeleton proving the registry seam (Phase 1 fills it)
      src/lib.rs                # Adapter trait + inventory registry + test
      src/adapters/mod.rs       # EXTENSION POINT doc comment
    a2d-run/                    # run dirs, manifest, worker lifecycle
      src/rundir.rs             # create layout, write/finalize manifest.json
      src/worker.rs             # spawn worker, job->stdin, stream stdout JSONL, tee to events.jsonl
  packages/                     # ALL Python, flat, src layout
    a2d-contracts/              # module a2d_contracts
      src/a2d_contracts/models/ # GENERATED pydantic v2, DO NOT EDIT headers
    a2d-worker-hf/              # console script `a2d-worker`
      src/a2d_core/
        registry.py             # generic decorator Registry
        worker.py               # stdin job -> validate -> emit EventEnvelope JSONL -> exit code
        ingest/ transform/handlers/ objectives/ data/ train/ sample/ eval/tasks/
                                # each __init__.py = docstring stating the extension contract
      tests/
  schema/                       # generated JSON Schema, checked in, CI drift-verified
  fixtures/configs/  fixtures/golden/   # README placeholders per §5
```

## Contract types (Phase 0 minimal)

All types live in `crates/a2d-contracts/src/lib.rs` with `#[derive(Serialize, Deserialize, JsonSchema)]`.
Every root type carries `schema_version` (from the contracts crate version) per SPEC-HANDOFF §7.

- `ConversionJob { schema_version, job_id, model_path, run_dir }` with `deny_unknown_fields`.
- `EventEnvelope { schema_version, job_id, seq, ts, event }`.
- `Event` internally tagged enum: `job_started | log | job_completed | job_failed`, with schemars `title` per variant so generated pydantic names stay readable.
- `Manifest { schema_version, a2d_version, job_id, created_at, model_path, status, finished_at }`.

Deliberately no `Progress` event and no `ConversionConfig` yet; Phase 2 adds them when something emits them.

## Key decisions

- **Layout conventions (adopted into SPEC §5):** flat `crates/` for all Rust, flat `packages/` for all Python with src layout, generated schema at top-level `schema/`.
  A future candle worker is `crates/a2d-worker-candle/`; no `workers/` super-directory.
- **Codegen chain:** schemars 1.x -> `export-schema` bin -> checked-in `schema/` -> `datamodel-code-generator` (pinned in `uv.lock`) -> `packages/a2d-contracts/src/a2d_contracts/models/` -> ruff format.
  CI drift check runs `scripts/codegen.sh` then `git diff --exit-code schema/ packages/a2d-contracts/`.
- **events.jsonl is written by a2d-run (CLI side), teeing worker stdout.**
  One writer, and the file survives a worker crash.
  The worker writes only artifacts (from Phase 2 on).
  This resolves the §2 diagram-vs-text ambiguity in the spec.
- **Worker stdout carries JSONL envelopes only;** anything unstructured goes to stderr.
  Exit codes: 0 success, 1 job failed, 2 contract violation.
- **Worker discovery (Phase 0 answer to §9.3):** `--worker-cmd` flag -> `A2D_WORKER_CMD` env -> dev default `uv run --project <repo>/packages/a2d-worker-hf a2d-worker`.
  Packaging for end users is deferred to Phase 7.
- **`a2d runs`:** filesystem scan of `--root` (default `./runs`) for `manifest.json`, per §9.2's leaning.
- **Python:** uv workspace with a single `uv.lock`; pydantic-only deps in Phase 0 (no torch); mypy strict + ruff; Python >= 3.11.
- **Registries, not switch statements (§3.3):** Rust uses the `inventory` crate; Python uses a small decorator `Registry` class.
  Phase 0 ships the mechanism plus a test for each, so Phase 1 adapters drop in without edits.
- **Run dir in Phase 0 is exactly `manifest.json` + `events.jsonl`** - no empty `model/` or `checkpoints/` dirs.

## Implementation order

Each step ends verifiable.

1. Repo bones: `git init`; move the three docs to `docs/`; LICENSE, CONTRIBUTING.md, .gitignore, .editorconfig, rust-toolchain.toml; README license update.
2. `crates/a2d-contracts` + round-trip tests + export-schema bin; generate `schema/`.
3. Root pyproject (uv workspace) + `packages/a2d-contracts` + `scripts/codegen.sh`; generate models + `uv.lock`.
4. `packages/a2d-worker-hf`: package, registry.py, worker.py, extension-point docstrings, tests.
5. `crates/a2d-run`: rundir + manifest + spawn/stream/tee + unit test against a fake inline worker.
6. `crates/a2d-detect` skeleton: trait + inventory registry + test.
7. `crates/a2d-cli`: clap enum, convert, runs, stubs; e2e test (skips if uv missing).
8. CI workflow; full lint sweep to green.

## Verification (exit criterion)

```sh
uv sync && cargo build --workspace
mkdir -p /tmp/fake-model
cargo run -p a2d-cli -- convert /tmp/fake-model --out runs/demo   # streams events, exit 0
cat runs/demo/manifest.json        # status: completed
cat runs/demo/events.jsonl         # job_started -> log -> job_completed, monotonic seq
cargo run -p a2d-cli -- runs       # lists the demo run
cargo run -p a2d-cli -- detect x   # exit 2, "lands in Phase 1"
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
bash scripts/codegen.sh && git diff --exit-code schema/ packages/a2d-contracts/
```

## Risks and ambiguities resolved

1. **events.jsonl writer:** spec §2 diagram shows the worker writing the run dir, but the contract text says events.jsonl mirrors the stdout stream.
   Resolved: a2d-run tees; revisit if a containerized worker ever outlives the CLI (platform mode).
2. **`a2d runs` root:** the spec never pins where runs live.
   Phase 0 uses `--root` defaulting to `./runs`.
3. **Worker discovery:** the flag/env/dev-default chain is a Phase 0 answer only; the compile-time default is meaningless for a distributed binary.
4. **datamodel-code-generator naming on tagged unions:** mitigated with schemars `title` per Event variant; worker wraps construction in small helpers anyway.
5. **Empty extension dirs:** git cannot track empty dirs, so every extension point carries a contract-stating `mod.rs`/`__init__.py`/README that doubles as playbook documentation.
