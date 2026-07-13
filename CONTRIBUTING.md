# Contributing to a2d

Thanks for your interest. This page is the short version: how to build, test, lint, and regenerate code.

## Prerequisites

- [rustup](https://rustup.rs/) - the Rust toolchain is pinned by `rust-toolchain.toml` (stable + rustfmt + clippy).
- [uv](https://docs.astral.sh/uv/) - manages the Python workspace and its single `uv.lock`.
  torch is pinned to the CPU-only PyTorch wheel index in the root `pyproject.toml`, so `uv sync` installs the ~100MB `+cpu` build instead of the default Linux wheel and its ~3GB of NVIDIA CUDA libraries.
  macOS wheels are unaffected (MPS still works); if you need CUDA locally, point the `torch` entry in `[tool.uv.sources]` at a CUDA index and re-lock, but don't commit that.

## The four commands

```sh
# build
cargo build --workspace

# test
cargo test --workspace
uv run pytest

# lint
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check .
uv run ruff format --check .
uv run mypy

# codegen (regenerate boundary types)
bash scripts/codegen.sh
```

## Boundary types are generated

The Rust ↔ Python boundary types are the single source of truth and are generated - do not hand-edit the
Python contract models.

To change a boundary type:

1. Edit the Rust types in `crates/a2d-contracts`.
2. Run `bash scripts/codegen.sh`. This exports JSON Schema to `schema/` and regenerates the pydantic models
   in `packages/a2d-contracts`.
3. Commit `crates/`, `schema/`, and `packages/a2d-contracts` together so they never drift. CI runs codegen
   and fails on any diff.

## The contribution / extension model

Read [`docs/SPEC-HANDOFF.md` section 3.3](docs/SPEC-HANDOFF.md) first. It is THE model for how a2d grows:
supporting a new model, attention variant, objective, weights format, or eval task each maps to exactly one
extension point you add a file to - never existing code you edit. Registries, not switch statements. If your
change requires editing an existing module to route to new behavior, it is in the wrong place.

## Pull requests

- Keep PRs focused: one change per PR.
  Extension-point additions (an adapter, a handler, an objective) are the ideal shape - one new file, one fixture or test.
- Run the four commands above before pushing; CI runs exactly the same gates and must be green.
- New non-trivial logic needs a test.
  Fixture-driven where possible: a new adapter means a new `fixtures/configs/` entry, which puts it in the parameterized gate matrix automatically.
- PRs are squash-merged, so the PR title becomes the commit message - write it like one.

## Finding work

Check the [roadmap](docs/SPEC-HANDOFF.md) for the current phase and the
[issue tracker](https://github.com/undeemed/a2d/issues) - issues labeled `good first issue` are scoped to a
single extension point or file.
