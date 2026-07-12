# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- **What a2d is + the extension playbook:** `docs/SPEC-HANDOFF.md` (esp. §3 open-closed model, §5 repo layout) and `docs/ARCHITECTURE.md` (the ML recipe: mask annealing, MDLM, D13 identity gate). Adding a model/capability is additive by policy — one file per quirk, never edit an existing adapter/handler.
- **Detect (Rust, config-only, no GPU):** `crates/a2d-detect/`. New `model_type` quirk = one file in `src/adapters/` delegating to `generic::detect` + `inferred=false`, one `mod` line in `adapters/mod.rs`, and one `fixtures/configs/<name>/{config.json,expected.json}` (auto-picked up by `tests/corpus.rs`). Capabilities are the unit of support, not model names; the gate blocks only conversion-blocking caps.
- **Convert handlers (Python worker):** `packages/a2d-worker-hf/src/a2d_core/transform/`. Two disjoint eager causal seams — GPT-2 bakes causality into per-layer `self.bias` (`attn.full`, `attention.py`); the RoPE family (Llama/Qwen2/Gemma) routes it through the 4D mask `_update_causal_mask` builds (`attn.gqa`, `gqa_attention.py`). `worker.py` picks the handler by structural introspection (`resolve_capabilities` in `apply.py`), not from job tags — the `ConversionJob` carries no capability set. D13 rule: patched@`alpha=0` must equal base logits bit-for-bit.
- **Hermetic tests only:** never download weights (Gemma is gated; CI is CPU/no-network). Build tiny random-weight configs in-process (see `tests/conftest.py` `tiny_gpt2`/`tiny_gqa`).
- **Full CI gate before shipping:** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`; `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict), `uv run pytest`; contracts are generated — after touching `crates/a2d-contracts/` run `bash scripts/codegen.sh` (CI fails on `schema/`+`packages/a2d-contracts/` drift).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
