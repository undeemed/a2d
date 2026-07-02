# a2d Phase 1 Implementation Plan - Detection, gate & ingest

**Status:** approved plan, pre-implementation.
**Source spec:** [`SPEC-HANDOFF.md`](SPEC-HANDOFF.md) §3.1 (capability taxonomy), §3.2 (two-tier config-only detection), §3.3 (open-closed extension), §4.1 (detect output/CLI), §4.3 (manifest records ModelSpec), §5 (repo layout for a2d-detect), §6 (Phase 1 scope and exit criteria).

## Context

Phase 0 shipped the walking skeleton: every layer exists and talks, and `a2d detect` is a stub that exits 2.
Phase 1 fills in the tool's spine - the part that classifies a model from its config alone - and it ships without a single GPU-hour.
The work is pure Rust in `crates/a2d-detect` (the `GenericAdapter` workhorse, a typed capability taxonomy, the gate, and the `a2d detect` CLI render) plus one torch-free Python ingest normalizer, exercised end to end from a `config.json` fixture corpus.
The `GenericAdapter` classifies 9 of the 10 corpus models and the entire reject set from standard HF config fields; known adapters exist only to pin trust and, for `gpt_oss`, to pin the one capability (`attn.sink`) that has no reliable generic config signal.
Detect types stay crate-internal in `a2d-detect/src/spec.rs` this phase; Phase 1 is where the shape of `ModelSpec` is still being discovered, and nothing in Python reads a `DetectReport` yet, so freezing it into codegen'd pydantic now would buy only drift churn.

**Exit criterion:** the curated `config.json` corpus classifies correctly - ACCEPT for GPT-2, Pythia, Llama, Qwen-dense, and OLMoE; REJECT with the correct reasons for Mistral and Gemma-2/3 (`attn.swa`), GPT-OSS (`attn.swa` + `attn.sink` + `weights.mxfp4`), and Mamba (`paradigm.ssm`) - AND config-only detect runs on a weights-less dir that holds just a `config.json`, printing the `hf download` hint.

## Layout to create

Follows SPEC-HANDOFF §5: `ModelSpec` and the capability tags live in `a2d-detect/src/spec.rs`, the gate in `gate.rs`, the `GenericAdapter` workhorse in `generic.rs`, and one file per quirky `model_type` under `adapters/`.
`fixtures/configs/` becomes the real config corpus feeding parameterized gate tests; `fixtures/golden/` stays a Phase 2 concern and is untouched.

```
crates/
  a2d-detect/
    Cargo.toml                   # EDIT: + serde (derive), serde_json, anyhow (workspace); NO a2d-contracts
    src/
      lib.rs                     # EDIT: extend Adapter trait with fn detect(&self, &RawConfig) -> ModelSpec;
                                 #       declare mods; add pub fn detect(dir) -> Result<DetectReport>
                                 #       (parse -> adapter/generic -> gate -> locate triple -> plan);
                                 #       fix DummyAdapter test for the new trait method
      spec.rs                    # NEW: Capability enum (as_str/reason/blocking), ModelSpec, Verdict,
                                 #      WeightsStatus, Plan, MaskStrategy, DetectReport (Serialize-only)
      gate.rs                    # NEW: gate(&ModelSpec) -> Verdict via Capability::blocking, deterministic order
      generic.rs                 # NEW: RawConfig alias accessors + generic::detect (workhorse) + GenericAdapter
      adapters/
        mod.rs                   # EDIT (sanctioned registry-table edit): mod gpt2; mod llama; mod qwen2; mod olmoe; mod gpt_oss;
        gpt2.rs                  # NEW: generic::detect + inferred=false (trusted)
        llama.rs                 # NEW: generic::detect + inferred=false
        qwen2.rs                 # NEW: generic::detect + inferred=false + use_sliding_window note
        olmoe.rs                 # NEW: generic::detect + inferred=false (trust-pin only; OLMoE has no shared experts)
        gpt_oss.rs               # NEW: generic::detect + pin attn.sink + inferred=false (reject-pinning; the 5th)
    tests/
      corpus.rs                  # NEW: parameterized fixture matrix (verdict + reason-set + weights vs sidecar)
  a2d-cli/
    Cargo.toml                   # EDIT: + a2d-detect path dep
    src/
      main.rs                    # EDIT: add --json to Detect; wire Detect -> detect::run (drop stub)
      detect.rs                  # NEW: human + --json render, exit codes 0/1/2
    tests/
      e2e.rs                     # EDIT: replace stub-exit-2 test with gpt2 / mamba / config-only-weightless / no-config-exit-2 cases
fixtures/configs/                # NEW corpus (config-only directories)
  gpt2/{config.json,expected.json}
  pythia/{config.json,expected.json}
  llama/{config.json,expected.json}
  qwen2/{config.json,expected.json}
  olmoe/{config.json,expected.json}
  mistral-v0.1/{config.json,expected.json}
  gemma2/{config.json,expected.json}
  gemma3/{config.json,expected.json}
  gpt-oss/{config.json,expected.json}
  mamba/{config.json,expected.json}
  README.md                      # NEW: sidecar convention + the mistral-must-be-v0.1 note
packages/a2d-worker-hf/
  src/a2d_core/ingest/
    __init__.py                  # EDIT: create INGEST = Registry("ingest") (expose .register); import the safetensors normalizer (registry-table edit)
    safetensors.py               # NEW: stdlib pass-through normalizer, self-registers, no new dep
  tests/test_ingest.py           # NEW: unit test for the pass-through normalizer
# UNTOUCHED on purpose (Decision 1 defers): a2d-contracts, export-schema.rs, scripts/codegen.sh,
#   schema/, generated pydantic, a2d-run, Manifest
```

## Detect types

All detect types live Rust-internal in `crates/a2d-detect/src/spec.rs`; NOT in `a2d-contracts` this phase (see Decision 1).
Every type derives `Serialize + Debug + Clone + PartialEq` only (except `Capability`, whose `Serialize` is hand-written to emit the dotted wire form via `as_str()` - see below) - `Serialize` powers both `--json` and the human render; there is no `Deserialize`, no `schemars`, and no `a2d-contracts` dependency.
A `// PHASE 2: promote to a2d-contracts + schemars, wire into Manifest` note sits at the top of `spec.rs` to record the promotion path.

```rust
// spec.rs

/// One namespaced capability tag from SPEC 3.1.
/// Exhaustive enum so classification is a compiler-checked match:
/// a new Phase-6 tag forces you to fill in as_str/reason/blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]  // NOT Serialize; see the hand-written impl below
pub enum Capability {
    ParadigmArTransformer, ParadigmSsm,
    AttnFull, AttnGqa, AttnSwa, AttnSink, AttnMla,
    PosRope, PosRopePartial, PosLearned, PosAlibi,
    FfnDense, FfnMoe, FfnMoeSharedExperts,
    NormRms, NormSandwich,
    HeadLogitSoftcap,
    WeightsBf16, WeightsMxfp4, WeightsGptq,
}

/// Hand-written so as_str() is the GENUINE single source of the wire spelling.
/// `#[derive(Serialize)]` would emit the PascalCase variant identifier ("AttnSwa"),
/// and rename_all="snake_case" would emit "attn_swa" - neither is the dotted wire
/// form "attn.swa" that SPEC 4.1's --json capability tags require.
impl serde::Serialize for Capability {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl Capability {
    /// The dotted wire form, e.g. "attn.swa". THE single source of the spelling;
    /// reused by Display, the hand-written Serialize impl above, and --json.
    pub fn as_str(self) -> &'static str { /* ... */ }
    /// Human gate reason; non-empty only for blocking caps.
    pub fn reason(self) -> &'static str { /* ... */ }
    /// True ONLY for the six caps Phase 1 cannot handle.
    pub fn blocking(self) -> bool {
        matches!(self,
            Capability::ParadigmSsm | Capability::AttnSwa | Capability::AttnSink
            | Capability::AttnMla | Capability::WeightsMxfp4 | Capability::WeightsGptq)
    }
}

/// Flat, pure description of the model. No AttentionSpec/FFNSpec nesting this phase.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub model_type: String,
    pub n_layers: u64,
    pub d_model: u64,
    pub vocab_size: u64,
    pub n_heads: u64,
    pub n_kv_heads: u64,
    pub sliding_window: Option<u64>,
    pub n_experts: Option<u64>,
    pub n_active_experts: Option<u64>,
    pub capabilities: Vec<Capability>,
    pub mask_token_id: Option<i64>,
    pub inferred: bool,          // true from GenericAdapter, false from a trusted known adapter
}

/// Architecture verdict. Internally tagged for future contract compatibility.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Supported,
    SupportedInferred,
    Unsupported { reasons: Vec<String> },
}

/// Files-on-disk axis, ORTHOGONAL to the verdict (a model can be Supported AND weightless).
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "weights", rename_all = "snake_case")]
pub enum WeightsStatus {
    Present { format: String },
    Missing { hint: String },    // "hf download <repo-id> --local-dir <dir>"
}

/// Report-only mask-token decision; the vocab mutation happens at convert-time (Phase 2).
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum MaskStrategy {
    ReuseId { id: i64, token: String },
    GrowVocab,
    Undetermined,                // no tokenizer files present (the config-only fixtures)
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Plan {
    pub objective: String,               // "mdlm" (SPEC 4.2 default)
    pub anneal_schedule: String,         // placeholder descriptor; numbers are a convert flag
    pub mask_token: MaskStrategy,
    pub estimated_memory_gb: Option<f64>,// coarse dims-only estimate, "(approx, weights-only)"
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct DetectReport {
    pub spec: ModelSpec,
    pub verdict: Verdict,
    pub weights: WeightsStatus,
    pub plan: Option<Plan>,              // Some for Supported/SupportedInferred, None for Unsupported
}
```

`RawConfig` lives in `generic.rs` and wraps the parsed `serde_json::Value` with alias-aware accessors (`u64(&[aliases])`, `u64_opt`, `f64_opt`, `bool_opt`, `str_at(dotted)`, `has_any`); it is the DRY core that keeps every adapter tiny.

## Key decisions

- **Decision 1 - types stay crate-internal in `a2d-detect/src/spec.rs`, NOT in `a2d-contracts`.**
  SPEC 5 literally places `ModelSpec` and the capability tags in `a2d-detect/src/spec.rs` and reserves contracts for cross-boundary types.
  Phase 1 does not wire convert or manifest, and nothing in Python reads a `DetectReport` (detect is pure Rust emitting text + `serde_json` `--json`), so `schemars` + `datamodel-codegen` + pydantic would have zero importers this phase.
  Phase 1 is also the phase that discovers `ModelSpec`'s shape, and Phase 2's patcher will add module paths / `router_path` / `aux_loss_coef`, so freezing a volatile type into a CI-drift-checked cross-language contract now maximizes churn.
  Promotion is a mechanical Phase-2 move done exactly when convert/manifest first read it: relocate the structs into contracts, add `#[derive(Deserialize, JsonSchema)]` plus a `#[schemars(title = "...")]` on each internally-tagged enum variant (`Verdict` / `WeightsStatus` / `MaskStrategy`) so datamodel-codegen emits readable pydantic names (the same reason `Event`'s variants carry titles today), add three `export-schema.rs` lines, and `Manifest` gains `model_spec: Option<ModelSpec>` + `capability_set: Vec<Capability>`.
  The ARCHITECTURE-draft `paths: HashMap` and the inline `supported`/`reasons` fields are dropped from the Phase-1 `ModelSpec` (paths belongs to Phase 2's transform layer; `supported`/`reasons` is the Verdict's job, not the description's).

- **Decision 2 - `Capability` is a typed exhaustive enum, one variant per SPEC 3.1 tag (plus `ParadigmSsm`, drawn from the SPEC 4.1 `unsupported` example rather than the 3.1 tag list, needed for the Mamba reject reason).**
  Chosen over a string newtype + table because an exhaustive enum makes classification a compiler-checked match: adding a Phase-6 tag forces you to fill in `as_str`/`reason`/`blocking`, with no runtime completeness test and no way to silently miss a new tag.
  Editing the enum is the sanctioned "1 capability tag" playbook row (SPEC 3.3), not the open-closed model-adding path, so it is fine to edit.
  The two SPEC classes collapse into the one predicate `blocking()`: it returns true only for the six caps Phase 1 cannot handle (`ParadigmSsm`, `AttnSwa`, `AttnSink`, `AttnMla`, `WeightsMxfp4`, `WeightsGptq`), and returns false for every implemented conversion cap and every fidelity cap (`PosRopePartial`, `NormSandwich`, `HeadLogitSoftcap`, `WeightsBf16`).
  Thus "required subset-of-implemented" becomes "no capability is `blocking()`", with implemented-ness folded into the one predicate (no separate `IMPLEMENTED` const to keep in sync), and fidelity tags cannot block by construction.
  Enabling GPT-OSS support later is just flipping `AttnSwa`/`AttnSink` `blocking()` to false (ARCHITECTURE 5's "flip the gate").

- **Decision 3 - `GenericAdapter` is the workhorse; the `Adapter` trait gains `fn detect(&self, cfg: &RawConfig) -> ModelSpec`.**
  `generic::detect(cfg)` is both the workhorse and the shared helper every known adapter calls, and it always sets `inferred=true`.
  Field rules, each citing a recon gotcha: `paradigm.ssm` if `ssm_cfg` present OR `model_type` in a small SSM set `{mamba, mamba2}` OR no attention-head field at all (this defeats the `MambaForCausalLM`-suffix trap - paradigm is NEVER read from the `*ForCausalLM` suffix).
  `n_layers = num_hidden_layers|n_layer`, `d_model = hidden_size|n_embd`, `n_heads = num_attention_heads|n_head` (GPT-2 legacy aliases), `n_kv_heads = num_key_value_heads` else `n_heads` (the Llama gotcha: missing kv means full); `attn.gqa` iff `n_kv < n_heads`, else `attn.full`.
  SWA (the Qwen2 gotcha): `attn.swa` only iff `sliding_window` present AND `> 0` AND `use_sliding_window != false`, which accepts Qwen2.5 (window=131072 but `use_sliding_window=false`) and rejects Mistral-v0.1 (4096, no toggle), Gemma-2/3, and GPT-OSS.
  Sink: any of `{attention_sink, attention_sinks, attn_sink, sink_size}` truthy gives `attn.sink`.
  Position: `rope` if any of `{rope_theta, rotary_emb_base, rope_scaling, rotary_pct}`, then `pos.rope.partial` (fidelity) if `rotary_pct|partial_rotary_factor < 1.0` (Pythia 0.25; missing means 1.0 means full); `pos.learned` is inferred from the ABSENCE of all rope fields (the GPT-2 gotcha - learned is never a positive field).
  `ffn.moe` is auto-detected from any expert alias `{num_experts, num_local_experts, n_routed_experts}`, `n_active` from `{num_experts_per_tok, num_local_experts_per_tok, moe_top_k}`, `ffn.moe.shared-experts` from `{num_shared_experts, n_shared_experts, shared_expert_intermediate_size, expert_shared_resource_gate}`; else `ffn.dense`.
  `norm.rms` if `rms_norm_eps` present (plain LayerNorm contributes no tag - the taxonomy has no LayerNorm tag).
  Weights: `quantization_config.quant_method` mxfp4 gives `weights.mxfp4`, gptq gives `weights.gptq` (both blocking); else `torch_dtype` bfloat16 gives `weights.bf16` (fidelity; bf16 is NOT quantized).
  Softcap: `attn_logit_softcapping|final_logit_softcapping` gives `head.logit-softcap` (fidelity; Gemma-2 records it, never blocks).
  Orchestration in `lib.rs` is `let spec = match find(model_type) { Some(a) => a.detect(cfg), None => generic::detect(cfg) }`.
  Known adapters (one self-registering file each, all delegate to `generic::detect` then tweak): `gpt2` (sets `inferred=false`), `llama` (`inferred=false`, the canonical dense reference SPEC names), `qwen2` (`inferred=false`, carries the `use_sliding_window` gotcha as a doc note while the guard lives in generic), `olmoe` (`inferred=false` only - trust-pin, no cap tweak).
  OLMoE (`allenai/OLMoE-1B-7B`) is a fine-grained dropless MoE with NO shared experts (`num_experts=64`, `num_experts_per_tok=8`, no shared-expert fields), so generic already classifies it as `ffn.moe` via the standard `num_experts` alias; the adapter's only legitimate job is pinning trust (`inferred=false`), and pinning `ffn.moe.shared-experts` would record a capability the model lacks and pollute the Phase-2 manifest cap set (SPEC 4.3).
  `ffn.moe.shared-experts` therefore stays exercised only by generic's alias set this phase; if a fixture that actually has shared experts is wanted (Qwen2-MoE or DeepSeek), it is added as its own fixture/adapter, not forced onto OLMoE.
  A fifth adapter `gpt_oss` is the one deliberate deviation from SPEC's named-four: it pins `attn.sink` (delegating swa + mxfp4 to generic's standard-field reads) because recon rates the sink config field low-confidence and the exit criterion hard-requires the `attn.sink` reason.
  Zero adapters for `pythia` (`gpt_neox`), `mistral`, `gemma2`, `gemma3`, and `mamba`: Pythia routes through generic to `supported(inferred)` (showcasing the "0 files" path with partial-rope recorded as fidelity), and generic rejects the whole reject set from standard fields (you never write an adapter for a model you reject).

- **Decision 4 - the gate reduces to one predicate over the capabilities.**
  `gate.rs` has `pub fn gate(spec: &ModelSpec) -> Verdict`: collect reasons from `spec.capabilities` filtered by `Capability::blocking`, mapped through `reason()`, emitted in the caps' canonical declaration order for deterministic output (GPT-OSS always yields `[attn.swa, attn.sink, weights.mxfp4]`).
  If reasons are empty, return `SupportedInferred` when `spec.inferred` else `Supported`; otherwise `Unsupported { reasons }`.
  Fidelity caps are structurally excluded (`blocking() == false`), so Gemma-2's `head.logit-softcap` sits in `spec.capabilities` but never in `reasons`.
  Reason strings are verbatim from recon: "sliding-window attention unsupported (attn.swa)", "attention sink unsupported (attn.sink)", "MXFP4 quantization unsupported (weights.mxfp4)", "state-space model, not transformer paradigm (paradigm.ssm)", plus `attn.mla` and `weights.gptq` for completeness.

- **Decision 5 - `a2d detect` exposes a library API and a thin CLI, with three exit codes.**
  `a2d-detect` exposes `pub fn detect(dir: &Path) -> anyhow::Result<DetectReport>` (load `config.json` -> `RawConfig`; `find(model_type)` known-adapter else `GenericAdapter`; gate; locate triple; build plan), so orchestration and triple-location live in the crate and the corpus test can assert the full report, not just the verdict.
  A new `crates/a2d-cli/src/detect.rs` mirrors `convert.rs` with `pub fn run(path: &Path, json: bool) -> Result<ExitCode>`, and `main.rs` replaces `stub("detect")` with the dispatch.
  Crucially, `detect::run` does NOT let the `detect()` bad-input `Err` propagate to `main`: it matches on the result, and on `Err` it prints `error: {e:#}` to stderr and returns `Ok(ExitCode::from(2))` itself.
  This keeps main's generic `Err => FAILURE` (exit 1) reserved for truly unexpected failures and prevents bad input from colliding with the unsupported=1 verdict path; the only way `run` returns an `Err` is a bug it did not anticipate.
  The only flag is `--json` (`serde_json::to_string_pretty(&DetectReport)` to stdout; human render is the default); NOT `--accept-inferred` (a Phase-2 convert flag; detect only reports, and prints "convert requires --accept-inferred" when supported_inferred), and NOT `--trust-remote-code` (detect is config-only, executes no model code).
  Exit codes reuse the Phase-0 convention: 0 = supported or supported_inferred (convertible, regardless of weights presence); 1 = unsupported (a clean, successful gate rejection); 2 = bad input (no `config.json`, malformed JSON, unreadable path, matching Phase 0's "2 = contract violation").
  Weights-missing does not change the exit code (supported + weightless exits 0 with the hint on stderr), so `a2d detect d && a2d convert d` composes.

- **Decision 6 - ingest is split by what the exit criteria actually require.**
  Required and built fully in Rust: triple location + missing-weights hint, inside `a2d-detect`'s `detect(dir)`, so the library API is self-contained and every fixture exercises the config-only weightless path.
  It finds `config.json` (absent means exit 2 bad input), lists tokenizer files (`tokenizer.json|tokenizer.model|tokenizer_config.json|vocab.json`+`merges.txt`), and globs weights (`*.safetensors` then `*.bin`/`*.pt`/`*.pth` then `*.gguf`), producing `WeightsStatus` - `Present{format}` on any hit else `Missing{hint}` where the hint is "hf download <repo-id> --local-dir <dir>" and `<repo-id>` is the config's `_name_or_path` if present else the literal `<repo-id>`.
  On the Python side, in scope and dep-free, first create the seam instance `INGEST = Registry("ingest")` in `ingest/__init__.py` (the generic `Registry` class already lives in `registry.py`; only this per-seam instance and its `register` decorator are new), then build only the safetensors pass-through normalizer at `packages/a2d-worker-hf/src/a2d_core/ingest/safetensors.py`, self-registering on `INGEST`, validating the safetensors header with a stdlib check (8-byte LE length prefix + `json.loads` of the header) and passing an already-canonical dir through unchanged, plus one unit test and the `__init__.py` import.
  **SPEC-SCOPE DEVIATION requiring orchestrator sign-off:** SPEC 6 names the Phase-1 ingest normalizers as "(safetensors pass-through, pickle -> safetensors)", and this plan ships ONLY the pass-through, deferring `pickle -> safetensors` to Phase 2 (docstring names the blocker).
  Rationale: `pickle -> safetensors` needs torch to load pickle tensors and is a code-exec surface, torch is banned until Phase 2, and it runs lazily at convert-time anyway (no Phase-1 caller and no exit-criterion dependency); gguf/onnx are out of scope.
  This is a named Phase-1 scope item being punted, not a silent drop: the exit-criteria owner must accept that Phase 1 ships only the pass-through normalizer before implementation begins.

- **Decision 7 - fixtures are directories with a sidecar, so dropping a fixture auto-adds it to the matrix.**
  Each fixture is a directory `fixtures/configs/<name>/` holding a real (trimmed but verbatim-shaped) `config.json` plus an `expected.json` sidecar.
  Directory-per-fixture (over flat files) means the corpus test calls the real `a2d_detect::detect(dir)` on each fixture and every fixture is naturally a weights-less dir, unifying the matrix with the CLI path and directly proving the config-only exit criterion for all 10 models.
  The sidecar (over a central table or embedding fields into the config) means dropping a new `<name>/{config.json,expected.json}` adds it to the matrix with zero edits to any existing file.
  Sidecar shape: `{ "verdict": "supported|supported_inferred|unsupported", "reasons": ["attn.swa", ...], "weights": "missing" }`, with `reasons` present only for unsupported.
  Sidecar `reasons` are the dotted capability tags, NOT the full human strings: the report's `Verdict::Unsupported.reasons` are human strings that each end in `(<tag>)`, so `corpus.rs` matches order-independently by asserting every sidecar tag appears in exactly one report reason and the counts are equal.
  Pinning tags (not prose) keeps sidecars stable when the human wording in `Capability::reason()` changes.

- **Decision 8 - mask-token handling is report-only; mechanics are Phase 2.**
  Detect decides and reports a `MaskStrategy` in the plan; the vocab mutation and embedding resize happen at convert-time.
  The heuristic is stdlib-only, reading `tokenizer_config.json`'s `added_tokens_decoder` map and `special_tokens_map.json` if present (so we get id + name without parsing the heavy `tokenizer.json`): (1) `config.json` `mask_token_id` present gives `ReuseId`; (2) else a special token whose content matches a mask pattern (`<mask>`/`[MASK]`, case-insensitive) gives `ReuseId`; (3) else a reserved/unused special slot (`<|reserved_special_token_*|>`, `<unused*>`, `<|extra_*|>`) gives `ReuseId` on the first such (reuse a spare, no vocab growth - the Llama-3 case); (4) else tokenizer files present but nothing reusable gives `GrowVocab`; (5) no tokenizer files at all (the config-only weightless dir the exit criterion tests, i.e. all 10 fixtures) gives `Undetermined` with the plan note "tokenizer not present; strategy resolved at convert".
  A ponytail ceiling comment records that we do not parse the full `tokenizer.json` (the `added_tokens_decoder` map covers modern tokenizers); the upgrade path is to parse `tokenizer.json` only if a model ships mask/reserved tokens exclusively there, which is Phase 2's materialize step when the tokenizer is loaded anyway.

## Implementation order

Each step ends verifiable.

1. **`a2d-detect` deps + `spec.rs`.**
   Add serde/serde_json/anyhow (workspace); define `Capability` (`as_str`/`reason`/`blocking`), `ModelSpec`, `Verdict`, `WeightsStatus`, `Plan`, `MaskStrategy`, `DetectReport`.
   Verify: `cargo build -p a2d-detect`; a unit test asserts `blocking() == true` for exactly the six unimplemented caps and `reason()` non-empty for each of them; a second unit test asserts a capability serializes to its dotted form (`serde_json::to_string(&Capability::AttnSwa) == "\"attn.swa\""`), proving the hand-written `Serialize` reuses `as_str()`.
2. **`gate.rs`.**
   `gate(&ModelSpec) -> Verdict` via `capabilities.filter(Capability::blocking)` in canonical order.
   Verify: unit tests - `AttnSwa` + `AttnSink` + `WeightsMxfp4` gives `Unsupported` with the three reasons in fixed order; a clean trusted dense spec gives `Supported`; the same spec untrusted gives `SupportedInferred`; a spec carrying a fidelity cap has that cap absent from `reasons`.
3. **`generic.rs`.**
   `RawConfig` alias accessors + `generic::detect` encoding every gotcha (the `use_sliding_window` guard, missing kv means full, `rotary_pct < 1` means partial fidelity, absence-of-rotary means learned, `ssm_cfg`/no-heads means ssm, `quant_method` means mxfp4/gptq, expert + shared aliases mean moe, softcap fidelity); `GenericAdapter` fallback sets `inferred=true`.
   Verify: inline-JSON unit tests for Qwen-like (no swa despite `sliding_window` set), Mistral-like (swa), Pythia (partial rope not blocking), Mamba (ssm).
4. **`lib.rs` orchestration.**
   Extend the `Adapter` trait with `detect(&RawConfig) -> ModelSpec`; fix the `DummyAdapter` test; add `pub fn detect(dir) -> Result<DetectReport>` orchestrating load `config.json` (exit-2-worthy anyhow error if absent/malformed) -> `find()`|generic -> gate -> locate triple + `WeightsStatus`/hint -> build `Plan` (mdlm/anneal/mask via Decision 8/coarse memory).
   Verify: `cargo test -p a2d-detect` (the existing inventory test stays green); a tempdir with only `config.json` yields `WeightsStatus::Missing` with an "hf download" hint.
5. **`adapters/*.rs`.**
   `gpt2`/`llama`/`qwen2`/`olmoe` (delegate to `generic::detect` + `inferred=false`; olmoe is trust-pin only, no cap tweak) and `gpt_oss` (delegate + pin `attn.sink` + `inferred=false`); each `inventory::submit!`; declare in `adapters/mod.rs`.
   Verify: `find(model_type)` resolves each; a per-adapter unit test asserts trusted + expected caps (gpt_oss gives swa + sink + mxfp4; olmoe gives `ffn.moe` and NOT `ffn.moe.shared-experts`).
6. **Fixtures + corpus test.**
   Create the 10 real config-only dirs `fixtures/configs/<name>/{config.json,expected.json}` + `tests/corpus.rs` (runtime-scan subdirs, run `detect(dir)`, assert verdict tag + reason SET (order-independent) + `weights == missing` against the sidecar; assert every dir has a sidecar; assert the corpus is non-empty with at least one supported and one unsupported).
   Verify: `cargo test -p a2d-detect` green across the full accept/reject matrix - the primary exit criterion.
7. **`a2d-cli` wiring.**
   Add the `a2d-detect` dep; write `src/detect.rs` (call `a2d_detect::detect(dir)`; on the bad-input `Err` print `error: {e:#}` and return `Ok(ExitCode::from(2))` rather than propagating; human render + `--json`; map verdict to exit 0/1); wire `main.rs` `Detect{path, --json}` and drop the stub.
   Verify: `a2d detect <gpt2 fixture dir>` prints SUPPORTED exit 0; `<mamba dir>` exit 1 with `paradigm.ssm`; a weightless dir prints the "hf download" hint on stderr, exit 0; a dir with no `config.json` prints an error and exits 2.
8. **Replace the e2e stub test.**
   Swap `detect_subcommand_is_stubbed_with_exit_2` for four cases (gpt2 exit 0/SUPPORTED, mamba exit 1/paradigm.ssm, config-only weightless exit 0/hf-download-hint, and a `config.json`-less dir exit 2/error proving `run` intercepts the bad-input `Err`).
   Verify: `cargo test --workspace`, `cargo fmt --check`, `cargo clippy --all-targets -D warnings`.
9. **Python ingest normalizer.**
   Create `INGEST = Registry("ingest")` in `ingest/__init__.py`; add `safetensors.py` (stdlib header check, self-registers on `INGEST`), import it in `ingest/__init__.py`, and `tests/test_ingest.py` builds a tiny valid + invalid header and asserts pass-through / validation.
   Verify: `uv run pytest`, `uv run ruff check`/`format --check`, `uv run mypy` (strict); confirm the codegen drift job is unaffected (no contracts change this phase).

## Verification (exit criterion)

```sh
cargo build --workspace

# Primary exit criterion: the full accept/reject corpus classifies correctly,
# and every fixture is a weights-less config-only dir.
cargo test -p a2d-detect            # corpus.rs matrix green: accept 5, reject 5, all weights=missing

# Spot-check accept (exit 0) and its plan block:
cargo run -p a2d-cli -- detect fixtures/configs/gpt2      # SUPPORTED, exit 0
cargo run -p a2d-cli -- detect fixtures/configs/qwen2     # SUPPORTED (GQA; sliding_window set but use_sliding_window=false), exit 0
cargo run -p a2d-cli -- detect fixtures/configs/pythia    # SUPPORTED (inferred); partial rope recorded, never blocks; exit 0

# Spot-check reject (exit 1) with correct reasons:
cargo run -p a2d-cli -- detect fixtures/configs/mistral-v0.1  # UNSUPPORTED: attn.swa (v0.1 has active window=4096), exit 1
cargo run -p a2d-cli -- detect fixtures/configs/gemma2        # UNSUPPORTED: attn.swa only (softcap is fidelity, not a reason), exit 1
cargo run -p a2d-cli -- detect fixtures/configs/gpt-oss       # UNSUPPORTED: attn.swa + attn.sink + weights.mxfp4, exit 1
cargo run -p a2d-cli -- detect fixtures/configs/mamba         # UNSUPPORTED: paradigm.ssm (despite MambaForCausalLM), exit 1

# Config-only detect on a weights-less dir prints the download hint and still exits 0:
mkdir -p /tmp/cfg-only && cp fixtures/configs/gpt2/config.json /tmp/cfg-only/
cargo run -p a2d-cli -- detect /tmp/cfg-only    # SUPPORTED, exit 0, "hf download ..." hint on stderr

# Bad input (no config.json) is exit 2, distinct from unsupported=1 (proves run() intercepts the Err):
mkdir -p /tmp/no-cfg && rm -f /tmp/no-cfg/config.json
cargo run -p a2d-cli -- detect /tmp/no-cfg      # error on stderr, exit 2

# All four gates green:
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

## Risks and ambiguities resolved

1. **`gpt_oss` is low-confidence in recon** (the exact attention-sink and `quantization_config` field names are uncertain).
   Mitigation: the `gpt_oss` adapter pins `attn.sink` (robust to the field name), mxfp4 comes from the standard `quantization_config.quant_method`, and swa from the standard active-window rule; the vendored real `gpt-oss` `config.json` is the fixture-of-record, and if generic under-reports, the fix stays in the one `gpt_oss` file.
2. **OLMoE is HF-gated** (recon could not fetch it directly).
   Resolved: `allenai/OLMoE-1B-7B` is a fine-grained dropless MoE with `num_experts=64`, `num_experts_per_tok=8`, and NO shared-expert fields, so generic classifies it as `ffn.moe` from the standard `num_experts` alias and the `olmoe` adapter only pins trust (`inferred=false`); it must NOT pin `ffn.moe.shared-experts` (a capability the model lacks).
   If the config cannot be vendored we hand-construct a faithful minimal `config.json` (64/8, no shared-expert keys) and mark it synthetic in the README/sidecar.
3. **Gemma-3 is low-confidence** (newer, less documented).
   Mitigation: no adapter (it routes through generic); vendor the real config and let its sidecar pin whatever active-window blocker it carries, so any surprise is a fixture + sidecar update, never code churn.
4. **Generic "no attention-head field means paradigm.ssm" could mislabel a malformed transformer config.**
   Accepted ceiling (ponytail comment): such a config is unconvertible anyway, and `ssm_cfg` presence is the primary positive SSM signal for the real corpus (Mamba).
5. **Writing a fifth adapter (`gpt_oss`) deviates from SPEC 6's named first-adapters** (gpt2/llama/qwen2/olmoe).
   Defensible: it exists solely to pin `attn.sink`, which is the sanctioned capability-pinning role of a known adapter (SPEC 3.2), it guards a hard exit criterion that recon flags as fragile via generic detection, and it is one small file that never adds a supported path.
6. **Detect exit code 1 = unsupported differs from Phase 0's stub 2.**
   Chosen so `detect && convert` composes and to reuse 2 for bad-input (Phase 0's contract-violation code); this is documented and the e2e stub test is replaced.
   To keep 1 (unsupported) and 2 (bad input) from colliding, `detect::run` intercepts `detect()`'s bad-input `Err` and returns `Ok(ExitCode::from(2))` itself rather than propagating it to main's generic `Err => FAILURE` (1); an e2e case asserts the exit-2 path directly.
7. **Deferring `ModelSpec`-into-contracts means Phase 2 must promote it.**
   This is a cheap mechanical move (add `Deserialize` + `schemars` derives with a `#[schemars(title=...)]` per internally-tagged enum variant, three `export-schema.rs` lines, `Manifest` gains `Option<ModelSpec>` + `capability_set`) done exactly when convert/manifest first read it - which is also when `ModelSpec`'s shape (paths/`router_path` added by the patcher) stabilizes, avoiding schema drift-check churn in the interim.
8. **The SPEC taxonomy has no LayerNorm tag,** so generic emits no norm tag for GPT-2/Pythia (recon's "norm.default" is not a real tag).
   Harmless (norm never gates); sidecars assert the reason set, not the full cap set, and this is flagged so no phantom tag is added.
9. **The Mistral fixture must be v0.1, not v0.2.**
   Mistral-7B-Instruct-v0.2 ships `sliding_window: null` (inactive) and would wrongly pass the gate; v0.1 has the active `sliding_window: 4096` that the reject test needs, so the blocking indicator is an ACTIVE window (not null and greater than 0), not mere presence of the field.
10. **The Phase-1 Python safetensors normalizer has no runtime caller until Phase 2 convert, and `pickle -> safetensors` (a named SPEC 6 Phase-1 ingest normalizer) is deferred.**
    The pass-through is accepted as a tested, dependency-free reference impl that anchors the ingest seam.
    The `pickle -> safetensors` deferral is a SPEC-scope deviation (see Decision 6), not a silent drop: it needs torch (banned until Phase 2), is a code-exec surface, and has no Phase-1 caller or exit-criterion dependency, but it requires explicit orchestrator / exit-criteria-owner sign-off that Phase 1 ships pass-through only.
