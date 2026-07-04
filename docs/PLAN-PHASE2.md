# a2d Phase 2 Implementation Plan - Conversion core: dense happy path

**Status:** approved plan, pre-implementation.
**Source spec:** [`SPEC-HANDOFF.md`](SPEC-HANDOFF.md) §4.2 (conversion pipeline: mask annealing, shift removal, identity gate, MDLM), §4.3 (run-dir target), §6 (Phase 2 scope and exit criteria), §9.4 (dllm wrap-vs-implement open decision); [`ARCHITECTURE.md`](ARCHITECTURE.md) M0 recipe (the three things conversion touches) and D13 (the identity test).

## Context

Phase 1 shipped the tool's spine: `a2d detect` classifies a model from its config alone, the gate rejects the unconvertible set, and the worker is still a no-op that only emits `job_started -> log -> job_completed`.
Phase 2 turns that no-op into a real GPT-2 to MDLM-diffusion conversion that runs end to end on this CUDA-less Mac.
The governing philosophy is narrow: own exactly the two things no library gives us correctly - the identity-preserving attention patch and the live identity gate - port MDLM's tiny `corrupt`/`loss` and its denoiser behind the `objectives/` and `sample/` registries (never depend on the research-grade `dllm`), and lean on HF `Trainer` for the fiddly, error-prone training/checkpoint/resume/RNG-determinism machinery.
Every ML claim is proven GPU-free by a seeded tiny-config GPT-2 whose patched-at-`anneal=0` logits are bit-identical to the base on CPU float32, so correctness never depends on a GPU or a network download.
The dev machine is macOS/Apple-Silicon with no CUDA, so "one local GPU" in the exit criterion is read as "the local accelerator (MPS) or CPU"; GPT-2 is small enough that the entire correctness surface runs on CPU and only the single "coherent text" acceptance run needs MPS.
This is also the phase that promotes `ModelSpec` + `Capability` from `a2d-detect` into `a2d-contracts` (the PLAN-PHASE1 Decision 1 commitment), done now because `a2d-run` must write `model_spec` into the manifest without depending on `a2d-detect`.

**Exit criterion:** `a2d convert` turns GPT-2 into a diffusion checkpoint on the local accelerator (MPS) or CPU; the identity gate is enforced as a hard pre-training stop (fail => abort, nothing trained); `a2d sample` produces coherent text; and the run dir matches SPEC §4.3 (`model/` HF triple + `a2d` config block, `manifest.json` with `ModelSpec` + `ConversionConfig` + identity result + data source + token count, `events.jsonl`, `checkpoints/`).

## Layout to create

Rust contract promotion + new structs land in `a2d-contracts`; `a2d-detect` re-exports the moved types; `a2d-run` grows the manifest and gains resume; the worker's `a2d_core` fills in the transform/objective/data/train/sample seams stubbed in Phase 0.
`fixtures/golden/` stays an empty placeholder (the live seeded-tiny-model gate is strictly better than a version-brittle snapshot; golden logits are a candle-track D6 artifact).

```
crates/
  a2d-contracts/
    src/lib.rs                 # EDIT: move in Capability (+Deserialize/FromStr/schema_with dotted enum) + ModelSpec
                               #       (+Deserialize/JsonSchema); add ConversionConfig, IdentityResult, SampleRequest;
                               #       Event += Progress/TrainStep/IdentityGate/Checkpoint/Metric (schemars titles);
                               #       ConversionJob += conversion_config (REQUIRED); Manifest += 6 Option+skip fields;
                               #       roundtrip + schema-enum tests
    src/bin/export-schema.rs   # EDIT: +1 line, sample_request.schema.json (all other new types ride as $defs of the 3 roots)
  a2d-detect/
    Cargo.toml                 # EDIT: + a2d-contracts path dep (new, one-way, no cycle)
    src/spec.rs                # EDIT: delete local Capability + ModelSpec bodies; pub use a2d_contracts::{Capability, ModelSpec};
                               #       keep Verdict/WeightsStatus/MaskStrategy/Plan/DetectReport (render-only, not cross-boundary)
  a2d-run/
    Cargo.toml                 # EDIT: + sha2 (source_hash)
    src/rundir.rs              # EDIT: extended manifest fields; read_manifest(); source_hash helper; resume path (reopen to Running)
    src/worker.rs              # EDIT: run_job(+model_spec,+conversion_config,+source_hash); accumulate identity/token_count
                               #       from the event stream into the manifest; resume_job()
    src/lib.rs                 # EDIT: export resume_job
  a2d-cli/
    src/main.rs                # EDIT: convert flags; wire Resume + Sample (drop stubs)
    src/convert.rs             # EDIT: detect + gate + accept-inferred -> build ConversionConfig -> run_job; render new events
    src/resume.rs              # NEW: read manifest -> a2d_run::resume_job
    src/sample.rs              # NEW: build SampleRequest, spawn a2d-sample, print text
    tests/e2e.rs               # EDIT: torch-free convert-smoke via fake worker (run-dir shape + manifest merge + exit codes)
schema/*.schema.json           # REGEN (checked in, CI drift-checked) + new sample_request.schema.json
packages/a2d-contracts/
  src/a2d_contracts/__init__.py  # EDIT: __all__ + TYPE_CHECKING + _MODULES += 'sample_request_schema' for SampleRequest
  src/a2d_contracts/models/      # REGEN via scripts/codegen.sh
packages/a2d-worker-hf/
  pyproject.toml               # EDIT: + torch/transformers/accelerate/safetensors; + a2d-sample script; mypy overrides
  src/a2d_core/
    worker.py                  # EDIT: convert pipeline (ingest -> materialize -> grow mask -> patch -> identity hard-stop ->
                               #       Trainer -> save model/ + a2d block); resume; LAZY heavy imports; emit new events
    sample_main.py             # NEW: a2d-sample entry (validate SampleRequest, denoise, print)
    device.py                  # NEW: select_device / select_dtype
    transform/attention.py     # NEW: AnnealState + patch GPT-2 eager seam (neutralize self.bias, inject clamp(log(alpha),finfo.min) additive mask; eager, use_cache=False)
    transform/apply.py         # NEW: load model, resolve/grow mask token, apply handlers by capability, wire AnnealState
    transform/identity.py      # NEW: base-vs-patched@0 float32 CPU gate -> IdentityResult (slice grown vocab)
    transform/handlers/__init__.py   # EDIT: TRANSFORM registry + import full_attention
    transform/handlers/full_attention.py  # NEW: @register("attn.full") installs the anneal patch
    objectives/__init__.py     # EDIT: OBJECTIVES registry + import mdlm
    objectives/base.py         # NEW: Objective protocol (corrupt, loss)
    objectives/mdlm.py         # NEW: @register("mdlm") corrupt collator + t-reweighted CE loss (ported MDLM)
    data/__init__.py           # EDIT: DATA registry + import jsonl, txt
    data/jsonl.py data/txt.py  # NEW: local readers -> tokenize -> concat-and-chunk torch Dataset (self-register)
    train/continual.py         # NEW: HF Trainer wiring (collator=mdlm.corrupt, compute_loss=mdlm.loss, resume_from_checkpoint)
    train/callbacks.py         # NEW: AnnealCallback (alpha schedule + TrainStep/Checkpoint events)
    sample/__init__.py         # EDIT: SAMPLERS registry + import denoiser
    sample/denoiser.py         # NEW: MDLM iterative confidence-reveal denoiser (remask policy inline, ponytail)
  tests/
    test_anneal.py test_identity.py test_bidir.py test_mdlm.py test_denoiser.py test_smoke_convert.py test_resume.py test_sample.py  # NEW (all CPU)
    test_worker.py             # EDIT: assert new pipeline events, not the no-op log
# SKIPPED (YAGNI, flagged): transform/head.py (MDLM reuses lm_head), sample/schedulers.py (P5),
#   train/finetune.py + MoERouterMonitor (P4), eval/* (P3), fixtures/golden/* (candle-track), the datasets dep
```

## Contract changes

All new/promoted types live in `crates/a2d-contracts/src/lib.rs` (the source of truth); codegen regenerates the pydantic mirrors.
The promoted `Capability` keeps its hand-written `Serialize` (the dotted `as_str()` wire form) verbatim and gains a hand-written `Deserialize` plus a hand-written `JsonSchema` impl (a manual `impl`, not a container `schema_with` - schemars only allows `schema_with` at field/variant level), because a naive derive would emit PascalCase variant names, not `attn.full`.

```rust
// PROMOTED from a2d-detect/src/spec.rs. Serialize stays; Deserialize + FromStr + a hand-written JsonSchema are new.
// Capability derives NONE of Serialize/Deserialize/JsonSchema: all three are hand-written so the dotted wire form
// ("attn.swa") stays the single source of truth. A manual `impl JsonSchema` (not a container `schema_with`, which
// schemars rejects at container level) describes the whole enum as one string with an explicit enum list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability { /* ...20 variants, unchanged... */ }

impl serde::Serialize for Capability {           // unchanged: dotted wire form
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> { s.serialize_str(self.as_str()) }
}
impl std::str::FromStr for Capability {          // NEW: inverts as_str(); exhaustive match
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> { /* "attn.swa" => Ok(Capability::AttnSwa), ... */ }
}
impl<'de> serde::Deserialize<'de> for Capability { // NEW: parse dotted form back
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?; s.parse().map_err(serde::de::Error::custom)
    }
}
impl schemars::JsonSchema for Capability {       // NEW: manual impl, NOT a derive or container schema_with
    fn schema_name() -> std::borrow::Cow<'static, str> { "Capability".into() }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // {"type":"string","title":"Capability","enum":[Capability::ALL.map(as_str)...]}
        schemars::json_schema!({ "type": "string", "title": "Capability",
            "enum": Capability::ALL.iter().map(|c| c.as_str()).collect::<Vec<_>>() })
    }
}

// PROMOTED unchanged in shape; gains Deserialize + JsonSchema.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ModelSpec { /* model_type, n_layers, d_model, vocab_size, n_heads, n_kv_heads,
    sliding_window, n_experts, n_active_experts, capabilities: Vec<Capability>, mask_token_id, inferred */ }

// NEW: the conversion knobs. REQUIRED on ConversionJob (transient, never persisted, so no back-compat reason to make it Option).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ConversionConfig {
    pub objective: String,          // "mdlm", resolved via the objectives registry
    pub data: String,               // required local jsonl/txt path
    pub anneal_steps: u64,          // attention causal->bidir window (drives ONLY the attention anneal)
    pub anneal_schedule: String,    // "linear" (cosine is a drop-in)
    pub seq_len: u64,               // default 512, <= GPT-2 n_positions 1024
    pub per_device_batch_size: u64, // default 8; -> TrainingArguments.per_device_train_batch_size
    pub grad_accum: u64,            // default 1; -> gradient_accumulation_steps
    pub lr: f64,                    // 1e-4
    pub max_steps: Option<u64>,     // exactly one of max_steps/max_tokens required
    pub max_tokens: Option<u64>,    // resolved to max_steps = ceil(max_tokens / tokens_per_step) at config-build
    pub mask_token: String,         // "reuse"|"grow", default resolved from detect's MaskStrategy
    pub keep_last: u64,             // default 3; maps to save_total_limit
    pub seed: u64,                  // 0
    pub device: String,             // "auto"|"cpu"|"mps"|"cuda"
    pub dtype: String,              // "float32"|"bfloat16", default float32
}

// NEW: the identity-gate record written into the manifest.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct IdentityResult { pub passed: bool, pub max_abs_diff: f64, pub tolerance: f64 }

// NEW: the a2d-sample worker request (the ONE new export-schema root).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct SampleRequest {
    pub schema_version: String, pub model_dir: String, pub prompt: String,
    pub canvas_len: u64, pub num_steps: u64, pub temperature: f64, pub seed: u64, pub device: String,
}

// EDIT: ConversionJob gains the required config.
#[serde(deny_unknown_fields)]
pub struct ConversionJob { /* schema_version, job_id, model_path, run_dir, */ pub conversion_config: ConversionConfig }

// EDIT: five new Event variants, each with a schemars title (as JobStarted/Log already carry).
pub enum Event {
    /* JobStarted, Log, */
    #[schemars(title = "Progress")]     Progress { stage: String, step: u64, total: Option<u64> },
    #[schemars(title = "TrainStep")]    TrainStep { step: u64, loss: f64, anneal: f64, lr: f64, tokens: u64 },
    #[schemars(title = "IdentityGate")] IdentityGate { passed: bool, max_abs_diff: f64, tolerance: f64 },
    #[schemars(title = "Checkpoint")]   Checkpoint { step: u64, path: String },
    #[schemars(title = "Metric")]       Metric { name: String, value: f64, step: u64 },
    /* JobCompleted, JobFailed */
}

// EDIT: Manifest grows six fields, ALL Option + skip_serializing_if so Phase-0/1 manifests still round-trip.
pub struct Manifest {
    /* schema_version, a2d_version, job_id, created_at, model_path, status, finished_at, */
    #[serde(skip_serializing_if = "Option::is_none")] pub model_spec: Option<ModelSpec>,
    #[serde(skip_serializing_if = "Option::is_none")] pub conversion_config: Option<ConversionConfig>,
    #[serde(skip_serializing_if = "Option::is_none")] pub identity: Option<IdentityResult>,
    #[serde(skip_serializing_if = "Option::is_none")] pub data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub source_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub token_count: Option<u64>,
}
```

**Codegen impact.**
`ModelSpec`, `Capability`, `ConversionConfig`, `IdentityResult`, and the new `Event` variants are all reachable as `$defs` of the three existing export roots (`ConversionJob`, `EventEnvelope`, `Manifest`), so `datamodel-codegen` emits them as classes into the three existing generated files and the `__init__._resolve` importlib scan finds them.
`export-schema.rs` therefore gains exactly ONE line (`sample_request.schema.json`), because `SampleRequest` is the only type not reachable from an existing root.
The Python `__init__.py` `_MODULES` tuple gains `'sample_request_schema'`, and `__all__` + the `TYPE_CHECKING` block gain `SampleRequest` (and `ConversionConfig` for typing).
`a2d-contracts` (Python) stays pure-pydantic and torch-free, so codegen is untouched by the worker's new torch dependency.
The CI guard is unchanged: `scripts/codegen.sh && git diff --exit-code schema/ packages/a2d-contracts/` plus `cargo test`, applied atomically (Rust and Python regenerate together).

## Key decisions

- **Decision 1 - implement MDLM directly; do NOT depend on `dllm` (resolves SPEC §9.4).**
  MDLM is a 2-method interface (`corrupt`/`loss`) plus a ~60-line denoiser; `dllm` (ZHZisZZ/dllm) is a script-shaped research framework that would jeopardize the single `uv.lock` and mypy-strict gates for a ~200-line payload, and ARCHITECTURE mandates owning the transform/patch layer regardless because the identity gate must be auditable line-by-line.
  What we DO lean on: `transformers` (`AutoModelForCausalLM` load/save, tokenizer, `GPT2Config`) and HF `Trainer` + `accelerate` for the training loop, optimizer, LR schedule, gradient accumulation, `save_total_limit` checkpointing, `resume_from_checkpoint`, and device placement - the lazy-correct choice because `Trainer` deletes exactly the RNG/optimizer/step-resume determinism code that is most error-prone to hand-roll.
  We do NOT add the `datasets` library: a ~40-line torch `Dataset` over jsonl/txt (tokenize + concat-and-chunk to `seq_len`) feeds `Trainer` directly, avoiding the pyarrow weight (parquet/hf-streaming is a one-file add behind the data registry later).
  `objectives/mdlm.py` carries the ponytail: `# ponytail: MDLM as weighted-MLM ~200 lines; swap for dllm.objectives.MDLM behind this same corrupt/loss protocol if P5 BD3LM justifies the dep`.

- **Decision 2 - patch GPT-2's causality at its REAL seam (neutralize `self.bias`, re-supply an annealed additive mask); bit-identical to base at `alpha=0` and PROVABLY bidirectional at `alpha=1`.**
  Pin `transformers==4.48.3`.
  In that release, as in every GPT-2 since the original, eager GPT-2 applies causality INSIDE the attention op: `GPT2Attention` masks with its lower-triangular `self.bias` buffer via `torch.where(causal, scores, mask_value=torch.finfo(float32).min)`, and the passed 4D `attention_mask` is padding-only.
  A model-level additive-mask factory therefore does NOT control GPT-2 causality, so we patch the seam that does: force `attn_implementation="eager"` and `use_cache=False`, then monkeypatch the eager attention path (`GPT2Attention._attn` / `eager_attention_forward`) to neutralize `self.bias` (register it all-True so `torch.where` never masks) and inject a single annealed additive mask in its place.
  A single mutable `AnnealState(alpha in [0,1])`, shared by all layers, drives that mask: on/below-diagonal entries are `0`, strictly-future (`j>i`) entries are `clamp(log(alpha), min=torch.finfo(float32).min)`.
  At `alpha=0` future entries are `finfo.min`, and because a finite pre-softmax score is negligible against `finfo.min` in float32 (`score + finfo.min == finfo.min` to the bit), the additive path reproduces base's `torch.where`-to-`finfo.min` scores EXACTLY, so base and patched@0 run identical scores and logits match to `0.0`.
  At `alpha=1` future entries are `0` and, because `self.bias` no longer masks, attention is fully bidirectional; intermediate `alpha` scales each future position's pre-softmax mass by exactly `alpha`, a smooth monotone causal-to-bidirectional reveal.
  This deliberately corrects the ML-recon's naive `(1-a)*(-inf) + a*0` blend, which is `-inf` for every `a<1` and therefore not an anneal at all.
  CRITICAL: the `alpha=0` identity gate CANNOT catch a patch that never reaches GPT-2's causality, because a no-op seam that leaves the model fully causal is bit-identical to base too; so bidirectionality is proven SEPARATELY by `test_bidir.py` (Decision 2's true regression guard, step 7), which asserts that perturbing a strictly-future token MOVES an earlier position's logits at `alpha=1` and does NOT move them at `alpha=0`.
  If a `transformers` bump moves or renames the seam, both the identity gate and `test_bidir` fail loudly and the fix stays localized to `transform/attention.py`.

- **Decision 3 - AR-shift removal is loss-side only; the identity test needs no shift code.**
  GPT-2's forward is already position-aligned (`logits[i] = P(next | <=i)`); the AR shift is purely a loss-side artifact.
  MDLM never passes `labels=` and never shifts - it feeds raw logits to `mdlm.loss`, which scores masked positions against clean targets at the SAME index.
  The identity test "restores the shift + full causality" simply by comparing raw forward logits at `alpha=0` on clean (uncorrupted) input, which needs no shift code because the comparison is on the forward output, not on a loss.

- **Decision 4 - ONE anneal knob (`anneal_steps`), driving only the attention anneal.**
  MDLM samples its own per-sequence diffusion time `t ~ U(0,1]` every step for corruption, so there is no separate mask-ratio schedule to expose; this is the correct MDLM reading and resolves the recon's "two annealings" ambiguity.
  `AnnealState.alpha = schedule(global_step)` is derived statelessly from the step by an `AnnealCallback` on `on_step_begin`, so resume is automatically correct with no persisted anneal state.
  Config-build VALIDATES `anneal_steps <= effective_max_steps` (the resolved `max_steps`, whether passed directly or derived from `max_tokens`): a violation is a hard config error, because a run that ends before anneal completes would deliver an only-partially-bidirectional checkpoint that `a2d sample` then forces to `alpha=1` it was never trained at.
  The final `alpha` actually reached is recorded in the `model/` `a2d` config block so a short run is auditable, not silently wrong.

- **Decision 5 - promote `ModelSpec` + `Capability` into `a2d-contracts` now; keep the rest in `a2d-detect`.**
  This is the PLAN-PHASE1 commitment, triggered because `a2d-run` must write `model_spec` into the manifest without depending on `a2d-detect` (`a2d-run` still does not; `a2d-cli` bridges them).
  `Capability` keeps its hand-written `Serialize`, gains hand-written `Deserialize` + `FromStr` + a hand-written `JsonSchema` impl emitting the dotted string enum, guarded by two lib.rs tests (serialize->deserialize->serialize roundtrip over `ALL`, and schema-enum == `as_str` list).
  `Verdict`/`WeightsStatus`/`MaskStrategy`/`Plan`/`DetectReport` STAY in `a2d-detect` (render-only, never crossing the worker boundary); `a2d-detect/src/spec.rs` becomes `pub use a2d_contracts::{Capability, ModelSpec}`.
  DELIBERATE deviation from PLAN-PHASE1's literal `capability_set` line: no separate manifest field - the set IS `model_spec.capabilities` (DRY, avoids two copies drifting); flagged in Risks.

- **Decision 6 - materialize is copy-on-normalize for free; the source dir is untouched by construction.**
  The worker only READS the source (`ingest.INGEST.get("safetensors")` validates headers and returns the dir; `AutoModelForCausalLM.from_pretrained(dir, attn_implementation="eager", torch_dtype=float32).eval()` + `AutoTokenizer.from_pretrained` load into memory), never mutates it.
  The deliverable `run_dir/model/` is written fresh by `save_pretrained` AFTER patch + train; intermediate states go to `run_dir/checkpoints/` via `Trainer`; so there is no `copytree`.
  `source_hash` = sha256 of the primary source safetensors file, computed by `a2d-run` (Rust owns manifest/provenance; add `sha2`; `# ponytail: hash the primary file, upgrade to header+shard-manifest if big-model provenance gets expensive`).

- **Decision 7 - mask-token default is GROW, not reuse.**
  GPT-2's BPE tokenizer has 50257 tokens (0-50256), no `<mask>`, no reserved slot, and `50256=<|endoftext|>` is load-bearing (eos AND the document-separator used during sequence packing).
  Reuse-of-eos is semantically broken for MDLM here: if the mask id equals the eos packing separator, `corrupt()` and `loss` cannot distinguish a real document boundary from a to-be-predicted mask.
  GROW mechanics: `tokenizer.add_special_tokens({"mask_token": "<|mdlm_mask|>"})` then `model.resize_token_embeddings(len(tokenizer))` gives exactly +1 row (50257) - the invariant's sanctioned single-row exception, and GPT-2 ties `wte`<->`lm_head` so one resize covers both; the new row is initialized from the mean of existing token embeddings.
  `--mask-token reuse` overrides to id 50256 only when the user explicitly accepts eos/mask conflation (e.g. non-packed data), ponytail-noted with its ceiling.
  Interaction with the gate: growing appends a logit COLUMN but leaves every existing row's values unchanged and the mask id never appears in the clean probe batch, so the gate slices patched logits to `[:, :, :base_vocab]` before comparing and stays bit-identical on shared columns; reuse keeps shapes identical so the slice is a no-op.

- **Decision 8 - continual pretrain leans on HF `Trainer` through three small hooks.**
  (1) `data_collator = objectives.mdlm.corrupt` (samples `t ~ U(0,1]` per sequence, replaces `Bernoulli(t)` tokens with `mask_token_id`, stashes clean + mask in the batch); (2) a `compute_loss` override pops clean + mask, calls `model(input_ids=noisy)` with NO labels (no shift), and returns `objectives.OBJECTIVES.get(cfg.objective).loss(logits, clean, mask)` = t-reweighted CE over masked positions only; (3) an `AnnealCallback` (the `AnnealScheduler` ARCHITECTURE names) sets `alpha` on `on_step_begin` and emits `TrainStep`/`Checkpoint` events.
  `TrainingArguments`: `output_dir=run_dir/checkpoints`, `learning_rate=cfg.lr`, `max_steps`, `per_device_train_batch_size=cfg.per_device_batch_size`, `gradient_accumulation_steps=cfg.grad_accum`, `save_steps`, `save_total_limit=cfg.keep_last` (this IS `--keep-last`), `seed=cfg.seed`, device/dtype from cfg.
  Token accounting is deterministic and reproducible from the manifest: `tokens_per_step = seq_len * per_device_batch_size * grad_accum` (single local device on this Mac), `--max-tokens` is resolved to `max_steps = ceil(max_tokens / tokens_per_step)` at config-build (exactly one of the two is required), and `manifest.token_count = actual_steps * tokens_per_step` - no reliance on `Trainer` batch-size defaults.
  RESUME is kept minimal: `a2d resume <run-dir>` reads the existing `manifest.json` for `model_path` + `conversion_config`, asserts `manifest.identity.passed` (trusts the already-run gate, SKIPS re-gating), and re-spawns the worker on the SAME run_dir (bypassing `create_run_dir`'s non-empty guard, flipping status back to `Running`, appending to the existing `events.jsonl`); the worker sees `resume=true` and calls `trainer.train(resume_from_checkpoint=latest)`, which restores optimizer/scheduler/RNG/step.

- **Decision 9 - sampling is a separate torch worker over the same stdin-JSON contract; the denoiser is one file.**
  `sample/denoiser.py` registers the MDLM iterative parallel denoiser under `"mdlm"` in a `SAMPLERS` registry (P5's BD3LM block sampler is another file, zero edits): canvas = fixed `prompt_ids` ++ `[mask_token_id]*M`, `alpha=1`, `use_cache=False`; for K steps do one forward over the whole canvas, `softmax(logits/temperature)`, take per-position max-prob confidence + argmax id, reveal the k most-confident still-masked positions on a linear top-k schedule, repeat until no mask remains, decode.
  The remask policy (confidence-ordered reveal) is folded inline with a ponytail note; `sample/schedulers.py` is created only in P5 when a second policy exists (no premature scaffolding).
  Delivery honors D9 (generated boundary types): `SampleRequest` is the one new export root, and a second `[project.scripts]` entry `a2d-sample` (`sample_main.py`) reads it on stdin, validates the generated pydantic, loads `model/`, denoises, and prints text to stdout (exit 0 ok / 2 contract violation), reusing the existing worker stdin-JSON + pydantic-validate pattern rather than inventing argv parsing; it does NOT touch manifest or events (sampling is a spot-check, not a run-dir-producing job).

## Implementation order

Each step ends verifiable; every check except step 12 runs GPU-free on this Mac.

1. **Contracts + codegen.**
   In `a2d-contracts/src/lib.rs` move in `Capability` (hand-written `Deserialize` + `FromStr` + a hand-written `JsonSchema` impl emitting the dotted string enum) + `ModelSpec` (`Deserialize`+`JsonSchema`); add `ConversionConfig` (incl. `per_device_batch_size`/`grad_accum` for reproducible token accounting), `IdentityResult`, `SampleRequest`; add `Event` variants `Progress`/`TrainStep`/`IdentityGate`/`Checkpoint`/`Metric` with schemars titles; add the required `conversion_config` to `ConversionJob`; add the six `Option` manifest fields; add the one `export-schema.rs` line for `SampleRequest`; extend the Python `__init__` `_MODULES` + `__all__`.
   CHECK (GPU-free): `cargo test -p a2d-contracts` (Capability serialize->deserialize->serialize over `ALL`, schema-enum == `as_str` list, old-shape manifest deserializes to None-fill); `bash scripts/codegen.sh && git diff --exit-code schema/ packages/a2d-contracts/` green.

2. **Re-point `a2d-detect` at contracts.**
   Add the `a2d-contracts` path dep; `spec.rs` re-exports `Capability`/`ModelSpec`; keep `Verdict`/`WeightsStatus`/`MaskStrategy`/`Plan`/`DetectReport` local.
   CHECK (GPU-free): `cargo test -p a2d-detect` (corpus matrix + `capability_serializes_to_dotted_form` still green); `cargo clippy -D warnings`.

3. **`a2d-run` manifest + resume.**
   `run_job` gains `model_spec` + `conversion_config` + `source_hash` (`sha2`); accumulate `identity`/`token_count` from the event stream into the manifest; add `read_manifest` + `resume_job` + the rundir resume path.
   CHECK (GPU-free): `cargo test -p a2d-run` (fake worker emits `IdentityGate`+`TrainStep`; manifest carries the new fields; resume reopens a non-empty dir without clobbering).

4. **`a2d-cli` convert/resume/sample wiring.**
   Convert flags + detect/gate/accept-inferred; `resume.rs`; `sample.rs`; `main` wiring; render the new events.
   CHECK (GPU-free): `cargo test --workspace`, `cargo fmt --check`, `cargo clippy --all-targets -D warnings`; `tests/e2e.rs` convert-smoke via fake worker asserts run-dir shape / manifest merge / exit codes 0/1/2; convert on an `Unsupported` fixture aborts exit 1 without spawning.

5. **Worker deps + device + skeleton.**
   `uv add torch/transformers/accelerate/safetensors`, the `a2d-sample` script, mypy overrides, `device.py`, LAZY torch imports (the exit-2 contract-violation path stays torch-free).
   CHECK (GPU-free): `uv sync`; `uv run python -c 'import torch,transformers,accelerate'`; `uv run mypy` strict + `ruff` green; `uv run pytest` existing green.

6. **transform: anneal + patch.**
   `attention.py` (`AnnealState` + the eager-seam patch for pinned `transformers==4.48.3`: neutralize `self.bias`, inject the `clamp(log(alpha), finfo.min)` additive mask), `apply.py`, the handlers registry, `full_attention.py`.
   CHECK (GPU-free): `pytest test_anneal.py` (`alpha(0)=0`, `alpha(N)=1`, monotone; the injected additive mask at `alpha=0` equals the causal `finfo.min` pattern).

7. **identity gate + bidirectionality guard.**
   `identity.py`: build a seeded tiny GPT-2, capture base logits, apply the patch, set `alpha=0`, return `IdentityResult`.
   CHECK (GPU-free, the headline proof): `pytest test_identity.py` asserts `max_abs_diff` over `logits[..., :base_vocab] == 0.0` base-vs-patched@0 on CPU float32 (tolerance `1e-6` guards reduction order); `pytest test_bidir.py` asserts that perturbing a strictly-future token MOVES an earlier position's logits at `alpha=1` and does NOT move them at `alpha=0` - the only check that proves the patch actually reaches GPT-2's causality, since the `alpha=0` gate passes even for a no-op seam.

8. **objectives.**
   Registry + `base.py` protocol + `mdlm.py` corrupt/loss.
   CHECK (GPU-free): `pytest test_mdlm.py` (`t=1` masks all => loss == plain CE; `-100`/unmasked positions ignored; hand-checked scalar).

9. **data + train.**
   `data/jsonl.py` + `data/txt.py` readers; `train/continual.py` (Trainer); `train/callbacks.py` (`AnnealCallback`, `save_total_limit=keep_last`, `resume_from_checkpoint`).
   CHECK (GPU-free): `pytest test_smoke_convert.py` runs 2 steps on the tiny model (`checkpoints/` written, `TrainStep`/`Checkpoint` events captured); `pytest test_resume.py` continues the step counter.

10. **sampling.**
    `sample/denoiser.py` + `sample_main.py` (`a2d-sample`) + `SampleRequest` wiring.
    CHECK (GPU-free): `pytest test_denoiser.py` (no mask id remains, prompt untouched, correct length); `a2d-sample` on the tiny model prints text.

11. **worker.py convert pipeline end to end.**
    ingest validate -> materialize -> grow mask token -> patch -> IDENTITY GATE (fail => emit `IdentityGate{passed:false}` + `job_failed` + exit 1, nothing trained) -> Trainer -> `save_pretrained` `model/` + the `a2d` block; emit `Progress`/`IdentityGate`/`TrainStep`/`Checkpoint`.
    CHECK (GPU-free): `pytest test_smoke_convert.py` + `test_worker.py` produce a SPEC-§4.3 run dir; the deliberately-broken-patch case aborts before any `TrainStep`.

12. **Real-GPT-2 acceptance (the ONE compute-bound step, on this Mac's MPS).**
    `a2d convert openai-community/gpt2 --device mps --data <local jsonl> --max-steps <modest>`, then `a2d sample -p '...'` shows coherent text; confirm the run dir matches SPEC §4.3.
    This is the inherently-long, human-observed exit item: CPU CI proves the machinery, MPS proves the exit criterion's "coherent text"; starting from pretrained weights keeps it to minutes-to-hours.

13. **Full gate sweep.**
    CHECK: `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`; `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`; `bash scripts/codegen.sh && git diff --exit-code`.

## Verification (exit criterion)

Every command below runs on this arm64/macOS Mac with CUDA absent and no network downloads, EXCEPT the single MPS acceptance run flagged inline as inherently compute-bound and shown small.

```sh
uv sync && cargo build --workspace

# --- Contracts round-trip and codegen drift (GPU-free) ---
cargo test -p a2d-contracts   # Capability serialize->deserialize->serialize over ALL; schema-enum==as_str; old-manifest None-fill
bash scripts/codegen.sh && git diff --exit-code schema/ packages/a2d-contracts/

# --- The ML core, proven GPU-free on a seeded tiny GPT-2 (no download) ---
uv run pytest packages/a2d-worker-hf/tests/test_anneal.py     # alpha(0)=0, alpha(N)=1, monotone; injected mask@0 == causal finfo.min
uv run pytest packages/a2d-worker-hf/tests/test_identity.py   # HEADLINE: max_abs_diff over logits[...,:base_vocab] == 0.0, CPU float32
uv run pytest packages/a2d-worker-hf/tests/test_bidir.py      # GUARD: future-token perturbation moves earlier logits @alpha=1, NOT @alpha=0
uv run pytest packages/a2d-worker-hf/tests/test_mdlm.py       # t=1 masks all -> loss == plain CE; -100 ignored; hand-checked scalar
uv run pytest packages/a2d-worker-hf/tests/test_denoiser.py   # no mask id remains; prompt untouched; correct length

# --- 1-2 step end-to-end smoke convert producing a SPEC-4.3 run dir (CPU, tiny model) ---
uv run pytest packages/a2d-worker-hf/tests/test_smoke_convert.py   # model/ triple + a2d block, manifest (model_spec+conversion_config+
                                                                   # identity+token_count), events.jsonl (IdentityGate+TrainStep+
                                                                   # Checkpoint), checkpoints/; broken-patch case aborts before TrainStep
uv run pytest packages/a2d-worker-hf/tests/test_sample.py         # a2d sample on the smoke output prints text
uv run pytest packages/a2d-worker-hf/tests/test_resume.py        # resume continues the step counter

# --- Torch-free Rust e2e (fake worker): run-dir shape + manifest merge + exit codes 0/1/2 ---
cargo test -p a2d-cli --test e2e

# --- All gates green ---
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest

# --- THE ONE compute-bound step (MPS, shown small): real GPT-2 -> coherent text ---
# Inherently long; human-observed; not part of CI. CPU above proves the machinery; this proves "coherent text".
cargo run -p a2d-cli -- convert openai-community/gpt2 --device mps --data ./fixtures/data/tiny.jsonl \
    --max-steps 500 --seq-len 512 --out runs/gpt2-diffusion
cargo run -p a2d-cli -- sample runs/gpt2-diffusion -p "The history of"   # coherent text on stdout
cat runs/gpt2-diffusion/manifest.json   # status completed; model_spec + conversion_config + identity{passed:true} + token_count
```

## Risks and ambiguities resolved

1. **A no-op anneal seam that never touches GPT-2's causality, and `transformers` version drift of that seam.**
   TOP RISK.
   GPT-2 bakes causality into `self.bias` INSIDE the attention op (`torch.where` to `finfo.min`), NOT into the model-level additive mask, so a patch aimed at a model-level mask factory would leave the model fully causal while still PASSING the `alpha=0` identity gate (which is bit-identical to base whether or not the seam is live); the conversion would then silently continual-pretrain a still-causal model.
   Mitigated: pin `transformers==4.48.3`; patch the actual eager seam (neutralize `self.bias` + inject the annealed additive mask, per Decision 2); and make `test_bidir.py` the real regression guard - it asserts an earlier position's logits CHANGE when a strictly-future token is perturbed at `alpha=1` and do NOT change at `alpha=0`, the exact property the `alpha=0` identity gate cannot verify.
   A `transformers` bump that moves the seam fails `test_bidir` (and the gate) loudly and is a one-file fix in `transform/attention.py`.

2. **bit-identical vs tolerance.**
   bf16 rounding would make `max_abs_diff != 0`.
   Resolved: the gate ALWAYS runs float32 on CPU regardless of `--dtype`; expect exactly `0.0` for eager+fp32, tolerance `1e-6` absorbs reduction order.

3. **grow-vocab adds a logit column that could look like an identity mismatch.**
   Resolved: the gate slices patched logits to `base_vocab` and feeds only clean (unmasked) input, so the new embedding row is never exercised; the mask id (`<|mdlm_mask|>`) is distinct from eos so packing separators are never confused with a mask.

4. **The ML-recon's linear `-inf` mask blend is numerically broken** (`-inf` for every `alpha<1`).
   Resolved by the `clamp(log(alpha), finfo.min)` additive penalty (a multiplicative-mass anneal); flagged as a deliberate correction of the recon.

5. **Bending HF `Trainer` for a custom objective** (custom collator, `compute_loss` without label shift, per-step `AnnealState` via callback).
   Mitigated: three well-trodden hooks; if `Trainer` proves too rigid, the fallback is a hand-written loop behind the same registries - the objective/mask/data seams are Trainer-agnostic.

6. **`ConversionJob` gains a REQUIRED `conversion_config` under `deny_unknown_fields`.**
   Safe because `ConversionJob` is transient (never persisted) and both sides regenerate together; the codegen drift check enforces Rust<->Python agreement, and the Phase-0/1 fake-worker tests are updated to supply a minimal config.

7. **Manifest back-compat.**
   New fields must not break reading Phase-0/1 manifests.
   Mitigated: all `Option` + `skip_serializing_if`; a serde test deserializes an old-shape manifest to prove None-fill.

8. **`SampleRequest` is a new export-schema root, so the Python `__init__` `_MODULES` tuple must be extended** (`_resolve` scans a fixed tuple).
   Flagged so it is not forgotten; a missing entry surfaces as an `ImportError` at test time.

9. **Deviation from PLAN-PHASE1's literal `capability_set` Manifest field.**
   Folded into `model_spec.capabilities` (DRY); defensible and consistent, flagged as a resolved deliberate deviation, not a dropped commitment.

10. **Mask-token reuse-vs-grow: reusing eos as mask is semantically hazardous with eos-packed data.**
    Resolved by defaulting to GROW for GPT-2; `--mask-token reuse` is opt-in only and ponytail-noted with its ceiling.

11. **"Coherent text" is compute-bound and subjective.**
    Mitigated: correctness (identity/wiring/run-dir/resume/sampler mechanics) is fully proven GPU-free on CPU; coherence is demonstrated small on real GPT-2 via MPS from pretrained weights and framed honestly as the single compute-limited exit item.
    The darwin/MPS constraint is explicit throughout: there is no CUDA locally, so device selection is `auto = cuda if available else (mps if available else cpu)`, `PYTORCH_ENABLE_MPS_FALLBACK=1` is documented for any un-MPS-compiled op, and dtype defaults to float32 (MPS float16 has precision quirks; bf16 is opt-in for training only).

12. **Resume trusts the recorded `manifest.identity.passed` instead of re-running the gate, and replays the data stream from the top (no persisted cursor).**
    Mitigated/accepted: resume asserts `identity.passed` before continuing (a failed run cannot silently resume into training); top-of-stream replay is fine for shuffled continual pretrain at GPT-2 scale, ponytail-flagged with a persisted-offset upgrade path.

13. **The generic `Metric` event widens the wire contract before P3 needs it** (mild YAGNI tension).
    Accepted: it is one variant, aligns with the house open-closed rule, and prevents a central Event-enum edit (with codegen churn) when P3 eval / P4 router telemetry land imminently.

14. **CI cost rises** (~800MB CPU torch wheel + slower pytest on the ubuntu python job).
    Accepted with `uv` caching; an optional-extra dependency split is the documented upgrade path.

15. **Scope discipline: MoE-router-under-anneal + finetune (P4), BD3LM + schedulers (P5), the eval harness and eval-parity (P3), golden fixtures (candle-track), `head.py`, and the `datasets` dep are explicitly NOT built in P2.**
    Flagged so they are not silently smuggled in; the MoE router monitor and eval parity in particular wait for their own phases because Phase 2 is the dense GPT-2 happy path only.