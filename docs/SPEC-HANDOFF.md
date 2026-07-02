# a2d — Local Tool Spec & Phased Roadmap

**What a2d is:** a local tool that converts open-weight autoregressive (AR) LLMs into diffusion language models. Point it at a locally downloaded model → it tells you whether and how it can convert it → runs the conversion on your hardware → writes an evaluated diffusion checkpoint you can load with standard tooling.

**Companion docs:** `ARCHITECTURE.md` (the ML conversion recipe — objectives, annealing, identity test). `LANDSCAPE.md` (prior art). This doc is the tool: lifecycle, extensibility model, contracts, repo layout, roadmap.

**Conventions:** phases only — no dates, sprints, or quarters. A phase is done when its exit criteria are observable. Design principle throughout: **supporting a new model must not require editing existing code** (open-closed; see §3).

---

## 0. Locked decisions

All decisions to date, in one place. Everything else follows from these.

| # | Decision | Choice | Note |
|---|----------|--------|------|
| D1 | Scope of conversion | **Architecture-preserving**: dense→dense, MoE→MoE | Cross-family out of scope |
| D2 | Product shape | **Local CLI tool** | Converter, not host. Platform mode (API + UI) and serving are deferred tracks (§8) |
| D3 | North star | **Universal adapter** | New AR transformer → convertible day-one, zero a2d code |
| D4 | Orchestration | **Rust** CLI binary | Detection, gate, run orchestration, progress rendering |
| D5 | Compute | **Python/HF worker (primary)**, separate process | The coverage engine; wraps the HF ecosystem |
| D6 | candle native path | **Deferred track** (§8) | Was justified by serving; parked with it |
| D7 | UI | **Terminal** (CLI + logs + JSONL events) | React/Bun frontend deferred with platform mode (§8) |
| D8 | Process boundary | **Subprocess + JSONL over pipes**; run dir as artifact store | Same contract later fronted by queue + REST/SSE in platform mode. No gRPC |
| D9 | Contract format | **JSON Schema generated from Rust types** | Python worker validates against it; TS generated later |
| D10 | Canonical model format | **HF triple**: `config.json` + tokenizer + `*.safetensors` | Everything normalizes to this at ingest (§4) |
| D11 | Model weights | **Must be pre-downloaded; a2d never fetches models** | Detect works config-only; convert requires local weights (§4.1) |
| D12 | First objective | **MDLM**, then **BD3LM** | Per ARCHITECTURE.md |
| D13 | Correctness gate | **Identity test** vs base-model logits | `anneal=0` ⇒ base behavior; gates every adapter/handler |

---

## 1. Goals & non-goals

**Goals**

1. **Universal adapter (north star).** When a new open-weight AR transformer ships (the GLM/Qwen/Llama lineage), converting it requires little-to-no a2d code: detection reads its config generically, and the HF worker runs it as soon as HF Transformers does. "Diffusion GLM 5.2" is: download it, `a2d convert` it.
2. **Gate before GPU (and before download).** `a2d detect` needs only `config.json` — you get the verdict and missing capabilities *before* committing bandwidth to 15 GB of weights or hours of compute.
3. **Clean extensibility.** Each kind of change (new model, new attention variant, new objective, new format) maps to exactly one extension point (§3), added — never woven in.
4. **Full local pipeline:** detect → gate → convert → evaluate → a loadable checkpoint, with live progress (metrics, anneal schedule, MoE router health) in the terminal and on disk.
5. **Type-safe boundaries.** Rust types are the single source of truth; the Python worker's contract models are generated. Hand-written boundary types are prohibited.

**Non-goals (for now)**

- Hosted platform: HTTP API, web UI, multi-tenant queue (deferred, §8 — the internal boundaries are built so it layers on without rework).
- Serving/hosting converted models. Sampling exists for eval and spot-checks (`a2d sample`), not as a product.
- Downloading models. a2d operates on local files only; it prints the `hf download` hint when weights are missing (D11). (Training *data* may stream — see §9.)
- candle-native compute (deferred, §8).
- Cross-family conversion (dense→MoE), multimodal towers, non-transformer paradigms (SSM/Mamba) — gate returns `unsupported` honestly rather than mis-converting.
- Fine-grained Rust↔Python coupling: the worker boundary is whole-job and separate-process — never per-step hooks, never PyO3 in-process.

---

## 2. System architecture (local)

```
 ┌──────────────────────────────────────────────────────────────┐
 │  a2d CLI — Rust binary                                        │
 │  detect · gate · plan · orchestrate runs · render progress    │  ← no GPU,
 │  (a2d-detect: config.json → ModelSpec → capability gate)      │    no model code
 └────────────┬─────────────────────────────▲───────────────────┘
        spawns │ ConversionJob (JSON, stdin)  │ EventEnvelopes (JSONL, stdout)
 ┌────────────▼─────────────────────────────┴───────────────────┐
 │  Python/HF worker — separate process                          │  ← GPU
 │  a2d_core: ingest-normalize · patch · train · sample · eval   │
 └────────────┬──────────────────────────────────────────────────┘
              │ writes
 ┌────────────▼──────────────────────────────────────────────────┐
 │  Run directory (the artifact store)                            │
 │  model/ · checkpoints/ · events.jsonl · manifest.json · eval/  │
 └────────────────────────────────────────────────────────────────┘
```

**Division of labor.** The CLI never loads weights or executes model code — it parses `config.json` (pure data), gates, plans the run, spawns the worker, renders progress, and manages run directories. The worker owns everything GPU: ingest normalization, the conversion recipe, training, eval. The process boundary gives crash isolation (a training OOM can't corrupt the orchestrator); the run directory gives durability (resume = replay manifest + latest checkpoint).

**Trust model (local).** Detection is always safe (data only). The worker executes HF modeling code; `trust_remote_code` is **off by default** and enabled per-run via an explicit `--trust-remote-code` flag. Container isolation of the worker is optional locally (`--containerized`), mandatory in platform mode (§8).

**Contract.** `ConversionJob` in on stdin; `EventEnvelope` stream out on stdout (progress, metrics, anneal state, router stats, logs, errors) — mirrored to `events.jsonl`; artifacts by path. Defined once in Rust, exported as JSON Schema; the worker validates against it. Versioned from day one. Platform mode later swaps pipes for a queue and adds REST/SSE — the message types don't change.

---

## 3. The extensibility model (the core of this spec)

The reason a2d stays clean as models ship: **"support" is a set of capabilities, not a list of models.** Detection normalizes any model into a `ModelSpec` + required capability tags; the gate checks `required ⊆ implemented`. Models are never special-cased downstream of detection.

### 3.1 Capability taxonomy

Namespaced tags, e.g.: `paradigm.ar-transformer` · `attn.full` · `attn.gqa` · `attn.swa` · `attn.sink` · `attn.mla` · `pos.rope` · `pos.rope.partial` · `pos.alibi` · `pos.learned` · `ffn.dense` · `ffn.moe` · `ffn.moe.shared-experts` · `norm.rms` · `norm.sandwich` · `head.logit-softcap` · `weights.bf16` · `weights.mxfp4` · `weights.gptq`.

Two tag classes with different consequences:
- **Conversion-blocking** (the recipe must handle it): paradigm, attention variants, quantized weights. Missing handler ⇒ `unsupported` (or ingest dequant).
- **Fidelity-only** (HF's forward pass already handles it; a2d just records it): sandwich norm, logit softcap, partial rotary. Never block; they matter only for the deferred candle track (D6), where ports must reproduce them.

### 3.2 Detection: two-tier, config-only

```
config.json ─► adapter registry (keyed by model_type)
                 ├─ known adapter   → precise ModelSpec     (trusted)
                 └─ GenericAdapter  → heuristic ModelSpec   (flagged "inferred")
              ModelSpec + required caps ─► gate ─► supported | missing[] | unsupported
```

- **`GenericAdapter` is the universal-adapter workhorse.** It reads the standard HF config vocabulary (`architectures: *ForCausalLM`, `num_hidden_layers`, `num_key_value_heads`, `sliding_window`, expert fields under their common aliases…). A new model with a conventional config needs **no adapter at all**.
- **Known adapters** exist only to normalize quirks (`num_experts` vs `num_local_experts`), pin trusted capability sets, and carry per-model notes. One small file each.
- **MoE vs dense is auto-detected** (expert fields present ⇒ MoE) — never user-specified. D1 makes target family == source family by construction.

### 3.3 The extension playbook

The contract this codebase is organized around — **what you touch when something new ships:**

| Event | You add | You edit | Size |
|-------|---------|----------|------|
| New standard dense/MoE AR transformer (conventional config, HF supports it) | nothing — GenericAdapter + HF worker cover it | nothing | **0 files** |
| Same, but quirky config field names | 1 adapter in `a2d-detect/src/adapters/` + 1 config fixture | nothing | ~50 lines |
| New **attention variant** (the next SWA-like scheme) | 1 capability tag + 1 transform handler in `a2d_core/transform/handlers/` + conformance test | nothing | 1 handler |
| New **objective** (post-MDLM/BD3LM research) | 1 module in `a2d_core/objectives/` (implements `corrupt`/`loss`) | nothing | 1 file |
| New **weights format** | 1 normalizer in `a2d_core/ingest/` | nothing | 1 file |
| New **eval task** | 1 task module in `a2d_core/eval/tasks/` | nothing | 1 file |
| candle port (only if D6 revived) | candle module + weight map + golden-logits test | nothing | per model |

Enforced by convention and CI, not hope:
- **Registries, not switch statements.** Adapters/handlers/objectives/normalizers self-register (Rust: `inventory`-style registry; Python: decorator or entry-point registry). No central `match model_type` to edit — adding = dropping in a file.
- **Fixture-driven conformance.** `fixtures/configs/` holds a corpus of real `config.json`s; the gate test suite is parameterized over it — new adapter ⇒ new fixture ⇒ automatically in the matrix. `fixtures/golden/` holds pinned base-model logits for the identity test (D13): every transform handler must leave `anneal=0` behavior identical to base.
- **CI enforces the playbook.** A PR adding a model/capability that modifies existing modules (outside generated registry tables) fails by policy.

### 3.4 What "universal" honestly means

Architecture-universal within the AR-transformer family (dense + MoE — the GLM/Qwen/Llama/OLMoE lineage), not paradigm-universal. Mamba/SSM, MLA, encoder-decoder, and already-non-AR models are cleanly rejected at the gate with reasons — never silently mis-converted.

---

## 4. A model's lifecycle: ingest → convert → output

The heart of the tool. Three stages, three commands (plus `sample`/`resume` conveniences).

### 4.1 Ingest — `a2d detect <model-dir>`

Input: a **local directory** holding a downloaded model (from `hf download`, git-lfs, wherever). a2d never downloads models (D11).

1. **Locate the triple.** Find `config.json`, tokenizer files, weight files. Weights present? If not: detect still runs (config-only), but the output says exactly what's missing and prints the `hf download <repo-id> --local-dir <dir>` hint. **Convert refuses to start without local weights.**
2. **Detect + gate.** Parse `config.json` → adapter registry → `ModelSpec` + required capabilities → gate verdict:
   - `supported` — proceed; shows the plan (objective, anneal schedule, mask-token strategy, est. memory).
   - `supported (inferred)` — GenericAdapter heuristics; convert requires explicit `--accept-inferred`.
   - `unsupported` — with reasons (`attn.mla`, `paradigm.ssm`, …). Nothing to run.
3. **Normalization plan** (executed lazily at convert-time, by the worker, **copy-on-normalize — the source dir is never mutated**): pickle → safetensors; gguf → dequant + HF-layout reconstruction when feasible; quantized weights → dequant to bf16. Output of ingest is a validated, canonical HF triple in the run workspace (§4.3) — or a refusal with a reason.

Because detect is config-only, the workflow "check the verdict *before* downloading 15 GB of weights" works: run detect on a dir containing just the downloaded `config.json`.

### 4.2 Convert — `a2d convert <model-dir> --out <run-dir> [flags]`

Runs in the Python worker (subprocess), streaming `EventEnvelope`s to the terminal and `events.jsonl`:

1. **Materialize** the canonical model into the run workspace (normalization from 4.1; resolve mask token: reuse an unused special id or `--grow-vocab`).
2. **Patch** (ARCHITECTURE.md recipe): attention → bidirectional-capable with the anneal scheduler at 0; remove the AR shift; wire the diffusion objective (MDLM default; `--objective bd3lm` later).
3. **Identity gate (D13, hard stop).** Forward the *base* model and the *patched-at-anneal-0* model on probe batches; logits must match within tolerance. Locally this is computed directly (no pinned fixtures needed — the base model is right there). Fail ⇒ abort with diagnostics, nothing trained.
4. **Continual pretrain.** Anneal causal→bidirectional per schedule; MDLM loss; data from `--data <local path>` (jsonl/txt/parquet) or `--data hf:<dataset>` (streaming — the one network use, for data only). MoE models: router aux loss kept, router-health events emitted. Checkpoints land in the run dir on a cadence; `a2d resume <run-dir>` picks up from the latest.
5. **(Optional) finetune stage** on instruction data — same machinery, `--stage finetune`.

### 4.3 Output — the run directory

```
<run-dir>/
  model/                    # ★ the deliverable: converted checkpoint as a standard HF triple
    config.json             #   + `a2d` block: objective, mask-token id, anneal state, sampler defaults
    tokenizer.json …
    model.safetensors
  manifest.json             # provenance: source path+hash, a2d version, ModelSpec, ConversionConfig,
                            #   capability set, identity-test result, data source, token count
  events.jsonl              # full EventEnvelope stream (progress, loss, anneal, router stats)
  checkpoints/              # intermediate states (resume; pruned by --keep-last N)
  eval/
    report.json             # EvalReport: likelihood bound, downstream tasks, tokens/sec vs AR base
    report.html             # rendered view (a2d eval --html)
```

- **The deliverable is `model/`** — HF-layout on purpose, so it loads with standard tooling (transformers + a diffusion sampler, the dLLM library) rather than being a2d-proprietary. The `a2d` config block is what a sampler needs to run it correctly.
- `a2d eval <run-dir>` populates `eval/`; `a2d sample <run-dir> -p "prompt"` spot-checks generation (iterative denoising in the worker).
- `manifest.json` makes every run reproducible and auditable — same inputs + manifest ⇒ same run.

### 4.4 Formats at ingest (reference)

| Format | Reality | Handling |
|--------|---------|----------|
| `.safetensors` | tensors + JSON header; safe, mmap-able | **Canonical.** Pass through. |
| `.pth` / `.pt` / `.bin` | PyTorch pickle | Convert → safetensors (pickle = code-exec risk; conversion happens in the worker). Requires `config.json`. |
| `.gguf` | llama.cpp bundle; usually quantized | Inference artifact; lossy training start. Dequant + reconstruct HF layout when feasible; else reject with reason. Low priority. |
| `.onnx` | lowered inference graph | Not recoverable to a trainable model. `unsupported`; point at the source checkpoint. |

Rule of thumb: GGUF/ONNX are *deployment* artifacts; conversion wants the full-precision **source checkpoint**.

---

## 5. Repo layout

Monorepo; every §3.3 extension point is a directory whose contents are registry-loaded.

Layout conventions: all Rust crates flat under `crates/` (root is a virtual manifest, directory name = crate name, per rust-analyzer's layout); all Python packages flat under `packages/` with src layout (per uv workspace convention); no single-child directories.

```
a2d/
  crates/                       # ALL Rust, flat
    a2d-contracts/              # canonical boundary types (serde + schemars)  [source of truth]
                                #   src/bin/export-schema.rs regenerates schema/
    a2d-cli/                    # the `a2d` binary: detect|convert|resume|eval|sample|runs
                                #   arg parsing, plan rendering, progress TUI, worker supervision
    a2d-detect/
      src/spec.rs               # ModelSpec, capability tags
      src/gate.rs               # required ⊆ implemented + reasons
      src/generic.rs            # GenericAdapter (heuristics)          [the workhorse]
      src/adapters/             # ◄ EXTENSION POINT: one file per quirky model_type
    a2d-run/                    # run dirs, manifest, resume, worker process lifecycle
  packages/                     # ALL Python, flat, src layout
    a2d-contracts/              # generated pydantic models            [generated]
      src/a2d_contracts/
    a2d-worker-hf/              # PRIMARY compute engine (subprocess; optional container)
      src/a2d_core/
        ingest/                 # ◄ EXTENSION POINT: format normalizers (copy-on-normalize)
        transform/
          patch.py              # attention→bidirectional, shift removal, anneal scheduler
          handlers/             # ◄ EXTENSION POINT: capability handlers (attn.full, ffn.moe, …)
        objectives/             # ◄ EXTENSION POINT: mdlm.py, bd3lm.py  (corrupt/loss iface)
        data/                   # local corpus readers + streaming, packing, noising collators
        train/                  # continual + finetune loops; checkpointing; MoE router monitor
        sample/                 # iterative denoiser; remasking policies
        eval/
          harness.py
          tasks/                # ◄ EXTENSION POINT: eval tasks
        worker.py               # stdin job → run → stdout JSONL events; contract validation
  schema/                       # generated JSON Schema                [checked in, CI-verified]
  fixtures/
    configs/                    # real config.json corpus → parameterized gate tests
    golden/                     # pinned logits for CI identity tests (local runs compute live)
  docs/                         # ARCHITECTURE.md · SPEC-HANDOFF.md · LANDSCAPE.md
```

(`ARCHITECTURE.md` §4's module sketch maps onto `packages/a2d-worker-hf/src/a2d_core/`, with introspection in the Rust `a2d-detect` crate. A future candle worker is `crates/a2d-worker-candle/` - no `workers/` super-directory. No `frontend/` — that returns with platform mode, §8.)

---

## 6. Phased roadmap (phases only)

Dependencies: linear, except P4/P5 both depend on P2 (P3 strongly recommended first) and can run in parallel.

### Phase 0 — Walking skeleton
**Goal:** every layer exists and talks.
**Scope:** monorepo scaffold per §5; CI; contracts codegen (Rust → schema → Python, drift-checked); `a2d-cli` skeleton (`a2d --help`, `a2d runs`); worker stub that reads a `ConversionJob` from stdin and emits `EventEnvelope`s; run-dir creation + manifest.
**Exit:** a no-op `a2d convert` spawns the worker, streams events to the terminal, and leaves a well-formed run dir with `manifest.json` + `events.jsonl`.

### Phase 1 — Detection, gate & ingest (no GPU)
**Goal:** the tool's spine, shippable without a single GPU-hour.
**Scope:** `a2d-detect` (GenericAdapter + first known adapters: gpt2, llama, qwen2, olmoe); capability taxonomy; gate with reasons; `a2d detect` output (spec, verdict, plan, missing-weights hint per D11); ingest normalizers (safetensors pass-through, pickle→safetensors); fixture corpus + parameterized gate tests.
**Exit:** the curated corpus classifies correctly — GPT-2/Pythia/Llama/Qwen-dense/OLMoE accepted; Mistral & Gemma-2/3 (`attn.swa`), GPT-OSS (`attn.swa`+`attn.sink`+`weights.mxfp4`), Mamba (`paradigm`) rejected with correct reasons; config-only detect works on a weights-less dir.

### Phase 2 — Conversion core: dense happy path
**Goal:** first real conversion, end to end, locally.
**Scope:** `a2d_core` implements ARCHITECTURE.md M0 on GPT-2: materialize, patch (mask annealing, shift removal), **identity gate as a hard pre-training stop** (live base-vs-patched logit comparison), MDLM objective, short continual pretrain on `--data`, checkpointing + `a2d resume`; terminal progress rendering.
**Exit:** GPT-2 → diffusion checkpoint via `a2d convert` on one local GPU; identity gate enforced; `a2d sample` produces coherent text; the run dir matches §4.3.

### Phase 3 — Evaluation
**Goal:** trustworthy results.
**Scope:** eval harness (MDLM likelihood bound, ≥2 downstream tasks, tokens/sec vs AR base); `a2d eval` filling `eval/report.json` + `--html` render; comparison vs the source AR model.
**Exit:** every conversion can produce a reproducible report; AR↔diffusion comparison renders.

### Phase 4 — MoE
**Goal:** prove "MoE = dense + a different FFN slot, same code path" (D1).
**Scope:** OLMoE end to end; `ffn.moe` handler (router/experts untouched by the patch); router-stability monitoring (aux loss, expert utilization) as first-class events in the terminal + `events.jsonl`; anneal-schedule ablation vs routing health.
**Exit:** OLMoE converts with routing balanced through annealing; router metrics visible live.

### Phase 5 — Objectives & sampling speed
**Goal:** the diffusion payoff.
**Scope:** BD3LM objective module; block-parallel + confidence-threshold sampler (`a2d sample` uses it; not hosting — D2); throughput in eval reports.
**Exit:** a BD3LM conversion shows measurable decode speedup over MDLM/AR in eval.

### Phase 6 — Capability expansion (boss level)
**Goal:** the hard architectures, purely additively.
**Scope:** `attn.swa` and `attn.sink` transform handlers (bidirectionalizing windowed/sink attention is recipe work, not just HF support) + conformance tests; `weights.mxfp4`/`weights.gptq` dequant ingest; the gate flips automatically — Mistral, Gemma 2/3, then GPT-OSS light up with **zero changes elsewhere**.
**Exit:** one SWA model converts end to end; GPT-OSS passes detect→convert (compute permitting). The playbook (§3.3) validated on its hardest case.

### Phase 7 — Polish & (optional) platform mode
**Goal:** great local UX; open the door to the platform track if wanted.
**Scope:** resumability hardening, multi-GPU training, run management (`a2d runs`, GC/`--keep-last`), packaging/distribution (single-command install), docs; *optionally* begin platform mode (§8): actix API + queue + React/Bun UI over the same contracts.
**Exit:** clean install→detect→convert→eval on a fresh machine; (if platform started) the API serves detect/convert backed by the same worker.

---

## 7. Cross-cutting workstreams

- **Identity test (D13)** — the universal correctness gate: every adapter and transform handler must reproduce base-model behavior at `anneal=0`. Locally computed live (base model is on disk); CI uses pinned golden logits. Catches the "silent attribute" class (sandwich norm, logit softcap, partial rotary) configs don't always declare.
- **Contracts & codegen** — one Rust source of truth; generated Python (and later TS); CI drift check; versioned format (it outlives the subprocess transport into platform mode).
- **Fixtures** — every supported *and* rejected model contributes a config fixture; goldens regenerated only by an explicit, reviewed job.
- **Trust boundary** — the worker executes third-party model code: `trust_remote_code` off by default, per-run opt-in flag, optional `--containerized`; mandatory containerization when platform mode arrives.
- **Observability** — `EventEnvelope` designed once: terminal rendering, `events.jsonl`, and (later) SSE/telemetry all consume the same stream.

---

## 8. Deferred tracks (parked, with revival conditions)

- **Platform mode** — actix-web API + queue + React/Bun frontend over the *same* contracts and worker (§2 note). *Revive when:* a2d needs to be shared, multi-user, or remote-GPU. The local tool's boundaries (CLI ↔ contracts ↔ worker ↔ run dir) were drawn so this is additive: pipes→queue, terminal→SSE/UI.
- **Serving/hosting.** *Revive when:* users want to run checkpoints where they made them. Pulls the sampler up to a `/generate` surface; re-opens the candle question.
- **candle native path (D6).** Rust compute for ported models — its case (fast Rust serving, no-Python deploys, no-remote-code surface) went with serving. *Revive when:* serving lands or an all-Rust runtime becomes a requirement. If revived: small models first (`native_max_params`), ports gated by golden-logits identity tests, swappable behind the job contract.
- **protobuf-as-IDL** for worker messages (D9): adopt only if JSON-Schema validation proves too loose across the boundary.

---

## 9. Open decisions (settle before/within Phase 0)

1. **Default training data**: ship a default streaming dataset (`hf:dclm`-style, network for data only) vs require `--data`? Leaning: require explicit `--data`, document recommendations.
2. **Run registry**: plain filesystem scan of run dirs vs a small local index (SQLite) for `a2d runs`. Leaning: filesystem first.
3. **Worker environment**: how the Python env ships — uv-managed venv pinned by the CLI vs user-supplied env + version check vs optional container image. Affects install UX more than architecture.
4. **Wrap the existing `dllm` library** inside `a2d_core` vs implement objectives directly (LANDSCAPE.md §E) — leaning wrap-where-possible; own the transform/patch layer regardless.

---

## 10. Risks

- **MoE router drift under bidirectional annealing** — the biggest ML unknown (P4). Mitigations: slow anneal, early router freeze, first-class router telemetry, OLMoE first (open data ⇒ debuggable routing).
- **GenericAdapter misdetection** on unconventional configs → wrong capability set. Mitigations: `inferred` flag + required `--accept-inferred`; fixture corpus grows with every miss.
- **Compute cost is the gate, not code** — continual pretraining is the expensive step, now on the *user's* hardware: set expectations per model size in docs; keep P2 targets small (GPT-2 scale); surface token-budget/throughput estimates in the `a2d detect` plan.
- **Eval comparability** — diffusion NLL ≠ AR perplexity; lead with the MDLM likelihood bound + downstream tasks + throughput (P3).
- **Third-party model code on the user's machine** — the standing cost of D5: remote-code off by default, opt-in flag, optional container; never auto-execute from a fresh download.
- **HF config field drift** — new models rename fields; by design that's a fixture + adapter (§3.3), never a refactor.

---

## 11. Prior art & positioning (summary — full scan in LANDSCAPE.md)

The *method* is proven: Dream 7B shipped AR-init diffusion at 7B (from Qwen2.5 weights); DiffuGPT/DiffuLLaMA validated the recipe at 127M–7B under 200B tokens; MDLM/BD3LM are established objectives. The *product* is the gap: dLLM is a research library; Dream/LLaDA/DiffusionGemma are one-off model releases; Mercury/Gemini Diffusion are closed commercial models. Nothing offers detect→gate→convert→eval over arbitrary open-weight models — locally, on your own checkpoints.

a2d's positioning: **the universal conversion adapter** — not a new method, not a flagship model. Its durable edge is breadth of input + the honest gate. Borrow aggressively: Dream's noise rescheduling, Fast-dLLM sampling tricks, the dLLM library as a worker dependency.

This framing is public-facing policy, not just internal strategy: the OSS `README.md` leads with "not a new method — the first tool that makes the method universal," credits the prior art in a table, and states the AR-transformer scope boundary plainly. Marketing copy must never drift toward claiming the science.

---

## 12. Definition of done

Every phase's exit criteria met in order; codegen prevents Rust↔Python drift; the identity test gates every adapter and handler; the fixture corpus covers every supported and rejected model; `trust_remote_code` never runs without explicit opt-in; and `a2d detect` truthfully reports what the tool can convert and what's missing — i.e., the extension playbook (§3.3) is not documentation but the observed behavior of the codebase.
