# a2d — AR→Diffusion Conversion Pipeline

**Working name:** `a2d` (autoregressive-to-diffusion)
**Goal:** A config-driven pipeline that takes an open-weight autoregressive (AR) LLM and continues training it into a diffusion language model (dLLM), the way Google did Gemma → DiffusionGemma — but for arbitrary checkpoints.
**Status:** Design draft.

---

## 1. Scope

### In scope (v1)
- **Architecture-preserving conversion only:** `dense → dense` and `MoE → MoE`. The output checkpoint has the *same* layer stack, attention shape, and FFN/MoE structure as the input — we only change the attention mask policy, the training objective, and the decoding loop.
- **Standard attention:** full (global) attention with RoPE or learned/ALiBi positions.
- **Objectives:** MDLM (absorbing-state masked diffusion) first; BD3LM (block diffusion) second.
- **Targets to validate against:** Llama / Qwen2.5 (dense), Qwen-MoE / OLMoE (MoE).

### Explicitly out of scope (for now)
- **Cross-family** conversion (dense → MoE or MoE → dense).
- **Exotic attention:** attention sinks, hybrid/SSM layers (sliding-window attention was gated out here originally; it has since landed as the `attn.swa` capability). *This is what makes GPT-OSS the "boss level" — it has SWA + sinks + native MXFP4. The introspect gate (§5) rejects it cleanly until we add the remaining capabilities.*
- Multimodal towers, vision adapters.
- Quantization-aware conversion (train in bf16; quantize after).

---

## 2. The key insight that makes the scope clean

An AR→diffusion conversion, done architecture-preserving, touches exactly **three** things:

1. **Attention masking** — causal → bidirectional. The model must see the whole "canvas" to denoise it. Done via *mask annealing* so we don't destroy the pretrained weights.
2. **Objective / shift** — drop AR's next-token shift; swap next-token cross-entropy for a diffusion loss (MDLM's weighted-MLM loss, or BD3LM's per-block loss).
3. **Decoding** — autoregressive decode → iterative parallel denoising.

It does **not** touch the FFN/MoE block, the experts, the router, the embeddings, or the attention projections — those are all inherited as-is.

**Consequence:** the converter is already FFN-agnostic, because it never touches the FFN. "Dense vs MoE" is just *what sits in the FFN slot*, and we preserve it. So **`dense→dense` and `MoE→MoE` are literally the same code path.** The only MoE-specific concern is keeping the router's load-balancing aux loss alive (and routing stable) while attention goes bidirectional. That isolated concern is the entire "MoE support" surface area.

---

## 3. Conversion recipe (the stages)

Follows the established adaptation recipe (AR2Diff / DiffuLLaMA-style continual pretraining):

```
 source AR ckpt
      │
 [0] introspect ──────────► ModelSpec  (+ supported? gate)
      │
 [1] patch (transform) ───► nn.Module: bidirectional-capable attn + diffusion head
      │                      (weights untouched; anneal=0 ⇒ bit-identical to base)
      │
 [2] continual pretrain ──► MDLM/BD3LM objective; attn annealed causal→bidir
      │                      (MoE: keep aux loss; optional router freeze early)
      │
 [3] task / instruct FT ──► same objective, downstream data
      │
 [4] sample ──────────────► iterative denoiser (block-parallel for BD3LM)
      │
 [5] eval ────────────────► likelihood bound, downstream, tokens/sec vs AR base
      ▼
 diffusion ckpt (same architecture family)
```

The continual-pretrain step (stage 2) is the only expensive part. Reference points: DiffuGPT/DiffuLLaMA converted 127M–7B models with **< 200B tokens**; small dense models are reachable on a single node.

---

## 4. Module layout

```
a2d/
  introspect/
    spec.py            # ModelSpec dataclass (normalized model description)
    registry.py        # model_type -> adapter
    adapters/
      generic_hf.py    # infer from HF config when possible (fallback)
      llama.py
      qwen2.py
      qwen2_moe.py
      gemma.py
  transform/
    attention.py       # causal -> bidirectional; annealing hook
    head.py            # remove AR shift; wire diffusion loss head
    apply.py           # ModelSpec + ConversionConfig -> patched nn.Module
  objectives/
    base.py            # interface: corrupt(batch) -> inputs ; loss(logits, target, mask)
    mdlm.py            # absorbing-state masked diffusion (NeurIPS'24)
    bd3lm.py           # block diffusion
  data/
    streaming.py       # corpus streaming + packing
    collators.py       # masking / noising collators per objective
  train/
    continual.py       # stage 2
    finetune.py        # stage 3
    callbacks.py       # AnnealScheduler, MoERouterMonitor (aux loss / utilization)
  sample/
    denoiser.py        # iterative denoising loop
    schedulers.py      # remasking policy: low-confidence / block-parallel
  eval/
    harness.py         # NLL bound, downstream tasks, throughput
  config/
    schema.py          # declarative config (pydantic)
  cli.py               # a2d convert | train | sample | eval
```

**Rule:** the *only* per-model code lives in `introspect/adapters/`. Everything downstream consumes the normalized `ModelSpec`, so adding a model = write one adapter (or lean on `generic_hf`).

---

## 5. `ModelSpec` — the generalization + gating layer

An adapter reads an HF model's config and module tree and emits a normalized description, so `transform/` never hardcodes module names. It's also the **gatekeeper** that enforces scope.

```python
@dataclass
class AttentionSpec:
    pos_emb: Literal["rope", "yarn", "alibi", "learned", "none"]
    is_causal: bool
    window: int | None          # not None  => sliding-window  (was UNSUPPORTED v1; attn.swa landed)
    attn_sink: bool             # True       => attention sinks => UNSUPPORTED v1
    n_heads: int
    n_kv_heads: int             # GQA/MQA
    head_dim: int

@dataclass
class FFNSpec:
    family: Literal["dense", "moe"]
    # MoE-only:
    n_experts: int | None = None
    n_active: int | None = None
    router_path: str | None = None     # dotted module path to the gate
    aux_loss_coef: float | None = None

@dataclass
class ModelSpec:
    model_id: str
    n_layers: int
    d_model: int
    vocab_size: int
    attention: AttentionSpec
    ffn: FFNSpec
    paths: dict[str, str]       # decoder_layer, attn_module, lm_head, embed
    mask_token_id: int | None   # reused special id, or None -> grow vocab by 1
    supported: bool
    reasons: list[str]          # why unsupported, if so
```

The gate is mechanical: `supported = (window is None) and (not attn_sink) and (ffn.family in {dense, moe})`. GPT-OSS fails on `window` + `attn_sink` and is rejected with a readable reason — no silent breakage. Adding GPT-OSS later = implement those two capabilities and flip the gate, not a rewrite. (The `window` half has since happened: the `attn.swa` handler landed and its gate flipped, so GPT-OSS's remaining reasons are the sink and its MXFP4 weights.)

The pipeline **detects** `family`; the user never sets it. Architecture-preserving means `target.family == source.family` by construction.

---

## 6. Declarative config

```yaml
source:
  model: Qwen/Qwen2.5-1.5B          # HF id or local path
  dtype: bf16

conversion:
  objective: mdlm                    # mdlm | bd3lm
  mask_token: reuse                  # reuse an unused special id | add (+1 vocab row)
  attention:
    target: bidirectional
    anneal: { schedule: linear, steps: 2000 }   # causal -> bidirectional
  block: { size: 32 }                # bd3lm only
  preserve:
    ffn: true                        # always true in v1
    router: true                     # moe only
    position_embedding: true

train:
  stage: continual                   # continual | finetune
  data: { stream: dclm, tokens: 5_000_000_000 }
  seq_len: 2048
  global_batch_tokens: 1_000_000
  lr: 1.0e-4
  moe:
    aux_loss_coef: inherit           # keep router balanced during conversion
    freeze_router_steps: 500         # stabilize routing early (optional)

sample:
  steps: 64
  block_parallel: true               # bd3lm
  remasking: low_confidence
  temperature: 0.7
```

---

## 7. Invariants & design decisions

- **Architecture-preserving invariant:** module set and tensor shapes are identical pre/post conversion, except optionally **+1 vocab row** for a mask token. This buys us:
  - the diffusion checkpoint loads back into the *same* model class;
  - existing kernels/inference paths mostly work;
  - a cheap correctness test (below).
- **The identity test:** with `anneal.steps` not yet started (fully causal) and the AR shift restored, the patched module must reproduce the base model's logits to within tight numerical tolerance (bit-identical if the attention kernel is unchanged). This is the unit test that guards every adapter and the transform layer.
- **MoE = dense + a different FFN, one code path.** MoE logic is confined to (a) the introspect adapter that finds the router, and (b) the `MoERouterMonitor` callback. `transform/` and `objectives/` stay FFN-agnostic.
- **Objectives are pluggable** behind a 2-method interface (`corrupt`, `loss`). MDLM first (simple, BERT-like weighted-MLM loss, gives a clean NLL bound); BD3LM second (block-parallel speed + long-range coherence).
- **No KV cache.** Diffusion decodes the full canvas each step; disable the cache rather than fight causal assumptions baked into it.
- **RoPE is fine bidirectionally** — relative positions don't assume causality. Just confirm no module asserts `is_causal=True`.

---

## 8. Build order

| Milestone | Deliverable | Proves |
|-----------|-------------|--------|
| **M0 — Dense happy path** | Qwen2.5/Llama + MDLM, small-corpus continual pretrain | Identity test passes; generates coherent text after convert |
| **M1 — Eval harness** | NLL bound + 2 downstream tasks + tokens/sec vs AR | Honest, repeatable comparison |
| **M2 — MoE** | Qwen-MoE/OLMoE adapter + router monitor | Routing stays balanced through annealing |
| **M3 — BD3LM + block sampling** | Block objective + parallel denoiser | The speed payoff (the DiffusionGemma selling point) |
| **Future — boss level** | Attention-sink capability → GPT-OSS (the SWA half landed: Mistral, Gemma 2/3 convert); cross-family; quant-aware | Gated off today by the introspect check |

---

## 9. Risks & open questions

- **MoE router under bidirectional attention.** Routing was learned causally; relaxing the mask shifts token representations and may unbalance experts. *Mitigations:* slow anneal, freeze router for the first N steps, monitor aux loss + expert utilization (the `MoERouterMonitor`). This is the single biggest unknown in the MoE path and the main reason M2 follows M1.
- **Mask token.** Reuse an unused special id (keeps the strict shape invariant) vs. grow vocab by one row (cleaner semantics, breaks the invariant by one row). Default: `reuse`, fall back to `add`.
- **Compute is the gate, not code.** Stage 2 is the cost; everything else is cheap. Budget a few B tokens for ≤2B-param models.
- **Eval parity.** Diffusion NLL isn't directly comparable to AR perplexity; rely on MDLM's likelihood bound for apples-to-apples, and lead with downstream + throughput.
- **Tokenizer is inherited** unchanged — good (no re-embedding), but means we can't fix tokenizer quirks during conversion.

---

## 10. Open design choices to settle next

1. Build the trainer in-house vs. wrap an existing dLLM training lib (e.g. `dllm`, MDLM repo) for stage 2.
2. First concrete MoE target: Qwen-MoE vs OLMoE (OLMoE is fully open incl. data — easier to reason about routing).
3. Anneal schedule shape: linear vs cosine vs stepwise mask-ratio — needs a small ablation in M0/M2.

---

### References
- DiffusionGemma — Gemma backbone → dLLM (vLLM blog): https://vllm.ai/blog/2026-06-10-diffusion-gemma
- Transfer Learning for Text Diffusion / AR2Diff: https://arxiv.org/pdf/2401.17181
- Scaling Diffusion LMs via Adaptation from AR Models (DiffuGPT/DiffuLLaMA): https://arxiv.org/html/2410.17891v2
- MDLM — Simple and Effective Masked Diffusion LMs (NeurIPS'24): https://arxiv.org/pdf/2406.07524
- `dllm` — diffusion LM training/inference library: https://github.com/ZHZisZZ/dllm
