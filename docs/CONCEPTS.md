# a2d Concepts - the plain-language map

The rest of `docs/` is precise and dense.
This page is the opposite: the smallest mental model that stops you feeling blind.
For the rigorous version see `SPEC-HANDOFF.md` (the tool) and `ARCHITECTURE.md` (the ML recipe).

## The whole idea in three concepts

1. **AR (autoregressive)** is every normal LLM.
It predicts the next token left-to-right, and a **causal mask** enforces the rule "a token cannot see the future."

2. **Diffusion LM (MDLM)** masks random tokens, predicts them all at once in **both directions** (bidirectional), and refines over a few passes.
The block of positions it works on is the **canvas**; turning masks back into tokens over N steps is **denoising**.

3. **a2d's whole job** is the conversion between them: take AR weights, **anneal** the causal mask off (the `attn.full` / `attn.gqa` transforms), and briefly retrain with a masking objective (**MDLM**).
One transform per attention seam plus one objective. That is the core.

Everything else in the vocabulary exists to decide *which models a2d will touch* - the complexity lives in honest **detection**, not in the small **conversion**.

## Glossary (one line each)

- **AR / autoregressive** - predict next token, left to right. Normal LLM.
- **Causal mask** - the rule that makes AR work: no peeking at future tokens.
- **Bidirectional** - a token sees left *and* right. What diffusion needs.
- **MDLM** - the masked-diffusion objective a2d uses: mask a random fraction, predict them, repeat.
- **Canvas** - the span of positions a diffusion model denoises (its workspace).
- **Anneal (`attn.full`, `attn.gqa`)** - a2d's trick: slowly turn the causal mask off so AR weights adapt to bidirectional. One transform per attention seam: GPT-2 bakes causality per layer (`attn.full`); the RoPE family (Gemma 1 / Qwen2 / Llama) routes it through one shared mask (`attn.gqa`).
- **Identity gate** - hard correctness check: at `anneal=0` the patched model must match the base model's logits, or convert aborts.
- **GQA** - grouped-query attention: fewer key/value heads than query heads (a memory trick). The mechanism rides along in HF's forward; the `attn.gqa` tag also names the RoPE-family anneal transform (see Anneal).
- **RoPE** - rotary position encoding (modern position scheme). Passthrough.
- **RMSNorm** - a normalization flavor. Passthrough.
- **Logit softcap** - squashes output logits (Gemma 2). Fidelity note only.
- **SWA (sliding window)** - a token only sees the nearest N tokens. a2d refuses today (Phase 6).
- **Attention sink** - first tokens are always attended to. Refuse today (Phase 6).
- **MLA** - DeepSeek's compressed latent attention. Refuse (out of scope).
- **SSM / Mamba** - not attention at all, a different sequence model. Refuse (out of scope).
- **MoE** - many expert sub-networks, routed per token. Detected; the Phase 4 target.
- **mxfp4 / gptq** - pre-quantized (compressed) weights. Refuse today; can't finetune compressed (Phase 6 dequant).
- **BD3LM / block diffusion** - AR *across* blocks, diffusion *within* a block. The DiffusionGemma recipe. Phase 5.
- **Entropy-bounded denoising / adaptive stopping** - stop denoising early when confident. DiffusionGemma-advanced, not built.

## The scary attributes are just three buckets

| Bucket | Tags | What a2d does |
|---|---|---|
| **Free (passthrough)** | `pos.rope`, `pos.rope.partial`, `pos.learned`, `norm.rms`, `norm.sandwich`, `head.logit-softcap`, `weights.bf16`, `ffn.dense` | Nothing. Detected, ride along, HF's forward handles them. No conversion work. |
| **a2d must handle (blocking)** | `attn.full` (done - GPT-2), `attn.gqa` (done - Gemma 1 / Qwen2 / Llama), `ffn.moe` (Phase 4), `attn.swa` / `attn.sink` / `weights.mxfp4` / `weights.gptq` (Phase 6), `attn.mla` / `paradigm.ssm` (out of scope) | Convert it, or the gate says `unsupported` honestly. Exactly these few flags are the "hard" list. |
| **DiffusionGemma-advanced** | block diffusion, encoder-decoder split, entropy-bounded denoising, adaptive stopping | Not built, not owed. Phase 5+ horizon, not this sprint. |

The middle row is the only place real work lives, and most of it a2d *refuses* rather than implements.

## You are here

Phases are ordered; `P4` and `P5` both only need `P2` and can run in parallel (`P3` recommended first).

- [x] **P0 - Walking skeleton.** Every layer exists and talks.
- [x] **P1 - Detection, gate & ingest.** Read `config.json`, classify supported/unsupported with reasons. No GPU.
- [x] **P2 - Conversion core (dense).** GPT-2 -> diffusion end to end, identity gate enforced.
  Extended after P3 to the GQA+RoPE family (Gemma 1 / Qwen2 / Llama) via the `attn.gqa` mask seam; the worker picks the seam from the model itself.
- [x] **P3 - Evaluation.** MDLM likelihood bound, downstream tasks, throughput vs AR; `eval/report.json` + `--html`.
- [ ] **P4 - MoE.** OLMoE end to end; `ffn.moe` handler; router-health telemetry. **<- next**
- [ ] **P5 - Objectives & sampling speed.** BD3LM (block diffusion) + block-parallel sampler. The DiffusionGemma recipe.
- [ ] **P6 - Capability expansion (boss level).** `attn.swa` / `attn.sink` handlers + quant dequant; unlocks Mistral, Gemma 2/3, GPT-OSS.
- [ ] **P7 - Polish & optional platform mode.**

Build surface right now is small: dense (soon MoE) model + flip attention bidirectional + mask-and-predict.
Everything else is a menu you are allowed to say "not yet" to.
