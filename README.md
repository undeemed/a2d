# a2d

[![CI](https://github.com/undeemed/a2d/actions/workflows/ci.yml/badge.svg)](https://github.com/undeemed/a2d/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Convert open-weight autoregressive LLMs into diffusion language models. Locally, with one command.**

> **Status: Phases 0-3 landed; the dense conversion core is real.**
> Detect & gate (config-only, no GPU), the conversion recipe (GPT-2 plus the GQA+RoPE family: Gemma 1, Qwen2, Llama), the identity gate, sampling, and the eval harness run end to end.
> MoE is next on the [roadmap](#roadmap); large real-model GPU conversions are still ahead.
> Contributions welcome - see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## What this is — and what it isn't

Let's be precise about the claim, because the field deserves it:

**a2d is not a new method.** Converting an AR model into a diffusion model by continued training is established research — [AR2Diff](https://arxiv.org/pdf/2401.17181) formulated it, [DiffuGPT/DiffuLLaMA](https://arxiv.org/abs/2410.17891) demonstrated it from 127M to 7B, and [Dream 7B](https://arxiv.org/abs/2508.15487) shipped a strong open model built exactly this way (initialized from Qwen2.5). "Recipes for any AR model" exist too ([Tiny-A2D](https://github.com/ZHZisZZ/dllm), the [dLLM](https://github.com/ZHZisZZ/dllm) library). If you want the science, start with those papers — and see [Prior art](#prior-art--credit), which a2d builds on directly.

**a2d is the first *tool* that makes that method universal, safe, and one command.** What exists today is research scripts, a training library, and one-off model releases. What doesn't exist — and what a2d is — is a product you point at an arbitrary local checkpoint and get:

- **A verdict before you spend anything.** `a2d detect` reads the model's config (no weights, no GPU) and tells you whether it can convert, how, and — when it can't — exactly why.
- **Automated conversion with a safety gate.** Attention surgery, an identity check that proves the surgery changed nothing before training starts, then the diffusion training recipe.
- **Reproducible runs.** Every conversion writes a manifest (source hash, config, capability set, test results) — same inputs, same run.
- **An honest no.** Architectures the recipe doesn't fit are rejected with reasons, never silently mis-converted.

## What "universal" means (scope, honestly)

Architecture-universal within the **AR-transformer family** — dense and Mixture-of-Experts. That covers the Llama / Qwen / GLM / Gemma / OLMoE lineage, i.e. most open-weight releases. It does **not** cover other paradigms: Mamba/SSM, encoder-decoder, or models that are already non-autoregressive. Those get a clean `unsupported` at the gate, with reasons.

New model drops that fit the family should work day-one with zero a2d code: detection is generic over the standard HF config vocabulary, and conversion runs on the HF ecosystem as soon as `transformers` supports the model. Support is defined by **capabilities** (attention variant, FFN family, weight format), not by a hardcoded model list — so gaps are visible, named, and additively fixable.

| Converts | Rejected (with reasons) |
|----------|------------------------|
| Dense AR transformers (GPT-2, Pythia, Llama, Qwen, Gemma 1…) | SSM/Mamba (`paradigm`) |
| MoE AR transformers (OLMoE, Qwen-MoE…) | MLA-attention models (`attn.mla`) — until a handler lands |
| Sliding-window / attention-sink models (Mistral, Gemma 2/3, GPT-OSS) — *planned, capability-gated* | ONNX-only exports (`format`) |

## How it works (30 seconds)

The model already knows language; that knowledge lives in weights a2d never touches. The conversion changes *how it reads* and *what it practices*:

1. **Detect & gate** — parse `config.json` → normalized spec → capability check.
2. **Patch** — open causal attention to bidirectional (gradually, via annealing); drop the next-token shift.
3. **Identity gate** — at anneal=0 the patched model must match the original's logits exactly; fail = abort, nothing wasted.
4. **Train** — masked-diffusion objective (MDLM; block diffusion later): fill-in-the-blank at varying mask ratios over a few billion tokens.
5. **Output** — a standard HF-layout checkpoint (+ provenance manifest and eval report) that loads with normal tooling.

```
# planned CLI
a2d detect  ./models/qwen2.5-1.5b            # verdict + plan, config-only
a2d convert ./models/qwen2.5-1.5b --out runs/qwen-diff --data ./corpus
a2d eval    runs/qwen-diff
a2d sample  runs/qwen-diff -p "The cat"
```

Weights must already be downloaded — a2d never fetches models. Third-party model code (`trust_remote_code`) never runs without an explicit flag.

## Roadmap

Phased, no dates: walking skeleton → detection & gate (no GPU) → dense conversion (GPT-2, then Gemma 1/Qwen2/Llama) → eval harness → MoE (OLMoE) → block diffusion & fast sampling → hard architectures (SWA/sinks/quantized: Mistral, Gemma 2/3, GPT-OSS) → polish. Details and exit criteria: [`docs/SPEC-HANDOFF.md`](docs/SPEC-HANDOFF.md).

## Prior art & credit

a2d packages other people's science. Read and cite them:

| Work | What it contributed |
|------|---------------------|
| [AR2Diff — Transfer Learning for Text Diffusion](https://arxiv.org/pdf/2401.17181) | The pretrain-AR → continue-as-diffusion formulation |
| [DiffuGPT / DiffuLLaMA](https://arxiv.org/abs/2410.17891) | Demonstrated adaptation 127M–7B, <200B tokens; the recipe a2d's core follows |
| [Dream 7B](https://arxiv.org/abs/2508.15487) | AR-init diffusion at scale (from Qwen2.5); context-adaptive noise rescheduling |
| [MDLM](https://arxiv.org/pdf/2406.07524) | The masked-diffusion objective a2d uses first |
| [BD3LM / block diffusion](https://arxiv.org/pdf/2406.07524) | Block-parallel objective/sampling (planned) |
| [LLaDA](https://arxiv.org/abs/2502.09992) | From-scratch proof that diffusion LMs scale competitively |
| [dLLM library / Tiny-A2D](https://github.com/ZHZisZZ/dllm) | Open training/eval infra and any-model recipes; a candidate dependency of a2d's worker |
| Full landscape | [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md) |

If a2d's framing ever drifts toward claiming the method — file an issue. The honest claim is the product.

## Design docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the ML conversion recipe (objectives, annealing, identity test)
- [`docs/SPEC-HANDOFF.md`](docs/SPEC-HANDOFF.md) — the tool: lifecycle, extensibility model, contracts, roadmap
- [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md) — prior art & positioning

## License

Apache-2.0. Licensed under the terms of the [`LICENSE`](LICENSE) file.
