# a2d — Prior Art & Competitive Landscape

A scan of work related to a2d (converting autoregressive open-weight LLMs into diffusion LMs). Companion to `ARCHITECTURE.md` and `SPEC-HANDOFF.md`.

## TL;DR — three takeaways

1. **a2d's *method* is established, not speculative.** AR→diffusion adaptation has been demonstrated from 127M up to 7B, and Dream 7B shipped the exact idea (init a diffusion LM from Qwen2.5 AR weights). You're building on proven science — "adapt > train from scratch" is the consensus.
2. **The gap a2d fills is productization.** Everything below is either a *research recipe*, a *one-off model release*, or a *Python/HF research library*. No one offers a general **detect → gate → convert → eval → monitor platform** across arbitrary open-weight models. That's the white space.
3. **Honest tension:** frontier labs now ship strong diffusion LMs off-the-shelf (Dream, LLaDA 2.0, Mercury, DiffusionGemma). So a2d's value is converting *your own / arbitrary* models on demand — not out-competing those flagship dLLMs.

---

## A. Direct prior art — AR→diffusion conversion (a2d's exact thesis)

- **Dream 7B** (HKU NLP + Huawei Noah's Ark, 2025). The closest analog to a2d. A 7B masked-diffusion LM **initialized from Qwen2.5-7B autoregressive weights**, then continued-trained on ~580B tokens; matches/beats similar-size AR models. Validates a2d's core bet — AR init dramatically accelerates diffusion training vs from-scratch — *and* that it's architecture-preserving (Qwen backbone). Difference: Dream is a single model release, not a reusable conversion pipeline.
- **DiffuGPT / DiffuLLaMA** — "Scaling Diffusion LMs via Adaptation from AR Models" (2024). Converts GPT-2 and LLaMA (127M–7B) into diffusion LMs with **<200B tokens**. This is the reference recipe a2d's Phase 2 implements (attention-mask annealing, shift removal, MDLM objective).
- **AR2Diff** — "Transfer Learning for Text Diffusion Models" (Google, 2024). The 3-step formulation: pretrain AR → continue as diffusion (bidirectional) → finetune. The conceptual backbone of the conversion.
- **Tiny-A2D** — open recipes converting *any* AR model (Qwen / LLaMA / GPT-2) into small diffusion models. Closest in spirit to a2d's "any model" framing, but at toy scale and as scripts, not a service.

**Implication:** a2d is *not* inventing the ML. Its contribution is making this repeatable, gated, and productized across architectures.

## B. From-scratch diffusion LMs (the path a2d deliberately avoids — but the quality bar)

- **LLaDA** (2025). 8B masked-diffusion LM trained from scratch on 2.3T tokens; rivals LLaMA3-8B. The proof that diffusion LMs scale competitively — and, implicitly, the expensive baseline that adaptation (a2d/Dream) undercuts.
- **LLaDA 2.0** (2025). Scales the family toward ~100B with MoE — directly relevant to a2d's MoE→MoE path.
- **Seed Diffusion** (ByteDance, 2025). Code-focused diffusion LM, ~2,146 tok/s, with block-level parallel sampling — a strong reference for a2d's Phase 6 (BD3LM + block-parallel decoding).

## C. Foundational discrete-diffusion methods (the objectives a2d uses)

- **D3PM** (Austin et al., 2021). The foundation of discrete denoising diffusion; introduced structured transition matrices (incl. absorbing/mask state).
- **SEDD — Score Entropy Discrete Diffusion** (2023). Score-matching for discrete data over a continuous-time Markov chain; first diffusion text model to beat GPT-2 on perplexity.
- **MDLM — Masked Diffusion LMs** (NeurIPS 2024). Absorbing-state masked diffusion with a simple weighted-MLM loss and a clean likelihood bound. **This is a2d's default Phase-2 objective.**
- **Block Diffusion / BD3LM**. Denoise contiguous spans for parallelism + long-range coherence. **a2d's Phase-6 objective.**

## D. Production / commercial diffusion LLMs (the market the outputs live in)

- **Mercury** (Inception Labs, 2025). First commercial-scale dLLM; >1,000 tok/s on one H100; Mercury Coder competitive with GPT-4o-mini class. **Mercury 2** (2026) pushes reasoning + speed further. The commercial proof point for "diffusion = speed."
- **Gemini Diffusion** (Google, I/O 2025). Frontier-scale text diffusion, experimental — signals big-lab seriousness.
- **DiffusionGemma** (Google, 2026). 26B MoE on a Gemma backbone; first dLLM natively supported in vLLM. Notable for a2d because it's *also* backbone-derived (Gemma) and proves serving-stack support is arriving.

## E. Tooling & ecosystem (what a2d builds on or competes with)

- **dLLM toolkit** (2026, `ZHZisZZ/dllm`). "What Hugging Face did for transformers, for diffusion LMs" — unifies training, fine-tuning, inference, and eval; built on HF Transformers / Accelerate / PEFT. **This is the closest existing tool to a2d** — but it's a Python/HF *library* for researchers, with no detection/capability gate, no service/API, no UI, and no Rust path. a2d's platform layer is the differentiation; the dLLM toolkit is roughly what a2d's *Python fallback worker* would wrap.
- **Fast-dLLM** (2026). 2–4× inference speedup via caching + confidence-threshold decoding — a technique a2d's sampler should adopt.
- **vLLM / SGLang**. Native dLLM serving was still nascent in early 2026 (DiffusionGemma is the first in vLLM); a2d's candle-native sampler is partly motivated by this thin serving support.
- **Surveys**: "Discrete Diffusion in Large Language and Multimodal Models: A Survey" (2025) and the Awesome-DLMs list — good for tracking the fast-moving field.

## F. Adjacent / multimodal

- **DiffusionVL, LLaDA-V** — apply the AR→diffusion conversion idea to vision-language models. Out of a2d's current scope, but the same recipe family.
- **d2** — improved techniques for *reasoning* diffusion LMs; relevant once a2d outputs need RL/reasoning post-training.

---

## What this means for a2d

- **De-risked thesis.** Dream 7B + DiffuLLaMA mean the conversion works at the scales a2d targets; the open questions are engineering (coverage, eval parity, MoE routing), not "does this work at all."
- **Clear differentiation.** Position a2d as the *conversion platform/product*, not a new method or a competing flagship model. The one-line pitch: "point it at any open-weight model and get a diffusion version, with a gate that tells you up front whether and how it'll convert." Nothing in this scan does that end-to-end.
- **Borrow aggressively.** MDLM/BD3LM objectives, Dream's AR-init + token-level noise rescheduling, Seed/Fast-dLLM's sampling speedups, and the dLLM toolkit (as the fallback engine) are all directly reusable.
- **Watch the moat.** As big labs ship more off-the-shelf dLLMs and serving matures (vLLM), a2d's durable value is *breadth of input models* + *self-serve productization*, not raw output quality. Keep the capability gate and "any model" coverage as the headline.

---

### References
- Dream 7B — https://arxiv.org/abs/2508.15487 · blog: https://hkunlp.github.io/blog/2025/dream/
- DiffuGPT/DiffuLLaMA (Scaling Diffusion LMs via Adaptation) — https://arxiv.org/abs/2410.17891
- AR2Diff (Transfer Learning for Text Diffusion Models) — https://arxiv.org/pdf/2401.17181
- LLaDA — https://arxiv.org/abs/2502.09992 · LLaDA 2.0 — https://arxiv.org/html/2512.15745
- Seed Diffusion — https://arxiv.org/html/2508.02193v1
- SEDD — https://arxiv.org/abs/2310.16834 · MDLM — https://arxiv.org/pdf/2406.07524
- Mercury — https://arxiv.org/abs/2506.17298 · Inception Labs — https://www.inceptionlabs.ai/
- DiffusionGemma (vLLM) — https://vllm.ai/blog/2026-06-10-diffusion-gemma
- dLLM toolkit — https://github.com/ZHZisZZ/dllm · paper: https://huggingface.co/papers/2602.22661
- Discrete Diffusion in LLMs & Multimodal — A Survey — https://arxiv.org/pdf/2506.13759
- d2 (reasoning diffusion LMs) — https://arxiv.org/pdf/2509.21474
