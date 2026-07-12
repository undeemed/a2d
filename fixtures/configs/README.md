# fixtures/configs

The `config.json` corpus that feeds the parameterized gate test (`crates/a2d-detect/tests/corpus.rs`).
Each fixture is a directory `<name>/` holding a real (trimmed but verbatim-shaped) `config.json` plus an
`expected.json` sidecar.
Every fixture is a weights-less dir (config only), so the corpus also proves the config-only detect path.

Dropping a new `<name>/{config.json,expected.json}` adds it to the matrix with zero edits to any existing
file (Decision 7). See `docs/SPEC-HANDOFF.md` section 3.3 and `docs/PLAN-PHASE1.md` Decision 7.

## Sidecar convention (`expected.json`)

```json
{ "verdict": "supported|supported_inferred|unsupported", "reasons": ["<dotted-tag>", ...], "weights": "missing" }
```

- `verdict` - the expected `Verdict` tag.
- `reasons` - present ONLY for `unsupported`; a list of dotted capability TAGS (e.g. `attn.swa`), NOT the
  human reason strings.
  The report's `Verdict::Unsupported.reasons` are human strings that each end in `(<tag>)`, so `corpus.rs`
  matches order-independently: every sidecar tag must appear in exactly one report reason, counts equal.
  Pinning tags (not prose) keeps sidecars stable when the human wording in `Capability::reason()` changes.
- `weights` - always `missing` here (config-only dirs).

## The corpus

| fixture        | source repo                     | model_type   | expected verdict     | reason tags                            |
|----------------|---------------------------------|--------------|----------------------|----------------------------------------|
| `gpt2`         | openai-community/gpt2           | gpt2         | supported            | -                                      |
| `pythia`       | EleutherAI/pythia-1.4b          | gpt_neox     | supported_inferred   | - (partial rope is fidelity, non-block)|
| `llama`        | NousResearch/Meta-Llama-3.1-8B  | llama        | supported            | -                                      |
| `qwen2`        | Qwen/Qwen2.5-7B                 | qwen2        | supported            | - (see use_sliding_window note)        |
| `gemma`        | unsloth/gemma-2b (Gemma 1)      | gemma        | supported            | - (full attn: no sliding_window)       |
| `olmoe`        | allenai/OLMoE-1B-7B-0924        | olmoe        | supported            | - (MoE, no shared experts)             |
| `mistral-v0.1` | mistralai/Mistral-7B-v0.1       | mistral      | unsupported          | attn.swa                               |
| `gemma2`       | google/gemma-2-9b               | gemma2       | unsupported          | attn.swa (softcap is fidelity)         |
| `gemma3`       | google/gemma-3-1b-it            | gemma3_text  | unsupported          | attn.swa                               |
| `gpt-oss`      | openai/gpt-oss-20b              | gpt_oss      | unsupported          | attn.swa, attn.sink, weights.mxfp4     |
| `mamba`        | state-spaces/mamba-130m-hf      | mamba        | unsupported          | paradigm.ssm                           |

## Notes on individual fixtures

- **`mistral-v0.1` must be v0.1, not v0.2.**
  Mistral-7B-Instruct-v0.2 ships `sliding_window: null` (inactive) and would wrongly PASS the gate.
  v0.1 has the active `sliding_window: 4096` the reject test needs.
  The blocking indicator is an ACTIVE window (present, `> 0`, and `use_sliding_window != false`), not mere
  presence of the field.
- **`qwen2` proves the SWA guard's negative case.**
  It sets `sliding_window: 131072` but `use_sliding_window: false`, so it is NOT `attn.swa` and stays
  supported.
- **`gemma` (Gemma 1) is convertible; `gemma2`/`gemma3` are not.**
  Gemma 1 has NO sliding window (`sliding_window` absent), so it stays full attention and passes the gate
  (`attn.gqa` from MQA `num_key_value_heads: 1`, `pos.rope`, `ffn.dense`, `norm.rms`). Gemma 2/3 add SWA and
  are gated out. Gemma's independent `head_dim: 256` and sqrt(hidden) embedding scaling are HF-forward
  concerns and never touch the capability set.
- **`gemma2` softcap is fidelity, not a blocker.**
  It carries `attn_logit_softcapping: 50.0` / `final_logit_softcapping: 30.0` (recorded as
  `head.logit-softcap`, non-blocking); the only reason is `attn.swa`.
- **`gpt-oss` `attn.sink` comes from the adapter, not the config.**
  The real config has no reliable attention-sink field; the `gpt_oss` adapter pins `attn.sink`.
  `attn.swa` (window 128) and `weights.mxfp4` (`quantization_config.quant_method`) come from generic reads.
- **`mamba` triggers `paradigm.ssm` via `ssm_cfg` presence and the absence of any attention-head field**,
  never from the `MambaForCausalLM` architecture suffix.

## Provenance / synthetic fixtures

- All configs are real Hugging Face `config.json` content, fetched from the source repo's
  `raw/main/config.json`, kept verbatim-shaped (real field names and values).
- **`gemma2` and `gemma3` are sourced from the `unsloth` open mirrors** (`unsloth/gemma-2-9b` and
  `unsloth/gemma-3-1b-it`) because the `google/*` originals are gated.
  The load-bearing field values are genuine Google values; the mirror-added `unsloth_fixed` /
  `unsloth_version` marker keys were stripped. These are faithful, NOT hand-constructed/synthetic.
- No fully synthetic (hand-constructed) fixtures were needed - every repo above was reachable.
