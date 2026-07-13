//! Gemma 3 (`gemma3_text`) adapter: trusted GQA/MQA + RoPE + RMSNorm dense
//! transformer that, unlike Gemma 1, uses sliding-window attention (`attn.swa`).
//!
//! Every small Gemma is a Gemma 3, so `gemma3_text` is the real conversion target.
//! Its capability set is what `generic::detect` already reads from the standard HF
//! fields - GQA/MQA (`num_key_value_heads < num_attention_heads`), RoPE
//! (`rope_theta`), dense FFN, RMSNorm (`rms_norm_eps`), bf16 weights - PLUS
//! `attn.swa` (a non-null `sliding_window`). Now that `attn.swa` is a supported,
//! non-blocking capability (the worker's `attn.swa` handler anneals the
//! sliding-window mask), a trusted Gemma 3 is Supported rather than rejected.
//!
//! Gemma 3 quirks that live purely in HF's own forward and NEVER touch the
//! capability set: the independent `head_dim` (256), sqrt(hidden) embedding
//! scaling, query-key norm, and per-layer local/global RoPE theta
//! (`rope_local_base_freq` vs `rope_theta`). The `sliding_window_pattern` (every
//! Nth layer is global/full) is likewise a forward-time layout detail. This
//! adapter only pins trust (`inferred = false`); classification stays in the
//! generic workhorse (open-closed, SPEC-HANDOFF 3.3).

use crate::generic;
use crate::spec::ModelSpec;
use crate::{Adapter, RawConfig, Registration};

struct Gemma3;

impl Adapter for Gemma3 {
    fn model_type(&self) -> &'static str {
        "gemma3_text"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        ModelSpec {
            inferred: false,
            ..generic::detect(cfg)
        }
    }
}

inventory::submit! {
    Registration { adapter: &Gemma3 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::gate;
    use crate::spec::{Capability, Verdict};

    /// The Gemma-3-270M / 1B field shape: MQA (kv=1), independent head_dim, RoPE +
    /// RMSNorm, GeGLU, and an active `sliding_window` - so the trusted spec carries
    /// `attn.swa` yet is Supported (swa is no longer a blocking capability).
    fn detect_gemma3(hidden_size: u64, intermediate_size: u64, n_layers: u64) -> ModelSpec {
        let json = format!(
            r#"{{
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3_text",
                "num_hidden_layers": {n_layers},
                "hidden_size": {hidden_size},
                "intermediate_size": {intermediate_size},
                "vocab_size": 262144,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "sliding_window": 512,
                "sliding_window_pattern": 6,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "rms_norm_eps": 1e-6,
                "attn_logit_softcapping": null,
                "final_logit_softcapping": null,
                "torch_dtype": "bfloat16"
            }}"#
        );
        Gemma3.detect(&RawConfig::new(serde_json::from_str(&json).unwrap()))
    }

    #[test]
    fn gemma3_is_trusted_gqa_rope_rms_with_swa() {
        // Gemma 3 270M shape (hidden 640) and 1B shape (hidden 1152) classify identically.
        for spec in [detect_gemma3(640, 2048, 26), detect_gemma3(1152, 6912, 26)] {
            assert!(!spec.inferred, "gemma3 is a trusted adapter");
            assert!(spec.capabilities.contains(&Capability::AttnGqa));
            assert!(spec.capabilities.contains(&Capability::PosRope));
            assert!(spec.capabilities.contains(&Capability::FfnDense));
            assert!(spec.capabilities.contains(&Capability::NormRms));
            // The whole point: sliding-window attention is classified...
            assert!(spec.capabilities.contains(&Capability::AttnSwa));
            // ...but no longer blocks, so a clean trusted Gemma 3 is Supported.
            assert_eq!(gate(&spec), Verdict::Supported);
            // Null softcap fields must NOT trip the head.logit-softcap fidelity tag.
            assert!(!spec.capabilities.contains(&Capability::HeadLogitSoftcap));
        }
    }
}
