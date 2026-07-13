//! Gemma 1 (`gemma`) adapter: trusted GQA/MQA + RoPE + RMSNorm dense transformer,
//! the driving target for the diffusion-Gemma conversion.
//!
//! Gemma 1 quirks that this adapter must NOT mis-classify: `head_dim` (256) is
//! independent of `hidden_size / num_attention_heads`; `num_key_value_heads = 1`
//! is MQA (kv < heads => `attn.gqa`); embeddings are scaled by sqrt(hidden_size)
//! and the MLP is GeGLU. All of those live in HF's own forward and never touch the
//! capability set, which is decided purely from config fields. Crucially Gemma 1
//! has NO sliding window (`sliding_window` absent/null), so it must stay full
//! attention (no `attn.swa`) - unlike Gemma 2/3, whose active window classifies
//! `attn.swa` (supported, non-blocking since the worker's swa handler landed).
//! Delegates classification to the generic workhorse and only pins trust
//! (`inferred = false`).

use crate::generic;
use crate::spec::ModelSpec;
use crate::{Adapter, RawConfig, Registration};

struct Gemma;

impl Adapter for Gemma {
    fn model_type(&self) -> &'static str {
        "gemma"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        ModelSpec {
            inferred: false,
            ..generic::detect(cfg)
        }
    }
}

inventory::submit! {
    Registration { adapter: &Gemma }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Capability;

    #[test]
    fn gemma1_is_trusted_gqa_rope_rms_without_swa() {
        // Gemma 1 2B (unsloth/gemma-2b) field shape: MQA (kv=1), independent head_dim,
        // RoPE + RMSNorm, GeGLU, tied embeddings, and NO sliding_window.
        let cfg = RawConfig::new(
            serde_json::from_str(
                r#"{
                    "model_type": "gemma",
                    "architectures": ["GemmaForCausalLM"],
                    "num_hidden_layers": 18, "hidden_size": 2048, "vocab_size": 256000,
                    "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256,
                    "intermediate_size": 16384, "hidden_act": "gelu_pytorch_tanh",
                    "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
                    "tie_word_embeddings": true, "torch_dtype": "bfloat16"
                }"#,
            )
            .unwrap(),
        );
        let spec = Gemma.detect(&cfg);
        assert!(!spec.inferred, "gemma is a trusted known adapter");
        assert!(spec.capabilities.contains(&Capability::AttnGqa));
        assert!(spec.capabilities.contains(&Capability::PosRope));
        assert!(spec.capabilities.contains(&Capability::FfnDense));
        assert!(spec.capabilities.contains(&Capability::NormRms));
        // Gemma 1 is full attention: no sliding window => convertible in this family.
        assert!(!spec.capabilities.contains(&Capability::AttnSwa));
    }
}
