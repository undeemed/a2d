//! Llama (`llama`) adapter: the canonical dense-transformer reference SPEC names
//! (GQA-capable, RoPE, RMSNorm). Delegates classification to the generic workhorse
//! and only pins trust (`inferred = false`).

use crate::generic;
use crate::spec::ModelSpec;
use crate::{Adapter, RawConfig, Registration};

struct Llama;

impl Adapter for Llama {
    fn model_type(&self) -> &'static str {
        "llama"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        ModelSpec {
            inferred: false,
            ..generic::detect(cfg)
        }
    }
}

inventory::submit! {
    Registration { adapter: &Llama }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Capability;

    #[test]
    fn llama_is_trusted_gqa_rope_rms() {
        let cfg = RawConfig::new(
            serde_json::from_str(
                r#"{
                    "model_type": "llama",
                    "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 128256,
                    "num_attention_heads": 32, "num_key_value_heads": 8,
                    "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "torch_dtype": "bfloat16"
                }"#,
            )
            .unwrap(),
        );
        let spec = Llama.detect(&cfg);
        assert!(!spec.inferred);
        assert!(spec.capabilities.contains(&Capability::AttnGqa));
        assert!(spec.capabilities.contains(&Capability::PosRope));
        assert!(spec.capabilities.contains(&Capability::NormRms));
    }
}
