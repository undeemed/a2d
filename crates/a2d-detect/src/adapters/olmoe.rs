//! OLMoE (`olmoe`, `allenai/OLMoE-1B-7B`) adapter: fine-grained dropless MoE with
//! 64 experts / top-8 and NO shared experts. Generic already classifies it as
//! `ffn.moe` from the `num_experts` alias, so this adapter is trust-pin ONLY
//! (`inferred = false`). It must NOT add `ffn.moe.shared-experts` - a capability
//! the model lacks; recording it would pollute the Phase-2 manifest cap set
//! (SPEC 4.3). A model that truly has shared experts gets its own fixture/adapter.

use crate::generic;
use crate::spec::ModelSpec;
use crate::{Adapter, RawConfig, Registration};

struct Olmoe;

impl Adapter for Olmoe {
    fn model_type(&self) -> &'static str {
        "olmoe"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        ModelSpec {
            inferred: false,
            ..generic::detect(cfg)
        }
    }
}

inventory::submit! {
    Registration { adapter: &Olmoe }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Capability;

    #[test]
    fn olmoe_is_moe_without_shared_experts() {
        let cfg = RawConfig::new(
            serde_json::from_str(
                r#"{
                    "model_type": "olmoe",
                    "num_hidden_layers": 16, "hidden_size": 2048, "vocab_size": 50304,
                    "num_attention_heads": 16, "num_key_value_heads": 16,
                    "num_experts": 64, "num_experts_per_tok": 8,
                    "rope_theta": 10000.0, "rms_norm_eps": 1e-5
                }"#,
            )
            .unwrap(),
        );
        let spec = Olmoe.detect(&cfg);
        assert!(!spec.inferred);
        assert!(spec.capabilities.contains(&Capability::FfnMoe));
        assert!(!spec.capabilities.contains(&Capability::FfnMoeSharedExperts));
    }
}
