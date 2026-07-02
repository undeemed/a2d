//! Qwen2 (`qwen2`) adapter: trusted GQA transformer.
//!
//! Note (the `use_sliding_window` gotcha): Qwen2.5 ships `sliding_window` set
//! (e.g. 131072) alongside `use_sliding_window = false`, so it must NOT be flagged
//! `attn.swa`. That guard lives in `generic::detect` (window present AND > 0 AND
//! `use_sliding_window != false`); this adapter only pins trust (`inferred = false`).

use crate::generic;
use crate::spec::ModelSpec;
use crate::{Adapter, RawConfig, Registration};

struct Qwen2;

impl Adapter for Qwen2 {
    fn model_type(&self) -> &'static str {
        "qwen2"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        ModelSpec {
            inferred: false,
            ..generic::detect(cfg)
        }
    }
}

inventory::submit! {
    Registration { adapter: &Qwen2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Capability;

    #[test]
    fn qwen2_is_trusted_gqa_without_swa() {
        let cfg = RawConfig::new(
            serde_json::from_str(
                r#"{
                    "model_type": "qwen2",
                    "num_hidden_layers": 28, "hidden_size": 3584, "vocab_size": 152064,
                    "num_attention_heads": 28, "num_key_value_heads": 4,
                    "sliding_window": 131072, "use_sliding_window": false,
                    "rope_theta": 1000000.0, "rms_norm_eps": 1e-6
                }"#,
            )
            .unwrap(),
        );
        let spec = Qwen2.detect(&cfg);
        assert!(!spec.inferred);
        assert!(spec.capabilities.contains(&Capability::AttnGqa));
        assert!(!spec.capabilities.contains(&Capability::AttnSwa));
    }
}
