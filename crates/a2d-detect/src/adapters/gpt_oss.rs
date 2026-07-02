//! GPT-OSS (`gpt_oss`) adapter: reject-pinning, the fifth adapter (a deliberate
//! deviation from SPEC's named-four). SWA and MXFP4 come from generic's standard
//! field reads (`sliding_window` active window, `quantization_config.quant_method`);
//! this adapter pins `attn.sink` because recon rates the sink config field
//! low-confidence and the exit criterion hard-requires the `attn.sink` reason. It
//! adds no supported path - all three caps are blocking. `inferred = false`.

use crate::generic;
use crate::spec::{Capability, ModelSpec};
use crate::{Adapter, RawConfig, Registration};

struct GptOss;

impl Adapter for GptOss {
    fn model_type(&self) -> &'static str {
        "gpt_oss"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        let mut spec = generic::detect(cfg);
        spec.inferred = false;
        if !spec.capabilities.contains(&Capability::AttnSink) {
            spec.capabilities.push(Capability::AttnSink);
        }
        spec
    }
}

inventory::submit! {
    Registration { adapter: &GptOss }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_pins_sink_and_carries_swa_mxfp4() {
        let cfg = RawConfig::new(
            serde_json::from_str(
                r#"{
                    "model_type": "gpt_oss",
                    "num_hidden_layers": 24, "hidden_size": 2880, "vocab_size": 201088,
                    "num_attention_heads": 64, "num_key_value_heads": 8,
                    "sliding_window": 128,
                    "num_local_experts": 32, "num_experts_per_tok": 4,
                    "rope_theta": 150000.0, "rms_norm_eps": 1e-5,
                    "quantization_config": {"quant_method": "mxfp4"}
                }"#,
            )
            .unwrap(),
        );
        let spec = GptOss.detect(&cfg);
        assert!(!spec.inferred);
        assert!(spec.capabilities.contains(&Capability::AttnSwa));
        assert!(spec.capabilities.contains(&Capability::AttnSink));
        assert!(spec.capabilities.contains(&Capability::WeightsMxfp4));
    }
}
