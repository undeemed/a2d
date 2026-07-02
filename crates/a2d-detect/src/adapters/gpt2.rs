//! GPT-2 (`gpt2`) adapter: trusted classic dense transformer with learned
//! positions and legacy config aliases (`n_layer`/`n_embd`/`n_head`). Delegates
//! classification to the generic workhorse and only pins trust (`inferred = false`).

use crate::generic;
use crate::spec::ModelSpec;
use crate::{Adapter, RawConfig, Registration};

struct Gpt2;

impl Adapter for Gpt2 {
    fn model_type(&self) -> &'static str {
        "gpt2"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        ModelSpec {
            inferred: false,
            ..generic::detect(cfg)
        }
    }
}

inventory::submit! {
    Registration { adapter: &Gpt2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Capability;

    #[test]
    fn gpt2_is_trusted_dense_with_learned_pos() {
        let cfg = RawConfig::new(
            serde_json::from_str(
                r#"{"model_type":"gpt2","n_layer":12,"n_embd":768,"vocab_size":50257,"n_head":12}"#,
            )
            .unwrap(),
        );
        let spec = Gpt2.detect(&cfg);
        assert!(!spec.inferred);
        assert!(spec.capabilities.contains(&Capability::AttnFull));
        assert!(spec.capabilities.contains(&Capability::PosLearned));
        assert!(spec.capabilities.contains(&Capability::FfnDense));
    }
}
