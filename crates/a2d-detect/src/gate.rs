//! The gate (Decision 4): one predicate over the capabilities.

use crate::spec::{Capability, ModelSpec, Verdict};

/// Reduce a spec to a verdict: collect the reasons of any blocking capability,
/// emitted in canonical declaration order (not the order caps were discovered),
/// so GPT-OSS always yields `[attn.sink, weights.mxfp4]` (its `attn.swa` is now a
/// supported, non-blocking cap). Fidelity caps are structurally excluded
/// (`blocking() == false`).
pub fn gate(spec: &ModelSpec) -> Verdict {
    let reasons: Vec<String> = Capability::ALL
        .into_iter()
        .filter(|c| c.blocking() && spec.capabilities.contains(c))
        .map(|c| c.reason().to_string())
        .collect();

    if reasons.is_empty() {
        if spec.inferred {
            Verdict::SupportedInferred
        } else {
            Verdict::Supported
        }
    } else {
        Verdict::Unsupported { reasons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with(caps: Vec<Capability>, inferred: bool) -> ModelSpec {
        ModelSpec {
            model_type: "test".to_string(),
            n_layers: 1,
            d_model: 8,
            vocab_size: 16,
            n_heads: 2,
            n_kv_heads: 2,
            sliding_window: None,
            n_experts: None,
            n_active_experts: None,
            capabilities: caps,
            mask_token_id: None,
            inferred,
        }
    }

    #[test]
    fn blocking_caps_reason_in_canonical_order() {
        // Scrambled input order proves the gate reorders to canonical.
        let spec = spec_with(
            vec![
                Capability::WeightsMxfp4,
                Capability::AttnMla,
                Capability::AttnSink,
            ],
            true,
        );
        match gate(&spec) {
            Verdict::Unsupported { reasons } => assert_eq!(
                reasons,
                vec![
                    Capability::AttnSink.reason().to_string(),
                    Capability::AttnMla.reason().to_string(),
                    Capability::WeightsMxfp4.reason().to_string(),
                ]
            ),
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    #[test]
    fn clean_trusted_dense_is_supported() {
        let spec = spec_with(
            vec![
                Capability::AttnFull,
                Capability::PosRope,
                Capability::FfnDense,
            ],
            false,
        );
        assert_eq!(gate(&spec), Verdict::Supported);
    }

    #[test]
    fn clean_untrusted_is_supported_inferred() {
        let spec = spec_with(vec![Capability::AttnFull], true);
        assert_eq!(gate(&spec), Verdict::SupportedInferred);
    }

    #[test]
    fn fidelity_cap_never_appears_in_reasons() {
        // One blocking + one fidelity cap: only the blocking one yields a reason.
        let spec = spec_with(
            vec![Capability::AttnSink, Capability::HeadLogitSoftcap],
            false,
        );
        match gate(&spec) {
            Verdict::Unsupported { reasons } => {
                assert_eq!(reasons, vec![Capability::AttnSink.reason().to_string()]);
                assert!(!reasons.iter().any(|r| r.contains("head.logit-softcap")));
            }
            other => panic!("expected unsupported, got {other:?}"),
        }
    }
}
