//! Detect types (SPEC 3.1 taxonomy + SPEC 4.1 report shape).
//!
//! PHASE 2: promote to a2d-contracts + schemars, wire into Manifest.
//! These types stay crate-internal this phase (Decision 1): nothing in Python
//! reads a `DetectReport` yet and `ModelSpec`'s shape is still being discovered,
//! so freezing them into codegen'd pydantic now would buy only drift churn. When
//! convert/manifest first read them, relocate into a2d-contracts, add
//! `#[derive(Deserialize, JsonSchema)]` plus a `#[schemars(title = "...")]` per
//! internally-tagged enum variant, add the `export-schema.rs` lines, and give
//! `Manifest` `model_spec: Option<ModelSpec>` + `capability_set: Vec<Capability>`.

/// One namespaced capability tag from SPEC 3.1 (plus `ParadigmSsm` for the Mamba
/// reject reason). Exhaustive enum so classification is a compiler-checked match:
/// a new Phase-6 tag forces you to fill in as_str/reason/blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] // NOT Serialize; see the hand-written impl below.
pub enum Capability {
    ParadigmArTransformer,
    ParadigmSsm,
    AttnFull,
    AttnGqa,
    AttnSwa,
    AttnSink,
    AttnMla,
    PosRope,
    PosRopePartial,
    PosLearned,
    PosAlibi,
    FfnDense,
    FfnMoe,
    FfnMoeSharedExperts,
    NormRms,
    NormSandwich,
    HeadLogitSoftcap,
    WeightsBf16,
    WeightsMxfp4,
    WeightsGptq,
}

/// Hand-written so `as_str()` is the GENUINE single source of the wire spelling.
/// `#[derive(Serialize)]` would emit the PascalCase variant identifier ("AttnSwa"),
/// and rename_all="snake_case" would emit "attn_swa" - neither is the dotted wire
/// form "attn.swa" that SPEC 4.1's --json capability tags require.
impl serde::Serialize for Capability {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Capability {
    /// Every variant in canonical declaration order (SPEC 3.1 tag order).
    /// Keep in sync with the enum; the exhaustive `match`es below are the
    /// compiler-checked safety net, this array only fixes iteration order.
    pub const ALL: [Capability; 20] = [
        Capability::ParadigmArTransformer,
        Capability::ParadigmSsm,
        Capability::AttnFull,
        Capability::AttnGqa,
        Capability::AttnSwa,
        Capability::AttnSink,
        Capability::AttnMla,
        Capability::PosRope,
        Capability::PosRopePartial,
        Capability::PosLearned,
        Capability::PosAlibi,
        Capability::FfnDense,
        Capability::FfnMoe,
        Capability::FfnMoeSharedExperts,
        Capability::NormRms,
        Capability::NormSandwich,
        Capability::HeadLogitSoftcap,
        Capability::WeightsBf16,
        Capability::WeightsMxfp4,
        Capability::WeightsGptq,
    ];

    /// The dotted wire form, e.g. "attn.swa". THE single source of the spelling;
    /// reused by Display, the hand-written Serialize impl above, and --json.
    pub fn as_str(self) -> &'static str {
        match self {
            Capability::ParadigmArTransformer => "paradigm.ar-transformer",
            Capability::ParadigmSsm => "paradigm.ssm",
            Capability::AttnFull => "attn.full",
            Capability::AttnGqa => "attn.gqa",
            Capability::AttnSwa => "attn.swa",
            Capability::AttnSink => "attn.sink",
            Capability::AttnMla => "attn.mla",
            Capability::PosRope => "pos.rope",
            Capability::PosRopePartial => "pos.rope.partial",
            Capability::PosLearned => "pos.learned",
            Capability::PosAlibi => "pos.alibi",
            Capability::FfnDense => "ffn.dense",
            Capability::FfnMoe => "ffn.moe",
            Capability::FfnMoeSharedExperts => "ffn.moe.shared-experts",
            Capability::NormRms => "norm.rms",
            Capability::NormSandwich => "norm.sandwich",
            Capability::HeadLogitSoftcap => "head.logit-softcap",
            Capability::WeightsBf16 => "weights.bf16",
            Capability::WeightsMxfp4 => "weights.mxfp4",
            Capability::WeightsGptq => "weights.gptq",
        }
    }

    /// Human gate reason; non-empty only for blocking caps and always ending in
    /// its "(<tag>)" so sidecars can match by tag. Verbatim from recon.
    pub fn reason(self) -> &'static str {
        match self {
            Capability::ParadigmSsm => "state-space model, not transformer paradigm (paradigm.ssm)",
            Capability::AttnSwa => "sliding-window attention unsupported (attn.swa)",
            Capability::AttnSink => "attention sink unsupported (attn.sink)",
            Capability::AttnMla => "multi-head latent attention unsupported (attn.mla)",
            Capability::WeightsMxfp4 => "MXFP4 quantization unsupported (weights.mxfp4)",
            Capability::WeightsGptq => "GPTQ quantization unsupported (weights.gptq)",
            _ => "",
        }
    }

    /// True ONLY for the six caps Phase 1 cannot handle. Every implemented
    /// conversion cap and every fidelity cap returns false, so fidelity tags
    /// cannot block by construction. Flipping AttnSwa/AttnSink here later is the
    /// entire "enable GPT-OSS" change (ARCHITECTURE 5's "flip the gate").
    pub fn blocking(self) -> bool {
        matches!(
            self,
            Capability::ParadigmSsm
                | Capability::AttnSwa
                | Capability::AttnSink
                | Capability::AttnMla
                | Capability::WeightsMxfp4
                | Capability::WeightsGptq
        )
    }
}

/// Flat, pure description of the model. No AttentionSpec/FFNSpec nesting this phase.
#[derive(serde::Serialize, Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub model_type: String,
    pub n_layers: u64,
    pub d_model: u64,
    pub vocab_size: u64,
    pub n_heads: u64,
    pub n_kv_heads: u64,
    pub sliding_window: Option<u64>,
    pub n_experts: Option<u64>,
    pub n_active_experts: Option<u64>,
    pub capabilities: Vec<Capability>,
    pub mask_token_id: Option<i64>,
    /// true from GenericAdapter, false from a trusted known adapter.
    pub inferred: bool,
}

/// Architecture verdict. Internally tagged for future contract compatibility.
#[derive(serde::Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    Supported,
    SupportedInferred,
    Unsupported { reasons: Vec<String> },
}

/// Files-on-disk axis, ORTHOGONAL to the verdict (a model can be Supported AND weightless).
#[derive(serde::Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "weights", rename_all = "snake_case")]
pub enum WeightsStatus {
    Present {
        format: String,
    },
    /// "hf download <repo-id> --local-dir <dir>"
    Missing {
        hint: String,
    },
}

/// Report-only mask-token decision; the vocab mutation happens at convert-time (Phase 2).
#[derive(serde::Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum MaskStrategy {
    ReuseId {
        id: i64,
        token: String,
    },
    GrowVocab,
    /// no tokenizer files present (the config-only fixtures)
    Undetermined,
}

#[derive(serde::Serialize, Debug, Clone, PartialEq)]
pub struct Plan {
    /// "mdlm" (SPEC 4.2 default)
    pub objective: String,
    /// placeholder descriptor; numbers are a convert flag
    pub anneal_schedule: String,
    pub mask_token: MaskStrategy,
    /// coarse dims-only estimate, "(approx, weights-only)"
    pub estimated_memory_gb: Option<f64>,
}

#[derive(serde::Serialize, Debug, Clone, PartialEq)]
pub struct DetectReport {
    pub spec: ModelSpec,
    pub verdict: Verdict,
    pub weights: WeightsStatus,
    /// Some for Supported/SupportedInferred, None for Unsupported.
    pub plan: Option<Plan>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_six_caps_block_and_carry_a_reason() {
        let blocking: Vec<Capability> = Capability::ALL
            .into_iter()
            .filter(|c| c.blocking())
            .collect();
        assert_eq!(
            blocking,
            vec![
                Capability::ParadigmSsm,
                Capability::AttnSwa,
                Capability::AttnSink,
                Capability::AttnMla,
                Capability::WeightsMxfp4,
                Capability::WeightsGptq,
            ]
        );
        for c in blocking {
            assert!(!c.reason().is_empty(), "{c:?} must carry a reason");
            assert!(
                c.reason().ends_with(&format!("({})", c.as_str())),
                "{c:?} reason must end in its dotted tag"
            );
        }
        // Non-blocking caps (implemented + fidelity) carry no reason.
        for c in Capability::ALL {
            if !c.blocking() {
                assert!(c.reason().is_empty(), "{c:?} must have an empty reason");
            }
        }
    }

    #[test]
    fn capability_serializes_to_dotted_form() {
        // Proves the hand-written Serialize reuses as_str(), not the derive.
        assert_eq!(
            serde_json::to_string(&Capability::AttnSwa).unwrap(),
            "\"attn.swa\""
        );
        assert_eq!(
            serde_json::to_string(&Capability::FfnMoeSharedExperts).unwrap(),
            "\"ffn.moe.shared-experts\""
        );
    }
}
