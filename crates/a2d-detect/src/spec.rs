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

// Promoted to a2d-contracts (Phase 2, Decision 5); re-exported so `crate::spec::`
// paths and the wire form stay in one place. Verdict/WeightsStatus/MaskStrategy/
// Plan/DetectReport stay local (render-only, never cross the worker boundary).
pub use a2d_contracts::{Capability, ModelSpec};

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
