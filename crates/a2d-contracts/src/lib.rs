//! Canonical boundary contract for a2d (D9): Rust types are the single source
//! of truth. JSON Schema is generated from these (see `bin/export-schema.rs`)
//! and the Python worker's pydantic models are generated from that schema.
//! Field names, serde attrs, and wire shape here are authoritative.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Contract version. Tied to the crate version so a bump is a deliberate,
/// visible act rather than a hidden constant edit.
pub const SCHEMA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// One namespaced capability tag from SPEC 3.1 (plus `ParadigmSsm` for the Mamba
/// reject reason). Exhaustive enum so classification is a compiler-checked match:
/// a new Phase-6 tag forces you to fill in as_str/reason/blocking.
///
/// Capability derives NONE of Serialize/Deserialize/JsonSchema: all three are
/// hand-written so the dotted wire form ("attn.swa") stays the single source of
/// truth. A naive derive would emit the PascalCase variant identifier, not the
/// dotted spelling SPEC 4.1's --json capability tags require.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Inverts `as_str()`; the exhaustive match keeps it in lockstep with the wire form.
impl std::str::FromStr for Capability {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "paradigm.ar-transformer" => Capability::ParadigmArTransformer,
            "paradigm.ssm" => Capability::ParadigmSsm,
            "attn.full" => Capability::AttnFull,
            "attn.gqa" => Capability::AttnGqa,
            "attn.swa" => Capability::AttnSwa,
            "attn.sink" => Capability::AttnSink,
            "attn.mla" => Capability::AttnMla,
            "pos.rope" => Capability::PosRope,
            "pos.rope.partial" => Capability::PosRopePartial,
            "pos.learned" => Capability::PosLearned,
            "pos.alibi" => Capability::PosAlibi,
            "ffn.dense" => Capability::FfnDense,
            "ffn.moe" => Capability::FfnMoe,
            "ffn.moe.shared-experts" => Capability::FfnMoeSharedExperts,
            "norm.rms" => Capability::NormRms,
            "norm.sandwich" => Capability::NormSandwich,
            "head.logit-softcap" => Capability::HeadLogitSoftcap,
            "weights.bf16" => Capability::WeightsBf16,
            "weights.mxfp4" => Capability::WeightsMxfp4,
            "weights.gptq" => Capability::WeightsGptq,
            other => return Err(format!("unknown capability tag: {other:?}")),
        })
    }
}

impl<'de> serde::Deserialize<'de> for Capability {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Manual impl (NOT a derive or container `schema_with`, which schemars rejects at
/// container level): one string schema whose `enum` is the dotted `as_str` list.
impl JsonSchema for Capability {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Capability".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "title": "Capability",
            "enum": Capability::ALL.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
        })
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
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
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

/// The conversion knobs. REQUIRED on `ConversionJob` (transient, never persisted,
/// so no back-compat reason to make it Option).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct ConversionConfig {
    /// "mdlm", resolved via the objectives registry.
    pub objective: String,
    /// Required local jsonl/txt path.
    pub data: String,
    /// Attention causal->bidir window (drives ONLY the attention anneal).
    pub anneal_steps: u64,
    /// "linear" (cosine is a drop-in).
    pub anneal_schedule: String,
    /// Default 512, <= GPT-2 n_positions 1024.
    pub seq_len: u64,
    /// Default 8; -> TrainingArguments.per_device_train_batch_size.
    pub per_device_batch_size: u64,
    /// Default 1; -> gradient_accumulation_steps.
    pub grad_accum: u64,
    pub lr: f64,
    /// Exactly one of max_steps/max_tokens required.
    pub max_steps: Option<u64>,
    /// Resolved to max_steps = ceil(max_tokens / tokens_per_step) at config-build.
    pub max_tokens: Option<u64>,
    /// "reuse"|"grow", default resolved from detect's MaskStrategy.
    pub mask_token: String,
    /// Default 3; maps to save_total_limit.
    pub keep_last: u64,
    pub seed: u64,
    /// "auto"|"cpu"|"mps"|"cuda".
    pub device: String,
    /// "float32"|"bfloat16", default float32.
    pub dtype: String,
}

/// The identity-gate record written into the manifest.
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct IdentityResult {
    pub passed: bool,
    pub max_abs_diff: f64,
    pub tolerance: f64,
}

/// The a2d-sample worker request (the ONE new export-schema root).
#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
pub struct SampleRequest {
    pub schema_version: String,
    pub model_dir: String,
    pub prompt: String,
    pub canvas_len: u64,
    pub num_steps: u64,
    pub temperature: f64,
    pub seed: u64,
    pub device: String,
}

/// Job sent to the worker on stdin.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConversionJob {
    pub schema_version: String,
    pub job_id: String,
    pub model_path: String,
    pub run_dir: String,
    pub conversion_config: ConversionConfig,
}

/// One line of the worker's stdout JSONL event stream (mirrored to
/// `events.jsonl`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct EventEnvelope {
    pub schema_version: String,
    pub job_id: String,
    pub seq: u64,
    /// RFC3339 timestamp.
    pub ts: String,
    pub event: Event,
}

/// Worker event payload. Internally tagged so each line is a flat object:
/// `{"type":"log","level":"info","message":"..."}`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    #[schemars(title = "JobStarted")]
    JobStarted { worker: String },
    #[schemars(title = "Log")]
    Log { level: LogLevel, message: String },
    #[schemars(title = "Progress")]
    Progress {
        stage: String,
        step: u64,
        total: Option<u64>,
    },
    #[schemars(title = "TrainStep")]
    TrainStep {
        step: u64,
        loss: f64,
        anneal: f64,
        lr: f64,
        tokens: u64,
    },
    #[schemars(title = "IdentityGate")]
    IdentityGate {
        passed: bool,
        max_abs_diff: f64,
        tolerance: f64,
    },
    #[schemars(title = "Checkpoint")]
    Checkpoint { step: u64, path: String },
    #[schemars(title = "Metric")]
    Metric { name: String, value: f64, step: u64 },
    #[schemars(title = "JobCompleted")]
    JobCompleted {},
    #[schemars(title = "JobFailed")]
    JobFailed { message: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Run provenance written to `<run-dir>/manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Manifest {
    pub schema_version: String,
    pub a2d_version: String,
    pub job_id: String,
    /// RFC3339 timestamp.
    pub created_at: String,
    pub model_path: String,
    pub status: RunStatus,
    pub finished_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_spec: Option<ModelSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversion_config: Option<ConversionConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<IdentityResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T>(v: &T) -> T
    where
        T: Serialize + serde::de::DeserializeOwned,
    {
        let json = serde_json::to_string(v).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    fn sample_config() -> ConversionConfig {
        ConversionConfig {
            objective: "mdlm".into(),
            data: "./fixtures/data/tiny.jsonl".into(),
            anneal_steps: 100,
            anneal_schedule: "linear".into(),
            seq_len: 512,
            per_device_batch_size: 8,
            grad_accum: 1,
            lr: 1e-4,
            max_steps: Some(500),
            max_tokens: None,
            mask_token: "grow".into(),
            keep_last: 3,
            seed: 0,
            device: "auto".into(),
            dtype: "float32".into(),
        }
    }

    #[test]
    fn conversion_job_roundtrip() {
        let job = ConversionJob {
            schema_version: SCHEMA_VERSION.to_string(),
            job_id: "job-1".into(),
            model_path: "/tmp/fake-model".into(),
            run_dir: "runs/demo".into(),
            conversion_config: sample_config(),
        };
        assert_eq!(job, roundtrip(&job));
    }

    #[test]
    fn event_envelope_roundtrip() {
        for event in [
            Event::JobStarted {
                worker: "a2d-worker-hf 0.1.0".into(),
            },
            Event::Log {
                level: LogLevel::Info,
                message: "no-op conversion".into(),
            },
            Event::Progress {
                stage: "materialize".into(),
                step: 1,
                total: Some(4),
            },
            Event::TrainStep {
                step: 2,
                loss: 3.5,
                anneal: 0.5,
                lr: 1e-4,
                tokens: 8192,
            },
            Event::IdentityGate {
                passed: true,
                max_abs_diff: 0.0,
                tolerance: 1e-6,
            },
            Event::Checkpoint {
                step: 2,
                path: "checkpoints/checkpoint-2".into(),
            },
            Event::Metric {
                name: "loss".into(),
                value: 3.5,
                step: 2,
            },
            Event::JobCompleted {},
            Event::JobFailed {
                message: "boom".into(),
            },
        ] {
            let env = EventEnvelope {
                schema_version: SCHEMA_VERSION.to_string(),
                job_id: "job-1".into(),
                seq: 0,
                ts: "2026-07-02T10:00:00Z".into(),
                event,
            };
            assert_eq!(env, roundtrip(&env));
        }
    }

    #[test]
    fn manifest_roundtrip() {
        let manifest = Manifest {
            schema_version: SCHEMA_VERSION.to_string(),
            a2d_version: "0.1.0".into(),
            job_id: "job-1".into(),
            created_at: "2026-07-02T10:00:00Z".into(),
            model_path: "/tmp/fake-model".into(),
            status: RunStatus::Running,
            finished_at: None,
            model_spec: None,
            conversion_config: None,
            identity: None,
            data_source: None,
            source_hash: None,
            token_count: None,
        };
        assert_eq!(manifest, roundtrip(&manifest));

        let finished = Manifest {
            status: RunStatus::Completed,
            finished_at: Some("2026-07-02T10:05:00Z".into()),
            model_spec: Some(ModelSpec {
                model_type: "gpt2".into(),
                n_layers: 12,
                d_model: 768,
                vocab_size: 50257,
                n_heads: 12,
                n_kv_heads: 12,
                sliding_window: None,
                n_experts: None,
                n_active_experts: None,
                capabilities: vec![Capability::AttnFull, Capability::PosLearned],
                mask_token_id: Some(50257),
                inferred: false,
            }),
            conversion_config: Some(sample_config()),
            identity: Some(IdentityResult {
                passed: true,
                max_abs_diff: 0.0,
                tolerance: 1e-6,
            }),
            data_source: Some("./fixtures/data/tiny.jsonl".into()),
            source_hash: Some("deadbeef".into()),
            token_count: Some(4096000),
            ..manifest
        };
        assert_eq!(finished, roundtrip(&finished));
    }

    /// Phase-0/1 manifests have none of the six new fields; they must still
    /// deserialize, filling the new fields with None.
    #[test]
    fn old_shape_manifest_deserializes_to_none() {
        let old = r#"{
            "schema_version": "0.1.0",
            "a2d_version": "0.1.0",
            "job_id": "job-1",
            "created_at": "2026-07-02T10:00:00Z",
            "model_path": "/tmp/fake-model",
            "status": "running",
            "finished_at": null
        }"#;
        let m: Manifest = serde_json::from_str(old).expect("old-shape manifest deserializes");
        assert_eq!(m.model_spec, None);
        assert_eq!(m.conversion_config, None);
        assert_eq!(m.identity, None);
        assert_eq!(m.data_source, None);
        assert_eq!(m.source_hash, None);
        assert_eq!(m.token_count, None);
    }

    #[test]
    fn enum_variants_roundtrip() {
        for level in [
            LogLevel::Debug,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
        ] {
            assert_eq!(level, roundtrip(&level));
        }
        for status in [RunStatus::Running, RunStatus::Completed, RunStatus::Failed] {
            assert_eq!(status, roundtrip(&status));
        }
    }

    #[test]
    fn event_tag_strings_on_the_wire() {
        let cases = [
            (Event::JobStarted { worker: "w".into() }, "job_started"),
            (
                Event::Log {
                    level: LogLevel::Info,
                    message: "m".into(),
                },
                "log",
            ),
            (Event::JobCompleted {}, "job_completed"),
            (
                Event::JobFailed {
                    message: "m".into(),
                },
                "job_failed",
            ),
        ];
        for (event, tag) in cases {
            let json = serde_json::to_string(&event).expect("serialize");
            assert!(
                json.contains(&format!("\"{tag}\"")),
                "expected tag {tag:?} in {json}"
            );
        }
    }

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
        for c in Capability::ALL {
            if !c.blocking() {
                assert!(c.reason().is_empty(), "{c:?} must have an empty reason");
            }
        }
    }

    #[test]
    fn capability_serializes_to_dotted_form() {
        assert_eq!(
            serde_json::to_string(&Capability::AttnSwa).unwrap(),
            "\"attn.swa\""
        );
        assert_eq!(
            serde_json::to_string(&Capability::FfnMoeSharedExperts).unwrap(),
            "\"ffn.moe.shared-experts\""
        );
    }

    /// Every dotted wire form parses back to its variant, and the full serialize
    /// -> deserialize -> serialize chain is stable over ALL.
    #[test]
    fn capability_roundtrips_over_all() {
        for cap in Capability::ALL {
            let json = serde_json::to_string(&cap).expect("serialize");
            let back: Capability = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(cap, back);
            assert_eq!(json, serde_json::to_string(&back).expect("re-serialize"));
        }
    }

    /// The hand-written JsonSchema emits a string schema whose `enum` list is
    /// exactly `ALL.map(as_str)` in declaration order.
    #[test]
    fn capability_schema_enum_matches_as_str() {
        let schema = serde_json::to_value(schemars::schema_for!(Capability)).expect("schema value");
        let enum_list = schema
            .get("enum")
            .and_then(|v| v.as_array())
            .expect("enum array");
        let expected: Vec<_> = Capability::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(enum_list.len(), expected.len());
        for (got, want) in enum_list.iter().zip(expected) {
            assert_eq!(got.as_str(), Some(want));
        }
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("string"));
    }
}
