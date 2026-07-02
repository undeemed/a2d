//! Canonical boundary contract for a2d (D9): Rust types are the single source
//! of truth. JSON Schema is generated from these (see `bin/export-schema.rs`)
//! and the Python worker's pydantic models are generated from that schema.
//! Field names, serde attrs, and wire shape here are authoritative.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Contract version. Tied to the crate version so a bump is a deliberate,
/// visible act rather than a hidden constant edit.
pub const SCHEMA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Job sent to the worker on stdin.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConversionJob {
    pub schema_version: String,
    pub job_id: String,
    pub model_path: String,
    pub run_dir: String,
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

    #[test]
    fn conversion_job_roundtrip() {
        let job = ConversionJob {
            schema_version: SCHEMA_VERSION.to_string(),
            job_id: "job-1".into(),
            model_path: "/tmp/fake-model".into(),
            run_dir: "runs/demo".into(),
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
        };
        assert_eq!(manifest, roundtrip(&manifest));

        let finished = Manifest {
            status: RunStatus::Completed,
            finished_at: Some("2026-07-02T10:05:00Z".into()),
            ..manifest
        };
        assert_eq!(finished, roundtrip(&finished));
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
}
