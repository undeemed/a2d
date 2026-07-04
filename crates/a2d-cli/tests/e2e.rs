//! Torch-free CLI integration tests: `a2d convert` drives detect + gate, then a
//! FAKE worker stub (a shell script, no python/network) so the run-dir shape,
//! manifest merge, and exit codes are all proven without heavy deps.

use std::process::Command;

use a2d_contracts::{Event, EventEnvelope, Manifest, RunStatus};

/// Path to a corpus fixture dir, relative to this crate's manifest dir.
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/configs")
        .join(name)
}

/// A fake worker that drains stdin then emits the full Phase-2 event set,
/// including an IdentityGate + TrainStep so manifest accumulation is exercised,
/// plus a garbage line to prove unparseable stdout is preserved and non-fatal.
const FAKE_WORKER: &str = r#"#!/bin/sh
cat >/dev/null
printf '%s\n' '{"schema_version":"0.1.0","job_id":"t","seq":0,"ts":"2026-07-02T10:00:00Z","event":{"type":"job_started","worker":"fake-worker"}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"t","seq":1,"ts":"2026-07-02T10:00:01Z","event":{"type":"progress","stage":"materialize","step":1,"total":4}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"t","seq":2,"ts":"2026-07-02T10:00:02Z","event":{"type":"identity_gate","passed":true,"max_abs_diff":0.0,"tolerance":1e-6}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"t","seq":3,"ts":"2026-07-02T10:00:03Z","event":{"type":"train_step","step":1,"loss":2.5,"anneal":0.5,"lr":1e-4,"tokens":4096}}'
printf '%s\n' 'not-json-garbage'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"t","seq":4,"ts":"2026-07-02T10:00:04Z","event":{"type":"checkpoint","step":1,"path":"checkpoints/checkpoint-1"}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"t","seq":5,"ts":"2026-07-02T10:00:05Z","event":{"type":"job_completed"}}'
"#;

/// Build a supported (gpt2) model dir with a dummy `model.safetensors` so detect
/// classifies it Supported+Present and `source_hash` succeeds; return the dir and
/// the `sh <script>` argv for the fake worker.
fn supported_model_and_worker(base: &std::path::Path) -> (std::path::PathBuf, String) {
    let model = base.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::copy(
        fixture("gpt2").join("config.json"),
        model.join("config.json"),
    )
    .unwrap();
    std::fs::write(model.join("model.safetensors"), b"not-real-weights").unwrap();

    let script = base.join("fake_worker.sh");
    std::fs::write(&script, FAKE_WORKER).unwrap();
    (model, format!("sh {}", script.to_str().unwrap()))
}

#[test]
fn convert_smoke_merges_manifest_and_writes_run_dir() {
    let base = tempfile::tempdir().expect("tempdir");
    let (model, worker_cmd) = supported_model_and_worker(base.path());
    let run_dir = base.path().join("run");

    let status = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .args(["convert"])
        .arg(&model)
        .arg("--out")
        .arg(&run_dir)
        .args(["--data", "./fixtures/data/tiny.jsonl"])
        .args(["--max-steps", "10", "--anneal-steps", "5"])
        .args(["--worker-cmd", &worker_cmd])
        .status()
        .expect("spawn a2d");
    assert_eq!(status.code(), Some(0), "convert-smoke should exit 0");

    // Manifest merged the CLI-supplied provenance + the event-stream accumulation.
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    assert_eq!(manifest.status, RunStatus::Completed);
    assert!(manifest.model_spec.is_some(), "model_spec written");
    let cfg = manifest
        .conversion_config
        .expect("conversion_config written");
    assert_eq!(cfg.max_steps, Some(10));
    assert_eq!(cfg.data, "./fixtures/data/tiny.jsonl");
    assert_eq!(
        manifest.identity.map(|i| i.passed),
        Some(true),
        "identity gate merged from the stream"
    );
    assert_eq!(manifest.token_count, Some(4096), "token_count merged");
    assert!(manifest.source_hash.is_some(), "source_hash written");

    // events.jsonl is present and terminates in job_completed.
    let events = std::fs::read_to_string(run_dir.join("events.jsonl")).expect("read events.jsonl");
    let envelopes: Vec<EventEnvelope> = events
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    assert!(matches!(
        envelopes.last().unwrap().event,
        Event::JobCompleted {}
    ));
}

#[test]
fn convert_unsupported_aborts_exit_1_without_spawning() {
    let base = tempfile::tempdir().expect("tempdir");
    let run_dir = base.path().join("run");
    // A worker command that would fail loudly IF spawned; it must never run.
    let sentinel = "this-worker-binary-must-never-be-spawned";

    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .args(["convert"])
        .arg(fixture("mamba")) // paradigm.ssm -> Unsupported
        .arg("--out")
        .arg(&run_dir)
        .args(["--data", "./fixtures/data/tiny.jsonl"])
        .args(["--max-steps", "10"])
        .args(["--worker-cmd", sentinel])
        .output()
        .expect("spawn a2d");

    assert_eq!(output.status.code(), Some(1), "unsupported -> exit 1");
    // The gate aborted before a2d-run touched the filesystem: no run dir spawned.
    assert!(!run_dir.exists(), "run dir must not be created on abort");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported") && stderr.contains("paradigm.ssm"),
        "stderr should explain the reject: {stderr}"
    );
}

#[test]
fn convert_both_token_budgets_is_usage_error_exit_2() {
    let base = tempfile::tempdir().expect("tempdir");
    let run_dir = base.path().join("run");

    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .args(["convert"])
        .arg(fixture("gpt2")) // supported; config-only is fine, we exit before hashing
        .arg("--out")
        .arg(&run_dir)
        .args(["--data", "./fixtures/data/tiny.jsonl"])
        .args(["--max-steps", "10", "--max-tokens", "100000"])
        .output()
        .expect("spawn a2d");

    assert_eq!(
        output.status.code(),
        Some(2),
        "both --max-steps and --max-tokens -> exit 2"
    );
    assert!(!run_dir.exists(), "no run dir on usage error");
}

#[test]
fn detect_supported_fixture_exits_0() {
    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .arg("detect")
        .arg(fixture("gpt2"))
        .output()
        .expect("spawn a2d");
    assert_eq!(output.status.code(), Some(0), "gpt2 is supported -> exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("SUPPORTED"),
        "stdout should say SUPPORTED: {stdout}"
    );
}

#[test]
fn detect_unsupported_fixture_exits_1_with_reason() {
    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .arg("detect")
        .arg(fixture("mamba"))
        .output()
        .expect("spawn a2d");
    assert_eq!(
        output.status.code(),
        Some(1),
        "mamba is unsupported -> exit 1"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("UNSUPPORTED") && stdout.contains("paradigm.ssm"),
        "stdout should reject with paradigm.ssm: {stdout}"
    );
}

#[test]
fn detect_config_only_dir_exits_0_with_download_hint() {
    let dir = tempfile::tempdir().expect("create config-only tempdir");
    std::fs::copy(
        fixture("gpt2").join("config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config.json");

    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .arg("detect")
        .arg(dir.path())
        .output()
        .expect("spawn a2d");
    assert_eq!(
        output.status.code(),
        Some(0),
        "config-only supported -> exit 0"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("hf download"),
        "stderr should carry the hf download hint: {stderr}"
    );
}

#[test]
fn detect_missing_config_exits_2() {
    let dir = tempfile::tempdir().expect("create empty tempdir");
    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .arg("detect")
        .arg(dir.path())
        .output()
        .expect("spawn a2d");
    // run() intercepts the bad-input Err itself -> exit 2, not main's generic FAILURE (1).
    assert_eq!(output.status.code(), Some(2), "no config.json -> exit 2");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error:"),
        "stderr should carry an error: {stderr}"
    );
}
