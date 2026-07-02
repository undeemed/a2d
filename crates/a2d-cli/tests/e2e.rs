//! The Phase 0 exit-criterion integration test: a no-op `a2d convert` spawns the worker,
//! streams events, and leaves a well-formed run dir (manifest.json + events.jsonl).

use std::process::Command;

use a2d_contracts::{Event, EventEnvelope, Manifest, RunStatus};

fn uv_available() -> bool {
    Command::new("uv")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn convert_no_op_writes_well_formed_run_dir() {
    if !uv_available() {
        eprintln!("skipping convert_no_op_writes_well_formed_run_dir: `uv` not found on PATH");
        return;
    }

    let model = tempfile::tempdir().expect("create model tempdir");
    let run_root = tempfile::tempdir().expect("create run-root tempdir");
    let run_dir = run_root.path().join("demo");

    let status = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .arg("convert")
        .arg(model.path())
        .arg("--out")
        .arg(&run_dir)
        .status()
        .expect("spawn a2d");
    assert!(status.success(), "`a2d convert` should exit 0");

    let manifest_raw =
        std::fs::read_to_string(run_dir.join("manifest.json")).expect("read manifest.json");
    let manifest: Manifest = serde_json::from_str(&manifest_raw).expect("parse manifest.json");
    assert_eq!(manifest.status, RunStatus::Completed);

    let events_raw =
        std::fs::read_to_string(run_dir.join("events.jsonl")).expect("read events.jsonl");
    let envelopes: Vec<EventEnvelope> = events_raw
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("parse EventEnvelope line"))
        .collect();
    assert!(!envelopes.is_empty(), "events.jsonl should have entries");
    assert!(
        matches!(envelopes.last().unwrap().event, Event::JobCompleted {}),
        "last event should be job_completed"
    );
}

/// Path to a corpus fixture dir, relative to this crate's manifest dir.
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/configs")
        .join(name)
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
