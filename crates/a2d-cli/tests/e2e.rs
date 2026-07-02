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

#[test]
fn detect_subcommand_is_stubbed_with_exit_2() {
    let output = Command::new(env!("CARGO_BIN_EXE_a2d"))
        .arg("detect")
        .arg("x")
        .output()
        .expect("spawn a2d");
    assert_eq!(output.status.code(), Some(2), "`a2d detect` should exit 2");
}
