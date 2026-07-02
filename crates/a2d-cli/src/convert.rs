use std::path::Path;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};

use a2d_contracts::{Event, EventEnvelope, LogLevel, RunStatus};

/// Resolve the worker command per PROTOCOL discovery order:
/// `--worker-cmd` flag -> `A2D_WORKER_CMD` env -> dev default (`uv run --project ...`).
fn resolve_worker_cmd(flag: Option<String>) -> String {
    if let Some(cmd) = flag {
        return cmd;
    }
    if let Ok(cmd) = std::env::var("A2D_WORKER_CMD") {
        if !cmd.trim().is_empty() {
            return cmd;
        }
    }
    // Dev default: repo root is two levels up from this crate's manifest dir.
    let repo = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    format!("uv run --project {repo}/packages/a2d-worker-hf a2d-worker")
}

fn level_str(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    }
}

/// Render one human-readable line per event for the terminal.
fn render_event(env: &EventEnvelope) {
    match &env.event {
        Event::JobStarted { worker } => println!("job started (worker: {worker})"),
        Event::Log { level, message } => println!("{}: {message}", level_str(*level)),
        Event::JobCompleted {} => println!("job completed"),
        Event::JobFailed { message } => eprintln!("job failed: {message}"),
    }
}

pub fn run(model_dir: &Path, run_dir: &Path, worker_cmd: Option<String>) -> Result<ExitCode> {
    if !model_dir.exists() {
        bail!("model dir does not exist: {}", model_dir.display());
    }

    let resolved = resolve_worker_cmd(worker_cmd);
    // ponytail: plain whitespace split is enough for Phase 0 (spec says shlex-free is fine);
    // upgrade to a real shell-word splitter if worker commands ever need quoting.
    let argv: Vec<String> = resolved.split_whitespace().map(str::to_string).collect();
    if argv.is_empty() {
        bail!("resolved worker command is empty");
    }

    // a2d-run owns run-dir creation, job_id generation, manifest lifecycle, spawning,
    // stdout->events.jsonl teeing, and manifest finalization. We just render and set exit code.
    let outcome = a2d_run::worker::run_job(model_dir, run_dir, &argv, render_event)
        .context("running conversion job")?;

    Ok(match outcome.status {
        RunStatus::Completed => {
            println!("run {} completed -> {}", outcome.job_id, run_dir.display());
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("run {} failed -> {}", outcome.job_id, run_dir.display());
            ExitCode::from(1)
        }
    })
}
