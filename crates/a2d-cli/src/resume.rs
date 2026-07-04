use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::Args;

use a2d_contracts::RunStatus;

/// Flags for `a2d resume`.
#[derive(Args)]
pub struct ResumeArgs {
    /// The run directory to resume (created by a prior `a2d convert`).
    pub run_dir: PathBuf,
    /// Override the worker command (default: A2D_WORKER_CMD env, else dev uv default).
    #[arg(long = "worker-cmd")]
    pub worker_cmd: Option<String>,
}

pub fn run(args: ResumeArgs) -> Result<ExitCode> {
    let resolved = crate::worker_cmd::resolve(args.worker_cmd, "A2D_WORKER_CMD", "a2d-worker");
    let argv: Vec<String> = resolved.split_whitespace().map(str::to_string).collect();
    if argv.is_empty() {
        bail!("resolved worker command is empty");
    }

    // a2d-run reads manifest.json for model_path + conversion_config, asserts the
    // recorded identity gate passed, reopens the run dir to Running, and re-spawns
    // the worker (which resumes from the latest checkpoint).
    let outcome = a2d_run::resume_job(&args.run_dir, &argv, crate::convert::render_event)
        .context("resuming conversion job")?;

    Ok(match outcome.status {
        RunStatus::Completed => {
            println!(
                "run {} completed -> {}",
                outcome.job_id,
                args.run_dir.display()
            );
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "run {} failed -> {}",
                outcome.job_id,
                args.run_dir.display()
            );
            ExitCode::from(1)
        }
    })
}
