use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod convert;
mod detect;
mod runs;

/// a2d - convert local autoregressive LLMs into diffusion language models.
#[derive(Parser)]
#[command(name = "a2d", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a local AR model into a diffusion checkpoint (Phase 0: no-op walking skeleton).
    Convert {
        /// Local model directory (must already contain downloaded weights).
        model_dir: PathBuf,
        /// Run directory to create for this conversion.
        #[arg(long = "out")]
        run_dir: PathBuf,
        /// Override the worker command (default: A2D_WORKER_CMD env, else dev uv default).
        #[arg(long = "worker-cmd")]
        worker_cmd: Option<String>,
    },
    /// List conversion runs found under a root directory.
    Runs {
        /// Root directory to scan (default: ./runs).
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Detect and gate a model from its config (Phase 1).
    Detect {
        /// Local model directory (must contain a config.json).
        path: PathBuf,
        /// Emit the full DetectReport as JSON instead of the human render.
        #[arg(long)]
        json: bool,
    },
    /// Resume a conversion run (Phase 2).
    Resume {
        #[allow(dead_code)]
        path: PathBuf,
    },
    /// Evaluate a converted checkpoint (Phase 3).
    Eval {
        #[allow(dead_code)]
        path: PathBuf,
    },
    /// Sample from a converted checkpoint (Phase 5).
    Sample {
        #[allow(dead_code)]
        path: PathBuf,
    },
}

/// A subcommand that isn't live yet: explain and exit 2 (contract-violation-style code).
fn stub(name: &str) -> ExitCode {
    eprintln!("error: 'a2d {name}' is not implemented yet (see the roadmap in docs/SPEC-HANDOFF.md section 6)");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Convert {
            model_dir,
            run_dir,
            worker_cmd,
        } => match convert::run(&model_dir, &run_dir, worker_cmd) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::FAILURE
            }
        },
        Command::Runs { root } => match runs::run(root) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::FAILURE
            }
        },
        Command::Detect { path, json } => match detect::run(&path, json) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::FAILURE
            }
        },
        Command::Resume { .. } => stub("resume"),
        Command::Eval { .. } => stub("eval"),
        Command::Sample { .. } => stub("sample"),
    }
}
