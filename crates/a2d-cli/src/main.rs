use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod convert;
mod detect;
mod resume;
mod runs;
mod sample;
mod worker_cmd;

/// a2d - convert local autoregressive LLMs into diffusion language models.
#[derive(Parser)]
#[command(name = "a2d", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a local AR model into a diffusion checkpoint.
    Convert(convert::ConvertArgs),
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
    /// Resume a conversion run.
    Resume(resume::ResumeArgs),
    /// Evaluate a converted checkpoint (Phase 3).
    Eval {
        #[allow(dead_code)]
        path: PathBuf,
    },
    /// Sample from a converted checkpoint.
    Sample(sample::SampleArgs),
}

/// A subcommand that isn't live yet: explain and exit 2 (contract-violation-style code).
fn stub(name: &str) -> ExitCode {
    eprintln!("error: 'a2d {name}' is not implemented yet (see the roadmap in docs/SPEC-HANDOFF.md section 6)");
    ExitCode::from(2)
}

fn dispatch<T>(result: anyhow::Result<T>, on_ok: impl FnOnce(T) -> ExitCode) -> ExitCode {
    match result {
        Ok(v) => on_ok(v),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Convert(args) => dispatch(convert::run(args), |code| code),
        Command::Runs { root } => dispatch(runs::run(root), |()| ExitCode::SUCCESS),
        Command::Detect { path, json } => dispatch(detect::run(&path, json), |code| code),
        Command::Resume(args) => dispatch(resume::run(args), |code| code),
        Command::Eval { .. } => stub("eval"),
        Command::Sample(args) => dispatch(sample::run(args), |code| code),
    }
}
