use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;

use a2d_contracts::{SampleRequest, SCHEMA_VERSION};

/// Flags for `a2d sample`.
#[derive(Args)]
pub struct SampleArgs {
    /// The run directory whose `model/` checkpoint to sample from.
    pub run_dir: PathBuf,
    /// The prompt to condition on.
    #[arg(short = 'p', long)]
    pub prompt: String,
    /// Total canvas length (prompt + masked positions to fill).
    #[arg(long = "canvas-len", default_value_t = 64)]
    pub canvas_len: u64,
    /// Number of denoising steps.
    #[arg(long = "num-steps", default_value_t = 32)]
    pub num_steps: u64,
    /// Sampling temperature.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Accelerator: "auto" | "cpu" | "mps" | "cuda".
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Override the sample worker command (default: A2D_SAMPLE_CMD env, else dev uv default).
    #[arg(long = "worker-cmd")]
    pub worker_cmd: Option<String>,
}

pub fn run(args: SampleArgs) -> Result<ExitCode> {
    let model_dir = args.run_dir.join("model");
    if !model_dir.is_dir() {
        bail!(
            "no model/ directory in run dir {} (run `a2d convert` first)",
            args.run_dir.display()
        );
    }

    let request = SampleRequest {
        schema_version: SCHEMA_VERSION.to_string(),
        model_dir: model_dir.to_string_lossy().into_owned(),
        prompt: args.prompt,
        canvas_len: args.canvas_len,
        num_steps: args.num_steps,
        temperature: args.temperature,
        seed: args.seed,
        device: args.device,
    };

    let resolved = crate::worker_cmd::resolve(args.worker_cmd, "A2D_SAMPLE_CMD", "a2d-sample");
    let argv: Vec<String> = resolved.split_whitespace().map(str::to_string).collect();
    let (program, rest) = argv
        .split_first()
        .ok_or_else(|| anyhow!("resolved sample worker command is empty"))?;

    // Spawn a2d-sample, feed it the request on stdin, capture the generated text.
    // stderr is inherited so worker diagnostics surface directly; the worker exits
    // 0 (ok) or 2 (contract violation) and we propagate that.
    let mut child = Command::new(program)
        .args(rest)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning sample worker: {resolved}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .context("sample worker stdin not captured")?;
        let doc = serde_json::to_vec(&request).context("serializing sample request")?;
        stdin
            .write_all(&doc)
            .context("writing sample request to worker stdin")?;
    }

    let mut text = String::new();
    child
        .stdout
        .take()
        .context("sample worker stdout not captured")?
        .read_to_string(&mut text)
        .context("reading sample worker stdout")?;
    let status = child.wait().context("waiting on sample worker")?;

    print!("{text}");

    Ok(match status.code() {
        Some(0) => ExitCode::SUCCESS,
        Some(code) => ExitCode::from(code as u8),
        None => ExitCode::FAILURE,
    })
}
