use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;

use a2d_contracts::{EvalRequest, SCHEMA_VERSION};

/// Flags for `a2d eval`.
#[derive(Args)]
pub struct EvalArgs {
    /// The run directory whose `model/` checkpoint to evaluate.
    pub run_dir: PathBuf,
    /// Held-out corpus for the likelihood bound + AR perplexity (local jsonl/txt).
    #[arg(long)]
    pub data: PathBuf,
    /// Downstream task to run; repeatable. Empty => all registered tasks.
    #[arg(long = "task")]
    pub tasks: Vec<String>,
    /// Eval context length. Defaults to the conversion's seq_len, else 128.
    #[arg(long = "seq-len")]
    pub seq_len: Option<u64>,
    /// Monte-Carlo t draws per chunk for the MDLM bound.
    #[arg(long = "mc-samples", default_value_t = 16)]
    pub mc_samples: u64,
    /// Cap on scored tokens.
    #[arg(long = "max-eval-tokens", default_value_t = 65536)]
    pub max_eval_tokens: u64,
    /// Sub-batch size for the MDLM likelihood forward; caps peak memory. 0 => one forward
    /// over all chunks (may OOM at large `--max-eval-tokens`).
    #[arg(long = "eval-batch-size", default_value_t = 8)]
    pub eval_batch_size: u64,
    /// Denoiser steps for the diffusion throughput measurement.
    #[arg(long = "num-steps", default_value_t = 32)]
    pub num_steps: u64,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Accelerator: "auto" | "cpu" | "mps" | "cuda".
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Also write eval/report.html.
    #[arg(long)]
    pub html: bool,
    /// Override the eval worker command (default: A2D_EVAL_CMD env, else dev uv default).
    #[arg(long = "worker-cmd")]
    pub worker_cmd: Option<String>,
}

pub fn run(args: EvalArgs) -> Result<ExitCode> {
    let model_dir = args.run_dir.join("model");
    if !model_dir.is_dir() {
        bail!(
            "no model/ directory in run dir {} (run `a2d convert` first)",
            args.run_dir.display()
        );
    }

    // Provenance for the AR comparison comes from the run's manifest (Decision 6).
    let manifest = a2d_run::read_manifest(&args.run_dir)
        .with_context(|| format!("reading manifest in {}", args.run_dir.display()))?;
    let seq_len = args
        .seq_len
        .or_else(|| manifest.conversion_config.as_ref().map(|c| c.seq_len))
        .unwrap_or(128);

    let request = EvalRequest {
        schema_version: SCHEMA_VERSION.to_string(),
        model_dir: model_dir.to_string_lossy().into_owned(),
        source_model: Some(manifest.model_path),
        source_hash: manifest.source_hash,
        data: args.data.to_string_lossy().into_owned(),
        tasks: args.tasks,
        seq_len,
        mc_samples: args.mc_samples,
        max_eval_tokens: args.max_eval_tokens,
        eval_batch_size: args.eval_batch_size,
        num_steps: args.num_steps,
        seed: args.seed,
        device: args.device,
        out_dir: args.run_dir.join("eval").to_string_lossy().into_owned(),
        html: args.html,
    };

    let resolved = crate::worker_cmd::resolve(args.worker_cmd, "A2D_EVAL_CMD", "a2d-eval");
    let argv: Vec<String> = resolved.split_whitespace().map(str::to_string).collect();
    let (program, rest) = argv
        .split_first()
        .ok_or_else(|| anyhow!("resolved eval worker command is empty"))?;

    // Spawn a2d-eval, feed it the request on stdin, pass its summary through. stderr is
    // inherited so worker diagnostics surface; exit 0 (ok) / 1 (eval failed) / 2 (contract).
    let mut child = Command::new(program)
        .args(rest)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning eval worker: {resolved}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .context("eval worker stdin not captured")?;
        let doc = serde_json::to_vec(&request).context("serializing eval request")?;
        stdin
            .write_all(&doc)
            .context("writing eval request to worker stdin")?;
    }

    let mut out = String::new();
    child
        .stdout
        .take()
        .context("eval worker stdout not captured")?
        .read_to_string(&mut out)
        .context("reading eval worker stdout")?;
    let status = child.wait().context("waiting on eval worker")?;

    print!("{out}");

    Ok(match status.code() {
        Some(0) => ExitCode::SUCCESS,
        Some(code) => ExitCode::from(code as u8),
        None => ExitCode::FAILURE,
    })
}
