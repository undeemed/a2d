use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::Args;

use a2d_contracts::{ConversionConfig, Event, EventEnvelope, LogLevel};
use a2d_detect::Verdict;

/// Flags for `a2d convert`, flattened into the subcommand.
#[derive(Args)]
pub struct ConvertArgs {
    /// Local model directory (must already contain downloaded weights).
    pub model_dir: PathBuf,
    /// Run directory to create for this conversion.
    #[arg(long = "out")]
    pub run_dir: PathBuf,
    /// Local training data file (jsonl or txt).
    #[arg(long)]
    pub data: String,
    /// Diffusion objective (resolved via the objectives registry).
    #[arg(long, default_value = "mdlm")]
    pub objective: String,
    /// Steps over which attention anneals causal -> bidirectional.
    #[arg(long = "anneal-steps", default_value_t = 100)]
    pub anneal_steps: u64,
    /// Anneal schedule shape.
    #[arg(long = "anneal-schedule", default_value = "linear")]
    pub anneal_schedule: String,
    /// Training sequence length (<= GPT-2 n_positions 1024).
    #[arg(long = "seq-len", default_value_t = 512)]
    pub seq_len: u64,
    /// Per-device train batch size.
    #[arg(long = "per-device-batch-size", default_value_t = 8)]
    pub per_device_batch_size: u64,
    /// Gradient accumulation steps.
    #[arg(long = "grad-accum", default_value_t = 1)]
    pub grad_accum: u64,
    /// Learning rate.
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    /// Train for exactly this many optimizer steps (XOR --max-tokens).
    #[arg(long = "max-steps")]
    pub max_steps: Option<u64>,
    /// Train until this many tokens have been seen (XOR --max-steps).
    #[arg(long = "max-tokens")]
    pub max_tokens: Option<u64>,
    /// Mask-token strategy: "grow" (default, +1 vocab row) or "reuse" (eos id).
    #[arg(long = "mask-token", default_value = "grow")]
    pub mask_token: String,
    /// Checkpoints to keep (maps to save_total_limit).
    #[arg(long = "keep-last", default_value_t = 3)]
    pub keep_last: u64,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Accelerator: "auto" | "cpu" | "mps" | "cuda".
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Compute dtype: "float32" | "bfloat16".
    #[arg(long, default_value = "float32")]
    pub dtype: String,
    /// Proceed even though the architecture was inferred by the generic adapter.
    #[arg(long = "accept-inferred")]
    pub accept_inferred: bool,
    /// Override the worker command (default: A2D_WORKER_CMD env, else dev uv default).
    #[arg(long = "worker-cmd")]
    pub worker_cmd: Option<String>,
}

fn level_str(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    }
}

/// Render one human-readable line per event for the terminal. Shared by
/// `convert` and `resume` (both stream the same worker event set).
pub(crate) fn render_event(env: &EventEnvelope) {
    match &env.event {
        Event::JobStarted { worker } => println!("job started (worker: {worker})"),
        Event::Log { level, message } => println!("{}: {message}", level_str(*level)),
        Event::Progress { stage, step, total } => match total {
            Some(total) => println!("progress: {stage} {step}/{total}"),
            None => println!("progress: {stage} {step}"),
        },
        Event::TrainStep {
            step,
            loss,
            anneal,
            lr,
            tokens,
        } => println!("step {step}: loss={loss:.4} anneal={anneal:.3} lr={lr:.2e} tokens={tokens}"),
        Event::IdentityGate {
            passed,
            max_abs_diff,
            tolerance,
        } => println!(
            "identity gate: {} (max_abs_diff={max_abs_diff:.2e}, tolerance={tolerance:.0e})",
            if *passed { "PASS" } else { "FAIL" }
        ),
        Event::Checkpoint { step, path } => println!("checkpoint @ step {step}: {path}"),
        Event::Metric { name, value, step } => println!("metric {name}={value:.4} @ step {step}"),
        Event::JobCompleted {} => println!("job completed"),
        Event::JobFailed { message } => eprintln!("job failed: {message}"),
    }
}

pub fn run(args: ConvertArgs) -> Result<ExitCode> {
    let model_dir = &args.model_dir;
    if !model_dir.exists() {
        bail!("model dir does not exist: {}", model_dir.display());
    }

    // Detect + gate from config.json alone. Bad input (no/malformed config) is a
    // usage error (exit 2), matching `a2d detect`; the worker is never spawned.
    let report = match a2d_detect::detect(model_dir) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("error: {e:#}");
            return Ok(ExitCode::from(2));
        }
    };

    match &report.verdict {
        // Unsupported architecture: abort exit 1 WITHOUT spawning the worker.
        Verdict::Unsupported { reasons } => {
            eprintln!("error: {} is unsupported", report.spec.model_type);
            for reason in reasons {
                eprintln!("  - {reason}");
            }
            return Ok(ExitCode::from(1));
        }
        // Inferred architecture demands an explicit opt-in.
        Verdict::SupportedInferred if !args.accept_inferred => {
            eprintln!(
                "error: architecture of {} was inferred by the generic adapter; \
                 re-run with --accept-inferred to proceed",
                report.spec.model_type
            );
            return Ok(ExitCode::from(2));
        }
        _ => {}
    }

    // Resolve the token budget: exactly one of --max-steps / --max-tokens.
    let tokens_per_step = args.seq_len * args.per_device_batch_size * args.grad_accum;
    let max_steps = match (args.max_steps, args.max_tokens) {
        (Some(s), None) => s,
        (None, Some(t)) => t.div_ceil(tokens_per_step),
        _ => {
            eprintln!("error: pass exactly one of --max-steps or --max-tokens");
            return Ok(ExitCode::from(2));
        }
    };

    // Decision 4: a run that ends before the anneal completes would ship an
    // only-partially-bidirectional checkpoint that `a2d sample` then forces to
    // alpha=1 it was never trained at.
    if args.anneal_steps > max_steps {
        eprintln!(
            "error: --anneal-steps ({}) must be <= effective max steps ({max_steps})",
            args.anneal_steps
        );
        return Ok(ExitCode::from(2));
    }

    let config = ConversionConfig {
        objective: args.objective,
        data: args.data,
        anneal_steps: args.anneal_steps,
        anneal_schedule: args.anneal_schedule,
        seq_len: args.seq_len,
        per_device_batch_size: args.per_device_batch_size,
        grad_accum: args.grad_accum,
        lr: args.lr,
        max_steps: Some(max_steps),
        max_tokens: args.max_tokens,
        mask_token: args.mask_token,
        keep_last: args.keep_last,
        seed: args.seed,
        device: args.device,
        dtype: args.dtype,
    };

    let source_hash =
        a2d_run::rundir::source_hash(model_dir).context("hashing model weights for provenance")?;

    let resolved = crate::worker_cmd::resolve(args.worker_cmd, "A2D_WORKER_CMD", "a2d-worker");
    // ponytail: plain whitespace split is enough (spec says shlex-free is fine);
    // upgrade to a real shell-word splitter if worker commands ever need quoting.
    let argv: Vec<String> = resolved.split_whitespace().map(str::to_string).collect();
    if argv.is_empty() {
        bail!("resolved worker command is empty");
    }

    // a2d-run owns run-dir creation, job_id, manifest lifecycle, spawning, the
    // stdout->events.jsonl tee, and manifest finalization. We render + set exit code.
    let outcome = a2d_run::run_job(
        model_dir,
        &args.run_dir,
        report.spec,
        config,
        source_hash,
        &argv,
        render_event,
    )
    .context("running conversion job")?;

    Ok(match outcome.status {
        a2d_contracts::RunStatus::Completed => {
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
