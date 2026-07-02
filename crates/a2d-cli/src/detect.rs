use std::path::Path;
use std::process::ExitCode;

use anyhow::Result;

use a2d_detect::{DetectReport, Verdict, WeightsStatus};

/// Run `a2d detect`: classify and gate a model from its `config.json` alone.
///
/// Bad input (no `config.json`, malformed JSON, unreadable path) is intercepted
/// here - it prints `error: {e:#}` to stderr and returns `Ok(ExitCode::from(2))`
/// rather than propagating, so main's generic `Err => FAILURE` (exit 1) stays
/// reserved for truly unexpected failures and never collides with the
/// unsupported=1 verdict path (Decision 5).
pub fn run(path: &Path, json: bool) -> Result<ExitCode> {
    let report = match a2d_detect::detect(path) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("error: {e:#}");
            return Ok(ExitCode::from(2));
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        render_human(&report);
    }

    // Weights-missing does not change the exit code; it just prints the download hint.
    if let WeightsStatus::Missing { hint } = &report.weights {
        eprintln!("weights not present: {hint}");
    }
    if matches!(report.verdict, Verdict::SupportedInferred) {
        eprintln!("note: inferred architecture; convert requires --accept-inferred");
    }

    Ok(match report.verdict {
        Verdict::Supported | Verdict::SupportedInferred => ExitCode::SUCCESS,
        Verdict::Unsupported { .. } => ExitCode::from(1),
    })
}

fn render_human(report: &DetectReport) {
    let spec = &report.spec;
    match &report.verdict {
        Verdict::Supported => println!("SUPPORTED: {}", spec.model_type),
        Verdict::SupportedInferred => println!("SUPPORTED (inferred): {}", spec.model_type),
        Verdict::Unsupported { reasons } => {
            println!("UNSUPPORTED: {}", spec.model_type);
            for reason in reasons {
                println!("  - {reason}");
            }
        }
    }

    println!(
        "  layers: {}  d_model: {}  heads: {}/{} (kv)  vocab: {}",
        spec.n_layers, spec.d_model, spec.n_heads, spec.n_kv_heads, spec.vocab_size
    );
    if !spec.capabilities.is_empty() {
        let caps: Vec<String> = spec.capabilities.iter().map(|c| c.to_string()).collect();
        println!("  capabilities: {}", caps.join(", "));
    }

    if let Some(plan) = &report.plan {
        println!("  plan: objective={}", plan.objective);
        if let Some(gb) = plan.estimated_memory_gb {
            println!("  estimated memory: {gb:.2} GB (approx, weights-only)");
        }
    }
}
