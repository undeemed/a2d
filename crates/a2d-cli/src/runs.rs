use std::path::PathBuf;

use anyhow::{Context, Result};

use a2d_contracts::{Manifest, RunStatus};

fn status_str(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Running => "running",
        RunStatus::Completed => "completed",
        RunStatus::Failed => "failed",
    }
}

/// Scan `root` one level deep for `<dir>/manifest.json` and print a table.
pub fn run(root: Option<PathBuf>) -> Result<()> {
    let root = root.unwrap_or_else(|| PathBuf::from("runs"));

    let mut manifests: Vec<Manifest> = Vec::new();
    if root.is_dir() {
        for entry in std::fs::read_dir(&root)
            .with_context(|| format!("reading runs root {}", root.display()))?
        {
            let manifest_path = entry?.path().join("manifest.json");
            if !manifest_path.is_file() {
                continue;
            }
            let data = std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("reading {}", manifest_path.display()))?;
            match serde_json::from_str::<Manifest>(&data) {
                Ok(m) => manifests.push(m),
                Err(e) => eprintln!("warning: skipping {}: {e}", manifest_path.display()),
            }
        }
    }

    if manifests.is_empty() {
        println!("no runs found under {}", root.display());
        return Ok(());
    }

    println!(
        "{:<38} {:<10} {:<26} MODEL_PATH",
        "JOB_ID", "STATUS", "CREATED_AT"
    );
    for m in manifests {
        println!(
            "{:<38} {:<10} {:<26} {}",
            m.job_id,
            status_str(m.status),
            m.created_at,
            m.model_path
        );
    }
    Ok(())
}
