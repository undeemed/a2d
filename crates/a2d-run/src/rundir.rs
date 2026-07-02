//! Run-directory layout and `manifest.json` lifecycle.

use std::fs;
use std::path::Path;

use a2d_contracts::{Manifest, RunStatus};
use anyhow::{bail, Context, Result};
use chrono::Utc;

/// Create the run directory, refusing to clobber existing work.
///
/// Succeeds if `run_dir` does not exist (creating it, parents included) or
/// exists but is empty. Errors if it exists and contains any entries.
pub fn create_run_dir(run_dir: &Path) -> Result<()> {
    if run_dir.exists() {
        let mut entries = fs::read_dir(run_dir)
            .with_context(|| format!("reading run dir {}", run_dir.display()))?;
        if entries.next().is_some() {
            bail!(
                "run dir {} already exists and is not empty",
                run_dir.display()
            );
        }
    } else {
        fs::create_dir_all(run_dir)
            .with_context(|| format!("creating run dir {}", run_dir.display()))?;
    }
    Ok(())
}

/// Write `<run_dir>/manifest.json` from the given manifest (pretty JSON).
pub fn write_manifest(run_dir: &Path, manifest: &Manifest) -> Result<()> {
    let path = run_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(manifest).context("serializing manifest")?;
    fs::write(&path, format!("{json}\n")).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Update `<run_dir>/manifest.json` to a terminal status and stamp `finished_at`.
///
/// Reads the existing manifest back so the initial fields are preserved verbatim.
pub fn finalize_manifest(run_dir: &Path, status: RunStatus) -> Result<()> {
    let path = run_dir.join("manifest.json");
    let data = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut manifest: Manifest =
        serde_json::from_str(&data).with_context(|| format!("parsing {}", path.display()))?;
    manifest.status = status;
    manifest.finished_at = Some(Utc::now().to_rfc3339());
    write_manifest(run_dir, &manifest)
}
