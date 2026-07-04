//! Run-directory layout and `manifest.json` lifecycle.

use std::fs;
use std::path::Path;

use a2d_contracts::{Manifest, RunStatus};
use anyhow::{bail, Context, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};

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

/// Read `<run_dir>/manifest.json` back into a [`Manifest`].
pub fn read_manifest(run_dir: &Path) -> Result<Manifest> {
    let path = run_dir.join("manifest.json");
    let data = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("parsing {}", path.display()))
}

/// Update `<run_dir>/manifest.json` to a terminal status and stamp `finished_at`.
///
/// Reads the existing manifest back so the initial fields are preserved verbatim.
pub fn finalize_manifest(run_dir: &Path, status: RunStatus) -> Result<()> {
    let mut manifest = read_manifest(run_dir)?;
    manifest.status = status;
    manifest.finished_at = Some(Utc::now().to_rfc3339());
    write_manifest(run_dir, &manifest)
}

/// Reopen an existing (non-empty) run dir for resume.
///
/// Bypasses [`create_run_dir`]'s non-empty guard: flips the manifest status back
/// to `Running` and clears `finished_at`. `events.jsonl` is untouched here - the
/// worker stream is opened in append mode, so resume events extend the existing
/// log rather than clobbering it.
pub fn reopen_run_dir(run_dir: &Path) -> Result<Manifest> {
    let mut manifest = read_manifest(run_dir)?;
    manifest.status = RunStatus::Running;
    manifest.finished_at = None;
    write_manifest(run_dir, &manifest)?;
    Ok(manifest)
}

/// sha256 (hex) of the primary source safetensors file, for manifest provenance.
///
/// ponytail: hash the primary `model.safetensors`; upgrade to a header +
/// shard-manifest digest if big-model / sharded provenance gets expensive.
pub fn source_hash(model_dir: &Path) -> Result<String> {
    let path = model_dir.join("model.safetensors");
    let mut file = fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| format!("hashing {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}
