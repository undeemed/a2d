//! Regenerates the checked-in JSON Schema (D9) from the canonical Rust types.
//! Output is deterministic: pretty JSON with serde_json's default sorted keys,
//! so CI can diff-check for Rust<->schema drift.

use std::fs;
use std::path::Path;

use a2d_contracts::{ConversionJob, EventEnvelope, Manifest};

fn main() -> std::io::Result<()> {
    // Workspace `schema/` dir, resolved relative to this crate at build time.
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../schema");
    fs::create_dir_all(dir)?;

    write(
        dir,
        "conversion_job.schema.json",
        schemars::schema_for!(ConversionJob),
    )?;
    write(
        dir,
        "event_envelope.schema.json",
        schemars::schema_for!(EventEnvelope),
    )?;
    write(dir, "manifest.schema.json", schemars::schema_for!(Manifest))?;

    Ok(())
}

fn write(dir: &str, name: &str, schema: schemars::Schema) -> std::io::Result<()> {
    let path = Path::new(dir).join(name);
    let json = serde_json::to_string_pretty(&schema).expect("schema serializes");
    fs::write(&path, format!("{json}\n"))?;
    println!("wrote {}", path.display());
    Ok(())
}
