//! Worker command resolution shared by `convert`/`resume` (a2d-worker) and
//! `sample` (a2d-sample).

/// Resolve a worker command per PROTOCOL discovery order:
/// `flag` -> `env_var` -> dev default (`uv run --project ... <default_script>`).
pub fn resolve(flag: Option<String>, env_var: &str, default_script: &str) -> String {
    if let Some(cmd) = flag {
        return cmd;
    }
    if let Ok(cmd) = std::env::var(env_var) {
        if !cmd.trim().is_empty() {
            return cmd;
        }
    }
    // Dev default: repo root is two levels up from this crate's manifest dir.
    let repo = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    format!("uv run --project {repo}/packages/a2d-worker-hf {default_script}")
}
