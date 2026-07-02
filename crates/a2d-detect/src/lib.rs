//! Config-only model detection and gating for a2d (Phase 1).
//!
//! `detect(dir)` parses a `config.json`, routes it to a known adapter or the
//! `GenericAdapter` workhorse, gates the resulting `ModelSpec`, locates the
//! model triple (config/tokenizer/weights) on disk, and builds a conversion
//! `Plan`. Adapters self-register from `src/adapters/*.rs` via `inventory`; the
//! registry is discovered at link time, no central list to edit (open-closed,
//! SPEC-HANDOFF section 3.3).

use std::path::Path;

use anyhow::{Context, Result};

mod adapters;
mod gate;
pub mod generic;
pub mod spec;

pub use generic::RawConfig;
pub use spec::{Capability, DetectReport, MaskStrategy, ModelSpec, Plan, Verdict, WeightsStatus};

/// A model adapter. Phase 1 adds `detect`, which inspects a parsed config and
/// returns a `ModelSpec`. Known adapters delegate to `generic::detect` and then
/// pin trust / capabilities.
pub trait Adapter: Sync {
    /// Stable identifier for the model family this adapter handles.
    fn model_type(&self) -> &'static str;
    /// Classify a parsed config into a `ModelSpec`.
    fn detect(&self, cfg: &RawConfig) -> ModelSpec;
}

/// Registry entry. Adapters register a `&'static dyn Adapter` through this
/// wrapper because `inventory::collect!` needs a concrete, nameable type.
pub struct Registration {
    pub adapter: &'static dyn Adapter,
}

inventory::collect!(Registration);

/// Iterate every registered adapter.
pub fn adapters() -> impl Iterator<Item = &'static dyn Adapter> {
    inventory::iter::<Registration>
        .into_iter()
        .map(|r| r.adapter)
}

/// Find the adapter for a given `model_type`, if one is registered.
pub fn find(model_type: &str) -> Option<&'static dyn Adapter> {
    adapters().find(|a| a.model_type() == model_type)
}

/// Detect and gate a model from its `config.json` alone (Decision 5/6).
///
/// Errors (bad input, exit-2-worthy) when `config.json` is absent, unreadable, or
/// malformed. Otherwise always succeeds - an unsupported architecture is a clean
/// verdict, not an error.
pub fn detect(dir: &Path) -> Result<DetectReport> {
    let config_path = dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    let cfg = RawConfig::new(value);

    let model_type = cfg.model_type().to_string();
    let spec = match find(&model_type) {
        Some(adapter) => adapter.detect(&cfg),
        None => generic::GenericAdapter.detect(&cfg),
    };

    let verdict = gate::gate(&spec);
    let weights = locate_weights(dir, &cfg);
    let plan = match verdict {
        Verdict::Unsupported { .. } => None,
        _ => Some(build_plan(
            &spec,
            resolve_mask_strategy(dir, spec.mask_token_id),
        )),
    };

    Ok(DetectReport {
        spec,
        verdict,
        weights,
        plan,
    })
}

/// Locate weights on disk (Decision 6): safetensors, then pytorch, then gguf; else
/// Missing with the `hf download` hint keyed off the config's `_name_or_path`.
fn locate_weights(dir: &Path, cfg: &RawConfig) -> WeightsStatus {
    if has_ext(dir, "safetensors") {
        WeightsStatus::Present {
            format: "safetensors".to_string(),
        }
    } else if ["bin", "pt", "pth"].iter().any(|&e| has_ext(dir, e)) {
        WeightsStatus::Present {
            format: "pytorch".to_string(),
        }
    } else if has_ext(dir, "gguf") {
        WeightsStatus::Present {
            format: "gguf".to_string(),
        }
    } else {
        let repo = cfg.str_opt("_name_or_path").unwrap_or("<repo-id>");
        WeightsStatus::Missing {
            hint: format!("hf download {} --local-dir {}", repo, dir.display()),
        }
    }
}

fn has_ext(dir: &Path, ext: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some(ext))
}

const TOKENIZER_FILES: [&str; 5] = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
];

/// Resolve the report-only mask-token strategy (Decision 8). The config-only
/// fixtures have no tokenizer files, so they resolve to `Undetermined`.
fn resolve_mask_strategy(dir: &Path, mask_token_id: Option<i64>) -> MaskStrategy {
    // (1) An explicit id in config.json wins outright.
    if let Some(id) = mask_token_id {
        let token = lookup_token_name(dir, id).unwrap_or_else(|| "<mask>".to_string());
        return MaskStrategy::ReuseId { id, token };
    }
    // (5) No tokenizer files at all -> resolved later, at convert time.
    if !TOKENIZER_FILES.iter().any(|f| dir.join(f).exists()) {
        return MaskStrategy::Undetermined;
    }
    let added = load_added_tokens(dir);
    // (2) An explicit mask token by content pattern (<mask>/[MASK], case-insensitive).
    if let Some((id, token)) = added
        .iter()
        .find(|(_, name)| {
            let low = name.to_ascii_lowercase();
            low == "<mask>" || low == "[mask]"
        })
        .cloned()
    {
        return MaskStrategy::ReuseId { id, token };
    }
    // (3) Reuse a reserved/unused slot (the Llama-3 case): no vocab growth.
    if let Some((id, token)) = added
        .iter()
        .find(|(_, name)| is_reserved_slot(name))
        .cloned()
    {
        return MaskStrategy::ReuseId { id, token };
    }
    // (4) Tokenizer present but nothing reusable -> grow the vocab at convert.
    MaskStrategy::GrowVocab
}

fn is_reserved_slot(name: &str) -> bool {
    name.starts_with("<|reserved_special_token_")
        || name.starts_with("<unused")
        || name.starts_with("<|extra_")
}

fn lookup_token_name(dir: &Path, id: i64) -> Option<String> {
    load_added_tokens(dir)
        .into_iter()
        .find(|(i, _)| *i == id)
        .map(|(_, name)| name)
}

// ponytail: we read only tokenizer_config.json's added_tokens_decoder map (id + content),
// which covers modern tokenizers; we do NOT parse the heavy tokenizer.json. Upgrade path:
// parse tokenizer.json only if a model ships mask/reserved tokens exclusively there, which is
// Phase 2's materialize step where the tokenizer is loaded anyway.
fn load_added_tokens(dir: &Path) -> Vec<(i64, String)> {
    let Ok(text) = std::fs::read_to_string(dir.join("tokenizer_config.json")) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let Some(map) = value
        .get("added_tokens_decoder")
        .and_then(|m| m.as_object())
    else {
        return Vec::new();
    };
    let mut out: Vec<(i64, String)> = map
        .iter()
        .filter_map(|(id_str, entry)| {
            let id = id_str.parse::<i64>().ok()?;
            let content = entry.get("content").and_then(serde_json::Value::as_str)?;
            Some((id, content.to_string()))
        })
        .collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

fn build_plan(spec: &ModelSpec, mask: MaskStrategy) -> Plan {
    Plan {
        objective: "mdlm".to_string(),
        anneal_schedule: "linear (placeholder; numbers set at convert)".to_string(),
        mask_token: mask,
        estimated_memory_gb: estimate_memory_gb(spec),
    }
}

// ponytail: coarse dims-only param count at 2 bytes/param (bf16 weights-only), rendered
// "(approx, weights-only)"; MoE total params are undercounted (only dense blocks summed).
// Upgrade path: sum safetensors shard byte lengths when weights are present.
fn estimate_memory_gb(spec: &ModelSpec) -> Option<f64> {
    if spec.d_model == 0 || spec.n_layers == 0 {
        return None;
    }
    let d = spec.d_model as f64;
    let layers = spec.n_layers as f64;
    let vocab = spec.vocab_size as f64;
    // ~12 * L * d^2 for the transformer blocks + 2 * vocab * d for embeddings + LM head.
    let params = 12.0 * layers * d * d + 2.0 * vocab * d;
    Some(params * 2.0 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyAdapter;
    impl Adapter for DummyAdapter {
        fn model_type(&self) -> &'static str {
            "dummy"
        }
        fn detect(&self, cfg: &RawConfig) -> ModelSpec {
            generic::detect(cfg)
        }
    }
    inventory::submit! {
        Registration { adapter: &DummyAdapter }
    }

    #[test]
    fn dummy_is_registered() {
        assert!(adapters().any(|a| a.model_type() == "dummy"));
        assert_eq!(find("dummy").map(|a| a.model_type()), Some("dummy"));
        assert!(find("nope").is_none());
    }

    #[test]
    fn config_only_dir_reports_missing_weights_with_hint() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"gpt2","n_layer":12,"n_embd":768,"vocab_size":50257,"n_head":12}"#,
        )
        .unwrap();

        let report = detect(dir.path()).unwrap();
        match report.weights {
            WeightsStatus::Missing { hint } => assert!(hint.contains("hf download")),
            other => panic!("expected missing weights, got {other:?}"),
        }
        // Config-only means no tokenizer files -> Undetermined mask strategy.
        let plan = report.plan.expect("gpt2 is supported so it carries a plan");
        assert_eq!(plan.mask_token, MaskStrategy::Undetermined);
    }

    #[test]
    fn missing_config_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(detect(dir.path()).is_err());
    }
}
