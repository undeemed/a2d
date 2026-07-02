//! The `GenericAdapter` workhorse (Decision 3): classify a model from standard
//! HF `config.json` fields alone. Every field rule below cites a recon gotcha.
//! `generic::detect` always sets `inferred = true`; known adapters call it and
//! then pin trust / capabilities.

use crate::spec::{Capability, ModelSpec};
use crate::Adapter;

const EXPERT_ALIASES: [&str; 3] = ["num_experts", "num_local_experts", "n_routed_experts"];
const EXPERT_ACTIVE_ALIASES: [&str; 3] = [
    "num_experts_per_tok",
    "num_local_experts_per_tok",
    "moe_top_k",
];
const SHARED_EXPERT_ALIASES: [&str; 4] = [
    "num_shared_experts",
    "n_shared_experts",
    "shared_expert_intermediate_size",
    "expert_shared_resource_gate",
];

/// Wraps the parsed `config.json` value with alias-aware accessors. This is the
/// DRY core that keeps every adapter tiny.
pub struct RawConfig {
    root: serde_json::Value,
}

fn is_truthy(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
    }
}

impl RawConfig {
    pub fn new(value: serde_json::Value) -> Self {
        Self { root: value }
    }

    /// The declared `model_type`, or "" if absent.
    pub fn model_type(&self) -> &str {
        self.str_opt("model_type").unwrap_or("")
    }

    /// First alias present as a u64.
    pub fn u64_opt(&self, aliases: &[&str]) -> Option<u64> {
        aliases
            .iter()
            .find_map(|k| self.root.get(*k).and_then(serde_json::Value::as_u64))
    }

    /// First alias present as a u64, else 0.
    pub fn u64(&self, aliases: &[&str]) -> u64 {
        self.u64_opt(aliases).unwrap_or(0)
    }

    /// First alias present as an i64 (mask token ids can be signed sentinels).
    pub fn i64_opt(&self, aliases: &[&str]) -> Option<i64> {
        aliases
            .iter()
            .find_map(|k| self.root.get(*k).and_then(serde_json::Value::as_i64))
    }

    /// First alias present as an f64.
    pub fn f64_opt(&self, aliases: &[&str]) -> Option<f64> {
        aliases
            .iter()
            .find_map(|k| self.root.get(*k).and_then(serde_json::Value::as_f64))
    }

    /// First alias present as a bool.
    pub fn bool_opt(&self, aliases: &[&str]) -> Option<bool> {
        aliases
            .iter()
            .find_map(|k| self.root.get(*k).and_then(serde_json::Value::as_bool))
    }

    /// A top-level key as a string.
    pub fn str_opt(&self, key: &str) -> Option<&str> {
        self.root.get(key).and_then(serde_json::Value::as_str)
    }

    /// A dotted nested path as a string, e.g. "quantization_config.quant_method".
    pub fn str_at(&self, dotted: &str) -> Option<&str> {
        let mut cur = &self.root;
        for part in dotted.split('.') {
            cur = cur.get(part)?;
        }
        cur.as_str()
    }

    /// True if any alias key is present and not null.
    pub fn has_any(&self, aliases: &[&str]) -> bool {
        aliases
            .iter()
            .any(|k| self.root.get(*k).is_some_and(|v| !v.is_null()))
    }

    /// True if any alias key is present and truthy (non-zero / non-empty / true).
    pub fn truthy_any(&self, aliases: &[&str]) -> bool {
        aliases
            .iter()
            .any(|k| self.root.get(*k).is_some_and(is_truthy))
    }
}

/// The workhorse: classify a parsed config into a `ModelSpec` (`inferred = true`).
pub fn detect(cfg: &RawConfig) -> ModelSpec {
    let model_type = cfg.model_type().to_string();

    let n_layers = cfg.u64(&["num_hidden_layers", "n_layer"]);
    let d_model = cfg.u64(&["hidden_size", "n_embd"]);
    let vocab_size = cfg.u64(&["vocab_size"]);
    let n_heads_opt = cfg.u64_opt(&["num_attention_heads", "n_head"]);
    let n_heads = n_heads_opt.unwrap_or(0);
    // The Llama gotcha: missing num_key_value_heads means full attention, not GQA.
    let n_kv_heads = cfg.u64_opt(&["num_key_value_heads"]).unwrap_or(n_heads);
    let sliding_window = cfg.u64_opt(&["sliding_window"]);
    let n_experts = cfg.u64_opt(&EXPERT_ALIASES);
    let n_active_experts = cfg.u64_opt(&EXPERT_ACTIVE_ALIASES);

    let mut caps = Vec::new();

    // Paradigm - NEVER read from the *ForCausalLM suffix (the MambaForCausalLM trap):
    // ssm_cfg present OR a known SSM model_type OR no attention-head field at all.
    // ponytail: "no attention head means ssm" can mislabel a malformed transformer
    // config, but such a config is unconvertible anyway and ssm_cfg is the primary
    // positive signal for the real corpus (Mamba).
    let is_ssm = cfg.has_any(&["ssm_cfg"])
        || matches!(model_type.as_str(), "mamba" | "mamba2")
        || n_heads_opt.is_none();
    caps.push(if is_ssm {
        Capability::ParadigmSsm
    } else {
        Capability::ParadigmArTransformer
    });

    if !is_ssm {
        // Attention family: GQA iff kv < heads, else full.
        caps.push(if n_kv_heads < n_heads {
            Capability::AttnGqa
        } else {
            Capability::AttnFull
        });
        // SWA guard (the Qwen2 gotcha): window present AND > 0 AND use_sliding_window != false.
        // Accepts Qwen2.5 (window=131072, use_sliding_window=false) as NOT swa; rejects
        // Mistral-v0.1 (4096, no toggle), Gemma-2/3, GPT-OSS.
        if sliding_window.is_some_and(|w| w > 0)
            && cfg.bool_opt(&["use_sliding_window"]) != Some(false)
        {
            caps.push(Capability::AttnSwa);
        }
        // Attention sink: any sink alias truthy.
        if cfg.truthy_any(&[
            "attention_sink",
            "attention_sinks",
            "attn_sink",
            "sink_size",
        ]) {
            caps.push(Capability::AttnSink);
        }

        // Position: rope iff any rope field; partial (fidelity) iff pct < 1.0 (Pythia 0.25);
        // learned is inferred from the ABSENCE of all rope fields (the GPT-2 gotcha - learned
        // is never a positive field).
        if cfg.has_any(&[
            "rope_theta",
            "rotary_emb_base",
            "rope_scaling",
            "rotary_pct",
        ]) {
            let partial = cfg
                .f64_opt(&["rotary_pct", "partial_rotary_factor"])
                .is_some_and(|p| p < 1.0);
            caps.push(if partial {
                Capability::PosRopePartial
            } else {
                Capability::PosRope
            });
        } else {
            caps.push(Capability::PosLearned);
        }
    }

    // FFN: MoE iff any expert-count alias; shared-experts iff any shared alias; else dense.
    if n_experts.is_some() {
        caps.push(Capability::FfnMoe);
        if cfg.has_any(&SHARED_EXPERT_ALIASES) {
            caps.push(Capability::FfnMoeSharedExperts);
        }
    } else {
        caps.push(Capability::FfnDense);
    }

    // Norm: RMSNorm iff rms_norm_eps present. Plain LayerNorm contributes no tag -
    // the taxonomy has no LayerNorm tag.
    if cfg.has_any(&["rms_norm_eps"]) {
        caps.push(Capability::NormRms);
    }

    // Weights: quantization is blocking and takes precedence; bf16 is fidelity
    // (bf16 is NOT quantized), added only when not quantized.
    match cfg.str_at("quantization_config.quant_method") {
        Some(m) if m.eq_ignore_ascii_case("mxfp4") => caps.push(Capability::WeightsMxfp4),
        Some(m) if m.eq_ignore_ascii_case("gptq") => caps.push(Capability::WeightsGptq),
        _ if cfg.str_opt("torch_dtype") == Some("bfloat16") => caps.push(Capability::WeightsBf16),
        _ => {}
    }

    // Head: logit softcap is fidelity (Gemma-2 records it, never blocks).
    if cfg.has_any(&["attn_logit_softcapping", "final_logit_softcapping"]) {
        caps.push(Capability::HeadLogitSoftcap);
    }

    ModelSpec {
        model_type,
        n_layers,
        d_model,
        vocab_size,
        n_heads,
        n_kv_heads,
        sliding_window,
        n_experts,
        n_active_experts,
        capabilities: caps,
        mask_token_id: cfg.i64_opt(&["mask_token_id"]),
        inferred: true,
    }
}

/// Fallback adapter used when no known `model_type` matches. Not registered in the
/// inventory (it is the default branch, not a lookup target); it only wraps
/// `generic::detect`, which already sets `inferred = true`.
pub struct GenericAdapter;

impl Adapter for GenericAdapter {
    fn model_type(&self) -> &'static str {
        "generic"
    }

    fn detect(&self, cfg: &RawConfig) -> ModelSpec {
        detect(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(json: &str) -> RawConfig {
        RawConfig::new(serde_json::from_str(json).unwrap())
    }

    #[test]
    fn qwen2_5_window_set_but_disabled_is_not_swa() {
        let spec = detect(&cfg(r#"{
            "model_type": "qwen2",
            "num_hidden_layers": 28, "hidden_size": 3584, "vocab_size": 152064,
            "num_attention_heads": 28, "num_key_value_heads": 4,
            "sliding_window": 131072, "use_sliding_window": false,
            "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "torch_dtype": "bfloat16"
        }"#));
        assert!(spec.capabilities.contains(&Capability::AttnGqa));
        assert!(!spec.capabilities.contains(&Capability::AttnSwa));
        assert!(spec.inferred);
    }

    #[test]
    fn mistral_v0_1_active_window_is_swa() {
        let spec = detect(&cfg(r#"{
            "model_type": "mistral",
            "num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 32000,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "sliding_window": 4096,
            "rope_theta": 10000.0, "rms_norm_eps": 1e-5
        }"#));
        assert!(spec.capabilities.contains(&Capability::AttnSwa));
    }

    #[test]
    fn pythia_partial_rope_is_fidelity_not_blocking() {
        let spec = detect(&cfg(r#"{
            "model_type": "gpt_neox",
            "num_hidden_layers": 6, "hidden_size": 512, "vocab_size": 50304,
            "num_attention_heads": 8,
            "rotary_pct": 0.25, "rotary_emb_base": 10000
        }"#));
        assert!(spec.capabilities.contains(&Capability::PosRopePartial));
        assert!(!Capability::PosRopePartial.blocking());
        assert!(!spec.capabilities.contains(&Capability::PosRope));
    }

    #[test]
    fn mamba_is_ssm_despite_causal_lm_suffix() {
        let spec = detect(&cfg(r#"{
            "model_type": "mamba",
            "num_hidden_layers": 64, "hidden_size": 2560, "vocab_size": 50280,
            "ssm_cfg": {}, "architectures": ["MambaForCausalLM"]
        }"#));
        assert!(spec.capabilities.contains(&Capability::ParadigmSsm));
        assert!(!spec
            .capabilities
            .contains(&Capability::ParadigmArTransformer));
    }
}
