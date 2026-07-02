//! Adapter registry seam for a2d.
//!
//! Phase 0 skeleton: proves the inventory-based extension seam links before
//! Phase 1 depends on it. Adapters self-register from `src/adapters/*.rs`; the
//! registry is discovered at link time, no central list to edit.

mod adapters;

/// A model adapter.
///
/// Phase 0 only exposes the identity of an adapter. Phase 1 adds detection
/// methods that inspect a model on disk and return a `ModelSpec`.
pub trait Adapter: Sync {
    /// Stable identifier for the model family this adapter handles.
    fn model_type(&self) -> &'static str;
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

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyAdapter;
    impl Adapter for DummyAdapter {
        fn model_type(&self) -> &'static str {
            "dummy"
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
}
