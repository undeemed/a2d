//! The primary Phase-1 exit criterion: the curated config-only corpus classifies
//! correctly. Every `fixtures/configs/<name>/` dir is a weights-less config-only
//! dir with an `expected.json` sidecar (Decision 7); this test runs the real
//! `a2d_detect::detect(dir)` on each and asserts the verdict tag, the weights
//! status, and - for rejects - the reason SET (order-independent, by dotted tag).

use std::path::PathBuf;

use a2d_detect::{detect, Verdict, WeightsStatus};

fn corpus_dir() -> PathBuf {
    // Fixtures live at <workspace>/fixtures/configs; this crate is <workspace>/crates/a2d-detect.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/configs")
        .canonicalize()
        .expect("fixtures/configs must exist")
}

fn verdict_tag(v: &Verdict) -> &'static str {
    match v {
        Verdict::Supported => "supported",
        Verdict::SupportedInferred => "supported_inferred",
        Verdict::Unsupported { .. } => "unsupported",
    }
}

#[test]
fn corpus_classifies_correctly() {
    let dir = corpus_dir();
    let mut supported = 0;
    let mut unsupported = 0;
    let mut checked = 0;

    for entry in std::fs::read_dir(&dir).expect("read fixtures/configs") {
        let path = entry.expect("dir entry").path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        // Every fixture dir must carry a sidecar.
        let sidecar_path = path.join("expected.json");
        let sidecar_raw = std::fs::read_to_string(&sidecar_path)
            .unwrap_or_else(|_| panic!("{name}: missing expected.json sidecar"));
        let sidecar: serde_json::Value = serde_json::from_str(&sidecar_raw)
            .unwrap_or_else(|_| panic!("{name}: malformed expected.json"));

        let report = detect(&path).unwrap_or_else(|e| panic!("{name}: detect failed: {e:#}"));

        // Verdict tag matches.
        let want_verdict = sidecar["verdict"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: sidecar has no verdict"));
        assert_eq!(
            verdict_tag(&report.verdict),
            want_verdict,
            "{name}: verdict mismatch"
        );

        // Every fixture is a weights-less config-only dir.
        let want_weights = sidecar["weights"].as_str().unwrap_or("missing");
        assert_eq!(
            want_weights, "missing",
            "{name}: sidecar weights must be missing"
        );
        assert!(
            matches!(report.weights, WeightsStatus::Missing { .. }),
            "{name}: expected missing weights, got {:?}",
            report.weights
        );

        // For rejects, the reason SET matches order-independently by dotted tag.
        if let Verdict::Unsupported { reasons } = &report.verdict {
            let want_tags: Vec<&str> = sidecar["reasons"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: unsupported sidecar needs reasons"))
                .iter()
                .map(|t| t.as_str().expect("reason tag is a string"))
                .collect();
            assert_eq!(
                want_tags.len(),
                reasons.len(),
                "{name}: reason count mismatch (want {want_tags:?}, got {reasons:?})"
            );
            for tag in &want_tags {
                let needle = format!("({tag})");
                let hits = reasons.iter().filter(|r| r.contains(&needle)).count();
                assert_eq!(
                    hits, 1,
                    "{name}: tag {tag} must appear in exactly one reason"
                );
            }
            unsupported += 1;
        } else {
            assert!(
                sidecar.get("reasons").is_none(),
                "{name}: supported sidecar must not carry reasons"
            );
            supported += 1;
        }
        checked += 1;
    }

    assert!(checked > 0, "corpus must be non-empty");
    assert!(supported >= 1, "corpus needs at least one supported model");
    assert!(
        unsupported >= 1,
        "corpus needs at least one unsupported model"
    );
}
