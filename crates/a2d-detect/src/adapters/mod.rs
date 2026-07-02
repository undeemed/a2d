//! EXTENSION POINT contract.
//!
//! One file per quirky `model_type`. Each self-registers via
//! `inventory::submit!`. Adding a model = dropping a new file here; never edit
//! existing files (open-closed, SPEC-HANDOFF section 3.3).
//!
//! Phase 0 is empty on purpose - it only proves the registry seam links.
//! Phase 1 fills this directory with real adapters.
