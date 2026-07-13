//! EXTENSION POINT contract.
//!
//! One file per quirky `model_type`. Each self-registers via `inventory::submit!`.
//! Adding a model = dropping a new file here and adding one `mod` line below (the
//! sole sanctioned edit to this table); never edit an existing adapter file
//! (open-closed, SPEC-HANDOFF section 3.3).

mod gemma;
mod gpt2;
mod gpt_oss;
mod llama;
mod olmoe;
mod qwen2;
