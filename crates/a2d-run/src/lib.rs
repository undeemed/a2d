//! Run-directory lifecycle and worker process management for a2d.
//!
//! `a2d-run` owns the CLI<->worker boundary: it creates the run dir, writes the
//! manifest, spawns the Python worker, feeds it the [`ConversionJob`] on stdin,
//! and tees the worker's JSONL event stream to `events.jsonl` while invoking a
//! caller-supplied callback for live rendering.
//!
//! [`ConversionJob`]: a2d_contracts::ConversionJob

pub mod rundir;
pub mod worker;

pub use worker::run_job;
