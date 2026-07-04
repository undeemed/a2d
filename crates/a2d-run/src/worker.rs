//! Worker process lifecycle: spawn, feed the job, tee the event stream.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

use a2d_contracts::{
    ConversionConfig, ConversionJob, Event, EventEnvelope, IdentityResult, Manifest, ModelSpec,
    RunStatus, SCHEMA_VERSION,
};
use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use uuid::Uuid;

use crate::rundir;

/// Outcome of a worker run that ran to completion.
///
/// Carries the generated `job_id` and the terminal status the manifest was
/// finalized to. Returned even when the worker exits non-zero, so the CLI owns
/// the process exit code; only spawn/IO faults surface as `Err`.
pub struct Outcome {
    pub job_id: String,
    pub status: RunStatus,
}

/// Run a no-op conversion job end to end.
///
/// Builds a [`ConversionJob`], creates the run dir, writes a `running` manifest,
/// spawns `worker_argv`, writes the job JSON to the worker's stdin (then closes
/// it), reads stdout line by line, appends each raw line verbatim to
/// `<run_dir>/events.jsonl` (flushed per line) and invokes `on_event` with the
/// parsed envelope. Unparseable stdout lines are still written to the file and
/// warned about on stderr, never fatal. On worker exit the manifest is finalized
/// to `completed` (exit 0) or `failed` (any non-zero exit) and the terminal
/// status is returned in the [`Outcome`]; a non-zero exit also surfaces the
/// worker's stderr on our stderr for debugging.
///
/// `worker_argv` is a pre-split argv (program + args, no shell, no shlex) per the
/// Phase 0 process protocol; the CLI resolves and splits the worker command.
///
/// `model_spec`, `conversion_config`, and `source_hash` are provenance the CLI
/// resolves (detect + config-build + [`rundir::source_hash`]) and are written
/// straight into the manifest; `identity` and `token_count` are accumulated from
/// the worker's own event stream (the `IdentityGate` event and the latest
/// `TrainStep.tokens`).
pub fn run_job(
    model_path: &Path,
    run_dir: &Path,
    model_spec: ModelSpec,
    conversion_config: ConversionConfig,
    source_hash: String,
    worker_argv: &[String],
    mut on_event: impl FnMut(&EventEnvelope),
) -> Result<Outcome> {
    let run_path = run_dir;

    let job = ConversionJob {
        schema_version: SCHEMA_VERSION.to_string(),
        job_id: Uuid::new_v4().to_string(),
        model_path: model_path.to_string_lossy().into_owned(),
        run_dir: run_dir.to_string_lossy().into_owned(),
        conversion_config: conversion_config.clone(),
    };

    rundir::create_run_dir(run_path)?;

    let manifest = Manifest {
        schema_version: SCHEMA_VERSION.to_string(),
        a2d_version: env!("CARGO_PKG_VERSION").to_string(),
        job_id: job.job_id.clone(),
        created_at: Utc::now().to_rfc3339(),
        model_path: job.model_path.clone(),
        status: RunStatus::Running,
        finished_at: None,
        model_spec: Some(model_spec),
        data_source: Some(conversion_config.data.clone()),
        conversion_config: Some(conversion_config),
        source_hash: Some(source_hash),
        // identity + token_count are accumulated from the event stream in finalize.
        identity: None,
        token_count: None,
    };
    rundir::write_manifest(run_path, &manifest)?;

    // Once the `running` manifest is on disk, every failure path must still stamp a
    // terminal status - otherwise a spawn/IO fault (e.g. the worker binary missing)
    // leaves the run dir wedged at "running" forever. run_worker owns spawn/stream/
    // wait and kills the child on error; here we translate its result to a manifest
    // status, finalizing to `failed` even when we propagate the error.
    let streamed = match run_worker(&job, run_path, worker_argv, &mut on_event) {
        Ok(v) => v,
        Err(e) => {
            let _ = rundir::finalize_manifest(run_path, RunStatus::Failed);
            return Err(e);
        }
    };

    finalize(run_path, job.job_id, streamed)
}

/// Resume an existing run dir on the SAME `run_dir`.
///
/// Reads the recorded `manifest.json` for `model_path` + `conversion_config`,
/// asserts the already-run identity gate passed (a failed run cannot silently
/// resume into training), reopens the non-empty dir to `Running`, and re-spawns
/// the worker. Resume is signaled to the worker purely by the checkpoints already
/// present in `run_dir/checkpoints` - there is no new contract field.
pub fn resume_job(
    run_dir: &Path,
    worker_argv: &[String],
    mut on_event: impl FnMut(&EventEnvelope),
) -> Result<Outcome> {
    let manifest = rundir::read_manifest(run_dir)?;

    match &manifest.identity {
        Some(id) if id.passed => {}
        Some(_) => bail!("cannot resume: recorded identity gate did not pass"),
        None => bail!("cannot resume: manifest has no identity result"),
    }
    let conversion_config = manifest
        .conversion_config
        .clone()
        .ok_or_else(|| anyhow!("cannot resume: manifest has no conversion_config"))?;

    let job = ConversionJob {
        schema_version: SCHEMA_VERSION.to_string(),
        job_id: manifest.job_id.clone(),
        model_path: manifest.model_path.clone(),
        run_dir: run_dir.to_string_lossy().into_owned(),
        conversion_config,
    };

    // Reopen the non-empty dir (flip to Running); events.jsonl is appended to.
    rundir::reopen_run_dir(run_dir)?;

    let streamed = match run_worker(&job, run_dir, worker_argv, &mut on_event) {
        Ok(v) => v,
        Err(e) => {
            let _ = rundir::finalize_manifest(run_dir, RunStatus::Failed);
            return Err(e);
        }
    };

    finalize(run_dir, job.job_id, streamed)
}

/// Merge the accumulated identity/token_count into the manifest, stamp the
/// terminal status, and surface the worker's stderr on a non-zero exit. Shared by
/// [`run_job`] and [`resume_job`].
fn finalize(run_path: &Path, job_id: String, streamed: Streamed) -> Result<Outcome> {
    let Streamed {
        status,
        stderr_output,
        identity,
        token_count,
    } = streamed;

    if identity.is_some() || token_count.is_some() {
        let mut manifest = rundir::read_manifest(run_path)?;
        if identity.is_some() {
            manifest.identity = identity;
        }
        if token_count.is_some() {
            manifest.token_count = token_count;
        }
        rundir::write_manifest(run_path, &manifest)?;
    }

    let final_status = if status.success() {
        RunStatus::Completed
    } else {
        RunStatus::Failed
    };
    rundir::finalize_manifest(run_path, final_status)?;

    if !status.success() {
        // The CLI owns the exit code (via Outcome.status); surface the worker's
        // stderr here so a failed run is still debuggable.
        let err = stderr_output.trim();
        if !err.is_empty() {
            eprintln!("a2d-run: worker exited with {status}: {err}");
        }
    }

    Ok(Outcome {
        job_id,
        status: final_status,
    })
}

/// What one worker run yielded: exit status, captured stderr, and the manifest
/// fields accumulated from the event stream.
struct Streamed {
    status: std::process::ExitStatus,
    stderr_output: String,
    identity: Option<IdentityResult>,
    token_count: Option<u64>,
}

/// Fold one event's provenance into the running manifest accumulators: the
/// identity gate carries the verdict; the latest `TrainStep` wins the token
/// count. Everything else is passthrough.
fn absorb_event(
    event: &Event,
    identity: &mut Option<IdentityResult>,
    token_count: &mut Option<u64>,
) {
    match event {
        Event::IdentityGate {
            passed,
            max_abs_diff,
            tolerance,
        } => {
            *identity = Some(IdentityResult {
                passed: *passed,
                max_abs_diff: *max_abs_diff,
                tolerance: *tolerance,
            });
        }
        Event::TrainStep { tokens, .. } => *token_count = Some(*tokens),
        _ => {}
    }
}

/// Spawn the worker, feed it the job, tee stdout to `events.jsonl`, and wait.
///
/// Owns the child process: on any error after a successful spawn the child is
/// killed and reaped before returning, so a mid-stream fault never orphans a
/// worker. Returns the terminal exit status, the worker's captured stderr, and
/// the identity result + latest token count observed on the event stream.
fn run_worker(
    job: &ConversionJob,
    run_path: &Path,
    worker_argv: &[String],
    on_event: &mut impl FnMut(&EventEnvelope),
) -> Result<Streamed> {
    // Caller supplies a pre-split argv (program first); no shell, no shlex.
    let (program, args) = worker_argv
        .split_first()
        .ok_or_else(|| anyhow!("empty worker command"))?;
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning worker: {}", worker_argv.join(" ")))?;

    // Everything past a successful spawn runs under this guard; if it bails we
    // kill+reap the child below so a failed run never leaks a worker process.
    let result = (|| {
        // Drain stderr on a thread so a chatty worker can't deadlock us while we
        // block reading stdout (both pipes have bounded buffers).
        let stderr = child.stderr.take().context("worker stderr not captured")?;
        let stderr_handle = thread::spawn(move || {
            let mut buf = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut buf);
            buf
        });

        // Write the single job document, then close stdin (drop).
        // ponytail: the job doc is a few hundred bytes and fits the pipe buffer, so a
        // blocking write before we read stdout is safe; write on a thread if jobs grow.
        {
            let mut stdin = child.stdin.take().context("worker stdin not captured")?;
            let doc = serde_json::to_vec(job).context("serializing job")?;
            stdin
                .write_all(&doc)
                .context("writing job to worker stdin")?;
        }

        let events_path = run_path.join("events.jsonl");
        let mut events_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&events_path)
            .with_context(|| format!("opening {}", events_path.display()))?;

        let mut identity: Option<IdentityResult> = None;
        let mut token_count: Option<u64> = None;

        let stdout = child.stdout.take().context("worker stdout not captured")?;
        for line in BufReader::new(stdout).lines() {
            let line = line.context("reading worker stdout")?;
            // a2d-run is the sole writer of events.jsonl: tee the raw line verbatim.
            writeln!(events_file, "{line}")
                .with_context(|| format!("appending to {}", events_path.display()))?;
            events_file.flush()?;
            match serde_json::from_str::<EventEnvelope>(&line) {
                Ok(envelope) => {
                    absorb_event(&envelope.event, &mut identity, &mut token_count);
                    on_event(&envelope);
                }
                Err(err) => {
                    eprintln!("a2d-run: warning: unparseable worker stdout line ({err}): {line}")
                }
            }
        }

        let stderr_output = stderr_handle.join().unwrap_or_default();
        let status = child.wait().context("waiting on worker")?;
        Ok(Streamed {
            status,
            stderr_output,
            identity,
            token_count,
        })
    })();

    if result.is_err() {
        // Best-effort: don't leak the worker if we bailed mid-stream.
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2d_contracts::Capability;
    use std::fs;

    // Self-contained fake worker: a shell script (no python, no network) that
    // drains stdin then prints canned envelopes, including an IdentityGate and a
    // TrainStep so the manifest-accumulation path is exercised, plus one garbage
    // line to exercise the "preserve unparseable line and continue" path.
    const FAKE_WORKER: &str = r#"#!/bin/sh
cat >/dev/null
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":0,"ts":"2026-07-02T10:00:00Z","event":{"type":"job_started","worker":"fake-worker 0.1.0"}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":1,"ts":"2026-07-02T10:00:01Z","event":{"type":"identity_gate","passed":true,"max_abs_diff":0.0,"tolerance":1e-6}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":2,"ts":"2026-07-02T10:00:02Z","event":{"type":"train_step","step":1,"loss":2.5,"anneal":0.5,"lr":1e-4,"tokens":4096}}'
printf '%s\n' 'not-json-garbage'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":3,"ts":"2026-07-02T10:00:03Z","event":{"type":"job_completed"}}'
"#;

    fn sample_model_spec() -> ModelSpec {
        ModelSpec {
            model_type: "gpt2".into(),
            n_layers: 12,
            d_model: 768,
            vocab_size: 50257,
            n_heads: 12,
            n_kv_heads: 12,
            sliding_window: None,
            n_experts: None,
            n_active_experts: None,
            capabilities: vec![Capability::AttnFull, Capability::PosLearned],
            mask_token_id: Some(50257),
            inferred: false,
        }
    }

    fn sample_config() -> ConversionConfig {
        ConversionConfig {
            objective: "mdlm".into(),
            data: "./fixtures/data/tiny.jsonl".into(),
            anneal_steps: 100,
            anneal_schedule: "linear".into(),
            seq_len: 512,
            per_device_batch_size: 8,
            grad_accum: 1,
            lr: 1e-4,
            max_steps: Some(500),
            max_tokens: None,
            mask_token: "grow".into(),
            keep_last: 3,
            seed: 0,
            device: "cpu".into(),
            dtype: "float32".into(),
        }
    }

    fn write_fake_worker(base: &Path) -> Vec<String> {
        let script = base.join("fake_worker.sh");
        fs::write(&script, FAKE_WORKER).unwrap();
        vec!["sh".to_string(), script.to_str().unwrap().to_string()]
    }

    #[test]
    fn fake_worker_completes_and_populates_manifest() {
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        let worker_argv = write_fake_worker(&base);
        let run_dir = base.join("run"); // does not exist yet

        let mut seen = Vec::new();
        let outcome = run_job(
            Path::new("/tmp/fake-model"),
            &run_dir,
            sample_model_spec(),
            sample_config(),
            "deadbeef".to_string(),
            &worker_argv,
            |ev| seen.push(ev.clone()),
        )
        .unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);

        // Callback fired once per parseable envelope; the garbage line is skipped.
        assert_eq!(seen.len(), 4);
        assert!(matches!(seen[0].event, Event::JobStarted { .. }));
        assert!(matches!(seen[3].event, Event::JobCompleted {}));

        let manifest = rundir::read_manifest(&run_dir).unwrap();
        assert_eq!(manifest.status, RunStatus::Completed);
        assert!(manifest.finished_at.is_some());
        // Provenance the CLI supplied lands in the manifest.
        assert_eq!(manifest.model_spec, Some(sample_model_spec()));
        assert_eq!(manifest.conversion_config, Some(sample_config()));
        assert_eq!(manifest.source_hash.as_deref(), Some("deadbeef"));
        assert_eq!(
            manifest.data_source.as_deref(),
            Some("./fixtures/data/tiny.jsonl")
        );
        // identity + token_count accumulated from the event stream.
        assert_eq!(
            manifest.identity,
            Some(IdentityResult {
                passed: true,
                max_abs_diff: 0.0,
                tolerance: 1e-6
            })
        );
        assert_eq!(manifest.token_count, Some(4096));

        // events.jsonl holds every raw stdout line, garbage included.
        let events_str = fs::read_to_string(run_dir.join("events.jsonl")).unwrap();
        assert_eq!(events_str.lines().count(), 5);
        assert!(events_str.contains("job_started"));
        assert!(events_str.contains("not-json-garbage"));
        assert!(events_str.contains("job_completed"));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resume_reopens_nonempty_dir_without_clobbering_events() {
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        let worker_argv = write_fake_worker(&base);
        let run_dir = base.join("run");

        // Initial run leaves a completed manifest (identity.passed) + 5-line events.
        run_job(
            Path::new("/tmp/fake-model"),
            &run_dir,
            sample_model_spec(),
            sample_config(),
            "deadbeef".to_string(),
            &worker_argv,
            |_| {},
        )
        .unwrap();
        let lines_before = fs::read_to_string(run_dir.join("events.jsonl"))
            .unwrap()
            .lines()
            .count();
        assert_eq!(lines_before, 5);

        // Resume the SAME non-empty dir: no clobber of the create_run_dir guard.
        let mut seen = Vec::new();
        let outcome = resume_job(&run_dir, &worker_argv, |ev| seen.push(ev.clone())).unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);

        // events.jsonl was appended to, not truncated.
        let lines_after = fs::read_to_string(run_dir.join("events.jsonl"))
            .unwrap()
            .lines()
            .count();
        assert_eq!(lines_after, lines_before * 2);

        // Manifest still carries the resumed run's provenance + gate result.
        let manifest = rundir::read_manifest(&run_dir).unwrap();
        assert_eq!(manifest.status, RunStatus::Completed);
        assert_eq!(manifest.conversion_config, Some(sample_config()));
        assert_eq!(manifest.identity.map(|i| i.passed), Some(true));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resume_rejects_unpassed_identity() {
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        let run_dir = base.join("run");
        fs::create_dir_all(&run_dir).unwrap();

        // A manifest whose gate never passed cannot be resumed.
        let mut manifest = Manifest {
            schema_version: SCHEMA_VERSION.to_string(),
            a2d_version: env!("CARGO_PKG_VERSION").to_string(),
            job_id: "j".into(),
            created_at: Utc::now().to_rfc3339(),
            model_path: "/tmp/fake-model".into(),
            status: RunStatus::Failed,
            finished_at: Some(Utc::now().to_rfc3339()),
            model_spec: Some(sample_model_spec()),
            conversion_config: Some(sample_config()),
            identity: Some(IdentityResult {
                passed: false,
                max_abs_diff: 1.0,
                tolerance: 1e-6,
            }),
            data_source: Some("./fixtures/data/tiny.jsonl".into()),
            source_hash: Some("deadbeef".into()),
            token_count: None,
        };
        rundir::write_manifest(&run_dir, &manifest).unwrap();
        assert!(resume_job(&run_dir, &["true".to_string()], |_| {}).is_err());

        // And a missing identity result is equally unresumable.
        manifest.identity = None;
        rundir::write_manifest(&run_dir, &manifest).unwrap();
        assert!(resume_job(&run_dir, &["true".to_string()], |_| {}).is_err());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn spawn_failure_finalizes_manifest_failed() {
        // A missing worker binary bails at spawn, after the `running` manifest is
        // written. The run dir must not be left stuck at "running".
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        let run_dir = base.join("run");
        let missing = base.join("this-binary-does-not-exist");
        let worker_argv = vec![missing.to_str().unwrap().to_string()];

        let result = run_job(
            Path::new("/tmp/fake-model"),
            &run_dir,
            sample_model_spec(),
            sample_config(),
            "deadbeef".to_string(),
            &worker_argv,
            |_| {},
        );
        assert!(result.is_err(), "spawn of a missing worker should error");

        let manifest = rundir::read_manifest(&run_dir).unwrap();
        assert_eq!(manifest.status, RunStatus::Failed);
        assert!(manifest.finished_at.is_some());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn create_run_dir_rejects_nonempty() {
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("stray"), b"x").unwrap();
        assert!(rundir::create_run_dir(&base).is_err());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn source_hash_of_primary_safetensors() {
        // Known sha256 of the bytes "abc".
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("model.safetensors"), b"abc").unwrap();
        assert_eq!(
            rundir::source_hash(&base).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = fs::remove_dir_all(&base);
    }
}
