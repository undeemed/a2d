//! Worker process lifecycle: spawn, feed the job, tee the event stream.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;

use a2d_contracts::{ConversionJob, EventEnvelope, Manifest, RunStatus, SCHEMA_VERSION};
use anyhow::{anyhow, Context, Result};
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
pub fn run_job(
    model_path: &Path,
    run_dir: &Path,
    worker_argv: &[String],
    mut on_event: impl FnMut(&EventEnvelope),
) -> Result<Outcome> {
    let run_path = run_dir;

    let job = ConversionJob {
        schema_version: SCHEMA_VERSION.to_string(),
        job_id: Uuid::new_v4().to_string(),
        model_path: model_path.to_string_lossy().into_owned(),
        run_dir: run_dir.to_string_lossy().into_owned(),
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
    };
    rundir::write_manifest(run_path, &manifest)?;

    // Once the `running` manifest is on disk, every failure path must still stamp a
    // terminal status - otherwise a spawn/IO fault (e.g. the worker binary missing)
    // leaves the run dir wedged at "running" forever. run_worker owns spawn/stream/
    // wait and kills the child on error; here we translate its result to a manifest
    // status, finalizing to `failed` even when we propagate the error.
    let (status, stderr_output) = match run_worker(&job, run_path, worker_argv, &mut on_event) {
        Ok(v) => v,
        Err(e) => {
            let _ = rundir::finalize_manifest(run_path, RunStatus::Failed);
            return Err(e);
        }
    };

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
        job_id: job.job_id,
        status: final_status,
    })
}

/// Spawn the worker, feed it the job, tee stdout to `events.jsonl`, and wait.
///
/// Owns the child process: on any error after a successful spawn the child is
/// killed and reaped before returning, so a mid-stream fault never orphans a
/// worker. Returns the terminal exit status and the worker's captured stderr.
fn run_worker(
    job: &ConversionJob,
    run_path: &Path,
    worker_argv: &[String],
    on_event: &mut impl FnMut(&EventEnvelope),
) -> Result<(std::process::ExitStatus, String)> {
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

        let stdout = child.stdout.take().context("worker stdout not captured")?;
        for line in BufReader::new(stdout).lines() {
            let line = line.context("reading worker stdout")?;
            // a2d-run is the sole writer of events.jsonl: tee the raw line verbatim.
            writeln!(events_file, "{line}")
                .with_context(|| format!("appending to {}", events_path.display()))?;
            events_file.flush()?;
            match serde_json::from_str::<EventEnvelope>(&line) {
                Ok(envelope) => on_event(&envelope),
                Err(err) => {
                    eprintln!("a2d-run: warning: unparseable worker stdout line ({err}): {line}")
                }
            }
        }

        let stderr_output = stderr_handle.join().unwrap_or_default();
        let status = child.wait().context("waiting on worker")?;
        Ok((status, stderr_output))
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
    use a2d_contracts::Event;
    use std::fs;

    // Self-contained fake worker: a shell script (no python, no network) that
    // drains stdin then prints canned envelopes, plus one garbage line to
    // exercise the "preserve unparseable line and continue" path.
    const FAKE_WORKER: &str = r#"#!/bin/sh
cat >/dev/null
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":0,"ts":"2026-07-02T10:00:00Z","event":{"type":"job_started","worker":"fake-worker 0.1.0"}}'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":1,"ts":"2026-07-02T10:00:01Z","event":{"type":"log","level":"info","message":"no-op conversion"}}'
printf '%s\n' 'not-json-garbage'
printf '%s\n' '{"schema_version":"0.1.0","job_id":"test-job","seq":2,"ts":"2026-07-02T10:00:02Z","event":{"type":"job_completed"}}'
"#;

    #[test]
    fn fake_worker_completes_and_tees_events() {
        let base = std::env::temp_dir().join(format!("a2d-run-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&base).unwrap();
        let script = base.join("fake_worker.sh");
        fs::write(&script, FAKE_WORKER).unwrap();
        let run_dir = base.join("run"); // does not exist yet
        let worker_argv = vec!["sh".to_string(), script.to_str().unwrap().to_string()];

        let mut seen = Vec::new();
        let outcome = run_job(Path::new("/tmp/fake-model"), &run_dir, &worker_argv, |ev| {
            seen.push(ev.clone())
        })
        .unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);

        // Callback fired once per parseable envelope; the garbage line is skipped.
        assert_eq!(seen.len(), 3);
        assert!(matches!(seen[0].event, Event::JobStarted { .. }));
        assert!(matches!(seen[2].event, Event::JobCompleted {}));

        // Manifest finalized to completed.
        let manifest_str = fs::read_to_string(run_dir.join("manifest.json")).unwrap();
        let manifest: Manifest = serde_json::from_str(&manifest_str).unwrap();
        assert_eq!(manifest.status, RunStatus::Completed);
        assert!(manifest.finished_at.is_some());

        // events.jsonl holds every raw stdout line, garbage included.
        let events_str = fs::read_to_string(run_dir.join("events.jsonl")).unwrap();
        assert_eq!(events_str.lines().count(), 4);
        assert!(events_str.contains("job_started"));
        assert!(events_str.contains("not-json-garbage"));
        assert!(events_str.contains("job_completed"));

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

        let result = run_job(Path::new("/tmp/fake-model"), &run_dir, &worker_argv, |_| {});
        assert!(result.is_err(), "spawn of a missing worker should error");

        let manifest_str = fs::read_to_string(run_dir.join("manifest.json")).unwrap();
        let manifest: Manifest = serde_json::from_str(&manifest_str).unwrap();
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
}
