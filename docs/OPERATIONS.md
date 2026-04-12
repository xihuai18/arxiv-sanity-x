# Operations Guide

This document focuses on deployment, observability, and runtime operations.

## Environment

- Recommended: `conda activate sanity`
- Configuration: `.env` in repo root (see `.env.example`)
- Inspect/validate config:
    - `python -m config.cli show`
    - `python -m config.cli validate`
    - `python -m config.cli doctor`

Notes:

- Canonical environment variable names are the `ARXIV_SANITY_*` names in `.env.example`.
- `python bin/run_services.py` does not start the daemon unless you add `--with-daemon`.
- `ARXIV_SANITY_OPENCODE_MANAGED` defaults to `true`, so the launcher will also start a local OpenCode service unless you explicitly disable it.
- When managed startup is enabled, the `opencode` binary must be installed on the host and available on `PATH`.

## Process Model

arxiv-sanity-X typically runs as **multiple processes**:

- **Web**: Flask app served by Gunicorn (`bin/up.sh`)
- **Huey consumer**: executes background tasks (summary generation, upload parsing, etc.)
- **Daemon (optional)**: scheduled pipeline (fetch → withdrawn cleanup → compute → summarize → email → backup)
- **Optional model services**: OpenCode server / embedding backend / MinerU backend

For local development, `bin/run_services.py` can start a full stack in one terminal.

Operationally, `tasks.py` is the async orchestration center for summary generation, upload parse/process/extract work, stale-task repair, and task-status visibility. If you deploy changes to `tasks.py` or async task semantics, restart the Huey consumer so worker code and web code stay in sync.

## Start Services

- Web (recommended): `./bin/up.sh`
- One-click launcher (starts multiple services): `python bin/run_services.py`
- One-click launcher with scheduled pipeline: `python bin/run_services.py --with-daemon`
- Huey consumer (required for async jobs): `python bin/huey_consumer.py tasks.huey -w 4 -k thread`
- Scheduler (fetch/compute/summaries/emails; not auto-started by web): `python -m tools daemon`

### Suggested Local Workflow

1. Initialize data once:

- `python -m tools arxiv_daemon -n 10000 -m 500`
- `python -m tools compute --num 20000`

2. Run the service stack:

- `python bin/run_services.py` (recommended), or:
    - Terminal A: `bash bin/up.sh`
    - Terminal B: `python bin/huey_consumer.py tasks.huey -w 4 -k thread`
    - Terminal C (optional): `python -m tools daemon`

### Notes

- `bin/up.sh` builds static assets via `npm run build:static` (best-effort) before starting Gunicorn.
- `bin/up.sh` sets `ARXIV_SANITY_PROCESS_ROLE=web` (fail-fast DB settings).
- `bin/huey_consumer.py` sets `ARXIV_SANITY_PROCESS_ROLE=worker` (more tolerant DB settings) and supports a memory cap via `ARXIV_SANITY_HUEY_MAX_MEMORY_MB`.
- `python bin/run_services.py` now handles external `SIGTERM` by running the same cleanup path as Ctrl+C, so child Gunicorn / Huey process groups are torn down instead of being left behind as orphaned processes.
- The launcher starts a local `opencode serve` by default and points child web/worker processes at that local instance for the current session.
- If `55555` or another web port still reports `address already in use` after stopping the launcher, confirm whether an older Gunicorn was started outside the current launcher session before assuming the latest launch failed.

## Observability

### Health Check

- `GET /health`
    - Non-strict liveness/degraded endpoint
    - Returns `200` for `ok`, `loading`, or `degraded` states
    - Returns `503` only on hard errors
- `GET /ready`
    - Strict readiness endpoint for launchers and probes
    - Returns `503` while papers are still loading or required dependencies are unavailable
    - Ready response shape (example): `{"status":"ok","papers":1234,"deps":{...}}`
    - Loading/error response shape (example): `{"status":"loading","message":"No papers loaded yet"}`

### Prometheus Metrics (Optional)

- Enable: `ARXIV_SANITY_ENABLE_METRICS=true`
- Optional protection key:
    - Set `ARXIV_SANITY_METRICS_KEY=...`
    - Send header `X-ARXIV-SANITY-METRICS-KEY: ...`
- Endpoint: `GET /metrics`
    - Returns `404` unless enabled
    - Under Gunicorn, metrics are **per-worker** (no cross-worker aggregation)

### Task Status (Huey)

- `GET /api/task_status/<task_id>`
    - For task owner, response includes `pid`, `model`, `error`, `priority`, and `stage`
    - `stage` is a coarse-grained progress marker (e.g. acquiring lock / LLM request / writing cache)
    - Some queued tasks may also return `queue_rank` / `queue_total`

### Operational Cleanup Helpers

- Lock cleanup: `python -m scripts cleanup_locks`
- Task-record cleanup: `python -m scripts cleanup_tasks`

Notes:

- Start with dry-run / filtered cleanup whenever possible, then add `--force` only after checking what will be removed.
- `cleanup_tasks` is an operator tool, not a replacement for runtime stale repair inside `tasks.py`; if queued/running states keep coming back, investigate worker health, queue lag, and task ownership first.

### Upload Task Behavior

- Upload mutation APIs now return `409` with `Paper is being deleted` when the target upload record is already marked `deleting`
- This applies to `POST /api/uploaded_papers/update_meta`, `POST /api/uploaded_papers/retry_parse`, `POST /api/uploaded_papers/parse`, `POST /api/uploaded_papers/process`, and `POST /api/uploaded_papers/extract_info`
- Upload task polling via `GET /api/task_status/<task_id>` may now expose upload-specific repair/cancel outcomes:
    - `failed` with `error=stale_running_repaired`
    - `canceled` with `error=superseded_upload_task`
- Deleting an upload now best-effort cancels linked upload parse/process/extract tasks and any linked summary task before the delete flow removes files and DB records
- Upload stale detection uses `ARXIV_SANITY_HUEY_UPLOAD_REPAIR_TTL`; set it higher than your expected upload queue backlog plus worst-case MinerU/LLM processing time
- During incident review, repeated `stale_running_repaired` or `superseded_upload_task` statuses usually indicate queue lag, worker starvation, or a replaced upload task pointer rather than a frontend bug

### Server-Sent Events (SSE)

- Stream: `GET /api/user_stream` (browser login required)
- Stats: `GET /api/sse_stats` (process-local)
- SSE IPC is SQLite-backed and designed to work across multiple Gunicorn workers.
- If SSE is enabled, prefer `gevent` worker class (recommended and auto-selected by `bin/up.sh` when available).
    - Optional hard fail: `ARXIV_SANITY_SSE_STRICT_WORKER_CLASS=true`

### Static Asset Manifest Behavior

- HTML templates resolve hashed assets through `static/dist/manifest.json`.
- Long-lived web workers now invalidate the in-memory manifest cache by file content, not only by filesystem `mtime`.
- This avoids a stale-hash window when `npm run build:static` rewrites `manifest.json` within the same timestamp granularity.
- If fresh builds still do not show up in HTML responses, first verify which process currently owns the web port; stale responses usually mean an older Gunicorn is still serving traffic.

### Logs

- `ARXIV_SANITY_LOG_LEVEL=WARNING|INFO|DEBUG|ERROR`
- `ARXIV_SANITY_LOG_FORMAT=text|json`

## Sentry (Optional Error Reporting)

Sentry is **disabled by default** and only initializes when both conditions are met:

- `ARXIV_SANITY_SENTRY_ENABLED=true`
- `ARXIV_SANITY_SENTRY_DSN` is set

Optional:

- `ARXIV_SANITY_SENTRY_ENVIRONMENT=prod`
- `ARXIV_SANITY_SENTRY_RELEASE=...`
- `ARXIV_SANITY_SENTRY_TRACES_SAMPLE_RATE=0.0`
- `ARXIV_SANITY_SENTRY_PROFILES_SAMPLE_RATE=0.0`

Notes:

- `send_default_pii` is disabled to avoid sending PII by default.

## Data Layout

Under `ARXIV_SANITY_DATA_DIR` (default `data/`):

- `papers.db`: fetched arXiv data
- `dict.db`: user data (tags/keywords/readinglist/task status/...)
- `features.p`: computed features
- `summary/`: cached summaries
- `huey.db`: Huey queue database
- `sse_events.db`: SSE cross-process event bus database (when enabled)
- `uploads/`: uploaded PDFs and derived artifacts (if enabled/used)
- `logs/`: runtime logs (if `ARXIV_SANITY_LOG_DIR` points here; launcher scripts may write logs here)

## User Data Git Backup (Optional)

If enabled, the daemon can snapshot `data/dict.db` and commit/push it to a git repository (e.g. GitHub).

- Enable: `ARXIV_SANITY_DAEMON_ENABLE_GIT_BACKUP=true`
- Source DB: `<ARXIV_SANITY_DATA_DIR>/dict.db`
- Backup repo directory (relative to project root): `ARXIV_SANITY_DAEMON_BACKUP_REPO_DIR=data-repo`
    - This directory can be a submodule checkout or a standalone git clone.
- Push control:
    - `ARXIV_SANITY_DAEMON_BACKUP_PUSH=true|false`
    - `ARXIV_SANITY_DAEMON_BACKUP_PUSH_REMOTE=` (optional)
    - `ARXIV_SANITY_DAEMON_BACKUP_PUSH_BRANCH=` (optional)
    - `ARXIV_SANITY_DAEMON_BACKUP_PUSH_RETRIES=3`

Notes:

- The backup uses SQLite's backup API for a consistent snapshot (safer than raw file copy).
- Ensure the backup repo has a configured remote and the runtime environment has git credentials for `git push`.
- If the backup repo is a git submodule, it may be checked out as a detached `HEAD`. Configure `ARXIV_SANITY_DAEMON_BACKUP_PUSH_BRANCH` (and optionally `..._PUSH_REMOTE`) so the daemon can push via `HEAD:<branch>`.

## Daemon Schedule

The scheduler in `tools/daemon.py` runs cron-style jobs in `settings.daemon.timezone` (set with `ARXIV_SANITY_DAEMON_TIMEZONE`, default `Asia/Shanghai`):

- `fetch_compute`: Mon–Fri 08:00 / 12:00 / 16:00 / 20:00
- `send_email`: Mon–Fri 18:00
- `backup_user_data`: Daily 20:00 (requires a git repo under `backup_repo_dir` + git remote/credentials)
- `cleanup_task_records`: Daily 03:00 (keeps task-status DB bounded)

During each `fetch_compute` run, the daemon can proactively scan the latest public papers already in `papers.db` and tombstone withdrawn-only papers via `python -m tools repair_paper_history --scan-recent-public <N> --apply`. Control this with:

- `ARXIV_SANITY_DAEMON_ENABLE_WITHDRAWN_CLEANUP=true|false`
- `ARXIV_SANITY_DAEMON_WITHDRAWN_CLEANUP_RECENT=250`

### Email Run Semantics

- `tools/daemon.py` computes the weekday / holiday-aware `time_delta` in `settings.daemon.timezone` and passes it to `tools/send_emails.py`; running `python -m tools send_emails` directly still follows the explicit CLI `--time-delta` you provide.
- Users with registered email addresses can now receive recommendation runs from positive-tag inputs, combined tags, or keyword-only inputs.
- Any per-user recommendation failure or per-recipient SMTP failure makes `tools/send_emails.py` exit non-zero.
- `tools/daemon.py` now warns on every non-zero mailer exit, so scheduler logs should treat that as a real delivery problem rather than a harmless partial success.
