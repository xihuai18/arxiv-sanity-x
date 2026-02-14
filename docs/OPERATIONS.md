# Operations Guide

This document focuses on deployment, observability, and runtime operations.

## Environment

- Recommended: `conda activate sanity`
- Configuration: `.env` in repo root (see `.env.example`)
- Inspect/validate config:
  - `python -m config.cli show`
  - `python -m config.cli validate`

## Process Model

arxiv-sanity-X typically runs as **multiple processes**:

- **Web**: Flask app served by Gunicorn (`bin/up.sh`)
- **Huey consumer**: executes background tasks (summary generation, upload parsing, etc.)
- **Daemon (optional)**: scheduled pipeline (fetch → compute → summarize → email → backup)
- **Optional model services**: LiteLLM gateway / embedding backend / MinerU backend

For local development, `bin/run_services.py` can start a full stack in one terminal.

## Start Services

- Web (recommended): `./bin/up.sh`
- One-click launcher (starts multiple services): `python bin/run_services.py`
- Huey consumer (required for async jobs): `python bin/huey_consumer.py`
- Scheduler (fetch/compute/summaries/emails): `python tools/daemon.py` (or `python -m tools daemon`)

### Suggested Local Workflow

1) Initialize data once:

- `python -m tools arxiv_daemon -n 10000 -m 500`
- `python -m tools compute --num 20000`

2) Run the service stack:

- `python bin/run_services.py` (recommended), or:
  - Terminal A: `bash bin/up.sh`
  - Terminal B: `python bin/huey_consumer.py`
  - Terminal C (optional): `python tools/daemon.py`

### Notes

- `bin/up.sh` builds static assets via `npm run build:static` (best-effort) before starting Gunicorn.
- `bin/up.sh` sets `ARXIV_SANITY_PROCESS_ROLE=web` (fail-fast DB settings).
- `bin/huey_consumer.py` sets `ARXIV_SANITY_PROCESS_ROLE=worker` (more tolerant DB settings) and supports a memory cap via `ARXIV_SANITY_HUEY_MAX_MEMORY_MB`.

## Observability

### Health Check

- `GET /health`
  - Returns `200` only when papers are loaded and service is ready
  - Returns `503` while cold-start loading or on errors
  - Ready response shape (example): `{"status":"ok","papers":1234,"deps":{...}}`
  - Loading response shape (example): `{"status":"loading","message":"No papers loaded yet"}`

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

### Server-Sent Events (SSE)

- Stream: `GET /api/user_stream` (browser login required)
- Stats: `GET /api/sse_stats` (process-local)
- SSE IPC is SQLite-backed and designed to work across multiple Gunicorn workers.
- If SSE is enabled, prefer `gevent` worker class (recommended and auto-selected by `bin/up.sh` when available).
  - Optional hard fail: `ARXIV_SANITY_SSE_STRICT_WORKER_CLASS=true`

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

The scheduler in `tools/daemon.py` runs cron-style jobs in `settings.daemon.timezone`:

- `fetch_compute`: Mon–Fri 08:00 / 12:00 / 16:00 / 20:00
- `send_email`: Mon–Fri 18:00
- `backup_user_data`: Daily 20:00 (requires a git repo under `backup_repo_dir` + git remote/credentials)
- `cleanup_task_records`: Daily 03:00 (keeps task-status DB bounded)
