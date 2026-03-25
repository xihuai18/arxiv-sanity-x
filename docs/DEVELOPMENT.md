# Development Guide

## Environment

- Recommended: `conda activate sanity`
- For subsystem-specific implementation notes, read the matching `.opencode/skills/` manual, then use `docs/INDEX.md` and `tests/README.md` as focused support docs.

## Install

This repo supports multiple workflows:

- Conda-based environment (recommended for heavy deps like CUDA/torch/vLLM)
- `pip install -r requirements.txt` for a batteries-included Python environment

For editable installs (optional):

- `pip install -e ".[dev]"`
- `npm install` for frontend asset builds (`npm run build:static`, `bin/up.sh`)

## Configuration CLI

- `python -m config.cli show`
- `python -m config.cli show --json`
- `python -m config.cli validate`
- `python -m config.cli doctor`
- `python -m config.cli env`

Notes:

- Canonical variable names live in `.env.example`; a small set of legacy aliases is still accepted for backward compatibility, but prefer canonical names for new configs.
- `python bin/run_services.py` does not start the scheduler daemon unless you add `--with-daemon`.

## Tests

Recommended (fast):

- `pytest tests/unit tests/integration -q -k "not daemon"`

Isolation tip (avoid touching real `data/`):

- `ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit tests/integration -q`

### Upload Task Regressions

When changing `backend/services/upload_service.py`, `backend/blueprints/api_uploads.py`, or the upload Huey tasks in `tasks.py`, run this targeted set in addition to the default unit/integration suite:

- `ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_upload_task_status_sse.py tests/unit/test_tasks_upload_deleting.py tests/integration/test_api_uploads.py -q`

This set covers:

- deleting upload records returning `409` from upload mutation APIs
- stale upload task repair and re-enqueue behavior
- superseded upload workers self-canceling instead of running outdated work
- upload task status / task id contract changes exposed through `/api/task_status/<task_id>`
- the dedicated upload stale TTL setting `ARXIV_SANITY_HUEY_UPLOAD_REPAIR_TTL`

### Task Orchestration Regressions

When changing `tasks.py`, async task visibility, cancel/supersede semantics, or worker-side progress/state repair logic, run this set in addition to the default unit/integration suite:

- `ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_tasks_summary_force_refresh.py tests/unit/test_summary_cancellation.py tests/unit/test_tasks_summary_status_events.py tests/unit/test_upload_task_status_sse.py tests/unit/test_tasks_upload_deleting.py tests/integration/test_api_summary.py tests/integration/test_api_uploads.py tests/integration/test_api_sse.py -q`

This set is the fastest regression net for:

- cooperative cancellation and generation epoch behavior
- force-refresh enqueue semantics and resolved-model cache purging
- upload task pointer repair / `pending_registration` windows / stale task cleanup
- owner-scoped task visibility exposed through `/api/task_status/<task_id>` and list overlays
- summary/upload SSE payloads that frontend polling or realtime consumers depend on

### Launcher / Static Manifest Regressions

When changing `bin/run_services.py`, `backend/utils/manifest.py`, Gunicorn launch behavior, or hashed static asset resolution, run this set in addition to the default unit/integration suite:

- `pytest tests/unit/test_manifest.py tests/unit/test_manifest_fallback.py tests/unit/test_run_services.py -q`

This set is the fastest regression net for:

- `run_services.py` handling external `SIGTERM` cleanly so child Gunicorn / Huey processes do not linger
- hashed asset resolution continuing to track fresh `static/dist/manifest.json` content even if filesystem `mtime` granularity is coarse
- fallback lookup still resolving hashed files when the manifest is missing or stale

## Running Locally

- Full stack (recommended): `python bin/run_services.py`
- Full stack + scheduled fetch/summary/email: `python bin/run_services.py --with-daemon`
- Web only:
    - Dev: `npm run build:static && python serve.py` (or `npm run build:dev && python serve.py` while iterating on frontend assets)
    - Gunicorn: `bash bin/up.sh`
- Huey consumer (required for async jobs): `python bin/huey_consumer.py tasks.huey -w 4 -k thread`
- Scheduler daemon (optional, not auto-started by web): `python -m tools daemon`
- If you changed `tasks.py` or async summary/upload behavior, restart the Huey consumer so the worker code matches the web process.

Notes:

- `python bin/run_services.py` now treats external `SIGTERM` the same way as Ctrl+C: the launcher exits through its normal cleanup path and stops child process groups instead of leaving orphaned Gunicorn / Huey processes behind.
- If a port such as `55555` still looks busy after you stop the launcher, suspect an older web process that was started outside the current launcher session rather than the latest `run_services.py` process.

## Frontend Build

- `npm run build:static` (production build)
- `npm run build:dev` (no hash, easier debugging)
- `npm run build:watch` (watch mode)

Notes:

- `./bin/up.sh` runs the static build automatically on startup.
- Build output is written to `static/dist/` (gitignored) and referenced via `static/dist/manifest.json`.
- Long-lived web processes now reload hashed asset mappings when `static/dist/manifest.json` content changes, even if the file timestamp does not move to a new second.
- If the browser still receives old hashed asset URLs after a fresh build, verify which process owns the web port first; stale HTML is more often caused by an older Gunicorn still serving requests than by browser cache.

## Frontend/UI Verification

For homepage filters, reading list, summary actions, tag dropdowns, or shared modal/toast changes, use this lightweight check sequence first:

- `npm run build:static`
- `pytest tests/unit/test_ui_accessibility_contract.py tests/unit/test_readinglist_ui_contract.py tests/unit/test_summary_toc_contract.py tests/unit/test_frontend_sse_wiring.py -q`

Recommended manual or Playwright spot checks after those tests pass:

- Homepage `/`: search mode switching, rank switching, advanced filters, tag suggestions, paper-card action spacing
- Reading list `/readinglist`: add/remove flow, confirmation dialog, mobile card actions
- Summary `/summary?...`: action bar layout, TOC collapse/expand on mobile, reading-list toggle
- Inspect `/inspect?...` and a 404 page: shared chrome, fallback scripts, recovery actions

Notes:

- The lightweight UI tests are source-contract tests and intentionally avoid importing the full Flask app.
- If your local Python environment is missing optional runtime packages, these tests are still the fastest safe regression net for frontend work.

### MathJax Rendering Notes

- There are two MathJax rendering modes in this repo:
    - Scan existing DOM with `CommonUtils.triggerMathJax()` (`typesetPromise()` / `typeset()`). This is used by homepage cards, inspect, and reading-list surfaces.
    - Programmatic CHTML conversion in `static/markdown_summary_utils.js` for the summary main body.
- If you add another programmatic conversion path using `MathJax.startup.document.convert()` or `MathJax.tex2chtmlPromise()`, you must sync the CHTML stylesheet before calling `updateDocument()`:
    - `MathJax.startup.document.outputJax.styleSheet(MathJax.startup.document)`
- Without that sync, some formulas can render as tiny fragments because the generated glyph classes do not have matching `mjx-c::before` rules yet.
- When changing summary math rendering, spot-check at least one long display equation (fractions, sums, integrals, aligned blocks) on `/summary?...` and verify the page console stays free of MathJax warnings/errors.

## CI

GitHub Actions workflows live in `.github/workflows/`:

- `ci.yml`: runs Python compile + unit/integration tests
- `check-dist.yml`: ensures `npm run build:static` works (and produces `static/dist/manifest.json`)

To reproduce the main CI test run locally:

- `ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit tests/integration -q -k "not daemon"`

## Open Source Release

If you publish an open source mirror, review the release checklist before pushing the public repo.

Notes:

- Review `docs/OPEN_SOURCE.md` before publishing.
- Destructive cleanup options in your release flow should remove local-only artifacts from the published tree.

## Pre-commit (Optional)

- Install hooks: `pre-commit install`
- Run on all files: `pre-commit run -a`
