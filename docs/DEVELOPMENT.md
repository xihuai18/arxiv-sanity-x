# Development Guide

## Environment

- Recommended: `conda activate sanity`

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

## Running Locally

- Full stack (recommended): `python bin/run_services.py`
- Full stack + scheduled fetch/summary/email: `python bin/run_services.py --with-daemon`
- Web only:
    - Dev: `npm run build:static && python serve.py` (or `npm run build:dev && python serve.py` while iterating on frontend assets)
    - Gunicorn: `bash bin/up.sh`
- Huey consumer (required for async jobs): `python bin/huey_consumer.py tasks.huey -w 4 -k thread`
- Scheduler daemon (optional, not auto-started by web): `python -m tools daemon`

## Frontend Build

- `npm run build:static` (production build)
- `npm run build:dev` (no hash, easier debugging)
- `npm run build:watch` (watch mode)

Notes:

- `./bin/up.sh` runs the static build automatically on startup.
- Build output is written to `static/dist/` (gitignored) and referenced via `static/dist/manifest.json`.

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

If you maintain a private fork and publish an open source mirror, use:

- `scripts/sync_to_opensource.sh`

See `docs/OPEN_SOURCE.md` for the checklist and safety notes.

## Pre-commit (Optional)

- Install hooks: `pre-commit install`
- Run on all files: `pre-commit run -a`
