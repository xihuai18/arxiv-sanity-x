# Development Guide

## Environment

- Recommended: `conda activate sanity`

## Install

This repo supports multiple workflows:

- Conda-based environment (recommended for heavy deps like CUDA/torch/vLLM)
- `pip install -r requirements.txt` for a batteries-included Python environment

For editable installs (optional):

- `pip install -e ".[dev]"`

## Configuration CLI

- `python -m config.cli show`
- `python -m config.cli show --json`
- `python -m config.cli validate`
- `python -m config.cli env`

## Tests

Recommended (fast):

- `pytest tests/unit tests/integration -q -k "not daemon"`

Isolation tip (avoid touching real `data/`):

- `ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit tests/integration -q`

## Running Locally

- Full stack (recommended): `python bin/run_services.py`
- Web only:
  - Dev: `python serve.py`
  - Gunicorn: `bash bin/up.sh`
- Huey consumer (required for async jobs): `python bin/huey_consumer.py`
- Scheduler daemon (optional): `python tools/daemon.py`

## Frontend Build

- `npm run build:static` (production build)
- `npm run build:dev` (no hash, easier debugging)
- `npm run build:watch` (watch mode)

Notes:

- `./bin/up.sh` runs the static build automatically on startup.
- Build output is written to `static/dist/` (gitignored) and referenced via `static/dist/manifest.json`.

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
