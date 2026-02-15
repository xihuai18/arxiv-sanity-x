# Open Source Release Guide

This doc is for maintainers who develop in a private repo but publish an open source mirror.

Goal: ship the **code + safe docs**, without leaking secrets, private data, or large local artifacts.

## What must NOT be published

- Runtime data: `data/` (DBs, caches, uploads, summaries, logs)
- Secrets/local config: `.env*`, `secret_key.txt`, `config/llm.yml`, SSH keys, API key files
- Local tool config: `.claude/`, `.factory/`, `.skills/`, IDE folders
- Virtualenvs: `.venv/`, `venv/`
- Build outputs: `static/dist/` (rebuildable)
- Submodule contents: `data-repo/` (and avoid publishing `.gitmodules` if it contains private URLs)

## Recommended release flow (mirror sync)

Use the sync helper (rsync-based):

```bash
# Preview
./scripts/sync_to_opensource.sh --dry-run

# Actual sync
./scripts/sync_to_opensource.sh
```

The script:

- excludes private/runtime files and common secret patterns
- rewrites `.gitmodules` to a dummy/public submodule URL (if present)
- runs a post-sync safety scan (forbidden files, private IPs, nested `.git`, large files)

## Manual safety checks (recommended)

Run these before pushing to a public repo:

```bash
# 1) Check for accidental secrets in tracked files
git ls-files | xargs rg -n \"sk-|BEGIN_PRIVATE_KEY|ghp_|github_pat_|AKIA\"

# 2) Check for private IPs / user paths
git ls-files | xargs rg -n \"172\\.16\\.|192\\.168\\.|\\b10\\.[0-9]{1,3}\\.|/home/|/Users/\"

# 3) Build sanity (should succeed; output is ignored)
npm run build:static
test -f static/dist/manifest.json
```

## Notes on `.gitmodules` / `data-repo/`

`data-repo/` is an optional submodule used for backing up `data/dict.db`.

- Do not publish the private submodule URL.
- Options for open source:
  - Remove submodule metadata entirely (recommended if you don't need it in public)
  - Keep `.gitmodules` but ensure it uses a public/dummy URL (the sync script rewrites it for the mirror)

## CI expectations

- Python CI should run without requiring real `data/` (tests will use a temp `ARXIV_SANITY_DATA_DIR`).
- Node build workflow only verifies that `npm run build:static` works and produces a manifest.
