# Open Source Release Guide

This doc is for maintainers preparing a public release of this repo.

Goal: ship the **code + safe docs**, without leaking secrets, sensitive data, or large local artifacts.

## What must NOT be published

- Runtime data: `data/` (DBs, caches, uploads, summaries, logs)
- Secrets/local config: `.env*`, `secret_key.txt`, `config/llm.yml`, SSH keys, API key files
- Local tool config: `.claude/`, `.factory/`, `.skills/`, `.playwright-cli/`, IDE folders, plus non-public parts of `.opencode/` (keep `.opencode/skills/` in the public tree)
- Virtualenvs: `.venv/`, `venv/`
- Build outputs: `static/dist/` (rebuildable)
- Submodule contents: `data-repo/` (and avoid publishing `.gitmodules` if it contains non-public URLs)
- Local test/runtime residue: `tmp/`, `coverage.xml`, `.hypothesis/`, `.tox/`, `.pytype/`, `static/test_mathjax.html`

## Recommended release checklist

Before publishing, make sure the release process will:

- exclude secrets, runtime files, and common sensitive patterns
- keep `.opencode/skills/` but exclude the rest of `.opencode/`
- rewrite `.gitmodules` to a dummy/public submodule URL when needed
- scrub local-only residue such as non-public `.opencode/` files, `.playwright-cli/`, `tmp/`, coverage/test artifacts, IDE folders, and `config/llm.yml`
- run a post-sync safety scan for forbidden files, local IPs, nested `.git`, and large files

## Release expectations

- Review the published tree itself instead of depending on unpublished local tooling.
- The release process should support overriding the target path instead of requiring a hardcoded location.
- Destructive cleanup options should delete excluded local-only files from the release tree as well as skipping them during sync.
- Review the exclude list before each release, especially if the repo has new local tooling under hidden directories.

## Manual safety checks (recommended)

Run these before pushing to a public repo:

```bash
# 1) Check for accidental secrets in tracked files
git ls-files | xargs rg -n \"sk-|BEGIN_PRIVATE_KEY|ghp_|github_pat_|AKIA\"

# 2) Check for local IPs / user paths
git ls-files | xargs rg -n \"172\\.16\\.|192\\.168\\.|\\b10\\.[0-9]{1,3}\\.|/home/|/Users/\"

# 3) Build sanity (should succeed; output is ignored)
npm run build:static
test -f static/dist/manifest.json
```

## Notes on `.gitmodules` / `data-repo/`

`data-repo/` is an optional submodule used for backing up `data/dict.db`.

- Do not publish a non-public submodule URL.
- Options for open source:
    - Remove submodule metadata entirely (recommended if you don't need it in public)
    - Keep `.gitmodules` but ensure it uses a public/dummy URL

## CI expectations

- Python CI should run without requiring real `data/` (tests will use a temp `ARXIV_SANITY_DATA_DIR`).
- Node build workflow only verifies that `npm run build:static` works and produces a manifest.
