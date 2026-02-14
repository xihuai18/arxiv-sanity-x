# Configuration Guide

This project uses **pydantic-settings** (`config/settings.py`) to load configuration from:

- Environment variables (preferred for deployment)
- A `.env` file in the repo root (recommended for local/dev; see `.env.example`)

## Inspect and Validate

- Show effective configuration: `python -m config.cli show`
- Validate configuration: `python -m config.cli validate`
- JSON output (debugging/automation): `python -m config.cli show --json`

## Common Files

- `.env.example`: configuration template (copy to `.env`)
- `config/llm_template.yml`: LiteLLM template (copy to `config/llm.yml` if you use LiteLLM)

## Key Configuration Areas

Most settings are `ARXIV_SANITY_...` variables. Some nested groups have their own prefixes:

- `ARXIV_SANITY_LLM_...`: LLM provider / OpenAI-compatible gateway settings
- `ARXIV_SANITY_EXTRACT_...`: metadata extraction model (uploads)
- `ARXIV_SANITY_EMBED_...`: embedding backend settings
- `ARXIV_SANITY_MINERU_...`: MinerU parsing backend settings
- `ARXIV_SANITY_EMAIL_...`: SMTP settings
- `ARXIV_SANITY_DAEMON_...`: scheduler pipeline settings
- `ARXIV_SANITY_HUEY_...`: Huey worker settings
- `ARXIV_SANITY_SSE_...`: SSE IPC settings (SQLite-backed event bus)
- `ARXIV_SANITY_DB_...`: SQLite retry/timeout tuning
- `ARXIV_SANITY_SEARCH_...`: search limits and guardrails

## Minimal Required Settings

For LLM summaries you typically need:

- `ARXIV_SANITY_LLM_BASE_URL`
- `ARXIV_SANITY_LLM_API_KEY`
- `ARXIV_SANITY_LLM_NAME`

If you enable MinerU API backend:

- `ARXIV_SANITY_MINERU_ENABLED=true`
- `ARXIV_SANITY_MINERU_BACKEND=api`
- `MINERU_API_KEY=...` (or `ARXIV_SANITY_MINERU_API_KEY`)

## Recommended Production-ish Settings

- Set a stable session key: `ARXIV_SANITY_SECRET_KEY=...` (or use `secret_key.txt`)
- Keep async jobs enabled: run a Huey consumer (`python bin/huey_consumer.py`)
- If enabling SSE, prefer gevent for Gunicorn (see `bin/up.sh` behavior)
