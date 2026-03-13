# Configuration Guide

This project uses **pydantic-settings** (`config/settings.py`) to load configuration from:

- Environment variables (preferred for deployment)
- A `.env` file in the repo root (recommended for local/dev; see `.env.example`)

## Inspect and Validate

- Show effective configuration: `python -m config.cli show`
- Validate configuration: `python -m config.cli validate`
- Diagnose common mistakes: `python -m config.cli doctor`
- JSON output (debugging/automation): `python -m config.cli show --json`
- Export current env template: `python -m config.cli env`
- Include secret values only when needed: `python -m config.cli show --json --include-secrets` or `python -m config.cli env --include-secrets`

## Common Scenarios

- Local web app: configure `ARXIV_SANITY_LLM_*`, then run `python bin/run_services.py`
- Automated pipeline: additionally start `python -m tools daemon` or use `python bin/run_services.py --with-daemon`
- Email recommendations: configure `ARXIV_SANITY_EMAIL_*` and `ARXIV_SANITY_RECO_API_KEY`
- MinerU API parsing: set `ARXIV_SANITY_MINERU_ENABLED=true`, `ARXIV_SANITY_MINERU_BACKEND=api`, and `ARXIV_SANITY_MINERU_API_KEY`

## Common Files

- `.env.example`: configuration template (copy to `.env`)
- `docs/DEFAULTS.md`: code defaults reference and baseline summary
- `config/llm_template.yml`: LiteLLM template (copy to `config/llm.yml` if you use LiteLLM)

If you copy `config/llm_template.yml` unchanged, also override `ARXIV_SANITY_LLM_NAME` and `ARXIV_SANITY_EXTRACT_MODEL_NAME` to aliases that exist in that template.

## Key Configuration Areas

Canonical operator-facing variables live in `.env.example`. Most settings are `ARXIV_SANITY_...` variables, with the following groups:

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
- `ARXIV_SANITY_RECO_...`: recommendation and email API behavior
- `ARXIV_SANITY_ARXIV_...`: fetched arXiv category groups

The preferred variable names are the canonical `ARXIV_SANITY_*` names shown in `.env.example` and `python -m config.cli env`.
A small set of legacy aliases is still accepted for backward compatibility, but new `.env` files and automation should use the canonical names.

If you want a single place to inspect actual code defaults, see `docs/DEFAULTS.md`.

## Minimal Required Settings

For LLM summaries you typically need:

- `ARXIV_SANITY_LLM_BASE_URL`
- `ARXIV_SANITY_LLM_API_KEY`
- `ARXIV_SANITY_LLM_NAME`

If you configure versioned GPT aliases such as `gpt-5.4`, `gpt-5.5`, or `gpt-6` in `config/llm.yml`:

- `tools/paper_summarizer.py` prefers the OpenAI Responses API for those models.
- When the alias exists in `config/llm.yml`, the summarizer will try to resolve the alias back to its upstream `api_base` / `api_key` / `extra_body` and call the upstream Responses endpoint directly.
- This avoids compatibility issues in gateways that proxy `/chat/completions` correctly but do not fully normalize `/responses` SSE payloads.
- If the YAML uses `os.environ/ENV_NAME` for `api_key`, that environment variable must exist in the process that runs the summarizer.
- If that environment variable is missing, the summarizer logs a warning and falls back to the configured gateway path instead of the direct upstream route.

If you enable MinerU API backend:

- `ARXIV_SANITY_MINERU_ENABLED=true`
- `ARXIV_SANITY_MINERU_BACKEND=api`
- `ARXIV_SANITY_MINERU_API_KEY=...`

If you do not have a MinerU API key, keep `ARXIV_SANITY_MINERU_ENABLED=false`.

For uploaded-paper metadata extraction:

- Default extract model: `ARXIV_SANITY_EXTRACT_MODEL_NAME=qwen3.5-plus`
- If `ARXIV_SANITY_EXTRACT_BASE_URL` / `ARXIV_SANITY_EXTRACT_API_KEY` are empty, extraction reuses the main LLM endpoint and credentials
- This means your main LLM gateway must know the `qwen3.5-plus` alias, or you should explicitly set `ARXIV_SANITY_EXTRACT_MODEL_NAME` to a model available on your endpoint

## Recommended Baseline

For a small local or single-user deployment, this is a sensible non-secret baseline:

- `ARXIV_SANITY_HOST=http://localhost:55555`
- `ARXIV_SANITY_LOG_LEVEL=INFO`
- `ARXIV_SANITY_WARMUP_DATA=true`
- `ARXIV_SANITY_WARMUP_ML=true`
- `ARXIV_SANITY_ENABLE_SCHEDULER=true`
- `ARXIV_SANITY_READY_REQUIRE_EMBEDDING=true`
- `ARXIV_SANITY_READY_REQUIRE_MINERU=true`
- `ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE=html`
- `ARXIV_SANITY_SUMMARY_HTML_SOURCES=ar5iv,arxiv`
- `ARXIV_SANITY_EMBED_USE_LLM_API=false`
- `ARXIV_SANITY_MINERU_ENABLED=false`
- `ARXIV_SANITY_DAEMON_FETCH_NUM=2000`
- `ARXIV_SANITY_DAEMON_SUMMARY_NUM=250`
- `ARXIV_SANITY_DAEMON_ENABLE_SUMMARY=true`
- `ARXIV_SANITY_DAEMON_ENABLE_EMBEDDINGS=true`
- `ARXIV_SANITY_HUEY_WORKERS=4`
- `ARXIV_SANITY_HUEY_WORKER_TYPE=thread`
- `ARXIV_SANITY_SSE_ENABLED=true`
- `ARXIV_SANITY_GUNICORN_PRELOAD=true`

If you want the automated pipeline (fetch -> compute -> summarize -> email), you must explicitly start the daemon with `python -m tools daemon` or `python bin/run_services.py --with-daemon`.

## Recommended Production-ish Settings

- Set a stable session key: `ARXIV_SANITY_SECRET_KEY=...` (or use `secret_key.txt`)
- Keep async jobs enabled: run a Huey consumer (`python bin/huey_consumer.py tasks.huey -w 4 -k thread`) or use `python bin/run_services.py`
- If enabling SSE, prefer gevent for Gunicorn (see `bin/up.sh` behavior)
