# Configuration Guide

This repo uses `pydantic-settings` (`config/settings.py`) and reads configuration from:

- environment variables
- a repo-root `.env` file

## Inspect and Validate

- `python -m config.cli show`
- `python -m config.cli show --json`
- `python -m config.cli validate`
- `python -m config.cli doctor`
- `python -m config.cli env`

## Text Models via OpenCode

Text generation is unified behind an [OpenCode](https://opencode.ai) HTTP server (`opencode serve`). Install and configure OpenCode first — see <https://opencode.ai> for installation instructions. The main settings are:

- `ARXIV_SANITY_OPENCODE_BASE_URL`
- `ARXIV_SANITY_OPENCODE_MANAGED`
- `ARXIV_SANITY_OPENCODE_HOST`
- `ARXIV_SANITY_OPENCODE_PORT`
- `ARXIV_SANITY_OPENCODE_USERNAME`
- `ARXIV_SANITY_OPENCODE_PASSWORD`
- `ARXIV_SANITY_OPENCODE_TIMEOUT`

Model selection is separate from the server location:

- `ARXIV_SANITY_LLM_NAME`: default summary model, can be an alias like `gpt-5.4` or a canonical `provider/model`
- `ARXIV_SANITY_EXTRACT_MODEL_NAME`: upload metadata extraction model, also accepts either an alias or `provider/model`

There is no separate `LLM_FALLBACK_MODELS` setting anymore. Alias members are the only automatic fallback path for text generation. If all candidates under the selected alias fail, the request fails.

Examples:

```bash
ARXIV_SANITY_OPENCODE_BASE_URL=http://127.0.0.1:53000
ARXIV_SANITY_OPENCODE_MANAGED=true
ARXIV_SANITY_LLM_NAME=gpt-5.4
ARXIV_SANITY_EXTRACT_MODEL_NAME=gpt-5.4-mini
```

If `ARXIV_SANITY_EXTRACT_MODEL_NAME` is empty, it falls back to `ARXIV_SANITY_LLM_NAME`.

By default, `python bin/run_services.py` launches `opencode serve` locally and points child processes at `http://<ARXIV_SANITY_OPENCODE_HOST>:<ARXIV_SANITY_OPENCODE_PORT>` for the current launcher session, even if `.env` still has a stale external `ARXIV_SANITY_OPENCODE_BASE_URL`. Set `ARXIV_SANITY_OPENCODE_MANAGED=false` only when you want to use an already-running external OpenCode service.

When you keep the default `ARXIV_SANITY_OPENCODE_MANAGED=true`, make sure the `opencode` binary is installed and available on `PATH`; otherwise `python bin/run_services.py` will fail early when it tries to start the managed local service.

## Embeddings Stay Independent

Embedding configuration does not reuse the main text-model settings.

- Local embedding service:
    - `ARXIV_SANITY_EMBED_USE_LLM_API=false`
    - `ARXIV_SANITY_EMBED_PORT=54000`
- Remote/OpenAI-compatible embedding API:
    - `ARXIV_SANITY_EMBED_USE_LLM_API=true`
    - `ARXIV_SANITY_EMBED_API_BASE` is required
    - `ARXIV_SANITY_EMBED_API_KEY` is optional depending on your provider

## MinerU / Email / Runtime

Other common groups remain unchanged:

- `ARXIV_SANITY_MINERU_*`
- `ARXIV_SANITY_EMAIL_*`
- `ARXIV_SANITY_DAEMON_*`
- `ARXIV_SANITY_HUEY_*`
- `ARXIV_SANITY_SSE_*`
- `ARXIV_SANITY_DB_*`
- `ARXIV_SANITY_ARXIV_*`

## Common Scenarios

- Local web app: keep `ARXIV_SANITY_OPENCODE_MANAGED=true`, configure `ARXIV_SANITY_LLM_NAME`, then run `python bin/run_services.py`
- External OpenCode service: set `ARXIV_SANITY_OPENCODE_MANAGED=false`, point `ARXIV_SANITY_OPENCODE_BASE_URL` at that service, then run `python bin/run_services.py`
- Automated pipeline: additionally start `python -m tools daemon` or use `python bin/run_services.py --with-daemon`
- MinerU API parsing: set `ARXIV_SANITY_MINERU_ENABLED=true`, `ARXIV_SANITY_MINERU_BACKEND=api`, `ARXIV_SANITY_MINERU_API_KEY=...`
- Remote embeddings: set `ARXIV_SANITY_EMBED_USE_LLM_API=true` and provide `ARXIV_SANITY_EMBED_API_BASE`

## Notes

- `config/llm.yml` and `config/llm_template.yml` have been removed. LiteLLM is no longer a dependency.
- `python bin/run_services.py` launches [OpenCode](https://opencode.ai) locally by default; set `ARXIV_SANITY_OPENCODE_MANAGED=false` to opt out.
- `/ready` validates OpenCode availability and required canonical model ids.
- OpenCode configuration (providers, API keys, model routing) is managed by OpenCode itself — see <https://opencode.ai> for details.
