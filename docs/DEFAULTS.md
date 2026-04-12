# Default Configuration Reference

This document summarizes the current code defaults.

## Root Defaults

| Variable                    | Default                  |
| --------------------------- | ------------------------ |
| `ARXIV_SANITY_DATA_DIR`     | `data`                   |
| `ARXIV_SANITY_SUMMARY_DIR`  | `<data_dir>/summary`     |
| `ARXIV_SANITY_LOG_DIR`      | `<data_dir>/logs`        |
| `ARXIV_SANITY_HOST`         | `http://localhost:55555` |
| `ARXIV_SANITY_SERVE_PORT`   | `55555`                  |
| `ARXIV_SANITY_PROCESS_ROLE` | ``                       |
| `ARXIV_SANITY_LOG_LEVEL`    | `WARNING`                |
| `ARXIV_SANITY_LOG_FORMAT`   | `text`                   |

## OpenCode / Text Models

| Variable                          | Default                                 |
| --------------------------------- | --------------------------------------- |
| `ARXIV_SANITY_OPENCODE_BASE_URL`  | ``                                      |
| `ARXIV_SANITY_OPENCODE_MANAGED`   | `true`                                  |
| `ARXIV_SANITY_OPENCODE_HOST`      | `127.0.0.1`                             |
| `ARXIV_SANITY_OPENCODE_PORT`      | `53000`                                 |
| `ARXIV_SANITY_OPENCODE_USERNAME`  | ``                                      |
| `ARXIV_SANITY_OPENCODE_PASSWORD`  | ``                                      |
| `ARXIV_SANITY_OPENCODE_TIMEOUT`   | `600`                                   |
| `ARXIV_SANITY_LLM_NAME`           | `openai/gpt-5.4`                        |
| `ARXIV_SANITY_LLM_SUMMARY_LANG`   | `zh`                                    |
| `ARXIV_SANITY_LLM_TIMEOUT`        | `600`                                   |
| `ARXIV_SANITY_EXTRACT_MODEL_NAME` | (falls back to `ARXIV_SANITY_LLM_NAME`) |
| `ARXIV_SANITY_EXTRACT_TIMEOUT`    | `600`                                   |

## Embedding / MinerU / Email

| Variable                                | Default                |
| --------------------------------------- | ---------------------- |
| `ARXIV_SANITY_EMBED_PORT`               | `54000`                |
| `ARXIV_SANITY_EMBED_USE_LLM_API`        | `false`                |
| `ARXIV_SANITY_EMBED_MODEL_NAME`         | `qwen3-embedding:0.6b` |
| `ARXIV_SANITY_EMBED_API_BASE`           | ``                     |
| `ARXIV_SANITY_EMBED_API_KEY`            | ``                     |
| `ARXIV_SANITY_MINERU_ENABLED`           | `false`                |
| `ARXIV_SANITY_MINERU_PORT`              | `52000`                |
| `ARXIV_SANITY_MINERU_BACKEND`           | `api`                  |
| `ARXIV_SANITY_MINERU_DEVICE`            | `cuda`                 |
| `ARXIV_SANITY_MINERU_MAX_WORKERS`       | `2`                    |
| `ARXIV_SANITY_MINERU_MAX_VRAM`          | `4`                    |
| `ARXIV_SANITY_MINERU_API_KEY`           | ``                     |
| `ARXIV_SANITY_MINERU_API_POLL_INTERVAL` | `5`                    |
| `ARXIV_SANITY_MINERU_API_TIMEOUT`       | `900`                  |
| `ARXIV_SANITY_EMAIL_FROM_EMAIL`         | ``                     |
| `ARXIV_SANITY_EMAIL_SMTP_SERVER`        | ``                     |
| `ARXIV_SANITY_EMAIL_SMTP_PORT`          | `465`                  |

## Notes

- Text generation uses [OpenCode](https://opencode.ai) as the sole LLM backend (`opencode serve`).
- Embeddings remain independent from the OpenCode text-model settings.
- See `.env.example` for the operator-facing template and `docs/CONFIGURATION.md` for setup guidance.
