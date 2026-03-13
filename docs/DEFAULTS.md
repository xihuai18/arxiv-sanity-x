# Default Configuration Reference

This document summarizes the current configuration defaults defined in code.

- Canonical source of truth: `config/settings_main.py`, `config/settings_services.py`, `config/settings_runtime.py`, `config/settings_features.py`
- Operator-facing template: `.env.example`
- Recommended baseline for real usage: `docs/CONFIGURATION.md`

Important distinction:

- `Code defaults`: what the app uses when you do not set a variable
- `Recommended baseline`: what we suggest you set explicitly for a normal local or small-team deployment

Notes:

- `ARXIV_SANITY_LLM_NAME=gpt-5.4` is the code default for the main summary model.
- `ARXIV_SANITY_EXTRACT_MODEL_NAME=qwen3.5-plus` is the code default for uploaded-paper metadata extraction.
- If `ARXIV_SANITY_EXTRACT_BASE_URL` is empty, extraction reuses the main LLM endpoint; make sure that endpoint can route `qwen3.5-plus`, or override the extract model explicitly.

## Root Defaults

| Variable                              | Default                  |
| ------------------------------------- | ------------------------ |
| `ARXIV_SANITY_DATA_DIR`               | `data`                   |
| `ARXIV_SANITY_SUMMARY_DIR`            | `<data_dir>/summary`     |
| `ARXIV_SANITY_LOG_DIR`                | `<data_dir>/logs`        |
| `ARXIV_SANITY_HOST`                   | `http://localhost:55555` |
| `ARXIV_SANITY_SERVE_PORT`             | `55555`                  |
| `ARXIV_SANITY_LITELLM_PORT`           | `53000`                  |
| `ARXIV_SANITY_PROCESS_ROLE`           | ``                       |
| `ARXIV_SANITY_LOG_LEVEL`              | `WARNING`                |
| `ARXIV_SANITY_LOG_FORMAT`             | `text`                   |
| `ARXIV_SANITY_ENABLE_SWAGGER`         | `false`                  |
| `ARXIV_SANITY_MAIN_CONTENT_MIN_RATIO` | `0.1`                    |

## LLM / Extract / Summary

| Variable                                       | Default                  |
| ---------------------------------------------- | ------------------------ |
| `ARXIV_SANITY_LLM_BASE_URL`                    | `http://localhost:53000` |
| `ARXIV_SANITY_LLM_API_KEY`                     | `no-key`                 |
| `ARXIV_SANITY_LLM_NAME`                        | `gpt-5.4`                |
| `ARXIV_SANITY_LLM_SUMMARY_LANG`                | `zh`                     |
| `ARXIV_SANITY_LLM_FALLBACK_MODELS`             | `auto`                   |
| `ARXIV_SANITY_LLM_TIMEOUT`                     | `600`                    |
| `ARXIV_SANITY_LLM_LITELLM_VERBOSE`             | `false`                  |
| `ARXIV_SANITY_EXTRACT_MODEL_NAME`              | `qwen3.5-plus`           |
| `ARXIV_SANITY_EXTRACT_BASE_URL`                | ``                       |
| `ARXIV_SANITY_EXTRACT_API_KEY`                 | ``                       |
| `ARXIV_SANITY_EXTRACT_TEMPERATURE`             | `0.1`                    |
| `ARXIV_SANITY_EXTRACT_MAX_TOKENS`              | `8192`                   |
| `ARXIV_SANITY_EXTRACT_TIMEOUT`                 | `600`                    |
| `ARXIV_SANITY_SUMMARY_MIN_CHINESE_RATIO`       | `0.25`                   |
| `ARXIV_SANITY_SUMMARY_DEFAULT_SEMANTIC_WEIGHT` | `0.5`                    |
| `ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE`         | `html`                   |
| `ARXIV_SANITY_SUMMARY_HTML_SOURCES`            | `ar5iv,arxiv`            |
| `ARXIV_SANITY_SUMMARY_BATCH_NUM`               | `500`                    |
| `ARXIV_SANITY_SUMMARY_FORCE_CACHE_ONLY`        | `true`                   |

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
| `ARXIV_SANITY_EMAIL_USERNAME`           | ``                     |
| `ARXIV_SANITY_EMAIL_PASSWORD`           | ``                     |
| `ARXIV_SANITY_EMAIL_API_WORKERS`        | `8`                    |

## Daemon / Huey / SSE / Gunicorn

| Variable                                         | Default                    |
| ------------------------------------------------ | -------------------------- |
| `ARXIV_SANITY_DAEMON_FETCH_NUM`                  | `2000`                     |
| `ARXIV_SANITY_DAEMON_FETCH_MAX`                  | `1000`                     |
| `ARXIV_SANITY_DAEMON_SUMMARY_NUM`                | `250`                      |
| `ARXIV_SANITY_DAEMON_SUMMARY_WORKERS`            | `2`                        |
| `ARXIV_SANITY_DAEMON_ENABLE_SUMMARY`             | `true`                     |
| `ARXIV_SANITY_DAEMON_ENABLE_EMBEDDINGS`          | `true`                     |
| `ARXIV_SANITY_DAEMON_ENABLE_PRIORITY_QUEUE`      | `true`                     |
| `ARXIV_SANITY_DAEMON_ENABLE_SUMMARY_QUEUE`       | `true`                     |
| `ARXIV_SANITY_DAEMON_PRIORITY_DAYS`              | `2.0`                      |
| `ARXIV_SANITY_DAEMON_PRIORITY_LIMIT`             | `200`                      |
| `ARXIV_SANITY_DAEMON_EMAIL_DRY_RUN`              | `false`                    |
| `ARXIV_SANITY_DAEMON_ENABLE_GIT_BACKUP`          | `true`                     |
| `ARXIV_SANITY_DAEMON_BACKUP_REPO_DIR`            | `data-repo`                |
| `ARXIV_SANITY_DAEMON_BACKUP_PUSH`                | `true`                     |
| `ARXIV_SANITY_DAEMON_BACKUP_PUSH_REMOTE`         | ``                         |
| `ARXIV_SANITY_DAEMON_BACKUP_PUSH_BRANCH`         | ``                         |
| `ARXIV_SANITY_DAEMON_BACKUP_PUSH_RETRIES`        | `3`                        |
| `ARXIV_SANITY_DAEMON_BACKUP_GIT_USER_NAME`       | `arxiv-sanity-daemon`      |
| `ARXIV_SANITY_DAEMON_BACKUP_GIT_USER_EMAIL`      | `daemon@localhost`         |
| `ARXIV_SANITY_DAEMON_SUBPROCESS_TIMEOUT_S`       | `7200`                     |
| `ARXIV_SANITY_DAEMON_WEB_CACHE_WARMUP_ON_UPDATE` | `true`                     |
| `ARXIV_SANITY_DAEMON_TIMEZONE`                   | `Asia/Shanghai`            |
| `ARXIV_SANITY_HUEY_DB_PATH`                      | `<data_dir>/huey.db`       |
| `ARXIV_SANITY_HUEY_WORKERS`                      | `4`                        |
| `ARXIV_SANITY_HUEY_WORKER_TYPE`                  | `thread`                   |
| `ARXIV_SANITY_HUEY_MAX_MEMORY_MB`                | `0`                        |
| `ARXIV_SANITY_HUEY_SUMMARY_PRIORITY_HIGH`        | `200`                      |
| `ARXIV_SANITY_HUEY_SUMMARY_PRIORITY_LOW`         | `10`                       |
| `ARXIV_SANITY_HUEY_SUMMARY_REPAIR_ON_START`      | `true`                     |
| `ARXIV_SANITY_HUEY_SUMMARY_REPAIR_TTL`           | `3600`                     |
| `ARXIV_SANITY_HUEY_SUMMARY_REPAIR_REQUEUE`       | `false`                    |
| `ARXIV_SANITY_HUEY_SUMMARY_REPAIR_ENABLE`        | `true`                     |
| `ARXIV_SANITY_HUEY_SUMMARY_REPAIR_INTERVAL`      | `900`                      |
| `ARXIV_SANITY_HUEY_FORCE_REPAIR`                 | `false`                    |
| `ARXIV_SANITY_HUEY_TASKS_SSE_ENABLED`            | `true`                     |
| `ARXIV_SANITY_HUEY_SQLITE_TIMEOUT_WEB`           | `2.0`                      |
| `ARXIV_SANITY_HUEY_SQLITE_TIMEOUT_WORKER`        | `10.0`                     |
| `ARXIV_SANITY_HUEY_ALLOW_THREAD_FALLBACK`        | `false`                    |
| `ARXIV_SANITY_SSE_ENABLED`                       | `true`                     |
| `ARXIV_SANITY_SSE_DB_PATH`                       | `<data_dir>/sse_events.db` |
| `ARXIV_SANITY_SSE_POLL_INTERVAL`                 | `0.05`                     |
| `ARXIV_SANITY_SSE_BATCH_SIZE`                    | `500`                      |
| `ARXIV_SANITY_SSE_RETENTION_SECONDS`             | `86400`                    |
| `ARXIV_SANITY_SSE_CLEANUP_INTERVAL`              | `60.0`                     |
| `ARXIV_SANITY_SSE_QUEUE_MAXSIZE`                 | `200`                      |
| `ARXIV_SANITY_SSE_MAX_CONNECTIONS_PER_USER`      | `2`                        |
| `ARXIV_SANITY_SSE_CONNECTION_LEASE_TTL_S`        | `90.0`                     |
| `ARXIV_SANITY_SSE_STRICT_WORKER_CLASS`           | `false`                    |
| `ARXIV_SANITY_SSE_PUBLISH_RETRY_QUEUE_MAXSIZE`   | `2000`                     |
| `ARXIV_SANITY_SSE_PUBLISH_RETRY_BACKOFF_MAX_S`   | `1.0`                      |
| `ARXIV_SANITY_SSE_PUBLISH_ASYNC`                 | `true`                     |
| `ARXIV_SANITY_GUNICORN_WORKERS`                  | `2`                        |
| `ARXIV_SANITY_GUNICORN_THREADS`                  | `2`                        |
| `ARXIV_SANITY_GUNICORN_PRELOAD`                  | `true`                     |
| `ARXIV_SANITY_GUNICORN_PRELOAD_CACHES`           | `true`                     |
| `ARXIV_SANITY_GUNICORN_WORKER_CLASS`             | ``                         |
| `ARXIV_SANITY_GUNICORN_FORCE_WORKERS`            | `false`                    |
| `ARXIV_SANITY_GUNICORN_EXTRA_ARGS`               | ``                         |
| `ARXIV_SANITY_GUNICORN_MAX_MEMORY_MB`            | `0`                        |

## Web / Lock / DB

| Variable                                           | Default                        |
| -------------------------------------------------- | ------------------------------ |
| `ARXIV_SANITY_CACHE_PAPERS`                        | `false`                        |
| `ARXIV_SANITY_WARMUP_DATA`                         | `true`                         |
| `ARXIV_SANITY_WARMUP_ML`                           | `true`                         |
| `ARXIV_SANITY_ENABLE_SCHEDULER`                    | `true`                         |
| `ARXIV_SANITY_RELOAD`                              | `false`                        |
| `ARXIV_SANITY_ACCESS_LOG`                          | `false`                        |
| `ARXIV_SANITY_SECRET_KEY`                          | ``                             |
| `ARXIV_SANITY_COOKIE_SAMESITE`                     | `Lax`                          |
| `ARXIV_SANITY_COOKIE_SECURE`                       | `false`                        |
| `ARXIV_SANITY_MAX_CONTENT_LENGTH`                  | `52428800`                     |
| `ARXIV_SANITY_SUMMARY_CACHE_STATS_REFRESH`         | `1800`                         |
| `ARXIV_SANITY_DATA_CACHE_REFRESH_MIN_INTERVAL`     | `60`                           |
| `ARXIV_SANITY_FEATURES_CACHE_REFRESH_MIN_INTERVAL` | `300`                          |
| `ARXIV_SANITY_ENABLE_CACHE_STATUS`                 | `false`                        |
| `ARXIV_SANITY_READY_REQUIRE_EMBEDDING`             | `true`                         |
| `ARXIV_SANITY_READY_REQUIRE_MINERU`                | `true`                         |
| `ARXIV_SANITY_ENABLE_METRICS`                      | `false`                        |
| `ARXIV_SANITY_METRICS_KEY`                         | ``                             |
| `ARXIV_SANITY_ASSET_CDN_ENABLED`                   | `true`                         |
| `ARXIV_SANITY_ASSET_NPM_CDN_BASE`                  | `https://cdn.jsdelivr.net/npm` |
| `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC`              | `3600`                         |
| `ARXIV_SANITY_MINERU_LOCK_STALE_SEC`               | `3600`                         |
| `ARXIV_SANITY_DB_TIMEOUT`                          | `120`                          |
| `ARXIV_SANITY_DB_MAX_RETRIES`                      | `5`                            |
| `ARXIV_SANITY_DB_RETRY_BASE_SLEEP`                 | `0.2`                          |
| `ARXIV_SANITY_DB_TIMEOUT_WEB`                      | `2`                            |
| `ARXIV_SANITY_DB_TIMEOUT_WORKER`                   | `120`                          |
| `ARXIV_SANITY_DB_MAX_RETRIES_WEB`                  | `3`                            |
| `ARXIV_SANITY_DB_MAX_RETRIES_WORKER`               | `5`                            |

## Search / Recommendation / Sentry / arXiv

| Variable                                     | Default                         |
| -------------------------------------------- | ------------------------------- |
| `ARXIV_SANITY_SEARCH_RET_NUM`                | `100`                           |
| `ARXIV_SANITY_SEARCH_MAX_RESULTS`            | `1000`                          |
| `ARXIV_SANITY_SEARCH_DISABLE_FULLSCAN`       | `false`                         |
| `ARXIV_SANITY_SEARCH_SEMANTIC_DISABLED`      | `false`                         |
| `ARXIV_SANITY_RECO_API_BASE_URL`             | `http://localhost:<serve_port>` |
| `ARXIV_SANITY_RECO_API_KEY`                  | ``                              |
| `ARXIV_SANITY_RECO_API_TIMEOUT`              | `45.0`                          |
| `ARXIV_SANITY_RECO_API_LIMIT`                | `1000`                          |
| `ARXIV_SANITY_RECO_MODEL_C`                  | `0.1`                           |
| `ARXIV_SANITY_RECO_NUM_THREADS`              | `0`                             |
| `ARXIV_SANITY_RECO_MAX_THREADS`              | `192`                           |
| `ARXIV_SANITY_RECO_WEB_NAME`                 | `Arxiv Sanity X`                |
| `ARXIV_SANITY_SENTRY_ENABLED`                | `false`                         |
| `ARXIV_SANITY_SENTRY_DSN`                    | ``                              |
| `ARXIV_SANITY_SENTRY_ENVIRONMENT`            | ``                              |
| `ARXIV_SANITY_SENTRY_RELEASE`                | ``                              |
| `ARXIV_SANITY_SENTRY_TRACES_SAMPLE_RATE`     | `0.0`                           |
| `ARXIV_SANITY_SENTRY_PROFILES_SAMPLE_RATE`   | `0.0`                           |
| `ARXIV_SANITY_ARXIV_CORE_TAGS`               | `cs.AI,cs.LG,stat.ML`           |
| `ARXIV_SANITY_ARXIV_LANG_TAGS`               | `cs.CL,cs.IR,cs.CV`             |
| `ARXIV_SANITY_ARXIV_AGENT_TAGS`              | `cs.MA,cs.RO,cs.HC,cs.GT,cs.NE` |
| `ARXIV_SANITY_ARXIV_APP_TAGS`                | `cs.SE,cs.CY`                   |
| `ARXIV_SANITY_ARXIV_EMPTY_RESPONSE_FALLBACK` | `3`                             |
| `ARXIV_SANITY_ARXIV_API_TIMEOUT`             | `30`                            |
| `ARXIV_SANITY_SVM_C`                         | `0.02`                          |
| `ARXIV_SANITY_SVM_MAX_ITER`                  | `5000`                          |
| `ARXIV_SANITY_SVM_TOL`                       | `0.001`                         |
| `ARXIV_SANITY_SVM_NEG_WEIGHT`                | `5.0`                           |

## Recommended Baseline

Use `docs/CONFIGURATION.md` and `.env.example` for a practical baseline.

For most local deployments, explicitly set at least:

- `ARXIV_SANITY_LLM_BASE_URL`
- `ARXIV_SANITY_LLM_API_KEY`
- `ARXIV_SANITY_LLM_NAME`
- `ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE=html`
- `ARXIV_SANITY_EMBED_USE_LLM_API=false`
- `ARXIV_SANITY_MINERU_ENABLED=false`
- `ARXIV_SANITY_HUEY_WORKERS=4`
- `ARXIV_SANITY_DAEMON_ENABLE_SUMMARY=true`

If you want automated fetch/compute/summary/email, you still need to explicitly start the daemon:

- `python -m tools daemon`
- or `python bin/run_services.py --with-daemon`
