---
name: arxiv-sanity-runtime-entry
description: 维护 Web 入口、Flask app factory、页面探针与全局运行时 hook 的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Runtime Entry Skill

## When to use me

- 你要改 `serve.py`、`backend/app.py`、`backend/blueprints/web.py`、`backend/blueprints/metrics.py` 或 `config/sentry.py`
- 你要排查启动失败、首请求异常、`/health` 或 `/ready` 不符合预期
- 你要改全局 hook、Jinja 注入、安全头、SSE runtime 启动时机

## Scope

- Covers: `serve.py`, `backend/app.py`, `backend/blueprints/web.py`, `backend/blueprints/metrics.py`, `backend/services/background.py`, `backend/services/health_service.py`, `backend/utils/memory_limit.py`, `config/sentry.py`
- Also touches: `backend/services/user_service.py`, `backend/utils/manifest.py`, `backend/utils/validation.py`, `backend/utils/sse.py`, `bin/up.sh`, `bin/run_services.py`
- Does not cover: 具体业务域实现；搜索、摘要、上传、SSE 细节分别看对应 skill

## Start here

1. `docs/INDEX.md`, `docs/DEVELOPMENT.md`, `docs/OPERATIONS.md`
2. `serve.py`
3. `backend/app.py`
4. `backend/blueprints/web.py`
5. `backend/services/user_service.py` 的 `before_request()` / `close_connection()`
6. `backend/services/background.py`, `backend/services/health_service.py`
7. `backend/blueprints/metrics.py`, `config/sentry.py`, `backend/utils/memory_limit.py`

## Core mental model

```text
serve.py
  -> set ARXIV_SANITY_PROCESS_ROLE=web
  -> optional gevent monkey patch
  -> optional memory limit / preload
  -> backend.app:create_app()
      -> register blueprints
      -> add security headers / cache policy / jinja helpers
      -> wire legacy before_request + teardown
      -> legacy.before_request() -> user_service.before_request()
      -> lazy start SSE runtime / background helpers on request path
```

- `serve.py` 不是纯薄壳，import 顺序会影响 gevent 和 preload 行为。
- `backend/app.py` 是全局语义中心，很多“看起来像某个页面的问题”其实是这里的 hook 或上下文注入引起的。
- `backend/blueprints/web.py` 只有 `/health`、`/ready` 是真实逻辑，其余大多转发到 `backend/legacy.py`。

## Design decisions

- `/ready` 比 `/health` 更严格，`bin/run_services.py` 等待的是 `/ready`，不是 `/health`。
- `/ready` 不只看依赖是否可达，还会校验主模型与 fallback model 是否满足当前 readiness 语义。
- SSE runtime 和 background service 都偏向惰性启动，以兼容 gunicorn preload 和多 worker 场景。
- HTML 响应默认 `no-store`，而 `static/dist/` 走 hashed asset + immutable cache；两套缓存策略不能混。
- memory limit 是运行时保护层的一部分，`serve.py` 和 `bin/huey_consumer.py` 都可能依赖它。

## Runbook

### 改启动入口

1. 先检查 `serve.py` 是否仍在早期设置 `ARXIV_SANITY_PROCESS_ROLE=web`
2. 若涉及 gevent，确认 monkey patch 仍早于会创建线程/锁/网络连接的导入
3. 用 `python serve.py` 和 `python bin/run_services.py --no-embed --no-mineru --no-litellm` 都走一遍

### 改 `/health` 或 `/ready`

1. 同时读 `backend/blueprints/web.py` 与 `backend/services/health_service.py`
2. 确认 launcher 依赖的严格 ready 语义没被稀释，包括 LLM fallback model 校验
3. 若新增探针，决定它是 strict ready 还是 degraded health

### 改 metrics / sentry / 全局 hook

1. 同时看 `backend/app.py`、`backend/blueprints/metrics.py`、`config/sentry.py`
2. 保持 import-time 无副作用；Sentry 和 metrics 失败不能拖垮正常请求
3. 补运行时探针与初始化测试

### 改全局模板注入或静态资源逻辑

1. 同时看 `backend/app.py`、`backend/utils/manifest.py`、`templates/base.html`
2. 构建前端产物：`npm run build:static`
3. 跑模板契约测试和 manifest 契约测试

## Gotchas

- 不要把有副作用的线程或连接启动搬到 import 期；这会和 preload 冲突。
- 未配置 secret key 时 session 会在重启后全部失效，这是当前设计，不是随机 bug。
- `/ready` 的依赖要求可能被 `bin/run_services.py` 的 `ARXIV_SANITY_READY_REQUIRE_EMBEDDING` / `ARXIV_SANITY_READY_REQUIRE_MINERU` 等环境变量覆盖，排障时不能只看 `.env`。
- background service 的真实挂载点在 `legacy.before_request()` -> `user_service.before_request()`，不要只在 `app.py` 里盲找。
- 改 `backend/app.py` 的 security headers 时，不要误伤 `csrf-token`、`nosniff`、静态资源缓存或 HTML no-store。
- API 版 `HTTPException` 包装如果改成 JSON，必须保留原始 response headers（例如 405 的 `Allow`），不要只返回新的 `(dict, status)` tuple。

## Validation

```bash
conda activate sanity
npm run build:static
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_run_services.py tests/unit/test_health_llm_fallback.py tests/unit/test_background_service.py tests/unit/test_metrics_endpoint.py tests/unit/test_sentry_init.py tests/unit/test_manifest.py tests/unit/test_manifest_fallback.py tests/unit/test_static_font_mimetypes.py tests/integration/test_app.py -q
python -m config.cli validate
python bin/run_services.py --no-embed --no-mineru --no-litellm
```

## Related skills

- `arxiv-sanity-routing-layer`
- `arxiv-sanity-legacy-core`
- `arxiv-sanity-ops-config`
- `arxiv-sanity-sse-bus`
- `arxiv-sanity-testing`
