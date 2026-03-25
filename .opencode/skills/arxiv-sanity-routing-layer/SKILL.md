---
name: arxiv-sanity-routing-layer
description: 维护 Flask blueprints、请求入口、schema 契约与路由分发边界的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Routing Layer Skill

## When to use me

- 你要新增或修改 `backend/blueprints/` 下的路由
- 你要改 API 请求/响应结构、Swagger 注释、HTTP 方法或 URL
- 你要确认某个接口真实逻辑到底在 blueprint、legacy 还是 service

## Scope

- Covers: `backend/blueprints/*.py`, `backend/schemas/*.py`
- Also touches: `backend/app.py`, `backend/legacy.py`, `backend/services/api_helpers.py`, `backend/utils/validation.py`
- Does not cover: 各业务域的内部实现细节；去看对应业务 skill

## Start here

1. `backend/app.py` 里的 `register_blueprint(...)`
2. `backend/blueprints/web.py`
3. `backend/blueprints/api_summary.py`
4. `backend/blueprints/api_search.py`
5. `backend/blueprints/api_papers.py`, `backend/blueprints/metrics.py`, `backend/blueprints/api_uploads.py`
6. `backend/services/api_helpers.py`, `backend/legacy.py`
7. `backend/schemas/*.py`

## Core mental model

```text
request
  -> blueprint function
     -> mostly backend.legacy.*
     -> sometimes backend.services.* directly
     -> sometimes local validation / schema parsing
  -> response json/html/sse
```

- 这个仓库的 blueprint 主要是“HTTP 入口整理层”，不是完整业务层。
- 最大例外是 `backend/blueprints/api_uploads.py`，它直接做较多真实逻辑。
- `backend/blueprints/web.py` 的 `/health` 和 `/ready`、`backend/blueprints/metrics.py`、`backend/blueprints/api_papers.py` 也都带有真实本地逻辑。

## Design decisions

- 绝大多数路由仍沿用 `legacy` 作为业务中枢，这是当前现实，不要假设 blueprint 已完全服务化。
- `backend/schemas/` 只覆盖一部分输入输出合同；旧接口里仍保留部分手写解析，尤其在 `legacy` 和 `api_uploads.py`。
- `_parse_api_request()` 现在会更积极地拒绝 malformed JSON 和错误类型输入；email / keyword / tag 这类字段不要再假设“传啥都能被字符串化”。
- metrics 是 process-local 视角，不是全局聚合。

## Runbook

### 新增一个 API

1. 先决定它是否应继续走 `legacy`，还是直接落 service
2. 在 `backend/blueprints/` 新增或修改 handler
3. 如需结构化请求/响应，补 `backend/schemas/`
4. 检查 `backend/app.py` 是否已注册 blueprint
5. 检查前端是否会调用该路径，并补契约测试

### 改现有 API 的权限或校验

1. 先查 blueprint 是否只是转发
2. 再查 `backend/legacy.py` 的 `_parse_api_request()` 或对应 service
3. 如果接口同时支持 form 和 JSON（如 `/login`、`/register_email`），确认响应模式和错误码仍分场景一致
4. 如涉及 mutation，确认 CSRF、login、owner-check、machine auth 是否仍符合预期
5. 如果返回字段会被页面脚本消费，补前后端契约测试而不只是 schema/unit 测试

### 改页面路由

1. 同时检查 `backend/blueprints/web.py`、`backend/legacy.py`、对应 `templates/*.html`
2. 若模板里有 `url_for` 或静态 `href/action`，跑 route contract 测试

## Gotchas

- 看到“模块化 blueprint”不要误以为业务也模块化；很多逻辑还在 `backend/legacy.py`。
- `backend/blueprints/__init__.py` 不是完整注册表，最终以 `backend/app.py` 为准。
- `api_papers.py` 和 `metrics.py` 虽然体量小，但都是前端或运维真实依赖的蓝图，不要因为“文件短”就跳过契约检查。
- 不同路由的前缀风格不完全统一，改路径时不要只凭肉眼猜。
- `POST /login`、`POST /register_email` 对 JSON 请求返回 JSON，对浏览器 form 仍可能 redirect；写脚本和写页面时别混淆。
- `POST /api/keyword_search` 的公开返回字段是 `pids` / `scores` / `total_count`，不是旧的 `papers` / `has_more`。
- `POST /api/tag_feedback` 对不存在 pid 返回 404；这是显式合同，不要再默默吞成 200/400。
- `api_uploads.py` 是风格例外，不能拿它推断其他蓝图都直接调 service。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/integration/test_app.py tests/integration/test_api_search.py tests/integration/test_api_summary.py tests/integration/test_api_tags.py tests/integration/test_api_readinglist.py tests/integration/test_api_uploads.py tests/integration/test_api_user.py tests/integration/test_api_sse.py tests/integration/test_api_papers.py -q
pytest tests/unit/test_schemas.py tests/unit/test_api_helpers.py tests/unit/test_api_helpers_parse_api_request.py tests/unit/test_frontend_backend_contract.py tests/unit/test_template_route_contract.py -q
```

## Related skills

- `arxiv-sanity-runtime-entry`
- `arxiv-sanity-legacy-core`
- `arxiv-sanity-frontend`
- `arxiv-sanity-testing`
- `arxiv-sanity-user-state-tags`
- `arxiv-sanity-upload-system`
