---
name: arxiv-sanity-user-state-tags
description: 维护登录、用户态、标签、关键词、阅读列表与相关 SSE 通知的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity User State and Tags Skill

## When to use me

- 你要改登录/登出、Profile、邮箱注册、标签反馈、关键词、reading list
- 你要排查“UI 操作成功但用户状态没同步”“标签消失”“reading list 计数不对”
- 你要理解 `dict.db` 里和用户相关的状态是如何传播到 UI 的

## Scope

- Covers: `backend/services/auth_service.py`, `backend/services/user_service.py`, `backend/services/tag_service.py`, `backend/services/keyword_service.py`, `backend/services/readinglist_service.py`, `backend/blueprints/api_user.py`, `backend/blueprints/api_tags.py`, `backend/blueprints/api_readinglist.py`, `backend/schemas/tags.py`, `backend/schemas/readinglist.py`
- Also touches: `backend/legacy.py`, `templates/profile.html`, `templates/index.html`, `templates/readinglist.html`, `static/paper_list.js`, `static/readinglist.js`
- Does not cover: 上传私有资源生命周期和摘要生成内部细节

## Start here

1. `backend/services/auth_service.py`
2. `backend/services/user_service.py`
3. `backend/services/tag_service.py`
4. `backend/services/keyword_service.py`
5. `backend/services/readinglist_service.py`
6. 对应的 `backend/legacy.py` handler

## Core mental model

```text
session user
  -> tags / neg_tags / keywords / combined_tags / readinglist in dict.db
  -> user_service builds view state
  -> mutations emit SSE user events
  -> frontend updates dropdowns, badges, reading list, summary status
```

- 这是“用户反馈闭环”模块：从点击到持久化再到前端刷新，都在这条链上。
- 认证模型是轻量 session 用户名，不是强账号系统。
- reading list 与 summary 状态有耦合，改动时常要顺带看摘要链。

## Design decisions

- 无密码登录适合个人或内网部署；公开部署应依赖外层认证/VPN。
- `POST /login`、`POST /register_email` 保持“浏览器 form 走 redirect、JSON 调用走 JSON 返回”的双模态契约。
- `user_service` 对 tags / neg_tags / combined_tags / keys 有 request-level cache；同一请求里别假设每次读取都会重新查库。
- 标签、关键词、reading list 变更后会发 SSE，保证多页面状态同步。
- combined tag 的组件校验基于正标签和负标签并集，不是只看正标签。
- 部分状态默认只围绕默认摘要模型持久化，这是当前折中，而非完整多模型状态仓储。

## Runbook

### 改登录或用户态

1. 同时看 `auth_service.py`、`user_service.py`、`backend/legacy.py`、`templates/profile.html`
2. 确认 session、默认用户、邮箱注册和页面提示仍一致
3. 确认 `/api/user_state` 仍返回当前 `user`，以及 JSON 请求的错误/成功结构没有漂移

### 改标签或关键词行为

1. 同时看 `tag_service.py` / `keyword_service.py` 与 `static/paper_list.js`
2. 确认 SSE 事件仍发出
3. 注意 tag rename/delete 事件还可能携带 `deleted_ctags` / `renamed_ctags`
4. 若改返回结构，补前后端契约测试

### 改 reading list 逻辑

1. 同时看 `readinglist_service.py`、`aslite/repositories.py`、`static/readinglist.js`
2. 如果牵涉摘要按钮或状态，连同 `summary` skill 一起检查
3. upload pid 进入 reading list 时要先看 parse 状态；parse 未完成时应清掉陈旧 summary 状态而不是沿用旧 `task_id`
4. reading list overlay 如果要判断摘要是否仍在 `queued/running`，先走 `summary_service.get_summary_status()` 触发修复，再决定是否展示 `task_id`

## Gotchas

- `default_user` 和 session 用户的存在会影响未登录时的行为判定。
- machine auth、CSRF、普通 session auth 可能在不同入口并存，先查真实入口再改。
- `register_email` 的 JSON 请求校验比 form 更严格：无效 JSON、`null`、数组等都应返回 400，且不能污染已有邮箱。
- upload pid 如果长得像带版本号，也必须保留原始字符串，不能被普通 pid 规范化逻辑截断。
- reading list 主表和索引表都要更新，否则列表会漂。
- reading list 最终展示的是 `_overlay_summary_status_for_user()` 合成后的 owner-scoped 视图，不是 reading list 表里旧字段的直译。
- 不要只读 `SummaryStatusRepository.get_status()` 就把 reading list 卡片判成仍在运行；陈旧任务修复和缓存/lock 现实在 `get_summary_status()`。
- 标签 dropdown 和 reading list UI 都强依赖 SSE 事件；后端 mutation 成功但没发事件，前端会看起来“坏了”。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_auth_service.py tests/unit/test_auth_service_email_validation.py tests/unit/test_user_service.py tests/unit/test_tag_service.py tests/unit/test_keyword_service.py tests/unit/test_readinglist_service.py tests/unit/test_repositories_atomic.py tests/unit/test_repositories_rmw_guards.py tests/integration/test_api_user.py tests/integration/test_api_tags.py tests/integration/test_api_tag_management.py tests/integration/test_api_readinglist.py tests/integration/test_api_login.py tests/integration/test_api_auth.py tests/integration/test_concurrent_user_state.py tests/integration/test_concurrent_readinglist_index.py -q
npm run build:static
pytest tests/unit/test_frontend_backend_contract.py tests/unit/test_readinglist_ui_contract.py tests/unit/test_frontend_sse_wiring.py -q
```

## Related skills

- `arxiv-sanity-routing-layer`
- `arxiv-sanity-legacy-core`
- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-sse-bus`
- `arxiv-sanity-data-layer`
- `arxiv-sanity-frontend`
