---
name: arxiv-sanity-legacy-core
description: 维护 backend/legacy.py 这条历史中枢、兼容层与业务编排层的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Legacy Core Skill

## When to use me

- 你发现某个接口明明改了 blueprint 或 service，但真实行为没变
- 你要改首页、summary 页面、reading list、标签、用户态或 SSE API
- 你要继续推进“从 legacy 迁出到 services”的工作

## Scope

- Covers: `backend/legacy.py`, `backend/core.py`
- Also touches: `backend/services/*.py`, `backend/blueprints/*.py`, `templates/*.html`, `tasks.py`, `aslite/repositories.py`
- Does not cover: 某个单一业务域的深度实现；读完本 skill 后跳到具体业务 skill

## Start here

1. `backend/core.py` 先确认它只是 `backend/legacy.py` 的 re-export 入口，不要在这里找独立运行时逻辑
2. `backend/legacy.py` 顶部的 imports、全局 helper、`_parse_api_request()`
3. `backend/legacy.py` 里的 `default_context()`、`main()`、`summary()`
4. `backend/legacy.py` 里的 summary API、reading list API、tag API、user API、SSE API
5. 对照对应 service，看它到底是 wrapper 还是独立实现

## Function map

| Domain                 | Where to start                                                                                | Next skill                      |
| ---------------------- | --------------------------------------------------------------------------------------------- | ------------------------------- |
| 首页 / 搜索            | `main()`, 搜索 API, `default_context()`                                                       | `arxiv-sanity-search-ranking`   |
| 摘要                   | `summary()`, `api_get_paper_summary()`, `api_trigger_paper_summary()`, `api_summary_status()` | `arxiv-sanity-summary-pipeline` |
| 用户 / 标签 / 阅读列表 | user API, tag API, reading list API                                                           | `arxiv-sanity-user-state-tags`  |
| SSE                    | `api_user_stream()`, `api_sse_stats()`                                                        | `arxiv-sanity-sse-bus`          |
| 上传关联兼容逻辑       | upload pid owner-check, upload summary helpers                                                | `arxiv-sanity-upload-system`    |

## Core mental model

```text
blueprint
  -> legacy handler
     -> parse request / permission / compatibility behavior
     -> call services or repositories or tasks
     -> render template or return json
```

- `backend/legacy.py` 不是可以随便删掉的死代码，它仍然是事实上的编排中心。
- 它同时承担三种角色：兼容层、页面层、部分业务层。
- 很多 service 已经存在，但真正的入口语义仍定义在这里。

## Design decisions

- `_parse_api_request()` 承担了 login、CSRF、machine auth、paper existence 的旧兼容语义。
- `_parse_api_request()` 现在既负责更严格的 JSON/type 校验，也负责保留 upload pid 的原始语义，不能只把它当薄薄的 request wrapper。
- `legacy.before_request()` 仍是请求入口的一部分，但真正的 request bootstrap / background startup 已部分迁到 `backend/services/user_service.py`。
- summary 相关接口刻意把“读缓存”和“触发生成”分开，避免页面自动生成摘要。
- 上传论文是私有资源，很多路径都重复做 owner-check，这是故意的边界，不要随手删。

## Runbook

### 改一个 legacy handler

1. 先确认它是否只是纯 wrapper
2. 如果不是，画出它涉及的 service/repository/template/task
3. 改完后检查相应 blueprint、前端模板注入和测试是否一起更新

### 迁出 legacy 逻辑到 service

1. 保留 HTTP 语义在入口层，先抽业务计算和状态机
2. 不要一次性大搬家；优先抽无 request context 的部分
3. 每次迁移都补对应回归测试，避免“路径变了、行为漂了”

### 调摘要或阅读列表接口

1. 同时看 `backend/legacy.py`、`backend/services/readinglist_service.py`、`backend/services/summary_service.py`、`tasks.py`
2. 确认 cache、DB 状态、task 状态和 SSE 事件都还对齐
3. 如果是 upload pid，先查 parse 状态和 owner；summary status 可能会先返回 parse 状态，活动 task_id 也可能因为权限被故意隐藏

## Gotchas

- `api_get_paper_summary()` 默认是 cache-only；真正入队生成走 `api_trigger_paper_summary()`。
- `backend/core.py` 是 `backend/legacy.py` 的纯 re-export；如果你从 IDE 跳转到 `core.py`，记得回到 `legacy.py` 看真实实现。
- 活动摘要任务如果不归当前用户所有，legacy 可能返回“有任务但不给 task_id”的隐藏语义，不能把 `task_id=None` 直接等同于“没有任务”。
- owner-only 的 `task_id`、`last_error` 等字段不要泄露给其他用户。
- 首页 `main()` 的 `rank/q/search_mode/time_filter` 契约已经被前端依赖，不要只改单一路径。
- upload pid 如果长得像版本化 arXiv id，也必须保留原始字符串；不要在 legacy 入口里把它误拆成普通 pid + version。
- upload pid 的存在性、权限和 ready 语义很容易混在一起，改这里要格外小心。
- 看起来重复的 helper 可能承载旧接口兼容，不要只因“重构洁癖”直接删。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/integration/test_app.py tests/integration/test_homepage_interactions.py tests/integration/test_api_search.py tests/integration/test_api_summary.py tests/integration/test_api_tags.py tests/integration/test_api_readinglist.py tests/integration/test_api_user.py tests/integration/test_api_sse.py -q
pytest tests/unit/test_api_helpers_parse_api_request.py tests/unit/test_frontend_backend_contract.py tests/unit/test_template_route_contract.py -q
```

## Related skills

- `arxiv-sanity-routing-layer`
- `arxiv-sanity-search-ranking`
- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-testing`
- `arxiv-sanity-user-state-tags`
- `arxiv-sanity-sse-bus`
