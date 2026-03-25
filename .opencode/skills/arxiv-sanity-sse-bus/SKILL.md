---
name: arxiv-sanity-sse-bus
description: 维护 SQLite-backed SSE 事件总线、用户事件流与前后端实时同步协议的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity SSE Bus Skill

## When to use me

- 你要改 `backend/utils/sse.py`、`backend/utils/sse_bus.py`、`backend/blueprints/api_sse.py`
- 你要排查“后端状态变了但页面没实时刷新”“多 tab 重复连接”“事件丢失或重复消费”
- 你要新增一种用户事件类型或改事件 payload

## Scope

- Covers: `backend/utils/sse.py`, `backend/utils/sse_bus.py`, `backend/blueprints/api_sse.py`
- Also touches: `backend/legacy.py`, `static/common_utils.js`, `static/paper_list.js`, `static/paper_summary.js`, `static/readinglist.js`, `tasks.py`
- Does not cover: 某个具体业务状态机本身；这里只管事件传输和消费协议

## Start here

1. `backend/utils/sse_bus.py`
2. `backend/utils/sse.py`
3. `backend/blueprints/api_sse.py`
4. `static/common_utils.js` 里的 event stream 管理
5. `tests/unit/test_sse.py` 和 `tests/unit/test_sse_sqlite_ipc.py`

## Core mental model

```text
producer (web/task/upload service)
  -> local fanout + sqlite sse bus
  -> /api/user_stream EventSource
  -> common_utils.js leader tab + BroadcastChannel + polling fallback
  -> page-specific consumers filter by event type / model / pid
```

- 这不是 Redis，也不是单进程内存队列，而是单机 SQLite IPC。
- 前端不是每个 tab 都开独立连接，而是有 leader tab 复用和 BroadcastChannel 协作。
- `/api/sse_stats` 是 process-local 视角，不是全局集群统计。

## Design decisions

- 用 SQLite 是为了避免额外依赖，同时支持 web worker 和 Huey worker 跨进程通信。
- 前端同时保留 polling fallback，是为了让摘要和上传状态在 SSE 不稳定时仍可追踪。
- 即使 SQLite publish 失败，当前进程也可能通过 local fanout 把事件送达现有连接；跨进程传播失败和“本进程前端完全收不到”不是一回事。
- payload 中的 `model`、`pid`、`origin_pid` 等字段是去重和正确路由的关键。

## Runbook

### 新增事件类型

1. 在 producer 侧统一定义 payload 结构
2. 确认 `sse.py` 会发到正确 user channel
3. 在前端 consumer 中显式按 `event.type` 和必要的 `model/pid` 过滤
4. 补 unit + integration 测试

### 排查事件不达前端

1. 检查 `/api/user_stream` 是否建立成功
2. 检查 `/api/sse_stats` 只作为当前进程参考，不要误判成全局
3. 检查 gevent worker / SSE 开关 / per-user connection lease / leader tab 逻辑
4. 检查 polling fallback 是否在接管，以及 BroadcastChannel 是否还在转发 leader tab 事件

## Gotchas

- gevent worker class 对 SSE 很关键；SSE 开着却走非 gevent worker 会很难排。
- 事件重复或丢失常常不是后端没发，而是前端 leader tab / BroadcastChannel / filtering 出问题。
- summary 事件在列表页通常要按 model 过滤，但 summary 页还要理解 `resolved_model` 命中当前视图的情况。
- `/api/user_stream` 受 per-user connection lease 保护；429 不一定是 bug，也可能是 leader tab 策略在工作。
- SSE 与 polling fallback 是混合策略，不是“一次失败后永久切换”的单向降级。
- 如果只看进程内统计，很容易误判多 worker 行为。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_sse.py tests/unit/test_sse_sqlite_ipc.py tests/unit/test_tasks_summary_status_events.py tests/unit/test_upload_task_status_sse.py tests/unit/test_frontend_sse_wiring.py tests/integration/test_api_sse.py -q
npm run build:static
```

## Related skills

- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-upload-system`
- `arxiv-sanity-user-state-tags`
- `arxiv-sanity-frontend`
