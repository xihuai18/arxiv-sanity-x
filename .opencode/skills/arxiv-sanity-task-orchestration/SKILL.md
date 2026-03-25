---
name: arxiv-sanity-task-orchestration
description: 维护 tasks.py、Huey 任务生命周期、任务状态与跨域异步编排的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Task Orchestration Skill

## When to use me

- 你要改 `tasks.py`、Huey enqueue/cancel/retry/repair、task status 写回或 worker 语义
- 你要排查“queued/running 卡住”“任务被 supersede 后还在写回”“前端状态和 task 状态不一致”
- 你要理解 summary/upload 两条异步链在任务层的共性约束

## Scope

- Covers: `tasks.py`
- Also touches: `backend/services/summary_service.py`, `backend/services/upload_service.py`, `backend/utils/sse.py`, `backend/legacy.py`, `aslite/repositories.py`, `bin/huey_consumer.py`
- Does not cover: 具体摘要生成实现或上传解析细节本身；读完本 skill 后跳到对应 domain skill

## Start here

1. `tasks.py` 顶部的 Huey 初始化、`_update_task_status()`、`_update_summary_status_db()`、`_update_readinglist_summary_status()`
2. `tasks.py` 里的 `cancel_summary_tasks()`、`enqueue_summary_task()`、`repair_stale_summary_tasks()`、`cleanup_tasks()`
3. `tasks.py` 里的 `generate_summary_task()`、`process_uploaded_pdf_task()`、`parse_uploaded_pdf_task()`、`extract_info_task()`
4. `backend/services/summary_service.py` 与 `backend/services/upload_service.py`
5. `tests/unit/test_tasks_summary_force_refresh.py`, `tests/unit/test_summary_cancellation.py`, `tests/unit/test_upload_task_status_sse.py`, `tests/unit/test_tasks_upload_deleting.py`

## Core mental model

```text
web request / service
  -> enqueue task + persist task status
  -> worker checks current ownership / epoch / locks
  -> worker updates summary/upload state + SSE
  -> read paths and frontend reconcile task reality
```

- `tasks.py` 不是简单的 worker 函数集合，而是异步状态机的编排中心。
- 任务“真相”不只在 Huey queue，还分布在 `task::*` 记录、summary/upload 业务状态、cache/lock 文件和 SSE 事件里。
- summary 与 upload 共用很多任务语义：current-task ownership、stale repair、owner-scoped visibility、best-effort cancel。

## Design decisions

- cooperative cancellation 依赖 generation epoch + task status + cache/lock 协作，而不是只靠 Huey revoke。
- `force_refresh` 会持久化在 task status 里，但不会直接塞进 Huey payload；这是为了兼容仍在运行的旧 worker。
- upload 记录上的 `parse_task_id` / `extract_task_id` / `summary_task_id` 是活动任务指针，不是长期审计字段。
- 任务状态与 SSE payload 是前端合同的一部分；owner-scoped 字段可以故意隐藏，不能随手放宽。

## Runbook

### 排查 queued/running 卡住

1. 查 `SummaryStatusRepository` 里的 summary status 和 `task::*` 记录
2. 查 `huey.db`、worker 进程、`/api/queue_stats`、`/api/task_status/<id>`
3. 对 summary 再看 `data/summary/` lock 文件和 `repair_stale_summary_tasks()` 是否应接管
4. 对 upload 再看 upload record 上的 task pointer 是否已 stale、terminal、missing 或 `pending_registration`

### 改摘要任务编排

1. 同时看 `enqueue_summary_task()`、`cancel_summary_tasks()`、`generate_summary_task()`
2. 保持 `/api/get_paper_summary` cache-only 与 `/api/trigger_paper_summary` enqueue-only 的分离
3. 如果改 `force_refresh`，保留 cancel + purge requested/resolved cache + task-status compatibility 语义

### 改上传任务编排

1. 同时看 `process_uploaded_pdf_task()`、`parse_uploaded_pdf_task()`、`extract_info_task()`
2. 保持 current-task ownership 检查，避免 superseded worker 回写旧状态
3. 删除 upload 时不要把 cooperative cancellation 需要的 epoch marker 一起删掉

### 改任务状态对外可见性

1. 检查 owner-scoped `task_id` / `task_user` / `error` 是否仍只对正确用户可见
2. 检查 reading list、upload list、summary page 和 polling/SSE 消费是否仍能接受被隐藏或被修复后的任务字段

## Gotchas

- 不要轻易给 Huey payload 新增 kwargs；旧 worker 可能直接 crash。`force_refresh` 的兼容写法就是典型例子。
- summary 的 stale repair 既发生在 worker/startup 侧，也发生在 read path；不要只看一边。
- upload 任务指针短暂缺失不一定是失败，可能只是 enqueue 注册窗口内的 `pending_registration`。
- `cleanup_tasks()`、upload delete、summary clear 等清理路径都要谨慎处理 task status、业务状态、SSE 和 reading list overlay 的对齐。
- `task_id` 为空不一定代表没有活动任务，也可能是 owner-scoped 隐藏语义。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_tasks_summary_force_refresh.py tests/unit/test_summary_cancellation.py tests/unit/test_tasks_summary_status_events.py tests/unit/test_upload_task_status_sse.py tests/unit/test_tasks_upload_deleting.py tests/unit/test_trigger_summary_async_enqueue_failures.py tests/integration/test_api_summary.py tests/integration/test_api_uploads.py tests/integration/test_api_sse.py -q
```

## Related skills

- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-upload-system`
- `arxiv-sanity-sse-bus`
- `arxiv-sanity-data-layer`
- `arxiv-sanity-testing`
- `arxiv-sanity-legacy-core`
