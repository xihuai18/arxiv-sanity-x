---
name: arxiv-sanity-summary-pipeline
description: 维护摘要生成、缓存、任务状态、SSE 通知与模型切换全链路的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Summary Pipeline Skill

## When to use me

- 你要改摘要生成、模型切换、缓存清理、任务状态、前端 summary 页行为
- 你要排查摘要卡在 queued/running、明明有缓存却读不到、错误状态不一致
- 你要修改 `tasks.py` 里的摘要任务编排或 `tools/paper_summarizer.py`

## Scope

- Covers: `backend/services/summary_service.py`, `tasks.py` 里的摘要链, `tools/paper_summarizer.py`, `tools/cleanup_summary_model_cache.py`, `backend/blueprints/api_summary.py`, `backend/utils/summary_utils.py`, `backend/schemas/summary.py`
- Also touches: `backend/legacy.py`, `backend/services/readinglist_service.py`, `backend/utils/sse.py`, `templates/summary.html`, `static/paper_summary.js`
- Does not cover: 上传解析细节；上传摘要权限与文件生命周期看 upload skill

## Start here

1. `backend/legacy.py` 里的 `summary()`、`api_get_paper_summary()`、`api_trigger_paper_summary()`、`api_summary_status()`
2. `backend/services/summary_service.py`
3. `tasks.py`
4. `tools/paper_summarizer.py`
5. `templates/summary.html` + `static/paper_summary.js`

## Core mental model

```text
summary page
  -> get cached summary / status
  -> user clicks Generate
  -> enqueue Huey task or thread fallback
  -> paper_summarizer builds summary
  -> cache files + summary_status DB + task status + SSE
  -> frontend consumes status and fetches content
```

- 读取摘要和触发摘要是两条不同链路，不能混为一谈。
- 状态至少分四层：缓存文件、summary_status、task status、前端本地 pending 状态。
- reading list / summary 页会用 `SummaryStatusRepository` 覆盖本地旧字段，readinglist 表里缓存的状态不是最终真相。
- 这是全仓库最复杂的状态机，改一处通常要联动多处。

## Design decisions

- `/api/get_paper_summary` 维持 cache-only 语义，避免页面自动生成摘要。
- 取消任务靠 generation epoch + task state + cache/lock 协作，而不是单纯 revoke。
- `force_refresh` 会保存在 task status 中，但不会直接进 Huey payload；这是为了兼容仍在跑旧代码的 worker。
- resolved model / fallback model 兼容是为了平滑模型名调整后的旧缓存读取。
- `resolved_model` 已经是前后端合同字段，不只是缓存兼容细节；SSE、API 和 summary page 都会消费它。
- `llm_fallback_attempts` 已经是完整尝试轨迹，成功 fallback 也会落进去，不能再把它当成“失败列表”。

## Runbook

### 排查 queued/running 卡住

1. 查 `summary_status` 记录
2. 查 `huey.db` 和 worker 是否运行
3. 查 `data/summary/` 下的 lock 文件
4. 查 `/api/task_status/<id>` 和 `/api/queue_stats`
5. 必要时用 `python tests/validate_summary_flow.py --pid <pid> --trigger --poll`
6. 如果是 reading list / upload 列表在卡住，确认这些读路径有没有先走 `get_summary_status()` 或对应 repair helper，而不是直接读 repository 旧值

### 改摘要模型或 prompt

1. 同时看 `tools/paper_summarizer.py`、`backend/services/opencode_service.py`、`config/settings_services.py`
2. 如果模型名改变，检查 resolved model / fallback 读取逻辑
3. 确认前端模型列表和默认模型仍一致

### `paper_summarizer.py` 子系统导航

1. PDF 获取与版本解析
2. MinerU 解析与中间缓存
3. HTML fallback 与 markdown 转换
4. LLM 调用与 API fallback
5. 摘要缓存写入与锁文件
6. 统一入口和错误传播

### 清缓存或强制重生成

1. 不要只删文件；同时看 `summary_status`、task state、generation epoch
2. force refresh 需要先 cancel 旧任务，再 purge requested/resolved model cache，再入队新任务

### 排查 summary 页 fallback 刷新异常

1. 同时看 `legacy.api_summary_status()`、`static/paper_summary.js` 和 `tests/live/test_summary_frontend_behavior.py`
2. 确认页面没有把 `event.model` 与 `event.resolved_model` 混成单一字段
3. 确认 cache miss 不会被页面自动 enqueue，而是仍走手动 generate 语义

### 排查 upload pid 的摘要状态

1. 先查 upload record 的 owner 和 `parse_status`；`/api/summary_status` 在 parse 未完成时会优先返回 parse 状态，而不是 summary miss。
2. 再查 `SummaryStatusRepository` 里的 `task_user` / `task_id`，确认有没有 owner-only 字段被刻意隐藏。
3. 如果 reading list 卡片状态怪异，先怀疑 overlay 逻辑和陈旧本地字段，而不是直接怀疑前端渲染。

## Gotchas

- cache miss 不应自动触发生成，必须由用户显式触发。
- owner-only 的错误详情不能返回给其他用户。
- upload pid 在 parse 未完成、owner 不匹配或记录失效时，会主动清掉陈旧 `summary_status` / `task_id`，这是故意的防误导行为。
- `tasks.py` 里的摘要链责任很重，改动时不要只盯单个 helper。
- stale repair 不只发生在 worker/startup；`get_summary_status()` 和 `legacy.api_summary_status()` 的读路径本身也会纠偏。
- summary 页在 miss 时仍是手动生成模型，不应因为前端重构而悄悄变成自动 enqueue。
- thread fallback 受配置控制，测试和线上行为可能不同。
- resolved model 兼容如果被破坏，会出现“缓存存在但 UI 一直说 miss”。
- reading list overlay 不应单靠 `SummaryStatusRepository.get_status()` 判断是否仍在运行；陈旧任务修复、cache/lock reality 和 fallback cache 命中都在 `get_summary_status()`。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_summary_service.py tests/unit/test_summary_service_extended.py tests/unit/test_summary_service_cache_only_nonblocking.py tests/unit/test_tasks_summary_force_refresh.py tests/unit/test_trigger_summary_async_enqueue_failures.py tests/unit/test_tasks_summary_cache_resolved_model.py tests/unit/test_summary_cancellation.py tests/unit/test_tasks_summary_status_events.py tests/unit/test_summary_status_repository_dedup.py tests/unit/test_paper_summarizer_llm_api_selection.py tests/integration/test_api_summary.py -q
npm run build:static
pytest tests/unit/test_summary_page_manual_generate.py tests/unit/test_summary_init_flow_contract.py tests/unit/test_summary_mathjax_readiness_contract.py tests/unit/test_summary_toc_contract.py tests/unit/test_summary_render_surfaces_contract.py tests/unit/test_frontend_sse_wiring.py -q
pytest tests/live/test_summary_frontend_behavior.py -q  # when localhost service is running and summary SSE / fallback behavior changed
```

## Related skills

- `arxiv-sanity-legacy-core`
- `arxiv-sanity-sse-bus`
- `arxiv-sanity-frontend`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-upload-system`
- `arxiv-sanity-testing`
