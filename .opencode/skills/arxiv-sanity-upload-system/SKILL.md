---
name: arxiv-sanity-upload-system
description: 维护 PDF 上传、解析、元数据抽取、相似论文检索与私有资源权限链的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Upload System Skill

## When to use me

- 你要改上传接口、文件校验、解析流程、抽取流程、相似论文或上传摘要联动
- 你要排查“上传成功但解析没跑”“删除后残留文件”“别人能探测到我的上传”
- 你要改 `backend/blueprints/api_uploads.py` 或 `backend/services/upload_service.py`

## Scope

- Covers: `backend/blueprints/api_uploads.py`, `backend/services/upload_service.py`, `backend/services/upload_similarity_service.py`, `backend/utils/upload_utils.py`, `backend/schemas/uploads.py`, `tasks.py` 里的上传任务部分
- Also touches: `tools/paper_summarizer.py`, `backend/utils/sse.py`, `backend/services/summary_service.py`, `templates/readinglist.html`, `static/readinglist.js`
- Does not cover: 普通 arXiv 论文的搜索和摘要基础设施；分别看 search/summary skill

## Start here

1. `backend/blueprints/api_uploads.py`
2. `backend/services/upload_service.py`
3. `backend/services/upload_similarity_service.py`
4. `backend/utils/upload_utils.py`
5. `tasks.py` 中 `process_uploaded_pdf_task()`、`parse_uploaded_pdf_task()`、`extract_info_task()`

## Core mental model

```text
upload file
  -> validate / quota / dedupe / owner check
  -> create uploaded_paper record + store original.pdf
  -> enqueue parse/extract/process task
  -> write parse/extract status + SSE
  -> optional metadata extraction / similarity / summary enqueue
```

- 上传链是状态机，不是简单 CRUD。
- 资源是私有的；存在性、可访问性、owner-check 必须一起考虑。
- reading list 页面同时承载了上传 UI，所以前端联动比看起来更重。
- `parse_task_id` / `extract_task_id` / `summary_task_id` 是活动任务指针，不是永久审计字段；终态后通常要清掉。

## Design decisions

- 上传蓝图直接调 service，是这个仓库里最“现代”的一条链。
- owner/no-leak 语义故意让未授权访问看起来像 404，避免泄露资源存在性。
- 解析、抽取、处理、摘要是可分步也可串联的状态流。
- upload task recovery 不是简单 stale check；还要区分活动 task、terminal/stale task、缺失指针恢复和短暂的 `pending_registration` 窗口。
- summary enqueue 失败会回写 summary status / reading list 状态，而不是只打日志；不补这条链，UI 会假装一切正常。

## Runbook

### 改上传约束或文件验证

1. 同时看 `api_uploads.py`、`upload_utils.py`、`upload_service.py`
2. 确认大小、MIME、PDF 签名、空文件等场景都被覆盖
3. 更新错误码或提示时同步前端 UI 文案

### 排查 parse/extract 卡住

1. 看 uploaded paper 记录里的状态和 task id
2. 看 Huey worker 和 `summary_status` / upload task status 写回
3. 看 SSE 是否发出 `upload_parse_status` / `upload_extract_status`
4. 看磁盘落点：`data/uploads/`, `data/mineru/`, `data/html_md/`
5. 读路径也要检查：`get_uploaded_papers_list()`、`/api/summary_status`、reading list overlay 是否会在展示前修复/隐藏陈旧 task pointer
6. 如果 record 上的 task id 缺失，别立刻判定失败；先看 `task::*` 里是否还能恢复 active task，或是否仍处在 enqueue 注册宽限窗内

### 删除或重试上传流程

1. 不要只删 PDF；要同时清 DB 记录、sha256 mapping、索引、任务状态和相关缓存
2. 失败重试要避免双入队
3. delete 是 two-phase 语义；清 status/task 记录时不要把 cooperative cancellation 还需要的 epoch marker 一起删掉

### 改上传后自动摘要

1. 只有 `parse_status=ok` 的 upload 才应该触发 summary；parse 还在 `queued/running/failed` 时要先处理解析状态。
2. 如果 enqueue summary 失败，要同步更新 summary status DB、reading list 状态和用户可见错误，而不是只留 worker 日志。
3. `/api/summary_status` 对 upload pid 会先暴露 parse 状态；不要把它实现成“缓存没命中就 not_found”。

## Gotchas

- 上传 pid 的权限语义和普通 arXiv pid 完全不同。
- 看起来像 `up_xxxv2` 的 upload pid 也要按原样保留，不能套普通 arXiv pid 的版本剥离逻辑。
- sha256 去重和 owner 绑定在一起，改去重策略时要先想清多用户语义。
- parse、extract、process 三种任务的状态字段不完全一样，别混用。
- `parse_task_id` / `extract_task_id` / `summary_task_id` 是“当前活动任务指针”；列表或状态接口应隐藏 terminal/stale id，不能把它们当长期展示字段。
- delete upload 时清理的是 summary/task/status/tag/reading list 等可重建状态，但不会顺手删除 generation epoch；那是故意保留给 cooperative cancellation 的。
- extract-info 现在会优先走专用 extract route，再按有效路由差异决定是否 fallback 到主 LLM，不再是永远独立的一条简化链。
- owner/no-leak 语义不只在 upload API；summary、image、MinerU image 等读路径也会刻意返回 404。
- 上传卡片的 TL;DR、summary、similarity 是分步补齐的，前端必须接受中间态。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_upload_utils.py tests/unit/test_upload_service_rollback.py tests/unit/test_upload_task_status_sse.py tests/unit/test_upload_similarity_service.py tests/unit/test_upload_service_extract_model_config.py tests/unit/test_tasks_upload_deleting.py tests/integration/test_api_uploads.py tests/integration/test_summary_upload_permissions.py -q
npm run build:static
pytest tests/integration/test_api_readinglist.py tests/unit/test_frontend_sse_wiring.py tests/unit/test_readinglist_ui_contract.py tests/unit/test_frontend_backend_contract.py -q
```

## Related skills

- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-sse-bus`
- `arxiv-sanity-user-state-tags`
- `arxiv-sanity-data-layer`
- `arxiv-sanity-routing-layer`
