---
name: arxiv-sanity-testing
description: 维护测试分层、回归命令、契约测试与复杂链路验证策略的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Testing Skill

## When to use me

- 你要决定一个改动至少该跑哪些测试
- 你要补回归测试，或排查 CI / 本地测试为什么失败
- 你要验证摘要、上传、SSE、前后端契约这种复杂链路

## Scope

- Covers: `tests/README.md`, `tests/conftest.py`, `tests/unit/`, `tests/integration/`, `tests/live/`, `tests/e2e/`, `tests/validate_summary_flow.py`
- Also touches: `.github/workflows/`, `docs/DEVELOPMENT.md`
- Does not cover: 某个业务模块内部逻辑；这里只回答“怎么验证它”

## Start here

1. `tests/README.md`
2. `tests/conftest.py`
3. `docs/DEVELOPMENT.md`
4. 按改动域跳到对应测试簇

## Core mental model

```text
unit         -> 逻辑与源码契约
integration  -> Flask test_client 接口协作
live         -> 真实服务探测
e2e          -> 真实服务器上的完整流
manual diag  -> tests/validate_summary_flow.py 等脚本
```

- 默认安全回归是 unit + integration。
- live/e2e 更像环境验证，不应替代单元和集成层。
- 这个仓库有大量“契约测试”，它们往往比目录结构更能说明真实依赖。

## Design decisions

- `tests/conftest.py` 默认切临时 `ARXIV_SANITY_DATA_DIR`，避免污染真实数据。
- 前端/模板有独立的契约测试，不要求总是启动浏览器。
- 摘要和上传排障保留了可直接运行的诊断脚本，方便非 CI 场景复现问题。

## Runbook

### 默认安全回归

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit tests/integration -q -k "not daemon"
```

### 按改动域补充

- 前端/模板：`npm run build:static` + template/frontend contract 测试
- summary 页 fallback/SSE 行为：在本地起服务且首页能看到公开论文时，补 `pytest tests/live/test_summary_frontend_behavior.py -q`
- SSE：`tests/unit/test_sse.py`、`tests/unit/test_sse_sqlite_ipc.py`、`tests/integration/test_api_sse.py`
- 摘要：`tests/unit/test_summary_*`、`tests/unit/test_tasks_*`、`tests/integration/test_api_summary.py`
- 上传：`tests/unit/test_upload_*`、`tests/integration/test_api_uploads.py`
- 任务编排：`tests/unit/test_tasks_summary_force_refresh.py`、`tests/unit/test_summary_cancellation.py`、`tests/unit/test_tasks_summary_status_events.py`、`tests/unit/test_upload_task_status_sse.py`、`tests/unit/test_tasks_upload_deleting.py`
- 配置/启动：`tests/unit/test_settings_*`、`tests/unit/test_run_services.py`

### 只改文档 / skill / AGENTS 时

1. 先核对每个引用的文件路径、命令、测试名仍存在
2. 再检查 AGENTS、skills、README/docs、`tests/README.md` 的交叉引用是否一致
3. 只有当文档声称某个行为或命令发生变化时，才补跑对应的 targeted tests

### 真机排障

```bash
python bin/run_services.py
python bin/huey_consumer.py tasks.huey -w 4 -k thread
python tests/validate_summary_flow.py --pid <pid> --trigger --poll
```

## Gotchas

- 如果手动设置了 `ARXIV_SANITY_DATA_DIR`，pytest 可能不再是隔离环境。
- live 和 e2e 失败先查环境，不要第一时间判代码坏。
- 模板内联 JS 不受 ESLint 保护，所以前端改动不能只依赖 lint。
- 复杂链路最容易漏的不是 happy path，而是状态切换、取消、权限隐藏和 SSE 过滤。
- `tests/unit/test_frontend_sse_wiring.py` 主要是源码契约守卫；如果你改的是 summary 页 fallback model 的实时刷新，记得再跑 live 浏览器测试。
- 只改 skill / 文档也可能把维护指引改坏；至少要做路径、命令和测试文件存在性检查。

## Validation checklist template

```text
1. 改动触达哪些文件？
2. 是否涉及 API / repository / task / SSE / upload / summary / template / config？
3. 默认 unit + integration 是否已跑？
4. 是否补跑了该模块专属测试簇？
5. 是否需要 build:static？
6. 是否需要 live/e2e 或 validate_summary_flow？
7. 是否新增了必须保留的回归测试？
```

## Related skills

- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-upload-system`
- `arxiv-sanity-sse-bus`
- `arxiv-sanity-frontend`
- `arxiv-sanity-ops-config`
