---
name: arxiv-sanity-ops-config
description: 维护配置系统、服务编排、离线工具与自动化调度的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Ops and Config Skill

## When to use me

- 你要改 `config/`、`bin/`、`tools/`、`scripts/`、`.env.example` 或运维文档
- 你要排查“服务起不来”“ready 永远不通过”“配置改了但行为没变”
- 你要规划本地开发、单机部署或自动化运维流程

## Scope

- Covers: `config/*.py`, `.env.example`, `bin/*`, `tools/__main__.py`, `tools/arxiv_daemon.py`, `tools/compute.py`, `tools/batch_paper_summarizer.py`, `tools/daemon.py`, `tools/send_emails.py`, `tools/rebuild_time_index.py`, `scripts/__main__.py`, `scripts/cleanup_locks.py`, `scripts/cleanup_tasks.py`, `scripts/check-dist-sync.sh`
- Also touches: `serve.py`, `tasks.py`, `docs/CONFIGURATION.md`, `docs/DEFAULTS.md`, `docs/OPERATIONS.md`, `docs/DEVELOPMENT.md`
- Does not cover: 摘要器内部实现和上传状态机细节；分别看对应 skill

## Start here

1. `docs/INDEX.md`, `docs/CONFIGURATION.md`, `docs/DEFAULTS.md`
2. `.env.example`, `config/settings_base.py`, `config/settings_main.py`, `config/settings_runtime.py`, `config/settings_services.py`, `config/settings_features.py`
3. `config/cli.py`
4. `bin/run_services.py`, `bin/up.sh`, `bin/huey_consumer.py`
5. `tools/__main__.py`, `scripts/__main__.py`, `scripts/cleanup_locks.py`, `scripts/cleanup_tasks.py`

## Core mental model

```text
config settings
  -> runtime / services / features groups
  -> launcher env overrides
  -> bin/* starts processes
  -> tools/* performs offline work
  -> daemon optionally chains fetch -> compute -> summarize -> email -> backup
```

- `from config import settings` 是唯一可信配置入口。
- `bin/run_services.py` 是本地全栈推荐入口，`/ready` 是编排层真正看的健康标准。
- OpenCode 现在默认由 launcher 托管启动；只有明确接外部服务时才需要把 `ARXIV_SANITY_OPENCODE_MANAGED=false`。
- `tools/daemon.py` 有真实副作用，开发环境不要随手跑。

## Design decisions

- 配置采用 pydantic-settings，统一处理默认值、路径解析、别名和类型校验。
- 本地配置固定从 repo root 的 `.env` 读取；旧环境变量别名仍保留一层兼容，但新文档和自动化应使用 canonical `ARXIV_SANITY_*` 名称。
- `python -m tools <command>` 是仓库偏好的工具入口，避免脚本直接执行带来的 path 差异。
- launcher 会通过环境变量覆写部分 ready 要求，所以“静态配置”和“最终运行配置”可能不同；典型如 `--no-embed` / `--no-mineru` 会覆写 `ARXIV_SANITY_READY_REQUIRE_EMBEDDING` / `ARXIV_SANITY_READY_REQUIRE_MINERU`。
- `ARXIV_SANITY_HUEY_UPLOAD_REPAIR_TTL` 这类运维阈值不只是 worker 细节，也会改变 upload stale repair 的用户可见语义。

## Runbook

### 增加一个配置项

1. 先决定它属于 root、runtime、services 还是 features
2. 在对应 settings 文件中加字段
3. 补 `config/cli.py` 的显示/校验
4. 更新文档和 `.env` 示例

### 本地拉起全栈

1. `python -m config.cli validate`
2. `python bin/run_services.py`
3. 如果只做后端快速排障，可用 `python serve.py` + 单独启动 Huey

### 做离线维护

1. 抓取：`python -m tools arxiv_daemon ...`
2. 重算特征：`python -m tools compute --use_embeddings`
3. 批量摘要：`python -m tools batch_paper_summarizer ...`
4. 修索引：`python -m tools rebuild_time_index`

### 改 launcher flag 或环境覆写

1. 同时检查 `bin/run_services.py`、`serve.py`、`backend/blueprints/web.py` 和 `backend/services/health_service.py`
2. 明确它改变的是静态 settings、子进程 env，还是 `/ready` 的最终约束
3. 更新 `.env.example` / `docs/CONFIGURATION.md` / `docs/DEFAULTS.md` / `docs/OPERATIONS.md`

### 清理本地维护状态

1. 用 `python -m scripts cleanup_locks` 处理残留 lock 文件
2. 用 `python -m scripts cleanup_tasks` 清理异常 task 记录前，先理解它和 `tasks.py` repair/cleanup 语义的边界
3. 把 `scripts/check-dist-sync.sh` 当成构建一致性检查，而不是通用部署脚本

### 排查 daemon / 邮件调度

1. 先确认 `settings.daemon.timezone` / `ARXIV_SANITY_DAEMON_TIMEZONE`，邮件时间窗和节假日回看都按这个时区算。
2. 再看 `tools/send_emails.py` 是否因为某个 user 或某个收件人失败而返回非 0。
3. 如果某个用户只有 keyword / combined tag 仍没收到邮件，确认它是否真的注册了邮箱，而不是只看正样本 tag 数量。

## Gotchas

- 不要在业务代码里自己读 `.env` 或 `os.environ`，会绕过 settings 的别名和默认值。
- `config.cli show/env` 会把 extract-info 配置打印出来；alias / readiness warning 主要看 `doctor` 或 `validate`，空 secret 也可能是有意保留为空，不是脱敏失败。
- `daemon.py` 可能抓真实数据、发邮件、git backup/push，开发时非常危险。
- `send_emails.py` 只要任一用户处理失败或任一收件人发送失败就会返回非 0；`daemon.py` 会把任何非 0 记成 warning。
- `/health` 不是 `/ready`；编排成功与否看 `/ready`。
- launcher 参数会改最终环境变量，排障时要看实际打印和子进程环境。
- `ARXIV_SANITY_OPENCODE_MANAGED` 默认是 `true`；如果你配置了外部 `ARXIV_SANITY_OPENCODE_BASE_URL` 却忘了关掉 managed，launcher 仍会优先拉起并使用本地 OpenCode。
- OpenCode `base_url`、显式 fallback model 可见性和 `/ready` 探针是联动的；只改其中一个文件经常不够。
- `ARXIV_SANITY_HUEY_UPLOAD_REPAIR_TTL` 太小会让长排队上传过早触发 stale repair，文档、默认值和运维告警要一起看。
- 发布到公开仓库时，要确保发布流程会清理非公开 `.opencode/` 内容、`tmp/`、coverage/test 产物等本地文件，不要把本地镜像流程当成通用分发工具。

## Validation

```bash
conda activate sanity
python -m config.cli show
python -m config.cli validate
python -m config.cli doctor
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_settings_base_env_file.py tests/unit/test_settings_path_resolution.py tests/unit/test_config_settings_legacy_aliases.py tests/unit/test_config_reload.py tests/unit/test_config_cli_extract.py tests/unit/test_opencode_settings.py tests/unit/test_opencode_service.py tests/unit/test_run_services.py tests/unit/test_health_llm_fallback.py tests/unit/test_background_service.py tests/unit/test_metrics_endpoint.py tests/unit/test_sentry_init.py tests/unit/test_daemon_simulation.py tests/unit/test_daemon_extended.py tests/unit/test_send_emails.py -q
```

## Related skills

- `arxiv-sanity-runtime-entry`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-data-layer`
- `arxiv-sanity-search-ranking`
- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-testing`
