---
name: arxiv-sanity-data-layer
description: 维护 aslite 数据层、SQLite KV、仓储、索引与底层状态库的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Data Layer Skill

## When to use me

- 你要改 `aslite/db.py`、`aslite/repositories.py`、`aslite/arxiv.py`
- 你要排查 SQLite 锁、索引漂移、reading list 或 summary status 异常
- 你要新增一个持久化能力或修一个跨表事务问题

## Scope

- Covers: `aslite/db.py`, `aslite/repositories.py`, `aslite/arxiv.py`
- Also touches: `backend/services/data_service.py`, `tasks.py`, `tools/rebuild_time_index.py`, `config/settings_runtime.py`
- Does not cover: 上层 HTTP 语义、前端行为、Huey 编排细节

## Start here

1. `aslite/db.py`
2. `aslite/repositories.py`
3. `docs/OPERATIONS.md`
4. `tests/unit/test_sqlitekv_*`
5. `tests/unit/test_repositories_*`

## Core mental model

```text
SQLite files
  papers.db  -> papers / metas / metas_time_index
  dict.db    -> tags / keywords / email / readinglist / uploaded_papers / summary_status
  huey.db    -> queue
  sse_events.db -> sse bus

business code
  -> repositories
     -> SqliteKV / transaction helpers
```

- `SqliteKV` 是 pickled blob KV，不是传统 ORM。
- 真正稳定的入口是 repository；不要在上层代码里自己拼 key 或直接改 SQLite 文件。
- 很多业务状态靠“主表 + 索引表”组合存在，必须一起维护。

## Design decisions

- SQLite 走 WAL + retry，是单机部署下的折中方案。
- `ARXIV_SANITY_PROCESS_ROLE` 会影响 DB timeout/retry，web 和 worker 的行为不完全一样。
- `summary_status`、`task::*`、`epoch::*` 在同一库里但不是同一语义层。
- repository 读路径已经带有一定自愈能力；reading list / uploaded papers 的索引漂移可能在读时被修回，而不是只在运维修复脚本里处理。
- upload SHA256 去重映射和 upload record 共存在同一张 KV 表里，只是用前缀 key 区分；排障时不要误以为它是独立索引库。

## Runbook

### 新增一个持久化能力

1. 优先在 `aslite/db.py` 加 `get_*_db()` 或复用现有 DB
2. 在 `aslite/repositories.py` 增加领域方法
3. 保证主表和索引表的原子更新
4. 在上层 service 调 repository，不直接拼 key

### 排查 reading list 或 upload 列表异常

1. 看主表记录是否存在
2. 再看 index 表是否一致
3. 如是“最新论文排序异常”，检查 `metas_time_index`
4. 必要时运行 `python -m tools rebuild_time_index`
5. 注意 repository 读路径可能已经做过 index 自愈；必要时分别核对“修复前原始状态”和“当前读结果”
6. 如果是 dict.db 上的用户态 RMW（reading list / keyword / combined tag / neg tag / epoch counter），确认实现用的是 `autocommit=False` + `transaction(mode="IMMEDIATE")`

### 排查 DB 锁竞争

1. 确认进程角色是 web 还是 worker
2. 看是否在长迭代中写库
3. 查事务模式是否应为 `IMMEDIATE`

## Gotchas

- 不要绕过 repository 直接写 SQLite，这会破坏事务、自愈逻辑和 key 约定。
- `features.p` 是 pickle 文件，环境变更后出错常常应重算，而不是手修。
- reading list、uploaded papers、summary status 都有自己的 key 规范；改 key schema 影响面很大。
- 对 dict.db 的 RMW 写法，不要先 `get(..., default)` 再裸写回；删除后的晚到更新应 no-op，递增 epoch/集合并集要放在 `IMMEDIATE` 事务里避免 lost update。
- WAL 模式下只看主 db 文件 mtime 不可靠，已有代码会同时看 wal/shm。
- prefix scan 走的是 SQLite `LIKE`，`%` / `_` 会被当字面量转义；设计前缀 key 时别忽视这个约束。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_sqlitekv_concurrency.py tests/unit/test_sqlitekv_lock_retry.py tests/unit/test_sqlitekv_prefix.py tests/unit/test_sqlitekv_tablename.py tests/unit/test_repositories_atomic.py tests/unit/test_repositories_transaction_atomicity.py tests/unit/test_repositories_rmw_guards.py tests/unit/test_data_service_sqlite_mtime.py -q
python -m tools rebuild_time_index --help
```

## Related skills

- `arxiv-sanity-search-ranking`
- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-upload-system`
- `arxiv-sanity-user-state-tags`
