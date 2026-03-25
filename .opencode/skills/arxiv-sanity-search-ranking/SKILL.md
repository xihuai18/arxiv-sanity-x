---
name: arxiv-sanity-search-ranking
description: 维护搜索、语义检索、混合排序、SVM 推荐与特征构建链的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Search Ranking Skill

## When to use me

- 你要改搜索排序、SVM 推荐、semantic search、hybrid 权重或特征构建
- 你要排查“搜不到”“推荐怪异”“embeddings 失效”“features.p 过期”
- 你要理解首页搜索链和 `/api/search` 的执行路径

## Scope

- Covers: `backend/services/search_service.py`, `backend/services/semantic_service.py`, `backend/services/data_service.py`, `backend/services/render_service.py`, `backend/utils/cache.py`, `tools/compute.py`, `backend/blueprints/api_search.py`
- Also touches: `backend/legacy.py`, `config/settings_features.py`, `config/settings_services.py`
- Does not cover: 摘要生成、上传解析、reading list 业务语义

## Start here

1. `backend/services/search_service.py`
2. `backend/services/semantic_service.py`
3. `backend/services/data_service.py`
4. `tools/compute.py`
5. `backend/services/render_service.py`
6. `backend/utils/cache.py`
7. `backend/legacy.py` 里首页和 search API 入口

## Core mental model

```text
query
  -> parse / normalize
  -> choose keyword / semantic / hybrid / svm path
  -> use features.p / features_new.p + papers/metas + optional embeddings
  -> rank pids
  -> render paper cards via render_service / legacy
```

- lexical、semantic、hybrid、SVM 共用同一批基础数据，但入口参数不同。
- `features.p` 是排序系统的核心资产；改排序前先确认特征文件是否匹配当前代码。
- 首页和 API 都会触发这条链，只是包装层不同。

## Design decisions

- semantic search 是可选增强，不应让没有 embedding 服务的环境完全不可用。
- hybrid 不是新的独立索引，而是 lexical 与 semantic 的融合。
- SVM 推荐依赖正负标签和上传特征时，权限和 feature 兼容性都要考虑。
- `features_new.p` 是 staged refresh 产物之一；读路径会在 `features.p` / `features_new.p` 之间择新或回退，不能只盯单个文件名。
- `POST /api/keyword_search` 在没传 `time_delta` 时不应偷偷退化成 recent-only；只有显式时间条件才裁剪候选集。

## Runbook

### 调整搜索公式或新增排序参数

1. 同时读 `search_service.py`、`legacy.py` 和 `templates/index.html`
2. 确认 query string、前端表单和后端默认值一致；尤其别把“省略 `time_delta`”和“显式 `time_delta<=0` 禁用时间过滤”混成一件事
3. 若涉及新特征，补 `tools/compute.py`
4. 如果参数或返回字段是 API/前端可见合同，连同 routing/frontend/tests 一起检查

### 排查 semantic search 无结果

1. 先看 `settings.search.semantic_disabled`
2. 再看 embedding 服务配置是否生效
3. 检查 `features.p` 是否包含 embeddings 相关字段
4. 必要时重算：`python -m tools compute --use_embeddings`

### 排查推荐结果异常

1. 看 tags / neg_tags / combined_tags 是否正常
2. 看上传样本特征是否与全局特征兼容
3. 检查时间过滤和 rank 参数是不是把候选集先剪没了

## Gotchas

- `data_service.py` 和 `semantic_service.py` 有锁顺序约束，乱加锁容易死锁。
- 语义搜索和混合搜索是可选路径，排障时先确认配置要求，不要默认服务必须在线。
- 关闭 lexical fullscan 并不一定直接返回空；`search_rank()` 还可能退回到 bounded title scan，排障时别漏掉这一层 fallback。
- embeddings / search cache 失效看的是 effective mtime（含 WAL / staged feature 文件），不是只看主文件名或主 db mtime。
- `logic=and` 的语义分两层：`/api/tags_search` 会在入口先做严格检查，只要请求里任一 tag 不在用户现有标签库里就直接返回空结果；但底层 `svm_rank` 更接近“只对存在的 tag 做交集”，排障时别把两条路径混为一谈。
- 首页大量搜索参数约定还在 `templates/index.html` 内联脚本里，不能只改后端。
- keyword / hybrid 缓存还绑定 `papers.db` mtime；改 time index、批量导入或替换 papers 库后，要把缓存失效一起考虑。
- upload 样本只参与 SVM 训练，不直接进入候选结果；其 feature fingerprint 也会影响缓存命中，别把“训练样本变了但结果没变”简单归咎于算法。
- `features.p` 的版本与环境不匹配时，表现可能像“排序退化”，不一定是算法 bug。

## Validation

```bash
conda activate sanity
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/test_search_service.py tests/unit/test_data_service.py tests/unit/test_data_service_async_refresh.py tests/unit/test_data_service_fast_lookup.py tests/unit/test_data_service_sqlite_mtime.py tests/unit/test_semantic_service_defensive.py tests/unit/test_semantic_service_embeddings_mtime_race.py tests/unit/test_cache.py tests/integration/test_api_search.py tests/integration/test_homepage_interactions.py -q
npm run build:static
pytest tests/unit/test_frontend_backend_contract.py tests/unit/test_template_route_contract.py -q
```

## Related skills

- `arxiv-sanity-data-layer`
- `arxiv-sanity-legacy-core`
- `arxiv-sanity-frontend`
- `arxiv-sanity-routing-layer`
- `arxiv-sanity-testing`
- `arxiv-sanity-ops-config`
