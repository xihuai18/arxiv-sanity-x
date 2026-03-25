---
name: arxiv-sanity-frontend
description: 维护模板、静态脚本、CSS、manifest 与页面级交互契约的手册。
license: MIT
compatibility: opencode
metadata:
    category: repo-maintenance
    repo: arxiv-sanity
    audience: maintainers
---

# arxiv-sanity Frontend Skill

## When to use me

- 你要改 `templates/`、`static/`、`static/css/`、`scripts/build.js`
- 你要新增页面脚本、修复 hashed asset、排查前后端契约或 SSE 前端接线
- 你要改首页、summary 页、reading list 页或 inspect 页的交互

## Scope

- Covers: `templates/*.html`, `static/*.js`, `static/css/`, `scripts/build.js`
- Also touches: `backend/app.py`, `backend/utils/manifest.py`, `backend/services/render_service.py`
- Does not cover: 后端业务状态机本身；状态来源去看对应后端 skill

## Start here

1. `templates/base.html`
2. `scripts/build.js`
3. `static/common_utils.js`
4. 页面级入口：`index.html` + `paper_list.js`, `summary.html` + `paper_summary.js`, `readinglist.html` + `readinglist.js`, `paper_detail.js`, `word_list.js`
5. 共享前端模块：`tag_dropdown_shared.js`, `tldr_utils.js`, `author_utils.js`, `markdown_core_utils.js`, `markdown_sanitizer_utils.js`, `markdown_renderer_utils.js`, `markdown_summary_dom_utils.js`, `markdown_summary_utils.js`
6. `backend/utils/manifest.py` + `static/css/main.css`

## Core mental model

```text
template renders html + injected globals
  -> hashed_static('dist/...') resolves manifest entry
  -> common_utils.js boots shared runtime
  -> page script reads injected globals and DOM anchors
  -> SSE / polling / csrfFetch connect UI with backend
```

- 这是 IIFE + global object 模式，不是现代 ESM bundle app。
- 很多页面契约依赖模板注入的全局变量和固定 DOM id/class。
- `static/dist/` 是产物，不要手改。

## Design decisions

- JS build 使用 `bundle: false`，页面脚本通过 `window.ArxivSanityCommon` 等全局对象复用能力。
- `templates/base.html` 会同步注入并加载 `static/common_utils.js`；大部分页面脚本默认它已经完成 runtime 初始化。
- CSS 只有一个入口 `static/css/main.css`，所有页面样式都从这里汇入。
- `backend/utils/manifest.py` 会在 manifest 缺失或漂移时 best-effort fallback 到 `static/dist/` 中的 hash 产物；`scripts/build.js` 也会尽量原子替换 dist 目录。
- summary 页资源加载复杂，是为了兼容 MathJax、markdown-it、CDN fallback 和延迟渲染。

## Runbook

### 新增页面脚本

1. 把新脚本加入 `scripts/build.js` 的 `JS_ENTRY_POINTS`
2. 模板里用 `hashed_static('dist/xxx.js')`
3. 如果有新 CSS，`@import` 到 `static/css/main.css`

### 排查页面脚本没生效

1. 先看 `static/dist/manifest.json` 有没有对应产物
2. 再看页面源代码里最终输出的静态资源 URL
3. 控制台确认 `window.ArxivSanityCommon` 和页面所需的全局对象存在

### 改后端注入的 globals 或 DOM anchors

1. 同时检查 `templates/*.html`、页面脚本和 `backend/app.py` / `backend/legacy.py` 的模板注入
2. 改 `id` / `class` / `data-*` 前先搜 `static/*.js`，许多 DOM anchor 同时承担行为契约
3. 如果影响 summary / reading list / 首页过滤条，补跑前后端契约测试

### 排查 summary / reading list 前端状态异常

1. 先看 `common_utils.js` 的 SSE / polling 是否正常
2. 再看页面脚本是否按 model 或 pid 正确过滤状态；summary 页成功事件既可能命中 requested model，也可能命中 `resolved_model`
3. 如涉及标签 dropdown，检查 `tag_dropdown_shared.js` 和相关 class 契约
4. upload 卡片如果 parse 还没完成，前端应接受“清空旧 summary badge/task id，只展示 parse 状态”的后端语义

## Gotchas

- 模板内联 JS 不在 `npm run lint` 覆盖范围内，改后要更主动自测。
- 许多 CSS class 同时承担行为契约，改名时要搜 JS。
- `paper_list.js` 是 React 渲染，`readinglist.js` 是原生 DOM，大思路不要混用。
- `window.ArxivSanityCommon`、`hashed_static(...)`、manifest fallback 和 `static/dist/` 的 hash 产物共同组成前端运行时；不要只盯某一个层面。
- reading list / summary 页不应盲信持久化在 readinglist 记录里的旧 `summary_status`；后端会用全局状态 overlay，并可能主动清掉陈旧 `task_id`。
- summary 页收到 `summary_status` SSE 时，当前选择模型可能是 fallback 后的 `resolved_model`；不要只按 `event.model` 决定是否刷新当前视图。
- upload 相关 `task_id` / `error` 可能是 owner-scoped 的，前端要接受 `status` 可见但 `task_id` 缺失的情况。
- summary 页的 MathJax 和 markdown 渲染依赖顺序敏感，表面像白屏时先查资源 ready gate。

## Validation

```bash
conda activate sanity
npm run build:static
pytest tests/unit/test_template_manifest_contract.py tests/unit/test_manifest_fallback.py tests/unit/test_template_route_contract.py tests/unit/test_frontend_backend_contract.py tests/unit/test_frontend_sse_wiring.py tests/unit/test_summary_init_flow_contract.py tests/unit/test_summary_toc_contract.py tests/unit/test_summary_mathjax_readiness_contract.py tests/unit/test_summary_render_surfaces_contract.py tests/unit/test_readinglist_ui_contract.py tests/unit/test_ui_accessibility_contract.py -q
pytest tests/live/test_summary_frontend_behavior.py -q  # when localhost service is running with visible paper data and you changed summary SSE/fallback behavior
```

## Related skills

- `arxiv-sanity-runtime-entry`
- `arxiv-sanity-search-ranking`
- `arxiv-sanity-summary-pipeline`
- `arxiv-sanity-task-orchestration`
- `arxiv-sanity-upload-system`
- `arxiv-sanity-user-state-tags`
