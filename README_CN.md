# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

基于现代机器学习技术的综合性 arXiv 论文浏览和推荐系统，集成 AI 智能总结、混合搜索功能和个性化推荐。采用 TF-IDF、语义嵌入和 LLM 集成等先进技术。

![Screenshot](arxiv-sanity-x.png)

## 📋 目录

### 入门

- [核心功能概览](#-核心功能概览)
- [快速开始](#-快速开始)
- [Docs](#docs)

### 使用

- [用户使用指南](#-用户使用指南)
- [AI 论文总结](#-ai-论文总结)
- [高级功能](#-高级功能)

### 配置

- [配置指南](#配置指南)
- [运行前准备](#-运行前准备与系统说明)

### 运维

- [数据目录与迁移](#-数据目录与迁移)
- [部署与安全](#-部署与安全说明)
- [常见问题](#-常见问题与排错)

### 开发

- [技术栈](#-技术栈)
- [项目结构](#-项目结构)
- [架构说明](#架构说明)
- [API 参考](#-api-参考)
- [开发指南](#-开发指南)

### 其他

- [更新日志](#-更新日志)
- [致谢](#-致谢)

---

## Docs

- 入口：[docs/INDEX.md](docs/INDEX.md)
- 仓库技能手册：[.opencode/skills/README.md](.opencode/skills/README.md)
- 运维：[docs/OPERATIONS.md](docs/OPERATIONS.md)
- 安全：[docs/SECURITY.md](docs/SECURITY.md)
- 开发：[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- 测试指南：[tests/README.md](tests/README.md)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
- 安全策略：[SECURITY.md](SECURITY.md)

## 🎯 核心功能概览

arxiv-sanity-X 是一个面向个人科研/工程阅读流的 arXiv 工作台：把「拉取与索引论文」「快速检索」「基于反馈的推荐」以及「AI 总结」整合到同一个网站里，帮助你更快找到重点、沉淀标签体系，并持续跟踪最新论文。

### 主要能力

| 功能              | 说明                                                    |
| ----------------- | ------------------------------------------------------- |
| 🔍 **多模式搜索** | 关键词（TF-IDF）、语义（Embedding）、混合搜索，权重可调 |
| 🎯 **智能推荐**   | 基于正/负反馈标签训练 SVM 分类器，个性化推荐            |
| 🤖 **AI 总结**    | HTML/PDF 解析 + LLM 生成结构化总结，支持多模型切换      |
| 🏷️ **标签系统**   | 正/负反馈、组合标签、关键词跟踪、阅读列表               |
| 📧 **邮件推荐**   | 每日自动推荐邮件，假期感知调度                          |
| 🔄 **自动化**     | 内置调度器：获取 → 计算 → 总结 → 邮件                   |

## 🛠️ 技术栈

### 后端

- **框架**：Flask，基于 Blueprint 的模块化架构
- **数据库**：SQLite，自定义 KV 存储（WAL 模式，支持压缩）
- **任务队列**：Huey（SQLite 后端），用于异步摘要 / 上传编排
- **配置管理**：pydantic-settings，类型安全的配置管理
- **实时通信**：Server-Sent Events (SSE) 实时推送

### 前端

- **模板引擎**：Jinja2，响应式 HTML/CSS
- **JavaScript**：混合页面运行时：IIFE/global scripts + 局部 React 页面 + 原生 DOM 交互
- **渲染**：MathJax 渲染 LaTeX，markdown-it 渲染 Markdown
- **构建**：esbuild，支持内容哈希缓存

### 机器学习/AI

- **搜索**：TF-IDF（scikit-learn）+ 语义嵌入（Ollama/OpenAI API）
- **推荐**：基于用户反馈训练的 SVM 分类器
- **摘要生成**：基于 OpenCode 的文本模型
- **PDF 解析**：MinerU（API 或本地 VLM）

### 基础设施

- **Web 服务器**：Gunicorn，多 worker 支持
- **调度器**：APScheduler，自动化流水线
- **服务组件**：OpenCode 服务、Ollama 嵌入、MinerU VLM

## 📁 项目结构

```
arxiv-sanity-x/
├── serve.py              # Flask 入口
├── tasks.py              # Huey 任务定义与编排中心
│
├── backend/              # Flask 应用
│   ├── app.py            # 应用工厂 & 初始化
│   ├── blueprints/       # 路由处理器（10 个 Blueprint）
│   │   ├── web.py        # 页面路由（/, /summary, /profile 等）
│   │   ├── api_user.py   # 登录/登出、用户状态、邮箱登记
│   │   ├── api_search.py # 搜索端点
│   │   ├── api_summary.py# 摘要生成 & 状态
│   │   ├── api_tags.py   # 标签管理
│   │   ├── api_papers.py # 论文数据 & 图片
│   │   ├── api_readinglist.py # 阅读列表
│   │   ├── api_uploads.py# 上传 PDF + 解析/抽取/相似度
│   │   ├── api_sse.py    # Server-Sent Events
│   │   └── metrics.py    # /metrics（Prometheus，可选）
│   ├── services/         # 业务逻辑层
│   │   ├── data_service.py    # 缓存 & 数据管理
│   │   ├── search_service.py  # TF-IDF、语义、混合搜索
│   │   ├── summary_service.py # 摘要生成 & 缓存
│   │   ├── semantic_service.py# 嵌入 & 向量搜索
│   │   └── ...
│   ├── schemas/          # Pydantic 请求/响应模型
│   └── utils/            # 工具函数（缓存、SSE、manifest）
│
├── aslite/               # 数据层
│   ├── db.py             # SqliteKV 封装 & 数据库访问
│   ├── repositories.py   # Repository 模式数据访问
│   └── arxiv.py          # arXiv API 客户端
│
├── config/               # 配置
│   ├── settings.py       # pydantic-settings 定义
│   ├── cli.py            # 配置 CLI 工具
│   └── ...               # OpenCode / 应用配置
│
├── tools/                # CLI 工具 & 自动化
│   ├── arxiv_daemon.py   # 从 arXiv 拉取论文
│   ├── compute.py        # TF-IDF & 嵌入计算
│   ├── daemon.py         # 定时任务调度器
│   ├── batch_paper_summarizer.py # 批量摘要生成
│   ├── paper_summarizer.py # 单篇论文摘要
│   └── send_emails.py    # 邮件推荐
│
├── bin/                  # 服务启动器
│   ├── run_services.py   # 一键多服务启动器
│   ├── up.sh             # Gunicorn 启动脚本
│   ├── huey_consumer.py  # Huey consumer 封装（内存限制 + worker 角色）
│   ├── embedding_serve.sh# Ollama 嵌入服务
│   ├── mineru_serve.sh   # MinerU VLM 服务
│   └── ...               # 可选本地服务启动脚本
│
├── static/               # 前端资源
│   ├── *.js              # JavaScript 源文件
│   ├── css/              # 样式表
│   ├── lib/              # 第三方库
│   └── dist/             # 构建产物（gitignore）
│
├── templates/            # Jinja2 HTML 模板
├── scripts/              # 构建 & 维护脚本
├── tests/                # 测试套件
├── data/                 # 运行时数据（gitignore）
│   ├── papers.db         # 论文元数据
│   ├── dict.db           # 用户数据（标签、关键词等）
│   ├── features.p        # 计算的特征
│   ├── huey.db           # Huey 任务队列数据库（SQLite）
│   ├── uploads/          # 上传的 PDF + 元信息
│   └── summary/          # 缓存的摘要
└── data-repo/            # 可选：用于备份 data/dict.db 的 git submodule
```

## 🧭 用户使用指南

本节介绍如何使用 arxiv-sanity-X 网站的各项功能。大多数操作都从首页开始。

### 1）登录

- 点击右上角 **Profile** 进入个人中心
- 输入用户名登录（无密码，适合个人/内网使用）
- 若要公网部署，建议放在统一认证/VPN 后面，并设置稳定会话密钥（`ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt`）
- （可选）在 Profile 页面登记通知邮箱：支持多个邮箱（逗号/空白/换行分隔），提交空值可清空。
- 对 `POST /login` 和 `POST /register_email` 的 JSON 调用会返回 JSON 成功/失败结果；浏览器 form 提交仍保持原来的跳转体验。

### 2）浏览与检索论文

**首页功能：**

- 默认按时间排序显示最新论文
- 点击论文标题查看详情，点击 arXiv 链接跳转原文
- 使用顶部搜索框进行检索（支持快捷键 `Ctrl+K`）

**搜索语法：**
| 语法 | 示例 | 说明 |
| ---- | ------------------------ | ------------------------ |
| 标题 | `ti:transformer` | 搜索标题包含 transformer |
| 作者 | `au:goodfellow` | 搜索作者 |
| 分类 | `cat:cs.LG` | 搜索特定 arXiv 分类 |
| ID | `id:2312.12345` | 按 arXiv ID 查找 |
| 短语 | `"large language model"` | 精确短语匹配 |
| 排除 | `-survey` 或 `!survey` | 排除包含该词的结果 |

**搜索模式切换：**

- **关键词**：最快，基于 TF-IDF，不依赖额外服务
- **语义**：基于向量相似度，需要启用 Embedding
- **混合**：结合关键词+语义，权重可调（推荐）

**首页筛选规则：**

- 输入搜索词后，前后端都会把排序限制在支持查询的模式（`search` 或 `time`）
- **Tags**、**PIDs**、**Logic**、**SVM C** 这类高级筛选，只有在 **Rank by** 为 `tags` 或 `pid` 时才真正生效
- 高级筛选里的标签输入支持联想建议，也支持键盘选择
- 对 `POST /api/keyword_search` 来说，不传 `time_delta` 会保留全量索引候选；脚本调用若想显式关闭时间过滤，请传 `time_delta <= 0`。

### 3）标签系统与个性化推荐

**打标签：**

- 在论文卡片上点击 **+** 按钮添加标签
- 支持正向标签（喜欢）和负向标签（不喜欢）
- 标签会训练个人 SVM 推荐模型

**使用标签推荐：**

- 在首页选择 **Tags** 排序模式
- 选择一个或多个标签，系统会推荐相似论文
- 组合标签（如 `RL,NLP`）可做交集推荐

### 4）AI 论文总结

- 点击论文进入详情页，点击 **Summary/总结** 按钮
- 首次生成需要等待 LLM 处理（通常 10-30 秒）
- 生成后会缓存，下次访问直接显示
- 支持切换不同 LLM 模型重新生成
- 可清除当前模型缓存或全部缓存

### 5）阅读列表

- 点击论文卡片上的书签/阅读列表按钮即可加入阅读列表
- 访问 `/readinglist` 页面管理阅读列表
- 可用于批量总结或稍后阅读
- 阅读列表页会复用首页的主要操作入口，包括 Similar / Inspect / Summary 以及私有上传论文相关操作

### 6）私有 PDF 上传

- 上传论文是私有资源，按 owner 隔离；非 owner 访问上传资源或修改接口时仍然看到 `404`
- 如果上传记录已经进入删除中，`POST /api/uploaded_papers/update_meta`、`POST /api/uploaded_papers/retry_parse`、`POST /api/uploaded_papers/parse`、`POST /api/uploaded_papers/process`、`POST /api/uploaded_papers/extract_info` 会统一返回 `409`，错误信息为 `"Paper is being deleted"`
- `parse` / `process` / `retry_parse` 只把当前记录指向的任务视为活跃任务；若旧任务已 stale，会先修复旧状态，再返回新的 `task_id`
- 轮询 `GET /api/task_status/<task_id>` 时，上传任务现在可能出现 `failed/stale_running_repaired` 或 `canceled/superseded_upload_task`
- 删除上传时会 best-effort 取消正在进行的 upload parse/process/extract 任务，以及关联的 summary 任务，再继续删文件和数据库记录
- 如果部署里 upload 队列积压较长，请调大 `ARXIV_SANITY_HUEY_UPLOAD_REPAIR_TTL`，避免仍然有效的上传任务被过早修复

### 7）其他功能

- **Stats 页面**：查看论文统计、每日新增图表
- **About 页面**：查看系统信息、支持的 arXiv 分类
- **邮件推荐**：配置 SMTP 后可接收每日推荐邮件（详见配置指南）

## 📦 数据目录与迁移

默认数据存放在 `data/`（由 `.env` / `config/settings.py` 中的 `ARXIV_SANITY_DATA_DIR` 决定）：

- `data/papers.db`：论文与元信息（由 arXiv 拉取）
- `data/dict.db`：用户数据（标签/负反馈/关键词/阅读列表/邮箱登记/总结状态等）
- `data/features.p`：由 [tools/compute.py](tools/compute.py) 生成的 TF‑IDF/混合特征
- `data/summary/`：LLM 总结缓存
- `data/pdfs/`、`data/mineru/`、`data/html_md/`：解析相关的中间缓存
- `data/uploads/`：上传的私有 PDF 与派生产物（如果使用上传功能）
- `data/huey.db`：Huey 任务队列数据库
- `data/sse_events.db`：SSE 跨进程事件总线（SQLite，启用时创建）
- `data-repo/`（可选）：daemon 用于备份 `data/dict.db` 的 git submodule

迁移到新机器时，通常至少复制：

- `data/papers.db`
- `data/dict.db`
- `data/features.p`（或在新环境重新运行 [tools/compute.py](tools/compute.py) 生成）
- `data/summary/`（可选：想保留已缓存总结时再带上）

如果使用 `data-repo/` git 备份，也可以从这里恢复：

- `data-repo/dict.db`

如何在 daemon 下使用 `data-repo/` 备份：

1. 初始化 submodule：`git submodule update --init --recursive`
2. `ARXIV_SANITY_DAEMON_ENABLE_GIT_BACKUP=true` 本来就是默认值；如果不想备份，请显式设为 `false`
3. 如果不想推送到远端，请设置 `ARXIV_SANITY_DAEMON_BACKUP_PUSH=false`；否则请确保 `data-repo/` 配好 remote，且运行环境具备 `git push` 权限

## 🔐 部署与安全说明

- 站点内置登录是“仅用户名、无密码”，适合个人/内网环境。
- 如果需要公网部署，务必放在统一认证/VPN/反向代理鉴权后面，并通过 `ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt` 设置稳定的会话密钥。
- 不要把真实 API Key 写进仓库；优先使用环境变量注入。

## 🧩 常见问题与排错

- **网站空白/没有论文**：通常是还没跑 [tools/arxiv_daemon.py](tools/arxiv_daemon.py) + [tools/compute.py](tools/compute.py)。
- **总结一直失败**：检查 `.env` 里的 `ARXIV_SANITY_OPENCODE_BASE_URL`、`ARXIV_SANITY_LLM_NAME`、`ARXIV_SANITY_EXTRACT_MODEL_NAME`。
- **总结不自动开始生成**：总结页在“缓存缺失”时不会自动入队，请手动点击 **Generate**；同时确保 Huey consumer 在跑（推荐：`python bin/run_services.py` 一键启动；或只启动 consumer：`python bin/huey_consumer.py tasks.huey -w 4 -k thread`）。
- **语义/混合检索没效果**：确认嵌入（Embedding）已启用，并用 [tools/compute.py](tools/compute.py) 重新生成特征（混合特征需要包含嵌入）。
- **按时间排序异常/变慢**：重建元数据时间索引：`python -m tools rebuild_time_index`。
- **MinerU 报错**：
    - API 后端：检查 `ARXIV_SANITY_MINERU_API_KEY`
    - 本地后端：检查 `ARXIV_SANITY_MINERU_BACKEND`，以及服务是否能在 `ARXIV_SANITY_MINERU_PORT` 访问
- **崩溃后卡住（锁文件）**：运行 `python -m scripts cleanup_locks`，或调整 `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC` / `ARXIV_SANITY_MINERU_LOCK_STALE_SEC`。
- **总结任务“卡死/幽灵任务”（Huey）**：先 dry-run `python -m scripts cleanup_tasks`，确认无误后加 `--force`；必要时用 `--flush-huey` 清空队列（谨慎）。
- **features.p 读取失败（NumPy 版本不匹配）**：在当前环境重新运行 [tools/compute.py](tools/compute.py) 生成特征文件。
- **Gunicorn 报 `WORKER TIMEOUT` / `SIGKILL`**：若日志里先出现 `WORKER TIMEOUT`，通常是 gunicorn 默认超时太短或冷启动/初始化阻塞。可通过 `ARXIV_SANITY_GUNICORN_EXTRA_ARGS="--timeout 600 --graceful-timeout 600"` 提高超时；并避免在开启大缓存时配置过多 worker。`bin/up.sh` 在 SSE 场景会优先选择 `gevent` 并自动设置较长超时。
- **gevent 的 MonkeyPatchWarning（ssl/urllib3）**：常见于 `--preload` 场景；若仍出现，可尝试 `ARXIV_SANITY_GUNICORN_PRELOAD=false` 或强制 `ARXIV_SANITY_GUNICORN_WORKER_CLASS=gthread`。
- **实时推送不工作（SSE）**：确认 `ARXIV_SANITY_SSE_ENABLED=true`，并访问 `GET /api/sse_stats` 查看每个进程的 SSE 队列/总线状态。

## ⚡ 快速开始

### 1. 安装

```bash
git clone https://github.com/xihuai18/arxiv-sanity-x && cd arxiv-sanity-x
pip install -r requirements.txt
npm install
```

还需要单独安装 [OpenCode](https://opencode.ai) CLI，并确保在当前 shell 里执行 `opencode --version` 能成功。默认启动流程会自动拉起 OpenCode。OpenCode 的安装和配置详见 <https://opencode.ai>。

### 2. 创建 `.env`

```bash
cp .env.example .env
```

### 3. 配置 OpenCode 与模型

文本模型现在统一走 OpenCode HTTP 服务。

```bash
ARXIV_SANITY_OPENCODE_BASE_URL=http://127.0.0.1:53000
ARXIV_SANITY_OPENCODE_MANAGED=true

ARXIV_SANITY_LLM_NAME=gpt-5.4
ARXIV_SANITY_EXTRACT_MODEL_NAME=gpt-5.4-mini
ARXIV_SANITY_LLM_SUMMARY_LANG=zh

ARXIV_SANITY_HOST=http://localhost:55555
ARXIV_SANITY_SERVE_PORT=55555
ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE=html
ARXIV_SANITY_SUMMARY_HTML_SOURCES=ar5iv,arxiv
```

说明：

- 模型选择器既可以是 `provider/model`，也可以是支持的 alias，比如 `gpt-5.4`。
- alias 成员会自动尝试；如果该 alias 下的候选都失败，就直接报错。
- 上传论文元信息提取也走 OpenCode。
- embedding 仍然独立，不会复用 OpenCode 文本模型配置。
- `ARXIV_SANITY_OPENCODE_MANAGED` 默认就是 `true`，所以 `bin/run_services.py` 会本地启动 `opencode serve`，并让子进程统一指向本地实例。
- 在默认 `managed=true` 的情况下，`python bin/run_services.py` 也要求本机已经安装并能在 `PATH` 中找到 `opencode`。

### 4. 如需配置 Embedding / MinerU

```bash
# 本地 embedding（默认）
ARXIV_SANITY_EMBED_USE_LLM_API=false
ARXIV_SANITY_EMBED_PORT=54000
ARXIV_SANITY_EMBED_MODEL_NAME=qwen3-embedding:0.6b

# 远程 embedding API（显式配置；不会再回退到文本模型配置）
# ARXIV_SANITY_EMBED_USE_LLM_API=true
# ARXIV_SANITY_EMBED_API_BASE=https://your-embedding-endpoint/v1
# ARXIV_SANITY_EMBED_API_KEY=...

# MinerU API
# ARXIV_SANITY_MINERU_ENABLED=true
# ARXIV_SANITY_MINERU_BACKEND=api
# ARXIV_SANITY_MINERU_API_KEY=...
```

### 5. 验证并启动

```bash
python -m config.cli validate
python -m config.cli doctor

python -m tools arxiv_daemon -n 10000 -m 500
python -m tools compute --num 20000

python bin/run_services.py
```

然后访问 `http://localhost:55555`。

## 🧰 运行说明

- 推荐全栈启动：`python bin/run_services.py`
- 全栈 + daemon：`python bin/run_services.py --with-daemon`
- 仅 Web：`npm run build:static && python serve.py`
- Gunicorn：`bash bin/up.sh`
- 仅 Huey worker：`python bin/huey_consumer.py tasks.huey -w 4 -k thread`

默认情况下，`bin/run_services.py` 会本地启动 `opencode serve`。只有在你明确要连接外部 OpenCode 服务时，才需要把 `ARXIV_SANITY_OPENCODE_MANAGED=false`。

## 🤖 AI 总结

总结主链路：

1. 获取 HTML/PDF 内容
2. 转换为 Markdown（`html` 或 `mineru`）
3. 通过 OpenCode 发送文本生成请求
4. 以 canonical model id 为 key 写入缓存
5. 对前端/API 保留 `summary_meta.llm_model`、`resolved_model`、usage 等字段

`GET /api/llm_models` 现在来自 OpenCode `/config/providers`，但仍返回兼容形态：

```json
{ "models": [{ "id": "openai/gpt-5.4" }], "default": "openai/gpt-5.4" }
```

## 📚 API 重点

- `GET /health`：非严格健康检查
- `GET /ready`：严格就绪检查（包含 OpenCode 与必需模型校验）
- `GET /api/llm_models`：OpenCode 可用模型列表
- `POST /api/get_paper_summary`：只读缓存摘要
- `POST /api/trigger_paper_summary`：触发异步摘要生成
- `POST /api/summary_status`：摘要状态
- `POST /api/uploaded_papers/extract_info`：通过 OpenCode 提取上传元信息

## 说明

- 文本生成需要 [OpenCode](https://opencode.ai)（`opencode serve`）作为 LLM 后端。详见[配置指南](#配置指南)。
- 默认启动器（`python bin/run_services.py`）在 `ARXIV_SANITY_OPENCODE_MANAGED=true`（默认）时会自动在本地启动 OpenCode。
- `config.cli` 与 `/ready` 都已切换到 OpenCode 语义。
- 代码默认值请查看 `docs/DEFAULTS.md`。

#### arxiv_daemon

```bash
python -m tools arxiv_daemon -n 10000 -m 500    # 获取最多 10000 篇，每次查询 500 篇
python -m tools arxiv_daemon --init             # 使用关键词搜索初始化
python -m tools arxiv_daemon --num-total 5000   # 限制所有分类总论文数
python -m tools arxiv_daemon --break-after 20   # 连续 20 批无新论文后停止
```

#### compute

```bash
python -m tools compute --num 20000             # TF-IDF 特征数
python -m tools compute --use_embeddings        # 启用嵌入（默认）
python -m tools compute --no-embeddings         # 禁用嵌入
python -m tools compute --embed_model nomic-embed-text  # 嵌入模型
python -m tools compute --embed_dim 512         # 嵌入维度
python -m tools compute --embed_batch_size 2048 # 批次大小
```

#### batch_paper_summarizer

```bash
python -m tools batch_paper_summarizer -n 100 -w 2         # 100 篇论文，2 个工作线程
python -m tools batch_paper_summarizer --priority          # 优先队列模式
python -m tools batch_paper_summarizer --priority-days 2   # 优先窗口
python -m tools batch_paper_summarizer --dry-run           # 仅预览
python -m tools batch_paper_summarizer -m "gpt-4o-mini"    # 指定模型
```

---

## 🤖 AI 论文总结

### 完整 AI 处理管道

1. **HTML/PDF 获取**：获取 arXiv/ar5iv HTML（默认）或 PDF，支持错误处理
2. **Markdown 解析**：HTML→Markdown（默认）或 minerU PDF 解析，支持结构识别
3. **LLM 处理**：使用多种兼容 OpenAI API 的模型生成全面总结
4. **质量控制**：中文文本比例验证和内容过滤
5. **智能缓存**：智能缓存机制，自动质量检查和存储优化

### OpenCode 模型示例

#### 默认总结模型

```python
ARXIV_SANITY_OPENCODE_BASE_URL = "http://127.0.0.1:53000"
ARXIV_SANITY_LLM_NAME = "openai/gpt-5.4"
```

#### 上传元信息提取模型

```python
ARXIV_SANITY_EXTRACT_MODEL_NAME = "openai/gpt-5.4-mini"
```

### 总结页面功能

- **清除当前模型（Clear Current Summary）**：仅删除当前模型的总结
- **清除所有缓存（Clear All）**：删除论文的所有缓存（总结、HTML、MinerU）

---

## 🔧 高级功能

### 嵌入模型

```bash
# 拉取并启动嵌入模型（Ollama）
ollama pull nomic-embed-text
bash bin/embedding_serve.sh  # 在 EMBED_PORT 启动

# 使用嵌入计算
python -m tools compute --use_embeddings --embed_model nomic-embed-text
```

### 自动化调度

**内置调度器：**

```bash
python -m tools daemon
```

内置 daemon 的调度时区由 `ARXIV_SANITY_DAEMON_TIMEZONE` 决定（默认 `Asia/Shanghai`）：

- **获取+计算**：工作日 8:00、12:00、16:00、20:00
- **发送邮件**：工作日 18:00
- **备份**：每日 20:00

说明：

- `tools/send_emails.py` 现在会把“只有 keyword / combined tag，但已经登记邮箱”的用户也纳入推荐发送候选。
- 只要某个用户的推荐流程失败，或某个收件人的 SMTP 发送失败，`tools/send_emails.py` 就会返回非 0；daemon 会把它记为 warning。

**手动 Cron：**

```cron
# 获取和计算（工作日每日 4 次）
0 9,13,17,21 * * 1-5 cd /path && python -m tools arxiv_daemon -n 1000 && python -m tools compute --use_embeddings

# 发送邮件（工作日下午 6 点）
0 18 * * 1-5 cd /path && python -m tools send_emails -t 2

# 生成总结（每日晚上 7 点）
0 19 * * * cd /path && python -m tools batch_paper_summarizer -n 200 -w 2
```

---

## 📚 API 参考

路由由 `backend/blueprints/` 下的 Flask Blueprint 提供。

如需查看 Swagger/OpenAPI 文档（默认关闭以减少暴露面），可设置 `ARXIV_SANITY_ENABLE_SWAGGER=true`，然后访问 `GET /apidocs/`。
如需开启 Prometheus 指标，可设置 `ARXIV_SANITY_ENABLE_METRICS=true`（可选鉴权：`ARXIV_SANITY_METRICS_KEY`，请求头 `X-ARXIV-SANITY-METRICS-KEY`）。
如需启用 Sentry（可选），可设置 `ARXIV_SANITY_SENTRY_ENABLED=true` 且配置 `ARXIV_SANITY_SENTRY_DSN=...`（可选：`ARXIV_SANITY_SENTRY_ENVIRONMENT`、`ARXIV_SANITY_SENTRY_RELEASE`、`ARXIV_SANITY_SENTRY_TRACES_SAMPLE_RATE`、`ARXIV_SANITY_SENTRY_PROFILES_SAMPLE_RATE`）。

### 页面路由（`web.py`）

| 路由               | 说明                    |
| ------------------ | ----------------------- |
| `GET /health`      | 存活/降级健康检查       |
| `GET /ready`       | 严格就绪检查            |
| `GET /`            | 首页，论文列表          |
| `GET /inspect`     | 调试检查页（需认证）    |
| `GET /summary`     | 论文总结页面            |
| `GET /profile`     | 用户个人中心            |
| `GET /stats`       | 系统统计页面            |
| `GET /about`       | 关于页面                |
| `GET /readinglist` | 阅读列表页面            |
| `GET /metrics`     | Prometheus 指标（可选） |

说明：`GET /health` 是非严格检查，冷启动或依赖降级时也可能返回 `200`，但 `status` 为 `loading` 或 `degraded`。如需严格就绪探针，请使用 `GET /ready`；在论文和必需依赖准备好之前它会返回 `503`。

### 搜索与推荐（`api_search.py`）

**首页 query（GET `/`）：**

| Query                                                                | 说明                 |
| -------------------------------------------------------------------- | -------------------- |
| `GET /?rank=search&q=<query>`                                        | 关键词搜索（TF-IDF） |
| `GET /?rank=search&q=<query>&search_mode=semantic`                   | 语义搜索             |
| `GET /?rank=search&q=<query>&search_mode=hybrid&semantic_weight=0.5` | 混合搜索             |
| `GET /?rank=tags&tags=<tag_list>&logic=<and\|or>`                    | 基于标签的 SVM 推荐  |
| `GET /?rank=time&time_filter=<days>`                                 | 时间过滤论文         |
| `GET /?rank=pid&pid=<paper_id>`                                      | 相似论文推荐         |

**JSON API：**

| 端点                       | 说明                                                       |
| -------------------------- | ---------------------------------------------------------- |
| `POST /api/keyword_search` | 关键词搜索（JSON，返回 `pids` / `scores` / `total_count`） |
| `POST /api/tag_search`     | 单标签搜索（需登录）                                       |
| `POST /api/tags_search`    | 多标签搜索（需登录）                                       |
| `GET /cache_status`        | 缓存状态页（需 `ARXIV_SANITY_ENABLE_CACHE_STATUS=true`）   |

说明：`tools/send_emails.py` 等脚本可配置 `ARXIV_SANITY_RECO_API_KEY`，并用 `X-ARXIV-SANITY-API-KEY` 头（或 `Authorization: Bearer ...`）在无浏览器会话时调用标签搜索接口；同时需要在 JSON 里提供 `{"user": "<username>"}`。对 `POST /api/keyword_search` 来说，不传 `time_delta` 不会再默认只搜最近论文；如要显式禁用时间过滤，请传 `time_delta <= 0`。

### 论文总结（`api_summary.py`）

| 端点                                   | 说明                         |
| -------------------------------------- | ---------------------------- |
| `POST /api/get_paper_summary`          | 获取论文总结（默认只读缓存） |
| `POST /api/get_paper_tldr`             | 获取缓存 TL;DR（尽力）       |
| `POST /api/trigger_paper_summary`      | 触发异步总结任务             |
| `POST /api/trigger_paper_summary_bulk` | 批量触发异步总结任务         |
| `GET /api/task_status/<task_id>`       | 查询 Huey 任务状态           |
| `GET /api/queue_stats`                 | Huey 队列统计                |
| `POST /api/summary_status`             | 获取总结状态（JSON）         |
| `POST /api/clear_model_summary`        | 清除特定模型的总结缓存       |
| `POST /api/clear_paper_cache`          | 清除论文所有缓存             |
| `GET /api/check_paper_summaries`       | 校验/重查缓存摘要            |

说明：对任务 owner，`GET /api/task_status/<task_id>` 会额外返回 `pid`、`model`、`error`、`priority`、`stage` 等字段（其中 `stage` 为粗粒度进度标记）；部分排队任务还可能返回 `queue_rank` / `queue_total`。

### 标签管理（`api_tags.py`）

| 端点                              | 说明                          |
| --------------------------------- | ----------------------------- |
| `POST /api/tag_feedback`          | 添加/移除正负反馈（JSON）     |
| `POST /api/tag_feedback_bulk`     | 批量添加/移除正负反馈（JSON） |
| `GET /api/tag_members`            | 获取标签成员                  |
| `POST /api/paper_titles`          | 批量获取论文标题              |
| `POST /add_tag/<tag>`             | 创建标签                      |
| `POST /add/<pid>/<tag>`           | 为论文添加标签                |
| `POST /sub/<pid>/<tag>`           | 移除论文标签                  |
| `POST /del/<tag>`                 | 删除标签                      |
| `POST /rename/<otag>/<ntag>`      | 重命名标签                    |
| `POST /add_ctag/<ctag>`           | 添加组合标签                  |
| `POST /del_ctag/<ctag>`           | 删除组合标签                  |
| `POST /rename_ctag/<otag>/<ntag>` | 重命名组合标签                |
| `POST /add_key/<keyword>`         | 添加跟踪关键词                |
| `POST /del_key/<keyword>`         | 移除跟踪关键词                |
| `POST /rename_key/<okey>/<nkey>`  | 重命名关键词                  |

### 论文资源（`api_papers.py`）

| 端点                                     | 说明                  |
| ---------------------------------------- | --------------------- |
| `GET /api/paper_image/<pid>/<filename>`  | 论文图片资源          |
| `GET /api/mineru_image/<pid>/<filename>` | MinerU 图片资源       |
| `GET /api/llm_models`                    | 获取可用 LLM 模型列表 |

### 阅读列表（`api_readinglist.py`）

| 端点                           | 说明                 |
| ------------------------------ | -------------------- |
| `POST /api/readinglist/add`    | 添加论文到阅读列表   |
| `POST /api/readinglist/remove` | 从阅读列表移除论文   |
| `GET /api/readinglist/paper`   | 获取单篇阅读列表论文 |
| `GET /api/readinglist/list`    | 获取阅读列表         |

### 用户与会话（`api_user.py`）

| 端点                   | 说明                     |
| ---------------------- | ------------------------ |
| `GET /api/user_state`  | 获取用户状态             |
| `POST /login`          | 用户登录                 |
| `GET/POST /logout`     | 用户登出                 |
| `POST /register_email` | 登记通知邮箱（支持多个） |

说明：`GET /api/user_state` 会返回当前 `user`。`POST /login` 与 `POST /register_email` 对浏览器 form 仍保持跳转语义，但对 JSON 请求会返回 JSON 结构，并且做更严格的类型校验。

### 实时推送（`api_sse.py`）

| 端点                   | 说明               |
| ---------------------- | ------------------ |
| `GET /api/user_stream` | 用户 SSE 流        |
| `GET /api/sse_stats`   | SSE 状态（进程内） |

### 上传（实验性）（`api_uploads.py`）

| 端点                                     | 说明                                     |
| ---------------------------------------- | ---------------------------------------- |
| `POST /api/upload_pdf`                   | 上传私有 PDF                             |
| `GET /api/uploaded_papers/list`          | 列出已上传论文                           |
| `POST /api/uploaded_papers/process`      | 处理上传（解析 + 抽取 + 总结）           |
| `POST /api/uploaded_papers/parse`        | 解析上传 PDF                             |
| `POST /api/uploaded_papers/extract_info` | 用 LLM 抽取元信息                        |
| `POST /api/uploaded_papers/update_meta`  | 更新上传论文元信息                       |
| `POST /api/uploaded_papers/delete`       | 删除上传论文                             |
| `POST /api/uploaded_papers/retry_parse`  | 重试解析                                 |
| `GET /api/uploaded_papers/pdf/<pid>`     | 下载上传 PDF                             |
| `GET /api/uploaded_papers/similar/<pid>` | 上传论文相似度搜索                       |
| `GET /api/uploaded_papers/tldr/<pid>`    | 获取上传论文 TL;DR（若有缓存摘要则复用） |

---

## 🔨 开发指南

### 环境搭建

```bash
# 克隆仓库
git clone https://github.com/xihuai18/arxiv-sanity-x && cd arxiv-sanity-x

# 创建 conda 环境（推荐）
conda create -n sanity python=3.10
conda activate sanity

# 安装依赖
pip install -r requirements.txt

# 安装 Node.js 依赖（用于前端构建）
npm install
```

### 前端开发

前端使用原生 JavaScript，通过 esbuild 打包：

```bash
# 生产构建（带内容哈希用于缓存）
npm run build:static

# 开发构建（无哈希，便于调试）
npm run build:dev

# 监听模式（文件变更自动重建）
npm run build:watch

# 检查 JavaScript 代码
npm run lint

# 格式化代码
npm run format
```

**注意**：`bin/up.sh` 启动脚本会自动运行构建，部署时通常不需要手动构建。

### 后端开发

```bash
# 运行开发服务器
# 如果 static/dist 缺失，先构建前端资源
npm run build:static
# 如需自动重载，请先设置 ARXIV_SANITY_RELOAD=true
python serve.py

# 或使用 gunicorn 进行类生产环境测试
bash bin/up.sh
```

### 配置管理

```bash
# 显示当前配置
python -m config.cli show

# 验证配置
python -m config.cli validate

# 生成环境变量模板
python -m config.cli env
```

### 测试

```bash
# 运行所有测试
pytest

# 运行特定类别的测试
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### 代码风格

- Python：遵循 PEP 8，使用类型提示
- JavaScript：ESLint + Prettier
- 日志：Python 中使用 `loguru`

### 架构说明

#### 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  入口层 (serve.py)                                          │
│  - Flask 应用创建                                           │
│  - Gunicorn 预加载实现 copy-on-write 内存共享               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  API 层 (backend/blueprints/)                               │
│  - 10 个 Flask Blueprint 按领域组织路由                     │
│  - 请求验证、认证、响应格式化                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  服务层 (backend/services/)                                 │
│  - 业务逻辑、缓存、搜索算法                                 │
│  - 跨 Blueprint 复用，便于独立测试                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  仓储层 (aslite/repositories.py)                            │
│  - 数据访问抽象，封装原始 DB 操作                           │
│  - 批量操作、类型提示、易于 Mock                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  数据层 (aslite/db.py)                                      │
│  - 自定义 SQLite 封装 (SqliteKV)，WAL 模式                  │
│  - 类字典接口，支持压缩                                     │
└─────────────────────────────────────────────────────────────┘
```

#### 核心设计模式

1. **Repository 模式**：`PaperRepository`、`TagRepository`、`ReadingListRepository` 等提供清晰的数据访问抽象
2. **服务层模式**：`data_service`（多级缓存）、`search_service`（查询解析与排序）、`summary_service`（总结生成编排）
3. **工厂模式**：`create_app()` 创建配置好的 Flask 实例，支持测试和 Gunicorn 预加载
4. **任务队列模式**：Huey + SQLite 后端实现异步总结生成，支持优先队列
5. **缓存旁路模式**：特征缓存（mtime 失效）、论文缓存（内存 LRU）、总结缓存（文件+锁）

#### 数据流：从 arXiv 到展示

```
arXiv API → arxiv_daemon.py → papers.db/dict.db
                    ↓
            compute.py → features.p (TF-IDF + Embeddings)
                    ↓
用户搜索 → search_service → 排序结果 → 前端渲染
                    ↓
点击总结 → Huey 任务 → HTML/PDF 解析 → LLM → 缓存 → SSE 推送
```

---

## 📈 更新日志

### Unreleased

### v3.3 - OpenCode 迁移与 LiteLLM 废弃

- 🔄 **LiteLLM → OpenCode**：用 [OpenCode](https://opencode.ai) 替换 LiteLLM 作为统一 LLM 后端。所有文本生成和结构化抽取现在通过 OpenCode HTTP 服务器（`opencode serve`）完成，默认由启动器管理（`ARXIV_SANITY_OPENCODE_MANAGED=true`）
- 🏗️ **模型别名系统**：新增 `config/model_aliases.py`，将短别名（如 `gpt-5.4`）映射到 OpenCode `provider/model` 规范 ID，并支持自动回退链
- 🔧 **新增 OpenCode 配置**：`ARXIV_SANITY_OPENCODE_*` 环境变量前缀，覆盖服务地址、端口、认证、超时 — 详见[配置指南](#配置指南)
- 🩺 **健康/就绪检查**：`/ready` 现在通过 OpenCode `/global/health` 和 `/config/providers` 验证服务健康与必需模型可用性
- 📄 **文档**：新增 `docs/` 指南（运维/安全/开发）并在 README 中链接
- 📡 **API 文档**：补充 `/api/task_status/<task_id>` owner-only 字段说明（包括 `stage`）
- 📡 **API 文档**：修正上传相关端点补齐 `/api` 前缀，并补充 `/api/uploaded_papers/process`
- 📊 **可观测性**：补充可选 Sentry（`ARXIV_SANITY_SENTRY_*`）与 Prometheus metrics（`/metrics`）说明
- 🗑️ **已移除**：`config/llm.yml`、`config/llm_template.yml` 及所有 LiteLLM 运行时依赖

### v3.2 - 上传功能、测试增强与安全加固

- 📤 **论文上传**：上传私有 PDF 文件，与论文库进行相似度搜索（实验性功能）
- 🧪 **测试套件增强**：全面的单元测试和集成测试，覆盖 API、服务和工具函数
- 🔒 **安全修复**：
    - 标签搜索 API（`/api/tag_search`、`/api/tags_search`）现在需要登录认证并验证用户身份
    - 邮箱验证支持现代长 TLD（最长 63 字符，如 `.engineering`、`.museum`）
    - 语义搜索增加 pid 列表缺失时的防御性检查，避免 IndexError
- 🛠️ **守护进程改进**：
    - `ARXIV_SANITY_DAEMON_ENABLE_EMBEDDINGS=false` 现在通过 `--no-embeddings` 标志正确禁用嵌入
    - 支持邮件干运行模式（`ARXIV_SANITY_DAEMON_EMAIL_DRY_RUN`）
- 🏗️ **架构重构**：
    - Repository 模式实现更清晰的数据访问（`aslite/repositories.py`）
    - 原生 SQLite3 替代 sqlitedict，提升并发性能
    - 跨进程数据库操作锁机制
- 🎨 **前端优化**：MathJax 集成重构、静态资源清理、同步加载优化

### v3.1 - 阅读列表与增强标签系统

- 📚 **阅读列表**：个人论文收藏功能，支持添加/移除论文，专属 `/readinglist` 页面
- 👍👎 **正负反馈标签**：增强的反馈系统，支持正向和负向标签状态用于 SVM 训练
- ⚖️ **SVM 负权重**：新增 `SVM_NEG_WEIGHT` 配置参数，控制显式负反馈的影响力
- 🔄 **实时状态同步**：基于 BroadcastChannel 的跨标签页和组件状态同步
- 📊 **摘要状态显示**：摘要生成的可视化状态指示器（排队中/运行中/完成/失败）
- 🏷️ **arXiv 标签分组**：arXiv 类别分组显示，关于页面动态更新
- 🎨 **UI 优化**：增强的标签下拉交互、确认对话框和视觉反馈

### v3.0 - UI 重设计与 HTML 总结

- 🎨 **UI 改版**：关于、个人中心、统计页面现代化布局重设计
- 📄 **HTML 总结**：ar5iv/arxiv HTML 解析（比 PDF 更快，结构更好）
- 🤖 **模型选择**：总结页面支持多 LLM 模型切换和自动重试
- 🔍 **增强搜索**：键盘快捷键（Ctrl+K）、高级过滤器、无障碍改进
- 📊 **统计图表**：每日论文数量柱状图可视化

<details>
<summary>📜 历史版本（v1.0 - v2.4）</summary>

### v2.4 - 多线程批量处理与服务完善

- ⚡ **并发优化**：真正的多线程并发论文总结处理
- 🔒 **线程安全**：文件级锁机制避免 minerU 解析冲突
- 📊 **统计增强**：详细的处理统计和失败原因分析
- 🔄 **重试机制**：智能重试失败的论文处理任务

### v2.3 - AI 论文总结系统

- ✨ **新功能**：完整的 AI 驱动论文总结系统
- 🧠 **MinerU 集成**：高级 PDF 解析，支持结构识别
- 📝 **总结界面**：新的 `/summary` 路由，支持异步加载

### v2.2 - 性能与稳定性改进

- ⚡ **性能提升**：增强统一数据缓存系统，支持智能自动重载
- 📈 **调度器增强**：将获取频率增加到每日 4 次

### v2.1 - API 与语义搜索

- ✨ **新功能**：语义搜索，支持关键词、语义和混合模式
- 🔗 **API 集成**：提供 RESTful API 端点用于推荐

### v2.0 - 增强机器学习功能

- ✨ **新功能**：混合 TF-IDF + 嵌入向量特征
- ⚡ **性能优化**：多核优化和 Intel scikit-learn 扩展

### v1.0 - 基础版本

- 📚 arXiv 论文获取和存储，使用 SQLite 数据库
- 🏷️ 用户标签和关键词系统
- 📧 邮件推荐服务
- 🤖 基于 SVM 的论文推荐

</details>

---

## 📝 许可证

本项目采用 MIT 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## ⭐ 致谢

- 原始 [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) 项目，作者 Andrej Karpathy
- [minerU](https://github.com/opendatalab/MinerU) 提供高级 PDF 解析功能
- [Ollama](https://github.com/ollama/ollama) 提供本地嵌入服务
- [vLLM](https://github.com/vllm-project/vllm) 提供 MinerU VLM 服务
