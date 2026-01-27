# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

基于现代机器学习技术的综合性 arXiv 论文浏览和推荐系统，集成 AI 智能总结、混合搜索功能和个性化推荐。采用 TF-IDF、语义嵌入和 LLM 集成等先进技术。

![Screenshot](arxiv-sanity-x.png)

## 📋 目录

### 入门
- [核心功能概览](#-核心功能概览)
- [快速开始](#-快速开始)

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

## 🎯 核心功能概览

arxiv-sanity-X 是一个面向个人科研/工程阅读流的 arXiv 工作台：把「拉取与索引论文」「快速检索」「基于反馈的推荐」以及「AI 总结」整合到同一个网站里，帮助你更快找到重点、沉淀标签体系，并持续跟踪最新论文。

### 主要能力

| 功能 | 说明 |
|------|------|
| 🔍 **多模式搜索** | 关键词（TF-IDF）、语义（Embedding）、混合搜索，权重可调 |
| 🎯 **智能推荐** | 基于正/负反馈标签训练 SVM 分类器，个性化推荐 |
| 🤖 **AI 总结** | HTML/PDF 解析 + LLM 生成结构化总结，支持多模型切换 |
| 🏷️ **标签系统** | 正/负反馈、组合标签、关键词跟踪、阅读列表 |
| 📧 **邮件推荐** | 每日自动推荐邮件，假期感知调度 |
| 🔄 **自动化** | 内置调度器：获取 → 计算 → 总结 → 邮件 |

## 🛠️ 技术栈

### 后端

- **框架**：Flask，基于 Blueprint 的模块化架构
- **数据库**：SQLite，自定义 KV 存储（WAL 模式，支持压缩）
- **任务队列**：Huey（SQLite 后端），用于异步摘要生成
- **配置管理**：pydantic-settings，类型安全的配置管理
- **实时通信**：Server-Sent Events (SSE) 实时推送

### 前端

- **模板引擎**：Jinja2，响应式 HTML/CSS
- **JavaScript**：原生 JS，esbuild 打包
- **渲染**：MathJax 渲染 LaTeX，markdown-it 渲染 Markdown
- **构建**：esbuild，支持内容哈希缓存

### 机器学习/AI

- **搜索**：TF-IDF（scikit-learn）+ 语义嵌入（Ollama/OpenAI API）
- **推荐**：基于用户反馈训练的 SVM 分类器
- **摘要生成**：OpenAI 兼容的 LLM API
- **PDF 解析**：MinerU（API 或本地 VLM）

### 基础设施

- **Web 服务器**：Gunicorn，多 worker 支持
- **调度器**：APScheduler，自动化流水线
- **服务组件**：LiteLLM 网关、Ollama 嵌入、MinerU VLM

## 📁 项目结构

```
arxiv-sanity-x/
├── serve.py              # Flask 入口
├── tasks.py              # Huey 任务定义
│
├── backend/              # Flask 应用
│   ├── app.py            # 应用工厂 & 初始化
│   ├── blueprints/       # 路由处理器（8 个 Blueprint）
│   │   ├── web.py        # 页面路由（/, /summary, /profile 等）
│   │   ├── api_user.py   # 用户认证 & 状态
│   │   ├── api_search.py # 搜索端点
│   │   ├── api_summary.py# 摘要生成 & 状态
│   │   ├── api_tags.py   # 标签管理
│   │   ├── api_papers.py # 论文数据 & 图片
│   │   ├── api_readinglist.py # 阅读列表
│   │   └── api_sse.py    # Server-Sent Events
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
│   └── llm.yml           # LiteLLM 网关配置
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
│   ├── embedding_serve.sh# Ollama 嵌入服务
│   ├── mineru_serve.sh   # MinerU VLM 服务
│   └── litellm.sh        # LiteLLM 网关
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
└── data/                 # 运行时数据（gitignore）
    ├── papers.db         # 论文元数据
    ├── dict.db           # 用户数据（标签、关键词等）
    ├── features.p        # 计算的特征
    └── summary/          # 缓存的摘要
```

## 🧭 用户使用指南

本节介绍如何使用 arxiv-sanity-X 网站的各项功能。大多数操作都从首页开始。

### 1）登录

- 点击右上角 **Profile** 进入个人中心
- 输入用户名登录（无密码，适合个人/内网使用）
- 若要公网部署，建议放在统一认证/VPN 后面，并设置稳定会话密钥（`ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt`）

### 2）浏览与检索论文

**首页功能：**

- 默认按时间排序显示最新论文
- 点击论文标题查看详情，点击 arXiv 链接跳转原文
- 使用顶部搜索框进行检索（支持快捷键 `Ctrl+K`）

**搜索语法：**
| 语法 | 示例 | 说明 |
|------|------|------|
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

- 点击论文卡片上的 **📚** 按钮加入阅读列表
- 访问 `/readinglist` 页面管理阅读列表
- 可用于批量总结或稍后阅读

### 6）其他功能

- **Stats 页面**：查看论文统计、每日新增图表
- **About 页面**：查看系统信息、支持的 arXiv 分类
- **邮件推荐**：配置 SMTP 后可接收每日推荐邮件（详见配置指南）

## 📦 数据目录与迁移

默认数据存放在 `data/`（由 `.env` / `config/settings.py` 中的 `ARXIV_SANITY_DATA_DIR` 决定）：

- `data/papers.db`：论文与元信息（由 arXiv 拉取）
- `data/dict.db`：用户数据（标签/负反馈/关键词/阅读列表/邮箱登记/总结状态等）
- `data/features.p`：由 [compute.py](compute.py) 生成的 TF‑IDF/混合特征
- `data/summary/`：LLM 总结缓存
- `data/pdfs/`、`data/mineru/`、`data/html_md/`：解析相关的中间缓存

迁移到新机器时，通常至少复制：

- `data/papers.db`
- `data/dict.db`
- `data/features.p`（或在新环境重新运行 [compute.py](compute.py) 生成）
- `data/summary/`（可选：想保留已缓存总结时再带上）

## 🔐 部署与安全说明

- 站点内置登录是“仅用户名、无密码”，适合个人/内网环境。
- 如果需要公网部署，务必放在统一认证/VPN/反向代理鉴权后面，并通过 `ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt` 设置稳定的会话密钥。
- 不要把真实 API Key 写进仓库；优先使用环境变量注入。

## 🧩 常见问题与排错

- **网站空白/没有论文**：通常是还没跑 [arxiv_daemon.py](arxiv_daemon.py) + [compute.py](compute.py)。
- **总结一直失败**：检查 `.env` 里的 `ARXIV_SANITY_LLM_API_KEY`、`ARXIV_SANITY_LLM_BASE_URL`、`ARXIV_SANITY_LLM_NAME`。
- **语义/混合检索没效果**：确认嵌入（Embedding）已启用，并用 [compute.py](compute.py) 重新生成特征（混合特征需要包含嵌入）。
- **MinerU 报错**：
    - API 后端：检查 `MINERU_API_KEY`（或 `ARXIV_SANITY_MINERU_API_KEY`）
    - 本地后端：检查 `ARXIV_SANITY_MINERU_BACKEND`，以及服务是否能在 `MINERU_PORT` 访问
- **崩溃后卡住（锁文件）**：运行 [cleanup_locks.py](cleanup_locks.py)，或调整 `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC` / `ARXIV_SANITY_MINERU_LOCK_STALE_SEC`。
- **features.p 读取失败（NumPy 版本不匹配）**：在当前环境重新运行 [compute.py](compute.py) 生成特征文件。

## ⚡ 快速开始

本项目 Web 本体开箱即用，但会依赖你选择的**外部模型服务**（LLM / Embedding / MinerU）。建议先选一个“运行档位”，再按步骤操作。

### 推荐运行档位

| 档位               | 你能得到什么                     | 需要什么                     | 适合                |
| ------------------ | -------------------------------- | ---------------------------- | ------------------- |
| **最小（仅 LLM）** | 浏览 + TF‑IDF 搜索 + LLM 总结    | LLM API key                  | 上手体验 / 资源有限 |
| **混合搜索**       | TF‑IDF + Embedding 混合搜索      | LLM API key + Embedding 后端 | 更好的检索相关性    |
| **完整（MinerU）** | 更强的 PDF→Markdown（公式/表格） | MinerU（API 或本地）         | 最佳总结可读性      |

### 1. 安装

```bash
# 克隆并安装
git clone https://github.com/xihuai18/arxiv-sanity-x && cd arxiv-sanity-x
pip install -r requirements.txt
```

### 2. 创建配置文件

```bash
# 必须：从模板创建 .env
cp .env.example .env

# 可选：创建 LiteLLM 配置（使用多模型网关时）
cp config/llm_template.yml config/llm.yml
```

### 3. 配置核心设置

编辑 `.env`（由 [.env.example](.env.example) 复制生成）。至少建议检查：**LLM 设置**，以及可选的**总结来源 / Embedding / MinerU**。

```bash
# LLM API（论文总结必需）
ARXIV_SANITY_LLM_BASE_URL=https://openrouter.ai/api/v1
ARXIV_SANITY_LLM_API_KEY=your-api-key
ARXIV_SANITY_LLM_NAME=deepseek/deepseek-chat-v3.1:free
ARXIV_SANITY_LLM_SUMMARY_LANG=zh

# Web
ARXIV_SANITY_HOST=http://localhost:55555
ARXIV_SANITY_SERVE_PORT=55555

# 总结来源（默认 HTML 快且稳定）
ARXIV_SANITY_SUMMARY_SOURCE=html
ARXIV_SANITY_SUMMARY_HTML_SOURCES=ar5iv,arxiv

# 邮件（可选，用于每日推荐）
ARXIV_SANITY_EMAIL_FROM_EMAIL=your_email@mail.com
ARXIV_SANITY_EMAIL_SMTP_SERVER=smtp.mail.com
ARXIV_SANITY_EMAIL_SMTP_PORT=465
ARXIV_SANITY_EMAIL_USERNAME=username
ARXIV_SANITY_EMAIL_PASSWORD=your-password

# Embedding（可选）
# ARXIV_SANITY_EMBED_USE_LLM_API=true
# ARXIV_SANITY_EMBED_MODEL_NAME=qwen3-embedding:0.6b

# MinerU（可选）
# ARXIV_SANITY_MINERU_ENABLED=true
# ARXIV_SANITY_MINERU_BACKEND=api
# MINERU_API_KEY=your-mineru-api-key
```

同时请检查 [tools/arxiv_daemon.py](tools/arxiv_daemon.py) 里的 arXiv 分类分组（`CORE/LANG/AGENT/APP/ALL_TAGS`），它决定你到底拉取/展示哪些领域的论文。

### 4. 验证配置

```bash
# 显示当前配置
python -m config.cli show

# 验证配置
python -m config.cli validate
```

### 5. 获取论文并启动

```bash
# 获取论文并计算特征
python -m tools arxiv_daemon -n 10000 -m 500
python -m tools compute --num 20000

# 一键启动所有服务
python bin/run_services.py

# 访问 http://localhost:55555
```

### 服务启动方式详解

根据你的需求，可以选择不同的启动方式：

#### 方式一：最简启动（仅 Web）

```bash
# 开发模式（支持热重载）
python serve.py

# 生产模式（Gunicorn）
./bin/up.sh
```

#### 方式二：一键启动（推荐）

```bash
# 启动 Web + 可选服务（Embedding/MinerU/LiteLLM）
python bin/run_services.py

# 常用选项
python bin/run_services.py --no-embed      # 不启动 Embedding 服务
python bin/run_services.py --no-mineru     # 不启动 MinerU 服务
python bin/run_services.py --no-litellm    # 不启动 LiteLLM 网关
python bin/run_services.py --with-daemon   # 同时启动定时任务调度器
```

#### 方式三：分别启动各服务

```bash
# 终端 1：Web 服务
./bin/up.sh

# 终端 2：Embedding 服务（可选）
./bin/embedding_serve.sh

# 终端 3：MinerU 服务（可选）
./bin/mineru_serve.sh

# 终端 4：LiteLLM 网关（可选）
./bin/litellm.sh

# 终端 5：定时任务调度器（可选）
python -m tools daemon
```

#### 方式四：一次性数据初始化

```bash
# 仅拉取论文和计算特征，不启动服务
python bin/run_services.py --fetch-compute 10000
```

> **提示**：如果你想在一个终端里把 embedding / minerU / litellm 一起拉起来，推荐用 [bin/run_services.py](bin/run_services.py)。但注意它会调用 bash 脚本（见下方"系统说明"）。

### 配置清单

| 项目               | 文件/位置                                      | 必需    | 说明                                                                             |
| ------------------ | ---------------------------------------------- | ------- | -------------------------------------------------------------------------------- |
| **核心配置**       | [.env](.env.example)                           | ✅ 必需 | 所有配置通过环境变量设置                                                         |
| **LLM 服务**       | `.env`                                         | ✅ 必需 | `ARXIV_SANITY_LLM_BASE_URL`、`ARXIV_SANITY_LLM_NAME`、`ARXIV_SANITY_LLM_API_KEY` |
| **arXiv 分类**     | [tools/arxiv_daemon.py](tools/arxiv_daemon.py) | ⚙️ 重要 | `CORE/LANG/AGENT/APP/ALL_TAGS` 决定拉取范围与 About 展示                         |
| **总结来源**       | `.env`                                         | ⚙️ 推荐 | `ARXIV_SANITY_SUMMARY_SOURCE=html\|mineru`                                       |
| **Embedding 后端** | `.env`                                         | ⚙️ 可选 | `ARXIV_SANITY_EMBED_*` 相关设置                                                  |
| **MinerU 后端**    | `.env`                                         | ⚙️ 可选 | `ARXIV_SANITY_MINERU_*` 相关设置 + `MINERU_API_KEY`                              |
| **邮件 SMTP**      | `.env`                                         | ⚙️ 可选 | `ARXIV_SANITY_EMAIL_*` 相关设置                                                  |
| **会话密钥**       | 环境变量/文件                                  | ⚙️ 推荐 | `ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt`（公网部署强烈建议）                |

---

## 🧰 运行前准备与系统说明

### Python

- 推荐 Python 3.10+
- 依赖见 [requirements.txt](requirements.txt)

### 你可能需要的外部服务

- **LLM 服务商**（OpenAI 兼容 API）：用于总结（必需）。
- **Ollama**（可选）：当你选择本地 embedding 时，由 [bin/embedding_serve.sh](bin/embedding_serve.sh) 启动。
- **MinerU**（可选）：
    - `api` 后端：走 mineru.net，需要 `MINERU_API_KEY`
    - 本地 VLM 后端：由 [bin/mineru_serve.sh](bin/mineru_serve.sh) 启动 `mineru-vllm-server`
- **LiteLLM**（可选）：多模型网关，由 [config/llm.yml](config/llm.yml) 配置。

### Windows 注意事项

部分启动脚本是 bash（[bin/up.sh](bin/up.sh)、[bin/embedding_serve.sh](bin/embedding_serve.sh)、[bin/mineru_serve.sh](bin/mineru_serve.sh)、[bin/litellm.sh](bin/litellm.sh)），而 [bin/run_services.py](bin/run_services.py) 会用 `bash` 调它们。

- Windows 建议使用 **WSL**（最省心）。
- 或使用能提供 bash 的环境。
- 只跑 Web 的话也可以直接 `python serve.py`，并把 embedding/MinerU 走 API 后端。

## 配置指南

### 配置概览

本项目使用 **pydantic-settings** 进行配置管理。所有配置通过环境变量或 `.env` 文件设置。

| 来源                                           | 作用                           | 必需    |
| ---------------------------------------------- | ------------------------------ | ------- |
| [.env](.env.example)                           | 所有配置设置                   | ✅ 必须 |
| [tools/arxiv_daemon.py](tools/arxiv_daemon.py) | arXiv 分类列表（论文采集范围） | ⚙️ 重要 |
| [config/llm.yml](config/llm.yml)               | LiteLLM 多模型网关             | ⚙️ 可选 |

**仓库中不包含的文件（.gitignore）：**

- `.env` - 从 [.env.example](.env.example) 复制
- `config/llm.yml` - 从 [config/llm_template.yml](config/llm_template.yml) 复制
- `secret_key.txt` - 可选，Flask 会话密钥
- `data/` - 运行时自动生成
- 本地嵌入模型（如 `qwen3-embed-0.6B/`）

---

### 1. .env 文件 - 核心配置

从 `.env.example` 复制到 `.env` 并配置以下部分：

#### 1.1 数据存储

```bash
ARXIV_SANITY_DATA_DIR=data                    # 数据存储根目录（推荐 SSD）
ARXIV_SANITY_SUMMARY_DIR=data/summary         # 论文总结缓存目录
```

#### 1.2 服务端口

```bash
ARXIV_SANITY_SERVE_PORT=55555      # Web 应用端口
ARXIV_SANITY_EMBED_PORT=54000      # Ollama 嵌入服务端口
ARXIV_SANITY_MINERU_PORT=52000     # MinerU VLM 服务端口
ARXIV_SANITY_LITELLM_PORT=53000    # LiteLLM 网关端口
```

#### 1.3 LLM API 配置

```bash
# 方式 1：直接 API（OpenRouter、OpenAI 等）
ARXIV_SANITY_LLM_BASE_URL=https://openrouter.ai/api/v1
ARXIV_SANITY_LLM_API_KEY=your-api-key
ARXIV_SANITY_LLM_NAME=deepseek/deepseek-chat-v3.1:free
ARXIV_SANITY_LLM_SUMMARY_LANG=zh

# 方式 2：通过 LiteLLM 网关（需要 config/llm.yml）
ARXIV_SANITY_LLM_BASE_URL=http://localhost:53000
ARXIV_SANITY_LLM_API_KEY=no-key
ARXIV_SANITY_LLM_NAME=or-mimo
```

#### 1.4 嵌入配置

```bash
# 使用 OpenAI 兼容 API 生成嵌入（默认）
ARXIV_SANITY_EMBED_USE_LLM_API=true
ARXIV_SANITY_EMBED_MODEL_NAME=qwen3-embedding:0.6b
ARXIV_SANITY_EMBED_API_BASE=       # 空 = 使用 LLM_BASE_URL
ARXIV_SANITY_EMBED_API_KEY=        # 空 = 使用 LLM_API_KEY

# 或使用本地 Ollama 服务
ARXIV_SANITY_EMBED_USE_LLM_API=false  # 使用 http://localhost:{EMBED_PORT}
```

#### 1.5 邮件服务

```bash
ARXIV_SANITY_EMAIL_FROM_EMAIL=your_email@mail.com
ARXIV_SANITY_EMAIL_SMTP_SERVER=smtp.mail.com
ARXIV_SANITY_EMAIL_SMTP_PORT=465
ARXIV_SANITY_EMAIL_USERNAME=username
ARXIV_SANITY_EMAIL_PASSWORD=your-password
ARXIV_SANITY_HOST=http://your-server:55555  # 邮件链接的公网地址
```

#### 1.6 论文总结配置

```bash
ARXIV_SANITY_SUMMARY_MIN_CHINESE_RATIO=0.25      # 缓存有效性的最低中文比例
ARXIV_SANITY_SUMMARY_DEFAULT_SEMANTIC_WEIGHT=0.5 # 混合搜索权重（0.0-1.0）
ARXIV_SANITY_SUMMARY_SOURCE=html                 # "html"（默认）或 "mineru"
ARXIV_SANITY_SUMMARY_HTML_SOURCES=ar5iv,arxiv    # HTML 来源优先顺序
```

#### 1.7 MinerU PDF 解析

```bash
ARXIV_SANITY_MINERU_ENABLED=true
ARXIV_SANITY_MINERU_BACKEND=api                  # "api"、"pipeline" 或 "vlm-http-client"
ARXIV_SANITY_MINERU_DEVICE=cuda                  # "cuda" 或 "cpu"（仅 pipeline）
ARXIV_SANITY_MINERU_MAX_WORKERS=2
ARXIV_SANITY_MINERU_MAX_VRAM=4
MINERU_API_KEY=your-mineru-api-key               # API 后端密钥
```

#### 1.8 SVM 推荐参数

```bash
ARXIV_SANITY_SVM_C=0.02
ARXIV_SANITY_SVM_MAX_ITER=5000
ARXIV_SANITY_SVM_TOL=0.001
ARXIV_SANITY_SVM_NEG_WEIGHT=5.0
```

---

### 2. arxiv_daemon.py - arXiv 分类

论文采集查询由 [tools/arxiv_daemon.py](tools/arxiv_daemon.py) 中的 `ALL_TAGS` 构建。自定义这些分组以控制采集哪些 arXiv 分类：

```python
# 默认分类组（按需编辑）
CORE = ["cs.AI", "cs.LG", "stat.ML"]           # 核心 AI/ML
LANG = ["cs.CL", "cs.IR", "cs.CV"]             # NLP、信息检索、计算机视觉
AGENT = ["cs.MA", "cs.RO", "cs.HC", "cs.GT", "cs.NE"]  # 智能体、机器人、人机交互
APP = ["cs.SE", "cs.CY"]                        # 软件工程、网络安全

ALL_TAGS = CORE + LANG + AGENT + APP
```

查询构建为 `cat:cs.AI OR cat:cs.LG OR ...`。根据您的研究兴趣添加或删除分类。

**常用 arXiv CS 分类：**

- `cs.AI` - 人工智能
- `cs.LG` - 机器学习
- `cs.CL` - 计算与语言（NLP）
- `cs.CV` - 计算机视觉
- `cs.RO` - 机器人学
- `cs.NE` - 神经与进化计算
- `stat.ML` - 统计机器学习

完整列表请参见 [arXiv 分类体系](https://arxiv.org/category_taxonomy)。

---

### 3. llm.yml - LiteLLM 网关

如果您想使用 LiteLLM 作为多 LLM 服务商的统一网关，请将 `config/llm_template.yml` 复制为 `config/llm.yml`。

```yaml
model_list:
    # OpenRouter - 免费模型
    - model_name: or-mimo # .env 中 ARXIV_SANITY_LLM_NAME 使用的别名
      litellm_params:
          model: openrouter/xiaomi/mimo-v2-flash:free
          api_base: https://openrouter.ai/api/v1
          api_key: YOUR_OPENROUTER_API_KEY # 替换为您的密钥
          max_tokens: 32768

    - model_name: or-glm
      litellm_params:
          model: openai/z-ai/glm-4.5-air:free
          api_base: https://openrouter.ai/api/v1
          api_key: YOUR_OPENROUTER_API_KEY

litellm_settings:
    drop_params: true
```

**使用方法：**

```bash
# 启动 LiteLLM 网关
litellm -c config/llm.yml --port 53000

# 或使用 run_services.py（自动启动 LiteLLM）
python bin/run_services.py
```

然后配置 `.env`：

```bash
ARXIV_SANITY_LLM_BASE_URL=http://localhost:53000
ARXIV_SANITY_LLM_API_KEY=no-key
ARXIV_SANITY_LLM_NAME=or-mimo  # 使用 llm.yml 中的别名
```

---

### 4. 配置 CLI 工具

项目提供了配置管理 CLI 工具：

```bash
# 显示当前配置
python -m config.cli show

# JSON 格式输出
python -m config.cli show --json

# 验证配置
python -m config.cli validate

# 生成环境变量模板
python -m config.cli env
```

#### 在代码中使用配置

```python
from config import settings

# 访问配置
print(settings.data_dir)
print(settings.llm.base_url)
print(settings.llm.api_key)
print(settings.mineru.enabled)
print(settings.email.smtp_server)
```

| 变量                              | 默认值 | 说明                                   |
| --------------------------------- | ------ | -------------------------------------- |
| `ARXIV_SANITY_MINERU_ENABLED`     | `true` | 启用/禁用 MinerU                       |
| `ARXIV_SANITY_MINERU_BACKEND`     | `api`  | `api`、`pipeline` 或 `vlm-http-client` |
| `ARXIV_SANITY_MINERU_DEVICE`      | `cuda` | pipeline 后端设备                      |
| `ARXIV_SANITY_MINERU_MAX_WORKERS` | `2`    | 最大并发 minerU 进程数                 |
| `ARXIV_SANITY_MINERU_MAX_VRAM`    | `3`    | 每进程最大显存（GB）                   |
| `MINERU_API_POLL_INTERVAL`        | `5`    | API 轮询间隔（秒）                     |
| `MINERU_API_TIMEOUT`              | `600`  | API 任务超时（秒）                     |

#### 锁与并发

| 变量                                  | 默认值 | 说明                                               |
| ------------------------------------- | ------ | -------------------------------------------------- |
| `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC` | `600`  | 总结缓存锁“过期清理”阈值（异常退出后建议保留默认） |
| `ARXIV_SANITY_MINERU_LOCK_STALE_SEC`  | `3600` | MinerU 解析 / GPU-slot 锁过期清理阈值              |

#### 嵌入

| 变量                             | 默认值 | 说明                  |
| -------------------------------- | ------ | --------------------- |
| `ARXIV_SANITY_EMBED_USE_LLM_API` | `true` | 使用 LLM API 生成嵌入 |

#### 守护进程/调度器

| 变量                             | 默认值 | 说明                     |
| -------------------------------- | ------ | ------------------------ |
| `ARXIV_SANITY_FETCH_NUM`         | `2000` | 每次获取的论文数         |
| `ARXIV_SANITY_FETCH_MAX`         | `1000` | 每次 API 查询最大结果数  |
| `ARXIV_SANITY_SUMMARY_NUM`       | `200`  | 每次总结的论文数         |
| `ARXIV_SANITY_SUMMARY_WORKERS`   | `2`    | 总结工作线程数           |
| `ARXIV_SANITY_DAEMON_SUMMARY`    | `1`    | 守护进程中启用总结生成   |
| `ARXIV_SANITY_DAEMON_EMBEDDINGS` | `1`    | 守护进程中启用嵌入       |
| `ARXIV_SANITY_PRIORITY_QUEUE`    | `1`    | 启用总结优先队列         |
| `ARXIV_SANITY_PRIORITY_DAYS`     | `2`    | 优先窗口（天）           |
| `ARXIV_SANITY_PRIORITY_LIMIT`    | `100`  | 最大优先论文数           |
| `ARXIV_SANITY_ENABLE_GIT_BACKUP` | `1`    | 启用 dict.db 的 git 备份 |

#### 网络 / 代理

- `http_proxy`、`https_proxy`：被 [arxiv_daemon.py](arxiv_daemon.py) 等出网请求使用。

#### Gunicorn（up.sh）

| 变量                            | 默认值 | 说明                 |
| ------------------------------- | ------ | -------------------- |
| `GUNICORN_WORKERS`              | `2`    | 工作进程数           |
| `GUNICORN_THREADS`              | `4`    | 每工作进程线程数     |
| `ARXIV_SANITY_GUNICORN_PRELOAD` | `1`    | 在主进程中预加载应用 |
| `GUNICORN_EXTRA_ARGS`           | ``     | 额外的 gunicorn 参数 |

---

### 5. 启动参数

#### run_services.py

```bash
# 一键启动（推荐）
python bin/run_services.py

# Web 服务器选项
python bin/run_services.py --web gunicorn    # 使用 gunicorn
python bin/run_services.py --web none        # 不启动 Web 服务器

# 跳过重型服务
python bin/run_services.py --no-embed        # 跳过 Ollama 嵌入
python bin/run_services.py --no-mineru       # 跳过 MinerU
python bin/run_services.py --no-litellm      # 跳过 LiteLLM 网关

# 总结来源
python bin/run_services.py --summary-source html
python bin/run_services.py --summary-source mineru

# 包含调度器守护进程
python bin/run_services.py --with-daemon

# 一次性：仅获取和计算
python bin/run_services.py --fetch-compute         # 默认 10000 篇论文
python bin/run_services.py --fetch-compute 1000    # 自定义数量
```

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

### LLM 服务商示例

#### OpenRouter（免费模型）

```python
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = "sk-or-v1-..."
LLM_NAME = "deepseek/deepseek-chat-v3.1:free"
```

#### OpenAI

```python
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "sk-..."
LLM_NAME = "gpt-4o-mini"
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
bash embedding_serve.sh  # 在 EMBED_PORT 启动

# 使用嵌入计算
python -m tools compute --use_embeddings --embed_model nomic-embed-text
```

### 自动化调度

**内置调度器：**

```bash
python -m tools daemon
```

调度计划（Asia/Shanghai 时区）：

- **获取+计算**：工作日 8:00、12:00、16:00、20:00
- **发送邮件**：工作日 18:00
- **备份**：每日 20:00

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

系统提供 54 个 API 端点，分布在 8 个 Blueprint 模块中。

### 页面路由 (web.py)

| 路由 | 说明 |
|------|------|
| `GET /` | 首页，论文列表 |
| `GET /summary` | 论文总结页面 |
| `GET /profile` | 用户个人中心 |
| `GET /stats` | 系统统计页面 |
| `GET /about` | 关于页面 |
| `GET /readinglist` | 阅读列表页面 |

### 搜索与推荐 (api_search.py)

| 端点 | 说明 |
|------|------|
| `GET /?rank=search&q=<query>` | 关键词搜索（TF-IDF） |
| `GET /?rank=search&q=<query>&search_mode=semantic` | 语义搜索 |
| `GET /?rank=search&q=<query>&search_mode=hybrid&semantic_weight=0.5` | 混合搜索 |
| `GET /?rank=tags&tags=<tag_list>&logic=<and\|or>` | 基于标签的 SVM 推荐 |
| `GET /?rank=time&time_filter=<days>` | 时间过滤论文 |
| `GET /?rank=pid&pid=<paper_id>` | 相似论文推荐 |
| `POST /api/tag_search` | 单标签搜索（需认证） |
| `POST /api/tags_search` | 多标签搜索（需认证） |

### 论文总结 (api_summary.py)

| 端点 | 说明 |
|------|------|
| `POST /api/get_paper_summary` | 获取/生成论文总结 |
| `POST /api/clear_model_summary` | 清除特定模型的总结缓存 |
| `POST /api/clear_paper_cache` | 清除论文所有缓存 |
| `GET /api/summary_status/<pid>` | 获取总结生成状态 |

### 标签管理 (api_tags.py)

| 端点 | 说明 |
|------|------|
| `GET /add/<pid>/<tag>` | 为论文添加正向标签 |
| `GET /sub/<pid>/<tag>` | 移除论文标签 |
| `GET /neg/<pid>/<tag>` | 为论文添加负向标签 |
| `GET /add_key/<keyword>` | 添加跟踪关键词 |
| `GET /del_key/<keyword>` | 移除跟踪关键词 |
| `POST /api/add_combined_tag` | 添加组合标签 |
| `POST /api/remove_combined_tag` | 移除组合标签 |

### 阅读列表 (api_readinglist.py)

| 端点 | 说明 |
|------|------|
| `POST /api/readinglist/add` | 添加论文到阅读列表 |
| `POST /api/readinglist/remove` | 从阅读列表移除论文 |
| `GET /api/readinglist/status/<pid>` | 获取论文阅读列表状态 |

### 用户管理 (api_user.py)

| 端点 | 说明 |
|------|------|
| `POST /api/login` | 用户登录 |
| `POST /api/logout` | 用户登出 |
| `POST /api/update_email` | 更新邮箱地址 |
| `GET /api/user_state` | 获取用户状态 |

### 实时推送 (api_sse.py)

| 端点 | 说明 |
|------|------|
| `GET /api/sse/user/<username>` | 用户专属 SSE 流 |
| `GET /api/sse/global` | 全局 SSE 流 |

### 系统

| 端点 | 说明 |
|------|------|
| `GET /cache_status` | 缓存状态（需认证） |
| `POST /api/upload_paper` | 上传私有 PDF（实验性） |

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
# 运行开发服务器（自动重载）
python serve.py

# 或使用 gunicorn 进行类生产环境测试
./bin/up.sh
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
│  - 8 个 Flask Blueprint 按领域组织路由                      │
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
- 📦 **LiteLLM 模板**：`llm_template.yml` 含 OpenRouter 免费模型配置

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
