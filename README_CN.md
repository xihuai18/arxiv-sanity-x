# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

基于现代机器学习技术的综合性 arXiv 论文浏览和推荐系统，集成 AI 智能总结、混合搜索功能和个性化推荐。采用 TF-IDF、语义嵌入和 LLM 集成等先进技术。

![Screenshot](arxiv-sanity-x.png)

## 📋 目录

- [核心功能概览](#-核心功能概览)
- [用户使用指南（Web）](#-用户使用指南web)
- [最低可运行要求](#-最低可运行要求)
- [数据目录与迁移](#-数据目录与迁移)
- [部署与安全说明](#-部署与安全说明)
- [常见问题与排错](#-常见问题与排错)
- [快速开始](#-快速开始)
- [运行前准备与系统说明](#-运行前准备与系统说明)
- [配置指南](#配置指南)
  - [配置概览](#配置概览)
  - [1. vars.py - 核心配置](#1-varspy---核心配置)
  - [2. arxiv_daemon.py - arXiv 分类](#2-arxiv_daemonpy---arxiv-分类)
  - [3. llm.yml - LiteLLM 网关](#3-llmyml---litellm-网关)
  - [4. 环境变量](#4-环境变量)
  - [5. 启动参数](#5-启动参数)
- [核心功能](#-核心功能)
- [使用指南](#-使用指南)
- [AI 论文总结](#-ai-论文总结)
- [高级功能](#-高级功能)
- [API 参考](#-api-参考)
- [更新日志](#-更新日志)

---

## 🎯 核心功能概览

arxiv-sanity-X 是一个面向个人科研/工程阅读流的 arXiv 工作台：把「拉取与索引论文」「快速检索」「基于反馈的推荐」以及「AI 总结」整合到同一个网站里，帮助你更快找到重点、沉淀标签体系，并持续跟踪最新论文。

主要能力包括：

- **论文采集与索引**：按你选择的 arXiv 分类拉取论文，落库到本地 SQLite（便于长期维护与增量更新）。
- **多种检索模式**：关键词检索（TF‑IDF）、语义检索（Embedding）、混合检索（权重可调）。
- **个人组织体系**：正/负反馈标签、组合标签、关键词跟踪、阅读列表等。
- **按需 AI 总结**：支持 HTML（ar5iv/arxiv）或 PDF（MinerU）解析后再用 LLM 总结，带缓存与状态追踪。
- **自动化流水线**：可选调度器完成 获取 → 计算 → 总结 → 邮件；并提供锁清理/备份等运维工具。

## 🧭 用户使用指南（Web）

下面这段是“怎么用网站”的速览地图。大多数操作都从首页开始。

### 1）登录

- 进入 Profile，用用户名登录（无密码）。
- 若要公网部署，建议放在统一认证/VPN 后面，并设置稳定会话密钥（`ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt`）。

### 2）检索与筛选论文

- 首页搜索支持字段过滤：`ti:`（标题）、`au:`（作者）、`cat:`（分类）、`id:`（arXiv ID）。
- 可切换搜索模式：
  - **关键词**：最快，不依赖额外服务
  - **语义**：需要嵌入（Embedding）
  - **混合**：如果启用了嵌入（Embedding），通常是最推荐的默认（权重可调）

### 3）打标签并获取推荐

- 给喜欢的论文打标签，用于训练“按标签推荐”。
- 负反馈（UI 支持时）可用于压制不想看的方向。
- 组合标签（例如“RL,NLP”）可以做交集推荐。

### 4）查看 AI 总结（按需生成）

- 打开论文详情页，点击 Summary/总结。
- 网站会使用你配置的 LLM 生成总结，并缓存结果。
- 可清理当前模型的总结缓存，或清除该论文的全部缓存产物。

### 5）阅读列表

- 把论文加入阅读列表，后续集中阅读。
- 当你打算批量总结时，阅读列表也可以作为“待处理队列”。

### 6）可选：每日邮件推荐

- 在 [vars.py](vars.py) 配好 SMTP，设置 `YOUR_EMAIL_PASSWD`，并把 `HOST` 设为正确的公网 base URL。
- 在 Profile 页面填写邮箱。
- 手动运行 [send_emails.py](send_emails.py)，或启用调度器 [daemon.py](daemon.py)。

## ✅ 最低可运行要求

如果你只想用“最小配置”跑通端到端（能浏览 + 搜索 + 按需生成总结），需要满足：

1. 从模板生成 [vars.py](vars.py)：复制 [vars_template.py](vars_template.py)。
2. 配好可用的 LLM：通常用环境变量提供 `YOUR_LLM_API_KEY`，并在 [vars.py](vars.py) 设置正确的 `LLM_BASE_URL` 与 `LLM_NAME`。
3. 至少执行一次“拉取 + 特征计算”：
   - 运行 [arxiv_daemon.py](arxiv_daemon.py)
   - 运行 [compute.py](compute.py)
4. 启动 Web：运行 [serve.py](serve.py)（或在支持 bash 的环境下使用 [run_services.py](run_services.py) 一键启动）。

其余（嵌入/Embedding、MinerU、LiteLLM、邮件、调度器）都属于可选增强。

## 📦 数据目录与迁移

默认数据存放在 `data/`（由 [vars.py](vars.py) 的 `DATA_DIR` 决定）：

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
- **总结一直失败**：检查 `YOUR_LLM_API_KEY`、[vars.py](vars.py) 的 `LLM_BASE_URL`、`LLM_NAME`。
- **语义/混合检索没效果**：确认嵌入（Embedding）已启用，并用 [compute.py](compute.py) 重新生成特征（混合特征需要包含嵌入）。
- **MinerU 报错**：
  - API 后端：检查 `MINERU_API_KEY`（或 `ARXIV_SANITY_MINERU_API_KEY`）
  - 本地后端：检查 `ARXIV_SANITY_MINERU_BACKEND`，以及服务是否能在 `MINERU_PORT` 访问
- **崩溃后卡住（锁文件）**：运行 [cleanup_locks.py](cleanup_locks.py)，或调整 `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC` / `ARXIV_SANITY_MINERU_LOCK_STALE_SEC`。
- **features.p 读取失败（NumPy 版本不匹配）**：在当前环境重新运行 [compute.py](compute.py) 生成特征文件。

## ⚡ 快速开始

本项目 Web 本体开箱即用，但会依赖你选择的**外部模型服务**（LLM / Embedding / MinerU）。建议先选一个“运行档位”，再按步骤操作。

### 推荐运行档位

| 档位 | 你能得到什么 | 需要什么 | 适合 |
| --- | --- | --- | --- |
| **最小（仅 LLM）** | 浏览 + TF‑IDF 搜索 + LLM 总结 | LLM API key | 上手体验 / 资源有限 |
| **混合搜索** | TF‑IDF + Embedding 混合搜索 | LLM API key + Embedding 后端 | 更好的检索相关性 |
| **完整（MinerU）** | 更强的 PDF→Markdown（公式/表格） | MinerU（API 或本地） | 最佳总结可读性 |

### 1. 安装

```bash
# 克隆并安装
git clone https://github.com/xihuai18/arxiv-sanity-x && cd arxiv-sanity-x
pip install -r requirements.txt
```

### 2. 创建配置文件

```bash
# 必须：从模板创建 vars.py
cp vars_template.py vars.py

# 可选：创建 LiteLLM 配置（使用多模型网关时）
cp llm_template.yml llm.yml
```

### 3. 配置核心设置

编辑 `vars.py`（由 [vars_template.py](vars_template.py) 复制生成）。至少建议检查：**路径**、**端口**、**LLM**，以及可选的**总结来源 / Embedding / MinerU**。

```python
# 存储
DATA_DIR = "data"  # 尽量放 SSD

# LLM API（论文总结必需）
LLM_BASE_URL = "https://openrouter.ai/api/v1"  # 或其他 LLM 服务商
LLM_API_KEY = os.environ.get("YOUR_LLM_API_KEY", "your_api_key")
LLM_NAME = "deepseek/deepseek-chat-v3.1:free"

# Web
SERVE_PORT = 55555

# 总结来源（默认 HTML 快且稳定）
SUMMARY_MARKDOWN_SOURCE = os.environ.get("ARXIV_SANITY_SUMMARY_SOURCE", "html")  # html/mineru
SUMMARY_HTML_SOURCES = os.environ.get("ARXIV_SANITY_HTML_SOURCES", "ar5iv,arxiv")

# 邮件（可选，用于每日推荐）
from_email = "your_email@mail.com"
smtp_server = "smtp.mail.com"
smtp_port = 465
email_username = "username"
email_passwd = os.environ.get("YOUR_EMAIL_PASSWD", "")
HOST = "http://your-server:55555"  # 邮件中链接的公网地址

# Embedding（可选）
# - 如果你不想跑本地 Ollama：保持 EMBED_USE_LLM_API=True（模板默认）
# - 如果你想用本地 Ollama：设置 EMBED_USE_LLM_API=False，并在 EMBED_PORT 启动 Ollama
# EMBED_USE_LLM_API = True
# EMBED_MODEL_NAME = "qwen3-embedding:0.6b"

# MinerU（可选）
# - api: 走 mineru.net（需要 MINERU_API_KEY）
# - vlm-http-client: 走本地 mineru-vllm-server（MINERU_PORT）
# - pipeline: 在 Python 内跑本地 pipeline（更重）
# MINERU_ENABLED = True
# MINERU_BACKEND = "api"
```

同时请检查 [arxiv_daemon.py](arxiv_daemon.py) 里的 arXiv 分类分组（`CORE/LANG/AGENT/APP/ALL_TAGS`），它决定你到底拉取/展示哪些领域的论文。

### 4. 设置环境变量

```bash
# 必需
export YOUR_LLM_API_KEY="your-llm-api-key"

# 可选
export MINERU_API_KEY="your-mineru-api-key"     # MinerU API 后端（PDF 解析）
# run_services.py 优先使用的别名（覆盖 vars.py）：
# export ARXIV_SANITY_MINERU_API_KEY="your-mineru-api-key"
export YOUR_EMAIL_PASSWD="your-email-password"  # 邮件推荐
export ARXIV_SANITY_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')"

# 可选 - 代理（用于 arXiv 拉取及其他 HTTP 出网）
# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"
```

### 5. 获取论文并启动

```bash
# 获取论文并计算特征
python3 arxiv_daemon.py -n 10000 -m 500
python3 compute.py --num 20000

# 一键启动所有服务
python3 run_services.py

# 访问 http://localhost:55555
```

如果你想在一个终端里把 embedding / minerU / litellm 一起拉起来，推荐用 [run_services.py](run_services.py)。但注意它会调用 bash 脚本（见下方“系统说明”）。

### 配置清单

| 项目 | 文件/位置 | 必需 | 说明 |
|------|----------|------|------|
| **核心配置** | [vars.py](vars.py) | ✅ 必需 | `DATA_DIR`、`SERVE_PORT`、LLM 以及可选邮件/MinerU/Embedding |
| **LLM 服务** | [vars.py](vars.py) + 环境变量 | ✅ 必需 | `LLM_BASE_URL`、`LLM_NAME` + 可用密钥（`YOUR_LLM_API_KEY` 或直接写 `LLM_API_KEY`） |
| **arXiv 分类** | [arxiv_daemon.py](arxiv_daemon.py) | ⚙️ 重要 | `CORE/LANG/AGENT/APP/ALL_TAGS` 决定拉取范围与 About 展示 |
| **总结来源** | 环境变量或 [vars.py](vars.py) | ⚙️ 推荐 | `ARXIV_SANITY_SUMMARY_SOURCE=html\|mineru`、`ARXIV_SANITY_HTML_SOURCES=ar5iv,arxiv` |
| **Embedding 后端** | 环境变量或 [vars.py](vars.py) | ⚙️ 可选 | `ARXIV_SANITY_EMBED_USE_LLM_API` + `EMBED_*`（走 API）或本地 Ollama（`EMBED_PORT`） |
| **MinerU 后端** | 环境变量或 [vars.py](vars.py) | ⚙️ 可选 | `ARXIV_SANITY_MINERU_BACKEND=api\|vlm-http-client\|pipeline` + 密钥/端口 |
| **邮件 SMTP** | [vars.py](vars.py) + 环境变量 | ⚙️ 可选 | SMTP 配置 + `HOST` + `YOUR_EMAIL_PASSWD` |
| **会话密钥** | 环境变量/文件 | ⚙️ 推荐 | `ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt`（公网部署强烈建议） |

---

## 🧰 运行前准备与系统说明

### Python

- 推荐 Python 3.10+
- 依赖见 [requirements.txt](requirements.txt)

### 你可能需要的外部服务

- **LLM 服务商**（OpenAI 兼容 API）：用于总结（必需）。
- **Ollama**（可选）：当你选择本地 embedding 时，由 [embedding_serve.sh](embedding_serve.sh) 启动。
- **MinerU**（可选）：
  - `api` 后端：走 mineru.net，需要 `MINERU_API_KEY`
  - 本地 VLM 后端：由 [mineru_serve.sh](mineru_serve.sh) 启动 `mineru-vllm-server`
- **LiteLLM**（可选）：多模型网关，由 [llm.yml](llm.yml) 配置。

### Windows 注意事项

部分启动脚本是 bash（[up.sh](up.sh)、[embedding_serve.sh](embedding_serve.sh)、[mineru_serve.sh](mineru_serve.sh)、[litellm.sh](litellm.sh)），而 [run_services.py](run_services.py) 会用 `bash` 调它们。

- Windows 建议使用 **WSL**（最省心）。
- 或使用能提供 bash 的环境。
- 只跑 Web 的话也可以直接 `python serve.py`，并把 embedding/MinerU 走 API 后端。

## 配置指南

### 配置概览

| 来源 | 作用 | 必需 |
| --- | --- | --- |
| [vars.py](vars.py) | 核心配置（路径、端口、LLM、邮件、MinerU、SVM） | ✅ 必须 |
| [arxiv_daemon.py](arxiv_daemon.py) | arXiv 分类列表（论文采集范围） | ⚙️ 重要 |
| [llm.yml](llm.yml) | LiteLLM 多模型网关 | ⚙️ 可选 |
| 环境变量 | API 密钥、运行开关、调度器参数 | ⚙️ 建议 |
| [up.sh](up.sh) / [run_services.py](run_services.py) | 服务启动参数 | ⚙️ 可选 |

**仓库中不包含的文件（.gitignore）：**

- `vars.py` - 从 [vars_template.py](vars_template.py) 复制
- `llm.yml` - 从 [llm_template.yml](llm_template.yml) 复制
- `secret_key.txt` - 可选，Flask 会话密钥
- `data/` - 运行时自动生成（除 `data/dict.db`）
- 本地嵌入模型（如 `qwen3-embed-0.6B/`）

---

### 1. vars.py - 核心配置

将 `vars_template.py` 复制为 `vars.py` 并配置以下部分：

#### 1.1 数据存储

```python
DATA_DIR = "data"                              # 数据存储根目录（推荐 SSD）
SUMMARY_DIR = os.path.join(DATA_DIR, "summary") # 论文总结缓存目录
```

#### 1.2 服务端口

```python
SERVE_PORT = 55555      # Web 应用端口
EMBED_PORT = 51000      # Ollama 嵌入服务端口
MINERU_PORT = 52000     # MinerU VLM 服务端口（vLLM）
LITELLM_PORT = 53000    # LiteLLM 网关端口
```

#### 1.3 LLM API 配置

```python
# 方式 1：直接 API（OpenRouter、OpenAI 等）
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = os.environ.get("YOUR_LLM_API_KEY", "your_api_key")
LLM_NAME = "deepseek/deepseek-chat-v3.1:free"  # 模型名称
LLM_SUMMARY_LANG = "zh"                         # 总结语言（zh/en）

# 方式 2：通过 LiteLLM 网关（需要 llm.yml）
LLM_BASE_URL = f"http://localhost:{LITELLM_PORT}"
LLM_API_KEY = "no-key"  # LiteLLM 处理认证
LLM_NAME = "or-mimo"    # llm.yml 中定义的模型别名
```

#### 1.4 嵌入配置

```python
# 使用 OpenAI 兼容 API 生成嵌入（默认）
EMBED_USE_LLM_API = True
EMBED_MODEL_NAME = "qwen3-embedding:0.6b"
EMBED_API_BASE = ""       # 空 = 使用 LLM_BASE_URL
EMBED_API_KEY = ""        # 空 = 使用 LLM_API_KEY

# 或使用本地 Ollama 服务
EMBED_USE_LLM_API = False  # 使用 http://localhost:{EMBED_PORT}
```

#### 1.5 邮件服务

```python
from_email = "your_email@mail.com"
smtp_server = "smtp.mail.com"
smtp_port = 465                    # 25 普通，465 SSL
email_username = "username"
email_passwd = os.environ.get("YOUR_EMAIL_PASSWD", "")
HOST = "http://your-server:55555"  # 邮件链接的公网地址
```

#### 1.6 论文总结配置

```python
SUMMARY_MIN_CHINESE_RATIO = 0.25          # 缓存有效性的最低中文比例
SUMMARY_DEFAULT_SEMANTIC_WEIGHT = 0.5     # 混合搜索权重（0.0-1.0）
SUMMARY_MARKDOWN_SOURCE = "html"          # "html"（默认）或 "mineru"
SUMMARY_HTML_SOURCES = "ar5iv,arxiv"      # HTML 来源优先顺序
```

#### 1.7 MinerU PDF 解析

```python
MINERU_ENABLED = True                     # 启用/禁用 MinerU
MINERU_BACKEND = "api"                    # "api"（默认）、"pipeline" 或 "vlm-http-client"
MINERU_DEVICE = "cuda"                    # "cuda"（默认）或 "cpu"（仅 pipeline 后端）
MINERU_MAX_WORKERS = 2                    # 并发 minerU 进程数（仅 pipeline）
MINERU_MAX_VRAM = 3                       # 每进程最大显存 GB（仅 pipeline+cuda）
MINERU_API_KEY = os.environ.get("MINERU_API_KEY", "")  # API 后端密钥
MINERU_API_POLL_INTERVAL = 5              # API 轮询间隔（秒）
MINERU_API_TIMEOUT = 600                  # API 任务超时（秒）
```

#### 1.8 SVM 推荐参数

```python
SVM_C = 0.02          # SVM 分类器 C 参数（正则化）
SVM_MAX_ITER = 5000   # 最大迭代次数
SVM_TOL = 1e-3        # 容差
SVM_NEG_WEIGHT = 5.0  # 显式负反馈样本权重
```

---

### 2. arxiv_daemon.py - arXiv 分类

论文采集查询由 [arxiv_daemon.py](arxiv_daemon.py) 中的 `ALL_TAGS` 构建。自定义这些分组以控制采集哪些 arXiv 分类：

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

如果您想使用 LiteLLM 作为多 LLM 服务商的统一网关，请将 `llm_template.yml` 复制为 `llm.yml`。

```yaml
model_list:
  # OpenRouter - 免费模型
  - model_name: or-mimo            # vars.py 中 LLM_NAME 使用的别名
    litellm_params:
      model: openrouter/xiaomi/mimo-v2-flash:free
      api_base: https://openrouter.ai/api/v1
      api_key: YOUR_OPENROUTER_API_KEY  # 替换为您的密钥
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
litellm -c llm.yml --port 53000

# 或使用 run_services.py（自动启动 LiteLLM）
python3 run_services.py
```

然后配置 `vars.py`：

```python
LLM_BASE_URL = f"http://localhost:{LITELLM_PORT}"
LLM_API_KEY = "no-key"
LLM_NAME = "or-mimo"  # 使用 llm.yml 中的别名
```

---

### 4. 环境变量

#### 必需

| 变量 | 说明 | 示例 |
|------|------|------|
| `YOUR_LLM_API_KEY` | LLM 服务商 API 密钥 | `sk-or-v1-...` |

#### 可选 - API 密钥

| 变量 | 说明 | 示例 |
|------|------|------|
| `MINERU_API_KEY` | MinerU API 密钥（API 后端 PDF 解析） | `...` |
| `ARXIV_SANITY_MINERU_API_KEY` | MinerU API 密钥别名（run_services.py 优先使用） | `...` |
| `YOUR_EMAIL_PASSWD` | SMTP 邮箱密码 | `...` |
| `ARXIV_SANITY_SECRET_KEY` | Flask 会话密钥（或使用 `secret_key.txt`） | `...` |
| `ARXIV_SANITY_EMBED_API_BASE` | 覆盖嵌入 API 基础 URL | `https://api.openai.com/v1` |
| `ARXIV_SANITY_EMBED_API_KEY` | 覆盖嵌入 API 密钥 | `sk-...` |

#### Web 与运行时

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_LOG_LEVEL` | `WARNING` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `ARXIV_SANITY_ACCESS_LOG` | `0` | 启用访问日志（`1`/`0`） |
| `ARXIV_SANITY_RELOAD` | `0` | 开发热重载模式 |
| `ARXIV_SANITY_CACHE_PAPERS` | `1` | 在内存中缓存完整论文表（`1`/`0`） |
| `ARXIV_SANITY_WARMUP_DATA` | `1` | 后台数据缓存预热 |
| `ARXIV_SANITY_WARMUP_ML` | `1` | 后台 ML 模型预热 |
| `ARXIV_SANITY_ENABLE_SCHEDULER` | `1` | 启用 APScheduler 缓存刷新 |
| `ARXIV_SANITY_ENABLE_CACHE_STATUS` | `0` | 启用 `/cache_status` 调试页面 |
| `ARXIV_SANITY_EMAIL_API_WORKERS` | `8` | 运行 [send_emails.py](send_emails.py) 时最多并发 API 调用数 |

#### Web 安全 / Cookie

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_COOKIE_SAMESITE` | `Lax` | Cookie SameSite 策略 |
| `ARXIV_SANITY_COOKIE_SECURE` | `0` | 是否启用 secure cookie（需要 HTTPS） |
| `ARXIV_SANITY_MAX_CONTENT_LENGTH` | `1048576` | 最大请求体大小（字节，默认 1 MiB） |

#### 总结来源

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_SUMMARY_SOURCE` | `html` | Markdown 来源：`html` 或 `mineru` |
| `ARXIV_SANITY_HTML_SOURCES` | `ar5iv,arxiv` | HTML 来源优先顺序 |

#### MinerU 后端

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_MINERU_ENABLED` | `true` | 启用/禁用 MinerU |
| `ARXIV_SANITY_MINERU_BACKEND` | `api` | `api`、`pipeline` 或 `vlm-http-client` |
| `ARXIV_SANITY_MINERU_DEVICE` | `cuda` | pipeline 后端设备 |
| `ARXIV_SANITY_MINERU_MAX_WORKERS` | `2` | 最大并发 minerU 进程数 |
| `ARXIV_SANITY_MINERU_MAX_VRAM` | `3` | 每进程最大显存（GB） |
| `MINERU_API_POLL_INTERVAL` | `5` | API 轮询间隔（秒） |
| `MINERU_API_TIMEOUT` | `600` | API 任务超时（秒） |

#### 锁与并发

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC` | `600` | 总结缓存锁“过期清理”阈值（异常退出后建议保留默认） |
| `ARXIV_SANITY_MINERU_LOCK_STALE_SEC` | `3600` | MinerU 解析 / GPU-slot 锁过期清理阈值 |

#### 嵌入

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_EMBED_USE_LLM_API` | `true` | 使用 LLM API 生成嵌入 |

#### 守护进程/调度器

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ARXIV_SANITY_FETCH_NUM` | `2000` | 每次获取的论文数 |
| `ARXIV_SANITY_FETCH_MAX` | `1000` | 每次 API 查询最大结果数 |
| `ARXIV_SANITY_SUMMARY_NUM` | `200` | 每次总结的论文数 |
| `ARXIV_SANITY_SUMMARY_WORKERS` | `2` | 总结工作线程数 |
| `ARXIV_SANITY_DAEMON_SUMMARY` | `1` | 守护进程中启用总结生成 |
| `ARXIV_SANITY_DAEMON_EMBEDDINGS` | `1` | 守护进程中启用嵌入 |
| `ARXIV_SANITY_PRIORITY_QUEUE` | `1` | 启用总结优先队列 |
| `ARXIV_SANITY_PRIORITY_DAYS` | `2` | 优先窗口（天） |
| `ARXIV_SANITY_PRIORITY_LIMIT` | `100` | 最大优先论文数 |
| `ARXIV_SANITY_ENABLE_GIT_BACKUP` | `1` | 启用 dict.db 的 git 备份 |

#### 网络 / 代理

- `http_proxy`、`https_proxy`：被 [arxiv_daemon.py](arxiv_daemon.py) 等出网请求使用。

#### Gunicorn（up.sh）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GUNICORN_WORKERS` | `2` | 工作进程数 |
| `GUNICORN_THREADS` | `4` | 每工作进程线程数 |
| `ARXIV_SANITY_GUNICORN_PRELOAD` | `1` | 在主进程中预加载应用 |
| `GUNICORN_EXTRA_ARGS` | `` | 额外的 gunicorn 参数 |

---

### 5. 启动参数

#### run_services.py

```bash
# 一键启动（推荐）
python3 run_services.py

# Web 服务器选项
python3 run_services.py --web gunicorn    # 使用 gunicorn
python3 run_services.py --web none        # 不启动 Web 服务器

# 跳过重型服务
python3 run_services.py --no-embed        # 跳过 Ollama 嵌入
python3 run_services.py --no-mineru       # 跳过 MinerU
python3 run_services.py --no-litellm      # 跳过 LiteLLM 网关

# 总结来源
python3 run_services.py --summary-source html
python3 run_services.py --summary-source mineru

# 包含调度器守护进程
python3 run_services.py --with-daemon

# 一次性：仅获取和计算
python3 run_services.py --fetch-compute         # 默认 10000 篇论文
python3 run_services.py --fetch-compute 1000    # 自定义数量
```

#### arxiv_daemon.py

```bash
python3 arxiv_daemon.py -n 10000 -m 500    # 获取最多 10000 篇，每次查询 500 篇
python3 arxiv_daemon.py --init             # 使用关键词搜索初始化
python3 arxiv_daemon.py --num-total 5000   # 限制所有分类总论文数
python3 arxiv_daemon.py --break-after 20   # 连续 20 批无新论文后停止
```

#### compute.py

```bash
python3 compute.py --num 20000             # TF-IDF 特征数
python3 compute.py --use_embeddings        # 启用嵌入（默认）
python3 compute.py --no-embeddings         # 禁用嵌入
python3 compute.py --embed_model nomic-embed-text  # 嵌入模型
python3 compute.py --embed_dim 512         # 嵌入维度
python3 compute.py --embed_batch_size 2048 # 批次大小
```

#### batch_paper_summarizer.py

```bash
python3 batch_paper_summarizer.py -n 100 -w 2         # 100 篇论文，2 个工作线程
python3 batch_paper_summarizer.py --priority          # 优先队列模式
python3 batch_paper_summarizer.py --priority-days 2   # 优先窗口
python3 batch_paper_summarizer.py --dry-run           # 仅预览
python3 batch_paper_summarizer.py -m "gpt-4o-mini"    # 指定模型
```

---

## 🚀 核心功能

- **🤖 AI 论文总结**：完整处理管道，包含 HTML（arXiv/ar5iv）解析或 `minerU` PDF 解析、LLM 总结和智能缓存系统
- **🔍 高级搜索引擎**：关键词、语义和混合搜索模式，支持可配置权重和智能时间过滤
- **🎯 智能推荐系统**：混合 TF-IDF + 嵌入特征，基于用户偏好训练动态 SVM 分类器
- **🏷️ 灵活组织管理**：个人标签支持正/负反馈、组合标签、关键词跟踪，支持 AND/OR 逻辑操作
- **📚 阅读列表**：个人论文收藏功能，支持添加/移除、摘要状态跟踪和专属管理页面
- **📧 邮件智能服务**：自动化每日推荐，个性化 HTML 模板和假期感知调度
- **⚡ 高性能优化**：多核处理、Intel 扩展、增量更新、Ollama 嵌入 + minerU(vLLM) 和智能缓存
- **🔗 现代化架构**：RESTful API、响应式 Web 界面、异步总结加载和全面错误处理
- **🔄 完全自动化**：内置调度器管理 获取→计算→总结→邮件 流程，智能资源管理

---

## 📖 使用指南

### 用户界面功能

- **账户系统（极简）**：
  - 登录只需要用户名（无密码），适合个人/内网。
  - 如果你要公网部署，建议放在统一认证 / VPN 后面，并设置稳定的会话密钥（`ARXIV_SANITY_SECRET_KEY` 或 `secret_key.txt`）。
- **高级搜索**：
  - **关键词搜索**：传统基于文本的搜索，使用 TF-IDF 评分
  - **语义搜索**：AI 驱动的相似性搜索，使用嵌入向量
  - **混合搜索**：结合关键词+语义，支持可调权重（0.0-1.0）
  - **标签推荐**：基于您的个人标签训练 SVM 推荐
  - **时间过滤**：智能过滤，即使在时间窗口外也保留已标记论文
- **组织工具**：
  - **个人标签**：单篇论文标记，支持 AND/OR 逻辑
  - **组合标签**：多标签分类（如 "RL,NLP"）处理复杂主题
  - **关键词跟踪**：跟踪所有论文中的特定术语
- **AI 论文总结**：
  - 点击"总结"获取 LLM 生成的总结
  - MathJax 渲染 LaTeX 公式
  - 异步加载和进度指示器
  - 缓存机制提升性能

### 每日邮件推荐（可选）

1. 在 [vars.py](vars.py) 配好 SMTP，并在环境变量设置 `YOUR_EMAIL_PASSWD`。
2. 在 [vars.py](vars.py) 设置 `HOST` 为**公网可访问的 base URL**（邮件里链接用它拼）。
3. 在网站 Profile 页面填写你的邮箱地址。
4. 手动运行 [send_emails.py](send_emails.py)，或直接运行调度器 [daemon.py](daemon.py)。

用户/标签多时可以调 `ARXIV_SANITY_EMAIL_API_WORKERS` 来限制并发，避免把本机/网络打满。

### 搜索语法

| 语法 | 示例 | 说明 |
|------|------|------|
| 字段过滤 | `ti:transformer`、`au:goodfellow`、`cat:cs.LG` | 搜索特定字段 |
| 短语 | `"diffusion model"` | 精确短语匹配 |
| 排除 | `-survey`、`!survey` | 排除术语 |
| arXiv ID | `id:2312.12345` | 按论文 ID 查找 |

**示例：**

- `ti:"graph neural network" cat:cs.LG` - 标题包含短语，分类为 cs.LG
- `au:goodfellow -survey` - 作者是 Goodfellow，排除综述
- `id:2312.12345` - 查找特定论文

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
python3 compute.py --use_embeddings --embed_model nomic-embed-text
```

### 自动化调度

**内置调度器：**

```bash
python3 daemon.py
```

调度计划（Asia/Shanghai 时区）：

- **获取+计算**：工作日 8:00、12:00、16:00、20:00
- **发送邮件**：工作日 18:00
- **备份**：每日 20:00

**手动 Cron：**

```cron
# 获取和计算（工作日每日 4 次）
0 9,13,17,21 * * 1-5 cd /path && python3 arxiv_daemon.py -n 1000 && python3 compute.py --use_embeddings

# 发送邮件（工作日下午 6 点）
0 18 * * 1-5 cd /path && python3 send_emails.py -t 2

# 生成总结（每日晚上 7 点）
0 19 * * * cd /path && python3 batch_paper_summarizer.py -n 200 -w 2
```

---

## 📚 API 参考

### 搜索与推荐

- `GET /?rank=search&q=<query>` - 关键词搜索
- `GET /?rank=search&q=<query>&search_mode=semantic` - 语义搜索
- `GET /?rank=search&q=<query>&search_mode=hybrid&semantic_weight=0.5` - 混合搜索
- `GET /?rank=tags&tags=<tag_list>&logic=<and|or>` - 基于标签的 SVM 推荐
- `GET /?rank=time&time_filter=<days>` - 时间过滤论文
- `GET /?rank=pid&pid=<paper_id>` - 相似论文

### 论文总结

- `GET /summary?pid=<paper_id>` - 查看总结页面
- `POST /api/get_paper_summary` - 获取总结 JSON
- `POST /api/clear_model_summary` - 清除特定模型的总结
- `POST /api/clear_paper_cache` - 清除所有论文缓存

### 标签与关键词管理

- `GET /add/<pid>/<tag>` - 为论文添加标签
- `GET /sub/<pid>/<tag>` - 从论文移除标签
- `GET /add_key/<keyword>` - 添加跟踪关键词
- `GET /del_key/<keyword>` - 移除跟踪关键词

### 系统

- `GET /stats` - 系统统计
- `GET /cache_status` - 缓存状态（需认证用户）

---

## 📈 更新日志

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
