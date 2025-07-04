# arxiv-sanity-X

增强型 arXiv 论文推荐系统，集成 AI 智能总结、语义搜索和个性化推荐功能。

![Screenshot](arxiv-sanity-x.png)

## 🚀 核心功能

- **AI 论文总结**：集成 LLM 和 minerU 的自动化论文总结，支持智能缓存
- **语义搜索**：关键词、语义和混合搜索，支持嵌入模型
- **智能推荐**：TF-IDF + 嵌入向量混合特征，SVM 分类器驱动
- **个性化标签**：个人和组合标签管理，支持 AND/OR 逻辑
- **邮件服务**：每日个性化推荐和关键词提醒
- **性能优化**：多核处理、Intel 扩展、vLLM 集成
- **API 支持**：RESTful 端点，支持推荐和总结功能


##  更新日志

### v2.3 - AI 论文总结系统
- ✨ **新功能**：完整的 AI 驱动论文总结系统，包含 [`paper_summarizer.py`](paper_summarizer.py)
- 🧠 **MinerU 集成**：使用 minerU 进行高级 PDF 解析，提供更好的文本提取
- 📝 **总结界面**：新的 `/summary` 路由，支持异步加载和 markdown 渲染
- 🔧 **批量处理**：[`batch_paper_summarizer.py`](batch_paper_summarizer.py) 支持并行总结生成
- ⚡ **智能缓存**：智能总结缓存，支持中文文本比例验证
- 🎨 **界面增强**：新的总结页面设计，支持 MathJax 数学公式渲染
- 📊 **配置完善**：在 [`vars_template.py`](vars_template.py) 中添加 LLM API 配置
- 🔄 **自动生成**：[`generate_latest_summaries.py`](generate_latest_summaries.py) 支持自动化批量处理

### v2.2 - 性能与稳定性改进
- ⚡ **性能提升**：增强数据缓存系统，支持智能自动重载
- 🔧 **嵌入优化**：简化 [`compute.py`](compute.py) 中的嵌入生成管道
- 📈 **调度器增强**：将获取频率从每日1次增加到每日4次（早6点、上午11点、下午4点、晚9点）
- 🛠️ **错误修复**：修复邮件推荐系统边界情况和空结果处理
- 🧠 **智能缓存**：统一论文和元数据缓存，支持自动文件变更检测
- 📊 **API 改进**：增强标签搜索 API，提供更好的错误处理和日志记录
- 🚀 **内存优化**：减少内存占用，提高查询性能

### v2.1 - API 与语义搜索
- ✨ **新功能**：语义搜索，支持关键词、语义和混合模式
- 🔗 **API 集成**：提供 RESTful API 端点用于推荐
- 🚀 **VLLM 支持**：使用 vLLM 进行高性能模型服务
- 🎯 **增强搜索**：混合搜索的可配置语义权重
- 🔧 **重构架构**：嵌入模型的 API 客户端实现

### v2.0 - 增强机器学习功能
- ✨ **新功能**：混合 TF-IDF + 嵌入向量特征
- ⚡ **性能**：多核优化和 Intel 扩展
- 🧠 **智能缓存**：智能特征缓存管理
- 📈 **增量处理**：高效嵌入生成
- 🎯 **改进算法**：增强推荐准确性
- 🔧 **更好的错误处理**：全面的日志记录和调试

### v1.0 - 基础版本
- 📚 arXiv 论文获取和存储
- 🏷️ 用户标签和关键词系统
- 📧 邮件推荐服务
- 🌐 网络界面和搜索功能
- 🤖 基于 SVM 的论文推荐

## 📋 目录
1. [安装与设置](#安装与设置)
2. [配置](#配置)
3. [系统架构](#系统架构)
4. [使用指南](#使用指南)
5. [AI 论文总结](#ai-论文总结)
6. [高级功能](#高级功能)
7. [API 参考](#api-参考)


## 🛠 安装与设置

### 系统要求
- Python 3.8 - 3.11
- 推荐：SSD 存储以获得数据库性能
- 内存：推荐 8GB+ 用于大数据集
- 可选：支持 CUDA 的 GPU 用于嵌入模型

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/xihuai18/arxiv-sanity-x
cd arxiv-sanity-x

# 安装 Python 依赖
pip install -r requirements.txt

# 可选：安装 Intel 扩展以提升性能
pip install scikit-learn-intelex
```

### 初始设置

1. **创建配置文件**
```bash
cp vars_template.py vars.py
# 编辑 vars.py 配置您的设置
```

2. **生成安全密钥**
```python
import secrets
print(secrets.token_urlsafe(16))
# 将输出保存到 secret_key.txt
```

3. **初始化数据库**
```bash
# 获取初始论文数据
python arxiv_daemon.py -n 50000 -m 1000

# 计算特征向量
python compute.py --num 50000 --embed_dim 512

# 启动网络服务
gunicorn -w 4 -b 0.0.0.0:5000 serve:app
```

## ⚙️ 配置

### 主要配置 (vars.py)
```python
# 数据库配置
DATA_DIR = "data"  # 使用 SSD 路径以获得最佳性能
HOST = "http://localhost:5000"  # 网络服务 URL

# 邮件服务配置
from_email = "your_email@example.com"
smtp_server = "smtp.example.com"
smtp_port = 465  # SSL 端口 (465) 或 TLS 端口 (587)
email_username = "your_username"
email_passwd = "your_app_password"

# LLM API 配置（用于 AI 总结功能）
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"  # 例如：智谱 AI API
LLM_API_KEY = "your_llm_api_key"  # 您的 LLM API 密钥
```

### 高级参数

#### 特征计算 (compute.py)
```bash
python compute.py \
  --num 50000 \              # TF-IDF 特征数量
  --min_df 20 \              # 最小文档频率
  --max_df 0.10 \            # 最大文档频率
  --ngram_max 1 \            # 最大 n-gram 大小
  --use_embeddings \         # 关闭嵌入向量
  --embed_model ./qwen3-embed-0.6B \  # 嵌入模型路径
  --embed_dim 512 \          # 嵌入维度
  --embed_batch_size 2048    # 嵌入生成批次大小
```

#### 邮件推荐 (send_emails.py)
```bash
python send_emails.py \
  -n 20 \                    # 每次推荐的论文数量
  -t 2.0 \                   # 时间窗口（天）
  -m 5 \                     # 每用户最少标记论文数
  --dry-run 0                # 设为 1 进行测试而不发送邮件
```

## 🏗 系统架构

### 组件概览
```
arxiv-sanity-X/
├── serve.py                    # Flask 网络服务器和 API
├── arxiv_daemon.py             # arXiv 数据获取守护进程
├── compute.py                  # 特征计算（TF-IDF + 嵌入）
├── send_emails.py              # 邮件推荐服务
├── daemon.py                   # 自动化任务调度器
├── paper_summarizer.py         # AI 论文总结模块
├── batch_paper_summarizer.py   # 批量论文总结处理
├── generate_latest_summaries.py # 自动生成最新论文总结
├── vllm_serve.sh               # vLLM 模型服务器启动脚本
├── aslite/                     # 核心库
│   ├── db.py                  # 数据库操作
│   └── arxiv.py               # arXiv API 接口
├── templates/                  # HTML 模板
│   └── summary.html           # 论文总结页面模板
├── static/                     # 静态网络资源
│   └── paper_summary.js       # 总结页面 JavaScript
└── data/                       # 数据存储
    ├── papers.db              # 论文数据库
    ├── features.pkl           # 特征缓存
    ├── dict.db                # 用户数据
    ├── pdfs/                  # 下载的 PDF 文件
    ├── mineru/                # MinerU 解析内容
    └── summary/               # 缓存的论文总结
```

### 数据流管道
1. **数据摄取**：[`arxiv_daemon.py`](arxiv_daemon.py) 从 arXiv API 获取论文
2. **特征处理**：[`compute.py`](compute.py) 生成 TF-IDF 和嵌入特征
3. **AI 总结**：[`paper_summarizer.py`](paper_summarizer.py) 使用 minerU 和 LLM 处理论文
4. **网络服务**：[`serve.py`](serve.py) 提供用户界面、推荐功能和总结显示
5. **邮件服务**：[`send_emails.py`](send_emails.py) 提供个性化推荐

### 自动化调度

**内置调度器：**
```bash
python daemon.py
```

**手动 Cron 设置：**
```cron
# 获取和计算特征（工作日下午 4 点）
0 16 * * 1-5 cd /path/to/arxiv-sanity-x && python arxiv_daemon.py -n 5000 && python compute.py

# 发送邮件推荐（工作日下午 6 点）
0 18 * * 1-5 cd /path/to/arxiv-sanity-x && python send_emails.py -t 1.5

# 备份用户数据（每日晚上 7 点）
0 19 * * * cd /path/to/arxiv-sanity-x && git add . && git commit -m "backup" && git push
```

## 📖 使用指南

### 用户界面功能

- **账户设置**：需要登录，在个人资料中配置推荐邮箱
- **搜索模式**：关键词、语义、混合、标签、时间筛选和相似性搜索
- **组织管理**：个人标签、组合标签、关键词跟踪、标签管理
- **AI 总结**：点击"总结"链接查看 LLM 生成的总结，支持 MathJax 渲染

### 邮件推荐
在个人资料中配置邮箱，接收每日标签推荐和关键词提醒。

## 🤖 AI 论文总结

### 使用方法
- **单篇总结**：点击任意论文的"总结"链接（`/summary?pid=<paper_id>`）
- **批量生成**：`python generate_latest_summaries.py --num_papers 100`
- **功能特性**：MathJax 公式渲染、智能缓存、异步加载

### 配置
```python
# 在 vars.py 中 - 添加 LLM API 配置
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"  # 智谱 AI 示例
LLM_API_KEY = "your_api_key_here"
```

## 🔧 高级功能

### 嵌入向量集成

通过 API 客户端架构支持现代嵌入模型如 Qwen3：

```bash
# 下载嵌入模型（示例）
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-embed-0.6B

# 启动 vLLM 模型服务器
bash vllm_serve.sh

# 使用 API 客户端启用嵌入计算
python compute.py --embed_model ./qwen3-embed-0.6B
```

### 嵌入模型与性能
```bash
# 下载并启动嵌入模型
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-embed-0.6B
bash vllm_serve.sh

# 启用嵌入计算
python compute.py --embed_model ./qwen3-embed-0.6B
```

特性：多核处理、Intel 扩展、智能缓存、增量更新。

## 📚 API 参考

### 核心端点

#### 搜索与推荐
- `GET /?rank=search&q=<query>` - 关键词搜索
- `GET /?rank=search&q=<query>&search_type=semantic` - 语义搜索
- `GET /?rank=search&q=<query>&search_type=hybrid&semantic_weight=<weight>` - 混合搜索
- `GET /?rank=tags&tags=<tag_list>` - 基于标签的推荐
- `GET /?rank=time&time_filter=<days>` - 时间筛选论文
- `GET /?rank=pid&pid=<paper_id>` - 相似论文

#### API 端点
- `GET /api/recommend/keywords/<keyword>` - 获取基于关键词的推荐
- `GET /api/recommend/tags/<tag_list>` - 通过 API 获取基于标签的推荐
- `POST /api/get_paper_summary` - 获取 AI 生成的论文总结（JSON: `{"pid": "paper_id"}`）

#### 论文总结
- `GET /summary?pid=<paper_id>` - 查看 AI 生成的论文总结，支持异步加载

#### 标签管理
- `GET /add/<pid>/<tag>` - 为论文添加标签
- `GET /sub/<pid>/<tag>` - 从论文移除标签
- `GET /del/<tag>` - 删除标签
- `GET /rename/<old_tag>/<new_tag>` - 重命名标签

#### 关键词管理
- `GET /add_key/<keyword>` - 添加跟踪关键词
- `GET /del_key/<keyword>` - 移除关键词

#### 系统信息
- `GET /stats` - 系统统计
- `GET /cache_status` - 缓存状态（需认证用户）

### SVM 参数与优化
- **SVM 参数**：C=0.02（正则化），逻辑模式：`and`/`or`，时间筛选
- **性能优化**：SSD 存储、16GB+ 内存、Intel 扩展、合适的批次大小
- **监控维护**：`/stats` 和 `/cache_status` 端点
