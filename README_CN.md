
# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

基于现代机器学习技术的综合性 arXiv 论文浏览和推荐系统，集成 AI 智能总结、混合搜索功能和个性化推荐。采用 TF-IDF、语义嵌入和 LLM 集成等先进技术。

![Screenshot](arxiv-sanity-x.png)

## 🚀 核心功能

- **🤖 AI 论文总结**：完整处理管道，包含 `minerU` PDF 解析、LLM 总结和智能缓存系统
- **🔍 高级搜索引擎**：关键词、语义和混合搜索模式，支持可配置权重和智能时间过滤
- **🎯 智能推荐系统**：混合 TF-IDF + 嵌入特征，基于用户偏好训练动态 SVM 分类器
- **🏷️ 灵活组织管理**：个人标签、组合标签、关键词跟踪，支持 AND/OR 逻辑操作
- **📧 邮件智能服务**：自动化每日推荐，个性化 HTML 模板和假期感知调度
- **⚡ 高性能优化**：多核处理、Intel 扩展、增量更新、vLLM 集成和智能缓存
- **🔗 现代化架构**：RESTful API、响应式 Web 界面、异步总结加载和全面错误处理
- **🔄 完全自动化**：内置调度器管理 获取→计算→总结→邮件 流程，智能资源管理

## 📈 更新日志

### v2.4 - 多线程批量处理与服务完善
- ⚡ **并发优化**：真正的多线程并发论文总结处理
- 🔒 **线程安全**：文件级锁机制避免 minerU 解析冲突
- 📊 **统计增强**：详细的处理统计和失败原因分析
- 🔄 **重试机制**：智能重试失败的论文处理任务
- 📈 **进度跟踪**：实时进度条和处理状态显示
- 🔧 **配置优化**：支持多种 LLM 服务商（OpenRouter、智谱AI等）
- 📊 **服务集成**：完善的 vLLM 和 minerU 服务集成
- 🎨 **界面增强**：更好的响应式设计和 MathJax 数学公式支持
- 🛠️ **错误处理**：增强的异常处理和重试机制

### v2.3 - AI 论文总结系统
- ✨ **新功能**：完整的 AI 驱动论文总结系统，包含 [`paper_summarizer.py`](paper_summarizer.py)
- 🧠 **MinerU 集成**：使用 minerU 进行高级 PDF 解析，提供更好的文本提取和结构识别
- 📝 **总结界面**：新的 `/summary` 路由，支持异步加载和 markdown 渲染
- 🔧 **批量处理**：[`batch_paper_summarizer.py`](batch_paper_summarizer.py) 支持并行总结生成和线程安全
- ⚡ **智能缓存**：智能总结缓存，支持中文文本比例验证和质量控制
- 🎨 **界面增强**：新的总结页面设计，支持 MathJax 数学公式渲染
- 📊 **配置完善**：在 [`vars.py`](vars.py) 中添加 LLM API 配置
- 🔄 **自动生成**：[`generate_latest_summaries.py`](generate_latest_summaries.py) 支持自动化批量处理

### v2.2 - 性能与稳定性改进
- ⚡ **性能提升**：增强统一数据缓存系统，支持智能自动重载和文件变更检测
- 🔧 **嵌入优化**：简化 [`compute.py`](compute.py) 中的嵌入生成管道，支持增量更新
- 📈 **调度器增强**：将获取频率从每日1次增加到每日4次（早6点、上午11点、下午4点、晚9点）
- 🛠️ **错误修复**：修复邮件推荐系统边界情况和空结果处理
- 🧠 **智能缓存**：统一论文和元数据缓存，支持自动文件变更检测
- 📊 **API 改进**：增强标签搜索 API，提供更好的错误处理和全面日志记录
- 🚀 **内存优化**：减少内存占用，提高大数据集查询性能

### v2.1 - API 与语义搜索
- ✨ **新功能**：语义搜索，支持关键词、语义和混合模式
- 🔗 **API 集成**：提供 RESTful API 端点用于推荐和论文总结
- 🚀 **VLLM 支持**：使用 vLLM 进行高性能模型服务和嵌入生成
- 🎯 **增强搜索**：混合搜索的可配置语义权重（0.0-1.0）
- 🔧 **重构架构**：嵌入模型的 API 客户端实现，支持适当错误处理

### v2.0 - 增强机器学习功能
- ✨ **新功能**：混合 TF-IDF + 嵌入向量特征，支持稀疏-稠密矩阵拼接
- ⚡ **性能优化**：多核优化和 Intel scikit-learn 扩展
- 🧠 **智能缓存**：智能特征缓存管理，支持自动重载检测
- 📈 **增量处理**：高效嵌入生成，支持增量更新
- 🎯 **改进算法**：通过混合特征方法增强推荐准确性
- 🔧 **更好的错误处理**：全面的日志记录和调试功能

### v1.0 - 基础版本
- 📚 arXiv 论文获取和存储，使用 SQLite 数据库
- 🏷️ 用户标签和关键词系统，支持灵活组织
- 📧 邮件推荐服务，支持 HTML 模板
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
8. [部署指南](#部署指南)

## 🛠 安装与设置

### 系统要求
- **Python**：3.8 - 3.11
- **存储**：推荐 SSD 以获得数据库性能（高效处理 40 万+论文）
- **内存**：推荐 8GB+（大特征矩阵的最佳性能需要 16GB+）
- **GPU**：可选 CUDA 兼容 GPU 用于嵌入模型和 minerU PDF 解析
- **网络**：稳定的互联网连接，用于 arXiv API、LLM API 调用和邮件服务

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
# 获取初始论文数据（CS 类别：AI、ML、CL 等）
python arxiv_daemon.py -n 50000 -m 1000

# 计算混合特征向量（TF-IDF + 嵌入）
python compute.py --num 50000 --embed_dim 512

# 启动网络服务
gunicorn -w 4 -b 0.0.0.0:5000 serve:app
```

## ⚙️ 配置

### 主要配置 (vars.py)

在 `vars.py` 中配置以下设置：

- **数据库配置**：数据存储路径和Web服务URL
- **邮件服务配置**：SMTP服务器设置，用于论文推荐邮件
- **LLM API 配置**：支持多种LLM服务商（OpenAI接口）
- **vLLM 服务端口**：嵌入模型和minerU服务的端口配置

项目支持多种LLM服务商，包括OpenRouter（推荐免费模型）、智谱AI（免费 Flash 系列模型）等。

### 高级参数

#### 特征计算 (compute.py)
```bash
python compute.py \
  --num 50000 \              # TF-IDF 特征数量
  --min_df 20 \              # 最小文档频率
  --max_df 0.10 \            # 最大文档频率
  --ngram_max 1 \            # 最大 n-gram 大小
  --use_embeddings \         # 启用嵌入向量
  --embed_model ./qwen3-embed-0.6B \  # 嵌入模型路径
  --embed_dim 512 \          # 嵌入维度
  --embed_batch_size 2048    # 嵌入生成批次大小
```

#### 批量论文总结 (batch_paper_summarizer.py)
```bash
python batch_paper_summarizer.py \
  -n 200 \                   # 处理论文数量
  -w 4 \                     # 工作线程数
  --max-retries 3 \          # 最大重试次数
  --no-skip-cached           # 重新处理已缓存论文
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
├── mineru_serve.sh             # minerU VLM 服务器启动脚本
├── embedding_serve.sh          # vLLM 嵌入服务器启动脚本
├── aslite/                     # 核心库
│   ├── db.py                  # 数据库操作（SQLite + 压缩）
│   └── arxiv.py               # arXiv API 接口
├── templates/                  # HTML 模板
│   ├── base.html              # 基础模板
│   ├── index.html             # 主界面模板
│   └── summary.html           # 论文总结页面模板
├── static/                     # 静态网络资源
│   ├── paper_list.js          # 主界面 JavaScript
│   ├── paper_summary.js       # 总结页面 JavaScript
│   └── style.css              # 样式表
└── data/                       # 数据存储
    ├── papers.db              # 论文数据库（SQLite）
    ├── features.p             # 特征缓存（pickle）
    ├── dict.db                # 用户数据（SQLite）
    ├── pdfs/                  # 下载的 PDF 文件
    ├── mineru/                # MinerU 解析内容
    └── summary/               # 缓存的论文总结
```

### 数据流管道
1. **数据摄取**：[`arxiv_daemon.py`](arxiv_daemon.py) 从 arXiv API 获取论文（每日4次：早6点、上午11点、下午4点、晚9点）
2. **特征处理**：[`compute.py`](compute.py) 生成混合 TF-IDF + 嵌入特征，支持增量更新
3. **AI 总结**：[`paper_summarizer.py`](paper_summarizer.py) 下载 PDF → minerU 解析 → LLM 总结
4. **网络服务**：[`serve.py`](serve.py) 提供响应式 UI、混合搜索、推荐功能和异步总结加载
5. **邮件服务**：[`send_emails.py`](send_emails.py) 提供个性化推荐，支持假期感知调度
6. **自动化管理**：[`daemon.py`](daemon.py) 协调整个流程，支持智能资源管理

### 服务依赖关系
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask Web     │    │  vLLM Embedding  │    │  minerU VLM     │
│   (port 5000)   │<-->│   (port 51000)   │    │  (port 52000)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         v                       v                       v
┌─────────────────────────────────────────────────────────────────┐
│                     SQLite 数据库存储                             │
│  papers.db | features.p | dict.db | summary/ | mineru/          │
└─────────────────────────────────────────────────────────────────┘
```

## 📖 使用指南

### 用户界面功能

- **账户系统**：用户认证，支持个人资料管理和邮箱配置
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

### 邮件智能服务
- **每日推荐**：基于您的标签的个性化论文建议
- **关键词提醒**：当匹配您关键词的论文出现时通知
- **假期感知**：假期期间调整推荐频率
- **HTML 模板**：丰富的邮件格式，直接链接到论文

## 🤖 AI 论文总结

### 完整 AI 处理管道
1. **PDF 下载**：自动 arXiv 论文获取，支持错误处理
2. **minerU 解析**：高级 PDF 文本提取，支持结构识别和图像处理
3. **LLM 处理**：使用多种兼容 OpenAI API 的模型生成全面总结
4. **质量控制**：中文文本比例验证和内容过滤
5. **智能缓存**：智能缓存机制，自动质量检查和存储优化

### 启动服务
```bash
# 启动 minerU VLM 服务
bash mineru_serve.sh

# 启动嵌入模型服务
bash embedding_serve.sh

# 启动主 Web 服务
python serve.py
```

### 使用命令
```bash
# 单篇论文总结（Web 界面）
# 点击"总结"链接或访问：/summary?pid=<paper_id>

# 批量处理（最新论文）
python generate_latest_summaries.py --num_papers 100

# 高级批量处理，自定义工作线程
python batch_paper_summarizer.py -n 200 -w 4 --max-retries 3

# 检查处理状态
python batch_paper_summarizer.py --dry-run  # 预览模式
```

### LLM 服务商支持
项目支持多种 LLM 服务商：

#### OpenRouter
```python
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = "sk-or-v1-..."
LLM_NAME = "deepseek/deepseek-chat-v3.1:free"  # 免费模型
```

#### OpenAI
```python
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "sk-..."
LLM_NAME = "gpt-4o-mini"
```

### 功能特性
- **MathJax 支持**：Web 界面中完整的 LaTeX 数学公式渲染
- **Markdown 输出**：丰富格式，包含标题、列表、代码块和数学表达式
- **异步加载**：非阻塞 Web 界面，支持进度指示器和实时更新
- **错误恢复**：自动重试机制，详细失败日志和优雅降级
- **线程安全**：并发处理，minerU 锁管理防止冲突
- **存储优化**：自动清理中间文件和智能缓存

## 🔧 高级功能

### 嵌入模型与性能
```bash
# 下载并启动嵌入模型
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-embed-0.6B
bash embedding_serve.sh

# 使用 API 客户端启用嵌入计算
python compute.py --embed_model ./qwen3-embed-0.6B --embed_api_base http://localhost:51000/v1
```

功能特性：
- 多核处理优化
- Intel scikit-learn 扩展提升性能
- 智能特征缓存，支持自动重载
- 增量嵌入生成
- 稀疏-稠密矩阵拼接，实现混合特征

### 性能优化
- **数据库**：压缩 SQLite 存储，支持智能缓存
- **特征**：混合稀疏 TF-IDF + 稠密嵌入，支持 L2 归一化
- **内存**：针对大数据集（40万+论文）优化，支持流式处理
- **计算**：多线程处理，可配置工作池和批处理

### 自动化调度

**内置调度器：**
```bash
python daemon.py
```

**手动 Cron 设置：**
```cron
# 获取和计算特征（工作日每日4次）
0 9,13,17,21 * * 1-5 cd /path/to/arxiv-sanity-x && python arxiv_daemon.py -n 1000 && python compute.py

# 发送邮件推荐（工作日下午 6 点）
0 18 * * 1-5 cd /path/to/arxiv-sanity-x && python send_emails.py -t 2

# 生成论文总结（每日晚上 7 点）
0 19 * * * cd /path/to/arxiv-sanity-x && python batch_paper_summarizer.py -n 200 -w 2

# 备份用户数据（每日晚上 8 点）
0 20 * * * cd /path/to/arxiv-sanity-x && git add . && git commit -m "backup" && git push
```

## 📚 API 参考

### 核心端点

#### 搜索与推荐
- `GET /?rank=search&q=<query>` - 关键词搜索，使用 TF-IDF 评分
- `GET /?rank=search&q=<query>&search_mode=semantic` - 语义搜索，使用嵌入向量
- `GET /?rank=search&q=<query>&search_mode=hybrid&semantic_weight=<weight>` - 混合搜索，可配置权重
- `GET /?rank=tags&tags=<tag_list>&logic=<and|or>` - 基于标签的 SVM 推荐
- `GET /?rank=time&time_filter=<days>` - 时间过滤论文，智能标签保留
- `GET /?rank=pid&pid=<paper_id>` - 相似论文，使用最近邻搜索

#### API 端点
- `POST /api/keyword_search` - 通过 API 进行关键词搜索
- `POST /api/tag_search` - 单标签推荐
- `POST /api/tags_search` - 多标签推荐，支持逻辑操作
- `POST /api/get_paper_summary` - 获取 AI 生成的论文总结（JSON: `{"pid": "paper_id"}`）

#### 论文总结
- `GET /summary?pid=<paper_id>` - 查看 AI 生成的论文总结，支持异步加载和 MathJax

#### 标签管理
- `GET /add/<pid>/<tag>` - 为论文添加标签
- `GET /sub/<pid>/<tag>` - 从论文移除标签
- `GET /del/<tag>` - 删除标签（需要确认）
- `GET /rename/<old_tag>/<new_tag>` - 重命名标签，应用于所有论文

#### 关键词管理
- `GET /add_key/<keyword>` - 添加跟踪关键词
- `GET /del_key/<keyword>` - 从跟踪中移除关键词

#### 系统信息
- `GET /stats` - 系统统计和数据库信息
- `GET /cache_status` - 缓存状态和性能指标（需认证用户）

### SVM 参数与优化
- **SVM 配置**：C=0.02（正则化）、平衡类权重、LinearSVC 提升速度
- **逻辑模式**：`and`（加权组合）/`or`（联合）用于多标签查询
- **性能调优**：SSD 存储、16GB+ 内存、Intel 扩展、适当批次大小
- **监控**：通过 `/cache_status` 实时缓存统计和性能监控

### 搜索模式
- **关键词**：传统 TF-IDF，支持多核并行处理
- **语义**：使用预计算嵌入的余弦相似性
- **混合**：加权组合，支持归一化和可配置语义权重（0.0-1.0）

---

## 📝 许可证
本项目采用 MIT 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

## 🤝 贡献
欢迎贡献！请随时提交 Pull Request。

## ⭐ 致谢
- 原始 [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) 项目，作者 Andrej Karpathy
- [minerU](https://github.com/opendatalab/MinerU) 提供高级 PDF 解析功能
- [vLLM](https://github.com/vllm-project/vllm) 提供高性能模型服务
