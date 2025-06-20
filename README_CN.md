# arxiv-sanity-X

基于 [arXiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) 构建的强大 arXiv 论文推荐系统，通过先进的机器学习技术、个性化推荐系统和自动化研究跟踪功能，显著加速学术研究工作流程。

![Screenshot](arxiv-sanity-x.png)

## 🚀 核心功能

### 主要特性
- **智能论文推荐**：结合 TF-IDF 与现代嵌入向量的混合特征系统，由 SVM 分类器驱动
- **个性化标签系统**：个人和组合标签管理，实现精细化兴趣跟踪
- **关键词监控**：自动跟踪指定的研究关键词，实时匹配相关论文
- **邮件推荐服务**：每日个性化论文推荐直接发送到您的收件箱
- **多维度搜索**：跨标题、作者、摘要等的高级搜索功能
- **多逻辑推荐**：支持标签组合推荐中的 AND/OR 逻辑
- **时间筛选**：专注于近期发表的论文，可配置时间窗口

### 性能优化
- **多核处理**：并行计算支持，充分利用所有可用 CPU 核心
- **Intel 扩展支持**：可选的 Intel scikit-learn 扩展，加速机器学习计算

### 机器学习能力
- **混合特征架构**：稀疏 TF-IDF 特征与密集嵌入向量相结合
- **现代嵌入模型**：支持 Qwen3 等先进嵌入模型
- **动态分类器**：为个性化推荐动态训练每个标签的 SVM 分类器



##  更新日志

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
5. [高级功能](#高级功能)
6. [API 参考](#api-参考)


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
├── serve.py              # Flask 网络服务器和 API
├── arxiv_daemon.py       # arXiv 数据获取守护进程
├── compute.py            # 特征计算（TF-IDF + 嵌入）
├── send_emails.py        # 邮件推荐服务
├── daemon.py             # 自动化任务调度器
├── aslite/               # 核心库
│   ├── db.py            # 数据库操作
│   └── arxiv.py         # arXiv API 接口
├── templates/           # HTML 模板
├── static/             # 静态网络资源
└── data/               # 数据存储
    ├── papers.db       # 论文数据库
    ├── features.pkl    # 特征缓存
    └── dict.db         # 用户数据
```

### 数据流管道
1. **数据摄取**：[`arxiv_daemon.py`](arxiv_daemon.py) 从 arXiv API 获取论文
2. **特征处理**：[`compute.py`](compute.py) 生成 TF-IDF 和嵌入特征
3. **网络服务**：[`serve.py`](serve.py) 提供用户界面和推荐功能
4. **邮件服务**：[`send_emails.py`](send_emails.py) 提供个性化推荐

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

#### 1. 账户管理
- **需要登录**：完整功能需要用户认证
- **个人资料设置**：在个人资料设置中配置推荐邮箱

#### 2. 论文发现
- **关键词搜索**：跨标题、作者和摘要搜索，具有优化排序
- **基于标签的推荐**：基于您标记的论文的 SVM 驱动推荐
- **时间筛选**：专注于特定时间段的论文
- **相似性搜索**：查找与特定论文 ID 相似的论文

#### 3. 组织系统
- **个人标签**：为感兴趣的论文创建个人标签
- **组合标签**：注册标签组合以获得更复杂的推荐
- **关键词跟踪**：设置研究关键词的自动监控
- **标签管理**：重命名、删除和组织您的标签系统

#### 4. 推荐模式
- **搜索**：基于关键词的论文发现
- **标签**：基于标记论文的 SVM 推荐
- **时间**：按时间顺序浏览最近论文
- **随机**：偶然的论文发现

### 邮件推荐

在个人资料中设置您的邮箱以接收：
- 基于您标签的每日推荐
- 关键词匹配的论文提醒
- 组合标签推荐
- 可自定义的推荐频率

## 🔧 高级功能

### 嵌入向量集成

支持现代嵌入模型如 Qwen3：

```bash
# 下载嵌入模型（示例）
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-embed-0.6B

# 启用嵌入计算
python compute.py --embed_model ./qwen3-embed-0.6B
```

### 性能优化

系统自动针对您的硬件进行优化：
- **多核利用**：自动检测并使用所有 CPU 核心
- **Intel 扩展**：使用 Intel scikit-learn 扩展的可选加速
- **内存管理**：智能缓存和内存优化

### 智能缓存

- **自动重载**：文件更改时特征缓存自动更新
- **增量处理**：仅为新论文计算嵌入
- **内存数据库**：论文加载到内存中以进行快速查询
- **智能缓存管理**：高效的缓存失效和更新

## 📚 API 参考

### 核心端点

#### 搜索与推荐
- `GET /?rank=search&q=<query>` - 关键词搜索
- `GET /?rank=tags&tags=<tag_list>` - 基于标签的推荐
- `GET /?rank=time&time_filter=<days>` - 时间筛选论文
- `GET /?rank=pid&pid=<paper_id>` - 相似论文

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

### SVM 参数

- **C 参数**：正则化强度（默认：0.02）
  - 较低值 = 更强正则化
  - 较高值 = 较少正则化
- **逻辑模式**：
  - `and`：所有标签都必须相关
  - `or`：任何标签都可以相关
- **时间筛选**：将推荐限制为最近论文（天数）

### 性能优化

1. **使用 SSD 存储**：将 `DATA_DIR` 设置为 SSD 路径以获得更快的 I/O
2. **足够内存**：推荐 16GB+ 内存用于大数据集
3. **Intel 扩展**：安装 `scikit-learn-intelex` 以获得 CPU 加速
4. **特征调优**：根据数据集大小调整 TF-IDF 特征数量
5. **批处理**：为您的硬件优化批次大小

### 监控与维护

```bash
# 检查系统状态
curl http://localhost:5000/stats

# 监控缓存性能
curl http://localhost:5000/cache_status
```
