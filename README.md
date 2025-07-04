
# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

Enhanced arXiv paper recommendation system with AI-powered summarization, semantic search, and intelligent personalized recommendations built on hybrid ML features.

![Screenshot](arxiv-sanity-x.png)

## 🚀 Key Features

- **🤖 AI Paper Summarization**: Complete pipeline with minerU PDF parsing, LLM generation, and intelligent caching
- **🔍 Advanced Search**: Keyword, semantic, and hybrid search modes with configurable weights
- **🎯 Smart Recommendations**: Hybrid TF-IDF + embedding features with dynamic SVM classifiers
- **🏷️ Tag Management**: Individual/combined tags with AND/OR logic and keyword monitoring
- **📧 Email Service**: Automated daily recommendations with personalized HTML templates
- **⚡ Performance**: Multi-core processing, Intel extensions, incremental updates, vLLM integration
- **🔗 API & Web**: RESTful APIs, responsive web interface, async loading summaries
- **🔄 Automation**: Built-in scheduler for fetching, computing, emailing, and backup


##  Changelog

### v2.3 - AI Paper Summarization
- ✨ **New**: Complete AI-powered paper summarization system with [`paper_summarizer.py`](paper_summarizer.py)
- 🧠 **MinerU Integration**: Advanced PDF parsing with minerU for better text extraction
- 📝 **Summary Interface**: New `/summary` route with async loading and markdown rendering
- 🔧 **Batch Processing**: [`batch_paper_summarizer.py`](batch_paper_summarizer.py) for parallel summary generation
- ⚡ **Smart Caching**: Intelligent summary caching with Chinese text ratio validation
- 🎨 **Enhanced UI**: New summary page design with MathJax support for mathematical formulas
- 📊 **Configuration**: Added LLM API configuration in [`vars_template.py`](vars_template.py)
- 🔄 **Auto Generation**: [`generate_latest_summaries.py`](generate_latest_summaries.py) for automated batch processing

### v2.2 - Performance & Stability Improvements
- ⚡ **Performance**: Enhanced data caching system with intelligent auto-reload
- 🔧 **Optimized Embedding**: Streamlined embedding generation pipeline in [`compute.py`](compute.py)
- 📈 **Scheduler Enhancement**: Increased fetch frequency from daily to 4x daily (6AM, 11AM, 4PM, 9PM)
- 🛠️ **Bug Fixes**: Fixed email recommendation system edge cases and empty result handling
- 🧠 **Smart Caching**: Unified papers and metas data caching with automatic file change detection
- 📊 **API Improvements**: Enhanced tag search API with better error handling and logging
- 🚀 **Memory Optimization**: Reduced memory footprint and improved query performance

### v2.1 - API & Semantic Search
- ✨ **New**: Semantic search with keyword, semantic, and hybrid modes
- 🔗 **API Integration**: RESTful API endpoints for recommendations
- 🚀 **VLLM Support**: High-performance model serving with vLLM
- 🎯 **Enhanced Search**: Configurable semantic weights for hybrid search
- 🔧 **Refactored Architecture**: API client implementation for embedding models

### v2.0 - Enhanced ML Features
- ✨ **New**: Hybrid TF-IDF + embedding vector features
- ⚡ **Performance**: Multi-core optimization and Intel extensions
- 🧠 **Smart Caching**: Intelligent feature cache management
- 📈 **Incremental Processing**: Efficient embedding generation
- 🎯 **Improved Algorithms**: Enhanced recommendation accuracy
- 🔧 **Better Error Handling**: Comprehensive logging and debugging

### v1.0 - Foundation
- 📚 arXiv paper fetching and storage
- 🏷️ User tagging and keyword systems
- 📧 Email recommendation service
- 🌐 Web interface and search functionality
- 🤖 SVM-based paper recommendations

## 📋 Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Configuration](#configuration)
3. [System Architecture](#system-architecture)
4. [Usage Guide](#usage-guide)
5. [AI Paper Summarization](#ai-paper-summarization)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)

## 🛠 Installation & Setup

### System Requirements
- Python 3.8 - 3.11
- Recommended: SSD storage for database performance
- Memory: 8GB+ recommended for large datasets
- Optional: CUDA-capable GPU for embedding models

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/xihuai18/arxiv-sanity-x
cd arxiv-sanity-x

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install Intel extensions for performance boost
pip install scikit-learn-intelex
```

### Initial Setup

1. **Create Configuration**
```bash
cp vars_template.py vars.py
# Edit vars.py with your settings
```

2. **Generate Security Key**
```python
import secrets
print(secrets.token_urlsafe(16))
# Save output to secret_key.txt
```

3. **Initialize Database**
```bash
# Fetch initial paper data
python arxiv_daemon.py -n 50000 -m 1000

# Compute feature vectors
python compute.py --num 50000 --embed_dim 512

# Start web service
gunicorn -w 4 -b 0.0.0.0:5000 serve:app
```

## ⚙️ Configuration

### Primary Configuration (vars.py)
```python
# Database Configuration
DATA_DIR = "data"  # Use SSD path for optimal performance
HOST = "http://localhost:5000"  # Web service URL

# Email Service Configuration
from_email = "your_email@example.com"
smtp_server = "smtp.example.com"
smtp_port = 465  # SSL port (465) or TLS port (587)
email_username = "your_username"
email_passwd = "your_app_password"

# LLM API Configuration (for AI summarization)
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"  # e.g., ZhipuAI API
LLM_API_KEY = "your_llm_api_key"  # Your LLM API key
```

### Advanced Parameters

#### Feature Computation (compute.py)
```bash
python compute.py \
  --num 50000 \              # Number of TF-IDF features
  --min_df 20 \              # Minimum document frequency
  --max_df 0.10 \            # Maximum document frequency
  --ngram_max 1 \            # Maximum n-gram size
  --use_embeddings \         # Disable embedding vectors
  --embed_model ./qwen3-embed-0.6B \  # Embedding model path
  --embed_dim 512 \          # Embedding dimensions
  --embed_batch_size 2048    # Batch size for embedding generation
```

#### Email Recommendations (send_emails.py)
```bash
python send_emails.py \
  -n 20 \                    # Papers per recommendation
  -t 2.0 \                   # Time window (days)
  -m 5 \                     # Minimum tagged papers per user
  --dry-run 0                # Set to 1 for testing without sending
```

## 🏗 System Architecture

### Component Overview
```
arxiv-sanity-X/
├── serve.py                    # Flask web server & API
├── arxiv_daemon.py             # arXiv data fetching daemon
├── compute.py                  # Feature computation (TF-IDF + embeddings)
├── send_emails.py              # Email recommendation service
├── daemon.py                   # Scheduler for automated tasks
├── paper_summarizer.py         # AI paper summarization module
├── batch_paper_summarizer.py   # Batch processing for paper summaries
├── generate_latest_summaries.py # Auto-generate summaries for latest papers
├── vllm_serve.sh               # vLLM model server startup script
├── aslite/                     # Core library
│   ├── db.py                  # Database operations
│   └── arxiv.py               # arXiv API interface
├── templates/                  # HTML templates
│   └── summary.html           # Paper summary page template
├── static/                     # Static web assets
│   └── paper_summary.js       # Summary page JavaScript
└── data/                       # Data storage
    ├── papers.db              # Paper database
    ├── features.pkl           # Feature cache
    ├── dict.db                # User data
    ├── pdfs/                  # Downloaded PDF files
    ├── mineru/                # MinerU parsed content
    └── summary/               # Cached paper summaries
```

### Data Flow Pipeline
1. **Data Ingestion**: [`arxiv_daemon.py`](arxiv_daemon.py) fetches papers from arXiv API
2. **Feature Processing**: [`compute.py`](compute.py) generates TF-IDF and embedding features
3. **AI Summarization**: [`paper_summarizer.py`](paper_summarizer.py) processes papers with minerU and LLM
4. **Web Service**: [`serve.py`](serve.py) provides user interface, recommendations, and summary display
5. **Email Service**: [`send_emails.py`](send_emails.py) delivers personalized recommendations

### Automated Scheduling

**Built-in Scheduler:**
```bash
python daemon.py
```

**Manual Cron Setup:**
```cron
# Fetch and compute features (weekdays 4 PM)
0 16 * * 1-5 cd /path/to/arxiv-sanity-x && python arxiv_daemon.py -n 5000 && python compute.py

# Send email recommendations (weekdays 6 PM)
0 18 * * 1-5 cd /path/to/arxiv-sanity-x && python send_emails.py -t 1.5

# Backup user data (daily 7 PM)
0 19 * * * cd /path/to/arxiv-sanity-x && git add . && git commit -m "backup" && git push
```

## 📖 Usage Guide

### User Interface Features

- **Account Setup**: Login required, configure email in profile for recommendations
- **Search Modes**: Keyword, semantic, hybrid, tag-based, time-filtered, and similarity search
- **Organization**: Personal tags, combined tags, keyword tracking, tag management
- **AI Summaries**: Click "Summary" link for LLM-generated summaries with MathJax support

### Email Recommendations
Configure email in profile for daily tag-based recommendations and keyword alerts.

## 🤖 AI Paper Summarization

### Complete AI Pipeline
1. **PDF Download**: Automatic arXiv paper fetching
2. **minerU Parsing**: Advanced PDF text extraction with structure recognition
3. **LLM Processing**: Generate comprehensive summaries using GLM-4-Flash or compatible models
4. **Quality Control**: Chinese text ratio validation and content filtering
5. **Smart Caching**: Intelligent caching with automatic quality checks

### Usage Commands
```bash
# Individual paper summary (web interface)
# Click "Summary" link or visit: /summary?pid=<paper_id>

# Batch processing (latest papers)
python generate_latest_summaries.py --num_papers 100

# Advanced batch processing with custom workers
python batch_paper_summarizer.py --num_papers 200 --max_workers 4 --skip_cached

# Check processing status
python batch_paper_summarizer.py --dry_run  # Preview mode
```

### Configuration
```python
# In vars.py - LLM API setup
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"  # ZhipuAI or OpenAI-compatible
LLM_API_KEY = "your_api_key_here"
```

### Features
- **MathJax Support**: Full LaTeX math formula rendering
- **Markdown Output**: Rich formatting with headers, lists, code blocks
- **Async Loading**: Non-blocking web interface with progress indicators
- **Error Recovery**: Automatic retry with detailed failure logging
- **Thread Safety**: Concurrent processing with minerU lock management

## 🔧 Advanced Features

### Embedding Models & Performance
```bash
# Download and start embedding model
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-embed-0.6B
bash vllm_serve.sh

# Enable embedding computation
python compute.py --embed_model ./qwen3-embed-0.6B
```

Features: Multi-core processing, Intel extensions, intelligent caching, incremental updates.

## 📚 API Reference

### Core Endpoints

#### Search & Recommendations
- `GET /?rank=search&q=<query>` - Keyword search
- `GET /?rank=search&q=<query>&search_type=semantic` - Semantic search
- `GET /?rank=search&q=<query>&search_type=hybrid&semantic_weight=<weight>` - Hybrid search
- `GET /?rank=tags&tags=<tag_list>` - Tag-based recommendations
- `GET /?rank=time&time_filter=<days>` - Time-filtered papers
- `GET /?rank=pid&pid=<paper_id>` - Similar papers

#### API Endpoints
- `GET /api/recommend/keywords/<keyword>` - Get keyword-based recommendations
- `GET /api/recommend/tags/<tag_list>` - Get tag-based recommendations via API
- `POST /api/get_paper_summary` - Get AI-generated paper summary (JSON: `{"pid": "paper_id"}`)

#### Paper Summarization
- `GET /summary?pid=<paper_id>` - View AI-generated paper summary with async loading

#### Tag Management
- `GET /add/<pid>/<tag>` - Add tag to paper
- `GET /sub/<pid>/<tag>` - Remove tag from paper
- `GET /del/<tag>` - Delete tag
- `GET /rename/<old_tag>/<new_tag>` - Rename tag

#### Keyword Management
- `GET /add_key/<keyword>` - Add keyword for tracking
- `GET /del_key/<keyword>` - Remove keyword

#### System Information
- `GET /stats` - System statistics
- `GET /cache_status` - Cache status (authenticated users only)

### SVM Parameters & Optimization
- **SVM**: C=0.02 (regularization), logic modes: `and`/`or`, time filtering
- **Performance**: SSD storage, 16GB+ RAM, Intel extensions, proper batch sizes
- **Monitoring**: `/stats` and `/cache_status` endpoints
