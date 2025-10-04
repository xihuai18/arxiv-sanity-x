# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

A comprehensive arXiv paper browsing and recommendation system featuring AI-powered summarization, hybrid search capabilities, and personalized recommendations. Built with modern ML techniques including TF-IDF, semantic embeddings, and LLM integration.

![Screenshot](arxiv-sanity-x.png)

## 🚀 Core Features

- **🤖 AI Paper Summarization**: Complete processing pipeline with `minerU` PDF parsing, LLM summarization, and intelligent caching system
- **🔍 Advanced Search Engine**: Keyword, semantic, and hybrid search modes with configurable weights and intelligent time filtering
- **🎯 Smart Recommendations**: Hybrid TF-IDF + embedding features with dynamic SVM classifiers trained on user preferences
- **🏷️ Flexible Organization**: Personal tags, combined tags, keyword tracking with AND/OR logic operations
- **📧 Email Intelligence**: Automated daily recommendations with personalized HTML templates and holiday-aware scheduling
- **⚡ High Performance**: Multi-core processing, Intel extensions, incremental updates, vLLM integration, and smart caching
- **🔗 Modern Architecture**: RESTful APIs, responsive web interface, async summary loading, and comprehensive error handling
- **🔄 Full Automation**: Built-in scheduler managing fetch→compute→summarize→email pipeline with intelligent resource management

## 📈 Changelog

### v2.4 - Multi-threading & Service Enhancement
- ⚡ **Concurrency Optimization**: True multi-threaded concurrent paper summarization processing
- 🔒 **Thread Safety**: File-level locking mechanism to avoid minerU parsing conflicts
- 📊 **Enhanced Statistics**: Detailed processing statistics and failure reason analysis
- 🔄 **Retry Mechanism**: Smart retry for failed paper processing tasks
- 📈 **Progress Tracking**: Real-time progress bars and processing status display
- 🔧 **Configuration Optimization**: Support for multiple LLM providers (OpenRouter, ZhipuAI, etc.)
- 📊 **Service Integration**: Complete vLLM and minerU service integration
- 🎨 **Interface Enhancement**: Better responsive design and MathJax mathematical formula support
- 🛠️ **Error Handling**: Enhanced exception handling and retry mechanisms

### v2.3 - AI Paper Summarization
- ✨ **New**: Complete AI-powered paper summarization system with [`paper_summarizer.py`](paper_summarizer.py)
- 🧠 **MinerU Integration**: Advanced PDF parsing with minerU for better text extraction and structure recognition
- 📝 **Summary Interface**: New `/summary` route with async loading and markdown rendering
- 🔧 **Batch Processing**: [`batch_paper_summarizer.py`](batch_paper_summarizer.py) for parallel summary generation with thread safety
- ⚡ **Smart Caching**: Intelligent summary caching with Chinese text ratio validation and quality control
- 🎨 **Enhanced UI**: New summary page design with MathJax support for mathematical formulas
- 📊 **Configuration**: Added LLM API configuration in [`vars.py`](vars.py)
- 🔄 **Auto Generation**: [`generate_latest_summaries.py`](generate_latest_summaries.py) for automated batch processing

### v2.2 - Performance & Stability Improvements
- ⚡ **Performance**: Enhanced unified data caching system with intelligent auto-reload and file change detection
- 🔧 **Optimized Embedding**: Streamlined embedding generation pipeline with incremental updates in [`compute.py`](compute.py)
- 📈 **Scheduler Enhancement**: Increased fetch frequency from daily to 4x daily (6AM, 11AM, 4PM, 9PM)
- 🛠️ **Bug Fixes**: Fixed email recommendation system edge cases and empty result handling
- 🧠 **Smart Caching**: Unified papers and metas data caching with automatic file change detection
- 📊 **API Improvements**: Enhanced tag search API with better error handling and comprehensive logging
- 🚀 **Memory Optimization**: Reduced memory footprint and improved query performance for large datasets

### v2.1 - API & Semantic Search
- ✨ **New**: Semantic search with keyword, semantic, and hybrid modes
- 🔗 **API Integration**: RESTful API endpoints for recommendations and paper summaries
- 🚀 **VLLM Support**: High-performance model serving with vLLM for embedding generation
- 🎯 **Enhanced Search**: Configurable semantic weights for hybrid search (0.0-1.0)
- 🔧 **Refactored Architecture**: API client implementation for embedding models with proper error handling

### v2.0 - Enhanced ML Features
- ✨ **New**: Hybrid TF-IDF + embedding vector features with sparse-dense matrix concatenation
- ⚡ **Performance**: Multi-core optimization and Intel scikit-learn extensions
- 🧠 **Smart Caching**: Intelligent feature cache management with automatic reload detection
- 📈 **Incremental Processing**: Efficient embedding generation with incremental updates
- 🎯 **Improved Algorithms**: Enhanced recommendation accuracy with hybrid feature approach
- 🔧 **Better Error Handling**: Comprehensive logging and debugging capabilities

### v1.0 - Foundation
- 📚 arXiv paper fetching and storage with SQLite database
- 🏷️ User tagging and keyword systems with flexible organization
- 📧 Email recommendation service with HTML templates
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
8. [Deployment Guide](#deployment-guide)

## 🛠 Installation & Setup

### System Requirements
- **Python**: 3.8 - 3.11
- **Storage**: SSD recommended for database performance (handles 400k+ papers efficiently)
- **Memory**: 8GB+ recommended (16GB+ for optimal performance with large feature matrices)
- **GPU**: Optional CUDA-capable GPU for embedding models and minerU PDF parsing
- **Network**: Stable internet for arXiv API, LLM API calls, and email services

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
# Fetch initial paper data (CS categories: AI, ML, CL, etc.)
python arxiv_daemon.py -n 50000 -m 1000

# Compute hybrid feature vectors (TF-IDF + embeddings)
python compute.py --num 50000 --embed_dim 512

# Start web service
gunicorn -w 4 -b 0.0.0.0:5000 serve:app
```

## ⚙️ Configuration

### Primary Configuration (vars.py)

Configure the following settings in `vars.py`:

- **Database Configuration**: Data storage paths and web service URL
- **Email Service Configuration**: SMTP server settings for paper recommendation emails
- **LLM API Configuration**: Support for multiple LLM providers (OpenAI-compatible API)
- **vLLM Service Ports**: Port configuration for embedding models and minerU services

The project supports multiple LLM providers, including OpenRouter (recommended free models), ZhipuAI (free Flash series models), and others.

### Advanced Parameters

#### Feature Computation (compute.py)
```bash
python compute.py \
  --num 50000 \              # Number of TF-IDF features
  --min_df 20 \              # Minimum document frequency
  --max_df 0.10 \            # Maximum document frequency
  --ngram_max 1 \            # Maximum n-gram size
  --use_embeddings \         # Enable embedding vectors
  --embed_model ./qwen3-embed-0.6B \  # Embedding model path
  --embed_dim 512 \          # Embedding dimensions
  --embed_batch_size 2048    # Batch size for embedding generation
```

#### Batch Paper Summarization (batch_paper_summarizer.py)
```bash
python batch_paper_summarizer.py \
  -n 200 \                   # Number of papers to process
  -w 4 \                     # Number of worker threads
  --max-retries 3 \          # Maximum retry attempts
  --no-skip-cached           # Reprocess cached papers
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
├── mineru_serve.sh             # minerU VLM server startup script
├── embedding_serve.sh          # vLLM embedding server startup script
├── aslite/                     # Core library
│   ├── db.py                  # Database operations (SQLite + compression)
│   └── arxiv.py               # arXiv API interface
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Main interface template
│   └── summary.html           # Paper summary page template
├── static/                     # Static web assets
│   ├── paper_list.js          # Main interface JavaScript
│   ├── paper_summary.js       # Summary page JavaScript
│   └── style.css              # Stylesheet
└── data/                       # Data storage
    ├── papers.db              # Paper database (SQLite)
    ├── features.p             # Feature cache (pickle)
    ├── dict.db                # User data (SQLite)
    ├── pdfs/                  # Downloaded PDF files
    ├── mineru/                # MinerU parsed content
    └── summary/               # Cached paper summaries
```

### Data Flow Pipeline
1. **Data Ingestion**: [`arxiv_daemon.py`](arxiv_daemon.py) fetches papers from arXiv API (4x daily: 6AM, 11AM, 4PM, 9PM)
2. **Feature Processing**: [`compute.py`](compute.py) generates hybrid TF-IDF + embedding features with incremental updates
3. **AI Summarization**: [`paper_summarizer.py`](paper_summarizer.py) downloads PDFs → minerU parsing → LLM summarization
4. **Web Service**: [`serve.py`](serve.py) provides responsive UI, hybrid search, recommendations, and async summary loading
5. **Email Service**: [`send_emails.py`](send_emails.py) delivers personalized recommendations with holiday-aware scheduling
6. **Automation**: [`daemon.py`](daemon.py) orchestrates the entire pipeline with intelligent resource management

### Service Dependencies
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flask Web     │    │  vLLM Embedding  │    │  minerU VLM     │
│   (port 5000)   │<-->│   (port 51000)   │    │  (port 52000)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         v                       v                       v
┌─────────────────────────────────────────────────────────────────┐
│                     SQLite Database Storage                     │
│  papers.db | features.p | dict.db | summary/ | mineru/          │
└─────────────────────────────────────────────────────────────────┘
```

## 📖 Usage Guide

### User Interface Features

- **Account System**: User authentication with profile management and email configuration
- **Advanced Search**:
  - **Keyword Search**: Traditional text-based search with TF-IDF scoring
  - **Semantic Search**: AI-powered similarity search using embedding vectors
  - **Hybrid Search**: Combines keyword + semantic with adjustable weights (0.0-1.0)
  - **Tag-based**: SVM recommendations trained on your personal tags
  - **Time Filtering**: Smart filtering that preserves tagged papers even outside time window
- **Organization Tools**:
  - **Personal Tags**: Individual paper tagging with AND/OR logic
  - **Combined Tags**: Multi-tag categories (e.g., "RL,NLP") for complex topics
  - **Keywords**: Track specific terms across all papers
- **AI Paper Summaries**:
  - Click "Summary" for LLM-generated summaries
  - MathJax rendering for LaTeX formulas
  - Async loading with progress indicators
  - Cached for performance

### Email Intelligence
- **Daily Recommendations**: Personalized paper suggestions based on your tags
- **Keyword Alerts**: Notifications when papers matching your keywords appear
- **Holiday Awareness**: Adjusts recommendation frequency during holidays
- **HTML Templates**: Rich email formatting with direct links to papers

## 🤖 AI Paper Summarization

### Complete AI Pipeline
1. **PDF Download**: Automatic arXiv paper fetching with error handling
2. **minerU Parsing**: Advanced PDF text extraction with structure recognition and image handling
3. **LLM Processing**: Generate comprehensive summaries using multiple OpenAI-compatible LLM providers
4. **Quality Control**: Chinese text ratio validation and content filtering
5. **Smart Caching**: Intelligent caching with automatic quality checks and storage optimization

### Start Services
```bash
# Start minerU VLM service
bash mineru_serve.sh

# Start embedding model service
bash embedding_serve.sh

# Start main web service
python serve.py
```

### Usage Commands
```bash
# Individual paper summary (web interface)
# Click "Summary" link or visit: /summary?pid=<paper_id>

# Batch processing (latest papers)
python generate_latest_summaries.py --num_papers 100

# Advanced batch processing with custom workers
python batch_paper_summarizer.py -n 200 -w 4 --max-retries 3

# Check processing status
python batch_paper_summarizer.py --dry-run  # Preview mode
```

### LLM Provider Support
The project supports multiple LLM providers:

#### OpenRouter
```python
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_API_KEY = "sk-or-v1-..."
LLM_NAME = "deepseek/deepseek-chat-v3.1:free"  # Free model
```

#### OpenAI
```python
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "sk-..."
LLM_NAME = "gpt-4o-mini"
```

### Features
- **MathJax Support**: Full LaTeX math formula rendering in web interface
- **Markdown Output**: Rich formatting with headers, lists, code blocks, and mathematical expressions
- **Async Loading**: Non-blocking web interface with progress indicators and real-time updates
- **Error Recovery**: Automatic retry with detailed failure logging and graceful degradation
- **Thread Safety**: Concurrent processing with minerU lock management to prevent conflicts
- **Storage Optimization**: Automatic cleanup of intermediate files and intelligent caching

## 🔧 Advanced Features

### Embedding Models & Performance
```bash
# Download and start embedding model
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./qwen3-embed-0.6B
bash embedding_serve.sh

# Enable embedding computation with API client
python compute.py --embed_model ./qwen3-embed-0.6B --embed_api_base http://localhost:51000/v1
```

Features:
- Multi-core processing optimization
- Intel scikit-learn extensions for performance
- Intelligent feature caching with automatic reload
- Incremental embedding generation
- Sparse-dense matrix concatenation for hybrid features

### Performance Optimization
- **Database**: Compressed SQLite storage with intelligent caching
- **Features**: Hybrid sparse TF-IDF + dense embeddings with L2 normalization
- **Memory**: Optimized for large datasets (400k+ papers) with streaming processing
- **Compute**: Multi-threading with configurable worker pools and batch processing

### Automated Scheduling

**Built-in Scheduler:**
```bash
python daemon.py
```

**Manual Cron Setup:**
```cron
# Fetch and compute features (weekdays 4x daily)
0 9,13,17,21 * * 1-5 cd /path/to/arxiv-sanity-x && python arxiv_daemon.py -n 1000 && python compute.py

# Send email recommendations (weekdays 6 PM)
0 18 * * 1-5 cd /path/to/arxiv-sanity-x && python send_emails.py -t 2

# Generate paper summaries (daily 7 PM)
0 19 * * * cd /path/to/arxiv-sanity-x && python batch_paper_summarizer.py -n 200 -w 2

# Backup user data (daily 8 PM)
0 20 * * * cd /path/to/arxiv-sanity-x && git add . && git commit -m "backup" && git push
```

## 📚 API Reference

### Core Endpoints

#### Search & Recommendations
- `GET /?rank=search&q=<query>` - Keyword search with TF-IDF scoring
- `GET /?rank=search&q=<query>&search_mode=semantic` - Semantic search using embeddings
- `GET /?rank=search&q=<query>&search_mode=hybrid&semantic_weight=<weight>` - Hybrid search with configurable weights
- `GET /?rank=tags&tags=<tag_list>&logic=<and|or>` - Tag-based SVM recommendations
- `GET /?rank=time&time_filter=<days>` - Time-filtered papers with smart tag preservation
- `GET /?rank=pid&pid=<paper_id>` - Similar papers using nearest neighbor search

#### API Endpoints
- `POST /api/keyword_search` - Keyword-based search via API
- `POST /api/tag_search` - Single tag recommendations
- `POST /api/tags_search` - Multi-tag recommendations with logic operations
- `POST /api/get_paper_summary` - Get AI-generated paper summary (JSON: `{"pid": "paper_id"}`)

#### Paper Summarization
- `GET /summary?pid=<paper_id>` - View AI-generated paper summary with async loading and MathJax

#### Tag Management
- `GET /add/<pid>/<tag>` - Add tag to paper
- `GET /sub/<pid>/<tag>` - Remove tag from paper
- `GET /del/<tag>` - Delete tag (with confirmation)
- `GET /rename/<old_tag>/<new_tag>` - Rename tag across all papers

#### Keyword Management
- `GET /add_key/<keyword>` - Add keyword for tracking
- `GET /del_key/<keyword>` - Remove keyword from tracking

#### System Information
- `GET /stats` - System statistics and database info
- `GET /cache_status` - Cache status and performance metrics (authenticated users only)

### SVM Parameters & Optimization
- **SVM Configuration**: C=0.02 (regularization), balanced class weights, LinearSVC for speed
- **Logic Modes**: `and` (weighted combination) / `or` (union) for multi-tag queries
- **Performance Tuning**: SSD storage, 16GB+ RAM, Intel extensions, proper batch sizes
- **Monitoring**: Real-time cache statistics and performance monitoring via `/cache_status`

### Search Modes
- **Keyword**: Traditional TF-IDF with multi-core parallel processing
- **Semantic**: Cosine similarity using pre-computed embeddings
- **Hybrid**: Weighted combination with normalization and configurable semantic weight (0.0-1.0)

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Acknowledgments
- Original [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) by Andrej Karpathy
- [minerU](https://github.com/opendatalab/MinerU) for advanced PDF parsing
- [vLLM](https://github.com/vllm-project/vllm) for high-performance model serving
