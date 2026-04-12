# arxiv-sanity-X

[中文](README_CN.md) | [English](README.md)

A comprehensive arXiv paper browsing and recommendation system featuring AI-powered summarization, hybrid search capabilities, and personalized recommendations. Built with modern ML techniques including TF-IDF, semantic embeddings, and LLM integration.

![Screenshot](arxiv-sanity-x.png)

## 📋 Table of Contents

### Getting Started

- [Core Overview](#-core-overview)
- [Quick Start](#-quick-start)
- [Docs](#docs)

### Usage

- [User Guide](#-user-guide)
- [AI Paper Summarization](#-ai-paper-summarization)
- [Advanced Features](#-advanced-features)

### Configuration

- [Configuration Guide](#configuration-guide)
- [Prerequisites & OS Notes](#-prerequisites--os-notes)

### Operations

- [Data Layout & Migration](#-data-layout--migration)
- [Deployment & Security](#-deployment--security-notes)
- [Troubleshooting](#-troubleshooting)

### Development

- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Architecture](#architecture)
- [API Reference](#-api-reference)
- [Development Guide](#-development-guide)

### Other

- [Changelog](#-changelog)
- [Acknowledgments](#-acknowledgments)

---

## Docs

- Entry point: [docs/INDEX.md](docs/INDEX.md)
- Repo skills manuals: [.opencode/skills/README.md](.opencode/skills/README.md)
- Operations: [docs/OPERATIONS.md](docs/OPERATIONS.md)
- Security: [docs/SECURITY.md](docs/SECURITY.md)
- Development: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- Test guide: [tests/README.md](tests/README.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)

## 🎯 Core Overview

arxiv-sanity-X is a personal research workbench for tracking arXiv papers. It combines (1) reliable paper ingestion, (2) fast search, and (3) feedback-driven recommendations, so you can quickly find what matters, save it, and keep up daily.

### Key Capabilities

| Feature                           | Description                                                                |
| --------------------------------- | -------------------------------------------------------------------------- |
| 🔍 **Multi-mode Search**          | Keyword (TF-IDF), semantic (Embedding), hybrid search with tunable weights |
| 🎯 **Smart Recommendations**      | SVM classifiers trained on positive/negative feedback tags                 |
| 🤖 **AI Summaries**               | HTML/PDF parsing + LLM-generated structured summaries, multi-model support |
| 🏷️ **Tag System**                 | Positive/negative feedback, combined tags, keyword tracking, reading list  |
| 📧 **Email Recommendations**      | Automated daily recommendation emails with holiday-aware scheduling        |
| 🔄 **Automation**                 | Built-in scheduler: fetch → compute → summarize → email                    |
| 📤 **PDF Uploads (Experimental)** | Upload private PDFs for parsing, summaries, and similarity search          |

## 🛠️ Tech Stack

### Backend

- **Framework**: Flask with Blueprint-based modular architecture
- **Database**: SQLite with custom KV store (WAL mode, compression support)
- **Task Queue**: Huey (SQLite backend) for async summary/upload orchestration
- **Configuration**: pydantic-settings for type-safe config management
- **Real-time**: Server-Sent Events (SSE) for live updates

### Frontend

- **Templates**: Jinja2 with responsive HTML/CSS
- **JavaScript**: Hybrid page runtime with IIFE/global scripts, page-specific React surfaces, and vanilla DOM flows
- **Rendering**: MathJax for LaTeX, markdown-it for Markdown
- **Build**: esbuild with content-hash caching

### ML/AI

- **Search**: TF-IDF (scikit-learn) + semantic embeddings (Ollama/OpenAI API)
- **Recommendations**: SVM classifiers trained on user feedback
- **Summarization**: OpenCode-backed text models
- **PDF Parsing**: MinerU (API or local VLM)

### Infrastructure

- **Web Server**: Gunicorn with multi-worker support
- **Scheduler**: APScheduler for automated pipelines
- **Services**: OpenCode server, Ollama embeddings, MinerU VLM

## 📁 Project Structure

```
arxiv-sanity-x/
├── serve.py              # Flask entry point
├── tasks.py              # Huey task definitions & orchestration center
│
├── backend/              # Flask application
│   ├── app.py            # App factory & initialization
│   ├── blueprints/       # Route handlers (10 blueprints)
│   │   ├── web.py        # Page routes (/, /summary, /profile, etc.)
│   │   ├── api_user.py   # Login/logout, user state, email registration
│   │   ├── api_search.py # Search endpoints
│   │   ├── api_summary.py# Summary generation & status
│   │   ├── api_tags.py   # Tag management
│   │   ├── api_papers.py # Paper data & images
│   │   ├── api_readinglist.py # Reading list
│   │   ├── api_uploads.py# Upload PDFs + parsing/extract/similarity
│   │   ├── api_sse.py    # Server-Sent Events
│   │   └── metrics.py    # /metrics (Prometheus, optional)
│   ├── services/         # Business logic layer
│   │   ├── data_service.py    # Cache & data management
│   │   ├── search_service.py  # TF-IDF, semantic, hybrid search
│   │   ├── summary_service.py # Summary generation & caching
│   │   ├── semantic_service.py# Embedding & vector search
│   │   └── ...
│   ├── schemas/          # Pydantic request/response models
│   └── utils/            # Helpers (cache, SSE, manifest)
│
├── aslite/               # Data layer
│   ├── db.py             # SqliteKV wrapper & DB access
│   ├── repositories.py   # Repository pattern for data access
│   └── arxiv.py          # arXiv API client
│
├── config/               # Configuration
│   ├── settings.py       # pydantic-settings definitions
│   ├── cli.py            # Config CLI tool
│   └── ...               # OpenCode / app configuration
│
├── tools/                # CLI tools & automation
│   ├── arxiv_daemon.py   # Paper fetching from arXiv
│   ├── compute.py        # TF-IDF & embedding computation
│   ├── daemon.py         # Scheduled task runner
│   ├── batch_paper_summarizer.py # Batch summary generation
│   ├── paper_summarizer.py # Single paper summarizer
│   └── send_emails.py    # Email recommendations
│
├── bin/                  # Service launchers
│   ├── run_services.py   # One-command multi-service launcher
│   ├── up.sh             # Gunicorn startup script
│   ├── huey_consumer.py  # Huey consumer wrapper (memory limit + worker role)
│   ├── embedding_serve.sh# Ollama embedding server
│   ├── mineru_serve.sh   # MinerU VLM server
│   └── ...               # Optional local service launchers
│
├── static/               # Frontend assets
│   ├── *.js              # Source JavaScript files
│   ├── css/              # Stylesheets
│   ├── lib/              # Third-party libraries
│   └── dist/             # Built assets (gitignored)
│
├── templates/            # Jinja2 HTML templates
├── scripts/              # Build & maintenance scripts
├── tests/                # Test suite
├── data/                 # Runtime data (gitignored)
│   ├── papers.db         # Paper metadata
│   ├── dict.db           # User data (tags, keywords, etc.)
│   ├── features.p        # Computed features
│   ├── huey.db           # Huey task queue DB (SQLite)
│   ├── uploads/          # Uploaded PDFs + metadata
│   └── summary/          # Cached summaries
└── data-repo/            # Optional git submodule for backing up data/dict.db
```

## 🧭 User Guide

This section covers how to use the arxiv-sanity-X website. Most workflows start from the homepage.

### 1) Sign in

- Click **Profile** in the top-right corner to access your profile page
- Enter a username to log in (no password required, suitable for personal/intranet use)
- If you plan to expose the site publicly, put it behind authentication/VPN and set a stable `ARXIV_SANITY_SECRET_KEY` (recommended; `secret_key.txt` is a local fallback and must not be committed)
- (Optional) Register notification emails on the Profile page. Multiple addresses are supported (comma/whitespace/newline separated). Submit an empty value to clear.
- JSON callers to `POST /login` and `POST /register_email` receive JSON success/error payloads; browser form posts keep the existing redirect UX.

### 2) Browse and Search Papers

**Homepage Features:**

- Papers are displayed by time (newest first) by default
- Click paper title to view details, click arXiv link to open original
- Use the search box at the top (keyboard shortcut: `Ctrl+K`)

**Search Syntax:**
| Syntax | Example | Description |
| -------- | ------------------------ | ------------------------------------ |
| Title | `ti:transformer` | Search titles containing transformer |
| Author | `au:goodfellow` | Search by author |
| Category | `cat:cs.LG` | Search specific arXiv category |
| ID | `id:2312.12345` | Find by arXiv ID |
| Phrase | `"large language model"` | Exact phrase match |
| Exclude | `-survey` or `!survey` | Exclude results containing the term |

**Search Mode Toggle:**

- **keyword**: Fastest, TF-IDF based, no extra services required
- **semantic**: Vector similarity based, requires Embedding enabled
- **hybrid**: Combines keyword + semantic, adjustable weight (recommended)

**Homepage Filter Rules:**

- If you type a search query, the UI and backend keep ranking in query-compatible modes (`search` or `time`)
- Advanced fields such as **Tags**, **PIDs**, **Logic**, and **SVM C** only apply when **Rank by** is `tags` or `pid`
- Tag inputs in advanced filters support inline suggestions and keyboard selection
- For `POST /api/keyword_search`, omitting `time_delta` keeps the full indexed corpus in play; use `time_delta <= 0` to disable time filtering explicitly in scripted calls.

### 3) Tagging System and Personalized Recommendations

**Adding Tags:**

- Click the **+** button on paper cards to add tags
- Supports positive tags (like) and negative tags (dislike)
- Tags train your personal SVM recommendation model

**Using Tag Recommendations:**

- Select **Tags** sort mode on the homepage
- Choose one or more tags, system will recommend similar papers
- Combined tags (e.g., `RL,NLP`) enable intersection recommendations

### 4) AI Paper Summaries

- Click a paper to enter detail page, click **Summary** button
- First generation requires LLM processing (typically 10-30 seconds)
- Results are cached, subsequent visits display instantly
- Switch between different LLM models to regenerate
- Clear current model cache or all caches as needed

### 5) Reading List

- Click the bookmark/reading-list button on paper cards to add papers to your reading list
- Visit `/readinglist` page to manage your reading list
- Useful for batch summarization or read-later queuing
- The reading list page mirrors the main paper-card actions, including Similar / Inspect / Summary links and private upload actions

### 6) Private PDF Uploads

- Uploaded PDFs are private and owner-scoped; non-owners still see `404` for upload resources and mutation APIs
- If an upload is already being deleted, mutation APIs such as `POST /api/uploaded_papers/update_meta`, `POST /api/uploaded_papers/retry_parse`, `POST /api/uploaded_papers/parse`, `POST /api/uploaded_papers/process`, and `POST /api/uploaded_papers/extract_info` now return `409` with `"Paper is being deleted"`
- `parse` / `process` / `retry_parse` treat only the currently tracked task as active; stale tracked tasks can be repaired and replaced with a new `task_id`
- Upload task polling via `GET /api/task_status/<task_id>` may now surface `failed` with `stale_running_repaired` or `canceled` with `superseded_upload_task`
- Deleting an upload best-effort cancels in-flight upload parse/process/extract work and any linked summary task before file/DB cleanup continues
- If your deployment has long upload queue delays, tune `ARXIV_SANITY_HUEY_UPLOAD_REPAIR_TTL` so valid upload tasks are not repaired too aggressively

### 7) Other Features

- **Stats page**: View paper statistics, daily addition charts
- **About page**: View system info, supported arXiv categories
- **Email recommendations**: Receive daily recommendations after configuring SMTP (see Configuration Guide)

## 📦 Data Layout & Migration

By default, data is stored under `data/` (configured by `ARXIV_SANITY_DATA_DIR` in `.env` / `config/settings.py`):

- `data/papers.db`: fetched papers + metadata
- `data/dict.db`: user data (tags, negative tags, keywords, reading list, email registry, summary status)
- `data/features.p`: TF‑IDF / hybrid features generated by [tools/compute.py](tools/compute.py)
- `data/summary/`: cached LLM summaries
- `data/pdfs/`, `data/mineru/`, `data/html_md/`: intermediate caches for parsing
- `data/uploads/`: uploaded private PDFs and derived artifacts (if you use uploads)
- `data/huey.db`: Huey task queue database
- `data/sse_events.db`: SSE cross-process event bus (SQLite, when enabled)
- `data-repo/` (optional): git submodule used by the daemon to back up `data/dict.db`

To migrate to a new machine, you typically copy at least:

- `data/papers.db`
- `data/dict.db`
- `data/features.p` (or regenerate it by running [tools/compute.py](tools/compute.py))
- `data/summary/` (optional, if you want to keep cached summaries)

If you use git backup via `data-repo/`, you can also restore from:

- `data-repo/dict.db`

To use `data-repo/` backup with the daemon:

1. Initialize the submodule: `git submodule update --init --recursive`
2. `ARXIV_SANITY_DAEMON_ENABLE_GIT_BACKUP=true` is already the default; set it to `false` if you want to disable backups
3. If you do not want remote pushes, set `ARXIV_SANITY_DAEMON_BACKUP_PUSH=false`; otherwise ensure `data-repo/` has a valid git remote and your runtime environment can `git push`

## 🔐 Deployment & Security Notes

- The built-in login is **username only** (no password). This is intended for personal / trusted environments.
- If you deploy on a public server, protect it behind authentication/VPN/reverse-proxy, and set a stable secret key via `ARXIV_SANITY_SECRET_KEY` or `secret_key.txt`.
- Do not commit your API keys. Prefer environment variables in `.env` or your shell environment.

## 🧩 Troubleshooting

- **The website is empty / no papers**: you likely didn’t run [tools/arxiv_daemon.py](tools/arxiv_daemon.py) + [tools/compute.py](tools/compute.py) yet.
- **Summaries always fail**: check `ARXIV_SANITY_OPENCODE_BASE_URL`, `ARXIV_SANITY_LLM_NAME`, and `ARXIV_SANITY_EXTRACT_MODEL_NAME` in `.env`.
- **Summary doesn’t start generating**: the Summary page will not auto-enqueue jobs on cache misses; click **Generate**. Ensure a Huey consumer is running (recommended: `python bin/run_services.py`, or run only the consumer with `python bin/huey_consumer.py tasks.huey -w 4 -k thread`).
- **Semantic/hybrid search has no effect**: ensure embeddings are enabled and you regenerated features with [tools/compute.py](tools/compute.py) (for hybrid features).
- **Time-sorted lists look wrong / slow**: rebuild the metadata time index: `python -m tools rebuild_time_index`.
- **MinerU errors**:
    - API backend: check `ARXIV_SANITY_MINERU_API_KEY`
    - local backend: check `ARXIV_SANITY_MINERU_BACKEND` and that the service is reachable on `ARXIV_SANITY_MINERU_PORT`
- **Stuck jobs after crash (locks)**: run `python -m scripts cleanup_locks` or tune `ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC` / `ARXIV_SANITY_MINERU_LOCK_STALE_SEC`.
- **Stuck/ghost summary tasks (Huey)**: dry-run `python -m scripts cleanup_tasks`, then rerun with `--force` (optionally `--flush-huey` to clear the entire queue). Use with care.
- **Cannot load features.p due to NumPy mismatch**: regenerate features by rerunning [tools/compute.py](tools/compute.py) under the current environment.
- **Gunicorn WORKER TIMEOUT / SIGKILL**: if logs show `WORKER TIMEOUT`, increase Gunicorn timeout (e.g. `ARXIV_SANITY_GUNICORN_EXTRA_ARGS="--timeout 600 --graceful-timeout 600"`), and avoid too many worker processes when caches are enabled. `bin/up.sh` auto-selects `gevent` for SSE and adds generous timeouts by default.
- **Gevent MonkeyPatchWarning (ssl/urllib3)**: this typically happens with `--preload`. If you still see it, try `ARXIV_SANITY_GUNICORN_PRELOAD=false` or force `ARXIV_SANITY_GUNICORN_WORKER_CLASS=gthread`.
- **Real-time updates not working (SSE)**: check `ARXIV_SANITY_SSE_ENABLED=true` and inspect `GET /api/sse_stats` for per-process queue/bus status.

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/xihuai18/arxiv-sanity-x && cd arxiv-sanity-x
pip install -r requirements.txt
npm install
```

Install the [OpenCode](https://opencode.ai) CLI separately and make sure `opencode --version` works in your shell. The default launcher flow starts OpenCode for you. For OpenCode installation and configuration details, see <https://opencode.ai>.

### 2. Create `.env`

```bash
cp .env.example .env
```

### 3. Configure OpenCode + Models

Text generation now always goes through an OpenCode HTTP server.

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

Notes:

- Model selectors can be a canonical `provider/model` or a supported alias such as `gpt-5.4`.
- Alias members are tried automatically. If every candidate under the selected alias fails, the request fails immediately.
- Upload metadata extraction also uses OpenCode.
- Embeddings are still separate and do **not** reuse the OpenCode text-model config.
- `ARXIV_SANITY_OPENCODE_MANAGED` defaults to `true`, so `bin/run_services.py` launches `opencode serve` locally and points child processes at that local instance.
- With the default `managed=true`, `python bin/run_services.py` requires the `opencode` binary to be installed and available on `PATH`.

### 4. Configure Embeddings / MinerU If Needed

```bash
# Local embeddings (default path)
ARXIV_SANITY_EMBED_USE_LLM_API=false
ARXIV_SANITY_EMBED_PORT=54000
ARXIV_SANITY_EMBED_MODEL_NAME=qwen3-embedding:0.6b

# Remote embedding API (explicit only; no fallback to text-model config)
# ARXIV_SANITY_EMBED_USE_LLM_API=true
# ARXIV_SANITY_EMBED_API_BASE=https://your-embedding-endpoint/v1
# ARXIV_SANITY_EMBED_API_KEY=...

# MinerU API backend
# ARXIV_SANITY_MINERU_ENABLED=true
# ARXIV_SANITY_MINERU_BACKEND=api
# ARXIV_SANITY_MINERU_API_KEY=...
```

### 5. Validate and Start

```bash
python -m config.cli validate
python -m config.cli doctor

python -m tools arxiv_daemon -n 10000 -m 500
python -m tools compute --num 20000

python bin/run_services.py
```

Open `http://localhost:55555`.

## 🧰 Runtime Notes

- Recommended full stack: `python bin/run_services.py`
- Full stack + daemon: `python bin/run_services.py --with-daemon`
- Web only: `npm run build:static && python serve.py`
- Gunicorn only: `bash bin/up.sh`
- Huey worker only: `python bin/huey_consumer.py tasks.huey -w 4 -k thread`

By default, `bin/run_services.py` also launches `opencode serve` locally. Set `ARXIV_SANITY_OPENCODE_MANAGED=false` only when you want to use an already-running external OpenCode service.

## 🤖 AI Summaries

Summary generation pipeline:

1. Fetch HTML/PDF content
2. Convert to Markdown (`html` or `mineru`)
3. Send text-generation requests to OpenCode
4. Cache summary files under canonical model ids
5. Expose `summary_meta.llm_model`, `resolved_model`, and usage metadata to the UI/API

`GET /api/llm_models` is now built from OpenCode `/config/providers` and still returns the legacy-compatible shape:

```json
{ "models": [{ "id": "openai/gpt-5.4" }], "default": "openai/gpt-5.4" }
```

## 📚 API Highlights

- `GET /health`: non-strict health check
- `GET /ready`: strict readiness check (includes OpenCode + required model validation)
- `GET /api/llm_models`: list available canonical text models from OpenCode
- `POST /api/get_paper_summary`: cache-only summary read
- `POST /api/trigger_paper_summary`: enqueue summary generation
- `POST /api/summary_status`: summary status lookup
- `POST /api/uploaded_papers/extract_info`: upload metadata extraction through OpenCode

## Notes

- Text generation requires [OpenCode](https://opencode.ai) (`opencode serve`) as the LLM backend. See [Configuration Guide](#configuration-guide) for setup.
- The default launcher (`python bin/run_services.py`) starts OpenCode locally when `ARXIV_SANITY_OPENCODE_MANAGED=true` (default).
- `config/cli` and `/ready` both use OpenCode semantics now.
- For exact defaults, see `docs/DEFAULTS.md`.

### Real-time Updates (`api_sse.py`)

| Endpoint               | Description             |
| ---------------------- | ----------------------- |
| `GET /api/user_stream` | User SSE stream         |
| `GET /api/sse_stats`   | SSE stats (per process) |

### Uploads (Experimental) (`api_uploads.py`)

| Endpoint                                 | Description                                         |
| ---------------------------------------- | --------------------------------------------------- |
| `POST /api/upload_pdf`                   | Upload private PDF                                  |
| `GET /api/uploaded_papers/list`          | List uploaded papers                                |
| `POST /api/uploaded_papers/process`      | Process upload (parse + extract + summary)          |
| `POST /api/uploaded_papers/parse`        | Parse uploaded PDF                                  |
| `POST /api/uploaded_papers/extract_info` | Extract metadata via LLM                            |
| `POST /api/uploaded_papers/update_meta`  | Update uploaded paper metadata                      |
| `POST /api/uploaded_papers/delete`       | Delete uploaded paper                               |
| `POST /api/uploaded_papers/retry_parse`  | Retry parsing                                       |
| `GET /api/uploaded_papers/pdf/<pid>`     | Download uploaded PDF                               |
| `GET /api/uploaded_papers/similar/<pid>` | Similarity search for an upload                     |
| `GET /api/uploaded_papers/tldr/<pid>`    | Get upload TL;DR (from cached summary if available) |

---

## 🔨 Development Guide

### Environment Setup

```bash
# Clone repository
git clone https://github.com/xihuai18/arxiv-sanity-x && cd arxiv-sanity-x

# Create conda environment (recommended)
conda create -n sanity python=3.10
conda activate sanity

# Install dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend build)
npm install
```

### Frontend Development

The frontend uses vanilla JavaScript with esbuild for bundling:

```bash
# Production build (with content hash for caching)
npm run build:static

# Development build (no hash, easier debugging)
npm run build:dev

# Watch mode (auto-rebuild on changes)
npm run build:watch

# Lint JavaScript files
npm run lint

# Format code
npm run format
```

**Note**: The `bin/up.sh` startup script automatically runs the build, so manual building is usually not needed for deployment.

### Backend Development

```bash
# Run development server
# Build frontend assets first if static/dist is missing
npm run build:static
# Set ARXIV_SANITY_RELOAD=true if you want auto-reload
python serve.py

# Or use gunicorn for production-like testing
bash bin/up.sh
```

### Configuration Management

```bash
# Show current configuration
python -m config.cli show

# Validate configuration
python -m config.cli validate

# Generate environment variable template
python -m config.cli env

# Include secret values when you explicitly need them
python -m config.cli env --include-secrets
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Code Style

- Python: Follow PEP 8, use type hints
- JavaScript: ESLint + Prettier
- Use `loguru` for logging in Python

### Architecture

#### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Entry Point (serve.py)                                     │
│  - Flask app creation                                       │
│  - Gunicorn preloading for copy-on-write memory sharing     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  API Layer (backend/blueprints/)                            │
│  - 10 Flask blueprints organizing routes by domain          │
│  - Request validation, authentication, response formatting  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Service Layer (backend/services/)                          │
│  - Business logic, caching, search algorithms               │
│  - Reusable across blueprints, testable in isolation        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Repository Layer (aslite/repositories.py)                  │
│  - Data access abstraction over raw DB operations           │
│  - Batch operations, type hints, easy mocking               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Data Layer (aslite/db.py)                                  │
│  - Custom SQLite wrapper (SqliteKV) with WAL mode           │
│  - Dict-like interface, compression support                 │
└─────────────────────────────────────────────────────────────┘
```

#### Core Design Patterns

1. **Repository Pattern**: `PaperRepository`, `TagRepository`, `ReadingListRepository` provide clean data access abstractions
2. **Service Layer Pattern**: `data_service` (multi-level caching), `search_service` (query parsing & ranking), `summary_service` (summary generation orchestration)
3. **Factory Pattern**: `create_app()` creates configured Flask instance, supports testing and Gunicorn preloading
4. **Task Queue Pattern**: Huey + SQLite backend for async summary generation with priority queue support
5. **Cache-Aside Pattern**: Features cache (mtime invalidation), papers cache (memory LRU), summary cache (file + locks)

#### Data Flow: arXiv to Display

```
arXiv API → arxiv_daemon.py → papers.db/dict.db
                    ↓
            compute.py → features.p (TF-IDF + Embeddings)
                    ↓
User Search → search_service → Ranked Results → Frontend Render
                    ↓
Click Summary → Huey Task → HTML/PDF Parse → LLM → Cache → SSE Push
```

---

## 📈 Changelog

### Unreleased

### v3.3 - OpenCode Migration & LiteLLM Deprecation

- 🔄 **LiteLLM → OpenCode**: Replaced LiteLLM with [OpenCode](https://opencode.ai) as the unified LLM backend. All text generation and structured extraction now go through an OpenCode HTTP server (`opencode serve`), managed by the launcher by default (`ARXIV_SANITY_OPENCODE_MANAGED=true`)
- 🏗️ **Model Alias System**: New `config/model_aliases.py` maps short display names (e.g., `gpt-5.4`) to canonical OpenCode `provider/model` IDs with automatic fallback chains
- 🔧 **New OpenCode Settings**: `ARXIV_SANITY_OPENCODE_*` env prefix for server URL, host/port, auth, timeout — see [Configuration Guide](#configuration-guide)
- 🩺 **Health/Readiness**: `/ready` now validates OpenCode service health and required model availability via OpenCode `/global/health` and `/config/providers`
- 📄 **Documentation**: added `docs/` guides (operations/security/development) and linked from README
- 📡 **API docs**: clarified `/api/task_status/<task_id>` owner-only fields (including `stage`)
- 📡 **API docs**: fixed Uploads endpoints to include `/api` prefix and added `/api/uploaded_papers/process`
- 📊 **Observability**: documented optional Sentry (`ARXIV_SANITY_SENTRY_*`) and Prometheus metrics (`/metrics`)
- 🗑️ **Removed**: `config/llm.yml`, `config/llm_template.yml`, and all LiteLLM runtime dependencies

### v3.2 - Upload, Testing & Security Hardening

- 📤 **Paper Upload**: Upload private PDFs for similarity search against the paper corpus (experimental)
- 🧪 **Test Suite Enhancement**: Comprehensive unit and integration tests for APIs, services, and utilities
- 🔒 **Security Fixes**:
    - Tag search APIs (`/api/tag_search`, `/api/tags_search`) now require authentication and validate user identity
    - Email validation accepts modern long TLDs (up to 63 characters, e.g., `.engineering`, `.museum`)
    - Semantic search guards against missing pid list to prevent IndexError
- 🛠️ **Daemon Improvements**:
    - `ARXIV_SANITY_DAEMON_ENABLE_EMBEDDINGS=false` now correctly disables embeddings via `--no-embeddings` flag
    - Email dry-run mode support (`ARXIV_SANITY_DAEMON_EMAIL_DRY_RUN`)
- 🏗️ **Architecture Refactoring**:
    - Repository pattern for cleaner data access (`aslite/repositories.py`)
    - Native SQLite3 replacing sqlitedict for improved concurrency
    - Cross-process locking for database operations
- 🎨 **Frontend Polish**: MathJax integration refactoring, static asset cleanup, synchronous loading optimization

### v3.1 - Reading List & Enhanced Tagging

- 📚 **Reading List**: Personal paper collection with add/remove functionality and dedicated `/readinglist` page
- 👍👎 **Positive/Negative Tagging**: Enhanced feedback system with positive and negative tag states for SVM training
- ⚖️ **SVM Negative Weight**: New `SVM_NEG_WEIGHT` config parameter for explicit negative feedback influence
- 🔄 **Real-time Sync**: BroadcastChannel-based state synchronization across browser tabs and components
- 📊 **Summary Status**: Visual status indicators (queued/running/ok/failed) for summary generation
- 🏷️ **arXiv Tag Groups**: Grouped display of arXiv categories with dynamic About page updates
- 🎨 **UI Polish**: Enhanced tag dropdown interactions, confirmation dialogs, and visual feedback

### v3.0 - UI Redesign & HTML Summarization

- 🎨 **UI Overhaul**: Redesigned About, Profile, Stats pages with modern layout and feature grids
- 📄 **HTML Summarization**: ar5iv/arxiv HTML parsing (faster than PDF, better structure)
- 🤖 **Model Selection**: Multiple LLM models with auto-retry in summary page
- 🔍 **Enhanced Search**: Keyboard shortcuts (Ctrl+K), advanced filters, accessibility improvements
- 📊 **Stats Chart**: Daily paper count visualization with bar chart

<details>
<summary>📜 Earlier Versions (v1.0 - v2.4)</summary>

### v2.4 - Multi-threading & Service Enhancement

- ⚡ **Concurrency Optimization**: True multi-threaded concurrent paper summarization processing
- 🔒 **Thread Safety**: File-level locking mechanism to avoid minerU parsing conflicts
- 📊 **Enhanced Statistics**: Detailed processing statistics and failure reason analysis
- 🔄 **Retry Mechanism**: Smart retry for failed paper processing tasks

### v2.3 - AI Paper Summarization

- ✨ **New**: Complete AI-powered paper summarization system
- 🧠 **MinerU Integration**: Advanced PDF parsing with structure recognition
- 📝 **Summary Interface**: New `/summary` route with async loading

### v2.2 - Performance & Stability

- ⚡ **Performance**: Enhanced unified data caching system with intelligent auto-reload
- 📈 **Scheduler Enhancement**: Increased fetch frequency to 4x daily

### v2.1 - API & Semantic Search

- ✨ **New**: Semantic search with keyword, semantic, and hybrid modes
- 🔗 **API Integration**: RESTful API endpoints for recommendations

### v2.0 - Enhanced ML Features

- ✨ **New**: Hybrid TF-IDF + embedding vector features
- ⚡ **Performance**: Multi-core optimization and Intel scikit-learn extensions

### v1.0 - Foundation

- 📚 arXiv paper fetching and storage with SQLite database
- 🏷️ User tagging and keyword systems
- 📧 Email recommendation service
- 🤖 SVM-based paper recommendations

</details>

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Acknowledgments

- Original [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) by Andrej Karpathy
- [minerU](https://github.com/opendatalab/MinerU) for advanced PDF parsing
- [Ollama](https://github.com/ollama/ollama) for local embedding serving
- [vLLM](https://github.com/vllm-project/vllm) for MinerU VLM serving
