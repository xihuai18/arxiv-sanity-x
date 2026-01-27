# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

arxiv-sanity-X is a comprehensive arXiv paper browsing and recommendation system featuring AI-powered summarization, hybrid search capabilities, and personalized recommendations. Built with Flask backend, vanilla JavaScript frontend, and modern ML techniques including TF-IDF, semantic embeddings, and LLM integration.

## Common Commands

### Development Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend build)
npm install

# Create configuration from template
cp .env.example .env
# Edit .env with your LLM API keys and settings

# Verify configuration
python -m config.cli show
python -m config.cli validate
```

### Data Initialization
```bash
# Fetch papers from arXiv (required before first run)
python -m tools arxiv_daemon -n 10000 -m 500

# Compute TF-IDF and embedding features (required before first run)
python -m tools compute --num 20000

# Optional: Generate summaries for recent papers
python -m tools batch_paper_summarizer -n 100 -w 2
```

### Running the Application

**Development mode (single process, auto-reload):**
```bash
python serve.py
```

**Production mode (Gunicorn with multiple workers):**
```bash
./bin/up.sh
```

**One-command startup (web + optional services):**
```bash
# Start all configured services
python bin/run_services.py

# Skip optional services
python bin/run_services.py --no-embed --no-mineru --no-litellm

# Include scheduler daemon
python bin/run_services.py --with-daemon
```

### Frontend Development
```bash
# Production build (with content hash for cache-busting)
npm run build:static

# Development build (no hash, easier debugging)
npm run build:dev

# Watch mode (auto-rebuild on changes)
npm run build:watch

# Lint and format
npm run lint
npm run format
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests (fast, no external dependencies)
pytest tests/integration/   # Integration tests (mocked services)
pytest tests/e2e/           # End-to-end tests
pytest tests/live/          # Live tests (requires running services)

# Run with coverage
pytest --cov=backend --cov=aslite --cov=tools

# Run specific test file
pytest tests/unit/test_frontend_backend_contract.py
```

### Code Quality
```bash
# Pre-commit hooks (runs automatically on git commit)
pre-commit install
pre-commit run --all-files

# Python formatting (Black, isort, autoflake)
black backend/ aslite/ tools/ config/
isort backend/ aslite/ tools/ config/

# JavaScript linting
npm run lint:fix
```

### Maintenance Tasks
```bash
# Clean up stale locks after crashes
python -m tools cleanup_locks

# Backup user data (dict.db) to git
python -m tools backup_dict

# Send email recommendations manually
python -m tools send_emails -t 2

# Run automated scheduler (fetch → compute → summarize → email)
python -m tools daemon
```

## Architecture Overview

### Layered Architecture

The codebase follows a strict layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  Entry Point (serve.py)                                     │
│  - Flask app creation                                       │
│  - Gunicorn preloading for copy-on-write memory sharing     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  API Layer (backend/blueprints/)                            │
│  - 8 Flask blueprints organizing routes by domain           │
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

### Key Design Patterns

**Repository Pattern** ([aslite/repositories.py](aslite/repositories.py)):
- Encapsulates data access logic with clean abstractions
- Static methods, stateless design for easy invocation
- Repositories: `PaperRepository`, `TagRepository`, `ReadingListRepository`, `SummaryStatusRepository`, `UserRepository`, `UploadedPaperRepository`
- Provides batch operations (`get_many()`, `items_with_prefix()`) and type hints

**Service Layer Pattern** ([backend/services/](backend/services/)):
- Separates business logic from routing and data access
- Key services:
  - `data_service.py`: Multi-level caching with mtime-based invalidation
  - `search_service.py`: Query parsing, TF-IDF/semantic/hybrid ranking
  - `summary_service.py`: Summary generation orchestration, cache management
  - `semantic_service.py`: Embedding generation and vector search

**Factory Pattern** ([backend/app.py](backend/app.py)):
- `create_app()` function creates configured Flask instance
- Enables testing with different configurations
- Supports Gunicorn preloading for performance

**Task Queue Pattern** ([tasks.py](tasks.py)):
- Asynchronous summary generation using Huey with SQLite backend
- Priority queue support for recent papers
- Status tracking in database for UI feedback via SSE

**Cache-Aside Pattern**:
- Features cache: Mtime-based invalidation, atomic file replacement
- Papers/metas cache: Optional in-memory caching with mtime checks
- Search cache: LRU with TTL for query results
- Summary cache: File-based with metadata, lock-based concurrency control

### Data Flow: arXiv to Display

**1. Paper Ingestion** ([tools/arxiv_daemon.py](tools/arxiv_daemon.py)):
```
arXiv API → parse_response() → PaperRepository.save_many() → papers.db
                                MetaRepository.save_many() → dict.db
```

**2. Feature Computation** ([tools/compute.py](tools/compute.py)):
```
papers.db → TfidfVectorizer → sparse matrix
                            ↓
         Embedding API → dense vectors
                            ↓
         concatenation → features.p (pickle)
```

**3. Search Request**:
```
User Query → parse_search_query() → {terms, phrases, filters, neg_terms}
                                   ↓
         keyword_rank() / semantic_rank() / hybrid_rank()
                                   ↓
         Cached features.p + TF-IDF/embeddings
                                   ↓
         Ranked PIDs + scores → fetch papers → render
```

**4. Summary Generation**:
```
User clicks Summary → /api/get_paper_summary
                                   ↓
         Check cache → if miss, enqueue Huey task
                                   ↓
         Huey worker: fetch HTML/PDF → parse to Markdown
                                   ↓
         LLM API → structured summary → cache to disk
                                   ↓
         SSE event → frontend updates status
```

### Frontend Architecture

**Build System** ([scripts/build.js](scripts/build.js)):
- esbuild for bundling and minification
- Content-hash filenames for cache-busting (e.g., `paper_list-ABC123.js`)
- Manifest file ([static/dist/manifest.json](static/dist/manifest.json)) maps original names to hashed names
- Atomic writes: builds to temp directory, replaces on success

**JavaScript Modules**:
- [static/paper_list.js](static/paper_list.js): Homepage paper cards, search, filtering
- [static/paper_summary.js](static/paper_summary.js): Summary page, LLM model selection, SSE updates
- [static/readinglist.js](static/readinglist.js): Reading list management
- [static/common_utils.js](static/common_utils.js): Shared utilities, API calls, SSE handling
- [static/markdown_summary_dom_utils.js](static/markdown_summary_dom_utils.js): Markdown rendering with MathJax

**Real-time Updates**:
- Server-Sent Events (SSE) via [backend/blueprints/api_sse.py](backend/blueprints/api_sse.py)
- Frontend subscribes to user-specific or global event streams
- Used for summary status updates, tag changes, reading list modifications

### Configuration Management

**pydantic-settings** ([config/settings.py](config/settings.py)):
- Type-safe configuration with validation
- All environment variables prefixed with `ARXIV_SANITY_`
- Nested settings: `settings.llm.base_url`, `settings.mineru.enabled`, etc.
- Field validators for consistency checks

**Configuration CLI** ([config/cli.py](config/cli.py)):
```bash
python -m config.cli show      # Display current configuration
python -m config.cli validate  # Validate configuration
python -m config.cli env       # Generate environment variable template
```

**Key Configuration Files**:
- `.env`: Core configuration (copy from [.env.example](.env.example))
- [config/llm.yml](config/llm.yml): LiteLLM multi-model gateway (optional)
- [tools/arxiv_daemon.py](tools/arxiv_daemon.py): arXiv categories to fetch (`ALL_TAGS`)

## Important Development Notes

### Database Concurrency

- SQLite uses WAL mode for concurrent reads/writes
- Repository layer provides cross-process locking for critical operations
- Huey tasks use file-based locks to prevent duplicate work
- After crashes, run `python -m tools cleanup_locks` to clear stale locks

### Caching Strategy

**Multi-level caching** for performance:
1. **In-memory LRU**: Papers, metas, search results (with TTL)
2. **File-based with mtime**: Features (features.p), embeddings
3. **Disk cache**: Summaries, HTML/PDF parsing results

**Cache invalidation**:
- Features cache: Regenerated by `compute.py`, mtime-checked on load
- Papers/metas cache: Mtime-checked against DB file
- Summary cache: Explicit clearing via API or CLI

### Testing Approach

**Contract Testing** ([tests/unit/](tests/unit/)):
- `test_frontend_backend_contract.py`: Validates API endpoints referenced in JS exist in Flask routes
- `test_frontend_sse_wiring.py`: Checks SSE event handling consistency
- `test_template_manifest_contract.py`: Ensures templates reference valid static assets
- `test_template_route_contract.py`: Validates template links match actual routes

**Integration Testing** ([tests/integration/](tests/integration/)):
- Tests for each API blueprint with Flask test client
- Mocks database and external services
- Tests authentication, validation, error handling

**Service Detection** ([tests/service_detection.py](tests/service_detection.py)):
- Checks if external services (LLM, embedding, MinerU) are available
- Skips tests requiring unavailable services

### Pre-commit Hooks

Configured in [.pre-commit-config.yaml](.pre-commit-config.yaml):
- **Python**: Black (line length 120), isort (black profile), autoflake, pyupgrade
- **JavaScript**: ESLint, Prettier
- **Custom**: Check static/dist sync, detect private keys, check large files

### Gunicorn Preloading

When using `./bin/up.sh` with `ARXIV_SANITY_GUNICORN_PRELOAD=1`:
- Master process loads papers/metas cache, features, embeddings
- Workers fork with shared memory (copy-on-write)
- Reduces memory usage and startup time
- See [serve.py](serve.py) for preloading logic

### Atomic File Operations

Critical for concurrent access:
- Features file ([tools/compute.py](tools/compute.py)): Write to staging file → rename
- Summary cache ([tools/paper_summarizer.py](tools/paper_summarizer.py)): Write to temp file → rename
- Frontend build ([scripts/build.js](scripts/build.js)): Build to temp dir → replace

### Search Query Syntax

Implemented in [backend/services/search_service.py](backend/services/search_service.py):
- Field filters: `ti:transformer`, `au:goodfellow`, `cat:cs.LG`, `id:2312.12345`
- Phrases: `"large language model"` (exact match)
- Negation: `-survey` or `!survey` (exclude terms)
- Combined: `ti:"graph neural network" cat:cs.LG -survey`

### Summary Generation Pipeline

1. **Source Selection** (configurable via `ARXIV_SANITY_SUMMARY_SOURCE`):
   - `html`: Fetch from ar5iv/arxiv HTML (fast, good structure)
   - `mineru`: Parse PDF with MinerU (best for tables/formulas)

2. **Markdown Parsing**:
   - HTML: [tools/paper_summarizer.py](tools/paper_summarizer.py) uses BeautifulSoup + markdownify
   - MinerU: [tools/paper_summarizer.py](tools/paper_summarizer.py) calls MinerU API or local VLM

3. **LLM Processing**:
   - OpenAI-compatible API (OpenRouter, OpenAI, LiteLLM gateway)
   - Structured prompt for comprehensive summaries
   - Chinese text ratio validation for quality control

4. **Caching**:
   - Summaries stored in `data/summary/{pid}/{model_name}.json`
   - Metadata includes source, timestamp, model, language
   - Lock-based concurrency control to prevent duplicate work

### Scheduler Automation

[tools/daemon.py](tools/daemon.py) uses APScheduler for automated pipelines:
- **Fetch + Compute**: Weekdays 8:00, 12:00, 16:00, 20:00 (Asia/Shanghai)
- **Email Recommendations**: Weekdays 18:00
- **Backup**: Daily 20:00 (git commit of dict.db)
- Holiday-aware time window adjustments

## Common Pitfalls

1. **Forgetting to rebuild frontend**: After modifying JS files, run `npm run build:static` or the changes won't appear
2. **Missing features.p**: If search returns no results, ensure you've run `python -m tools compute`
3. **Stale locks after crashes**: Run `python -m tools cleanup_locks` to clear
4. **NumPy version mismatch**: If features.p fails to load, regenerate with `python -m tools compute`
5. **Empty homepage**: Ensure you've run `python -m tools arxiv_daemon` to fetch papers
6. **Summary generation fails**: Check LLM API configuration in `.env` (API key, base URL, model name)
7. **Semantic search has no effect**: Ensure embeddings are enabled and features regenerated with `--use_embeddings`

## File Organization

### Backend Structure
- [backend/app.py](backend/app.py): Flask app factory, blueprint registration
- [backend/blueprints/](backend/blueprints/): 8 blueprints for different API domains
- [backend/services/](backend/services/): Business logic layer
- [backend/schemas/](backend/schemas/): Pydantic request/response models
- [backend/utils/](backend/utils/): Helpers (cache, SSE, manifest)

### Data Layer
- [aslite/db.py](aslite/db.py): Custom SQLite wrapper (SqliteKV)
- [aslite/repositories.py](aslite/repositories.py): Repository pattern implementation
- [aslite/arxiv.py](aslite/arxiv.py): arXiv API client

### Tools & Automation
- [tools/arxiv_daemon.py](tools/arxiv_daemon.py): Paper fetching from arXiv
- [tools/compute.py](tools/compute.py): TF-IDF and embedding computation
- [tools/batch_paper_summarizer.py](tools/batch_paper_summarizer.py): Batch summary generation
- [tools/daemon.py](tools/daemon.py): Scheduled task runner

### Configuration
- [config/settings.py](config/settings.py): pydantic-settings definitions
- [config/cli.py](config/cli.py): Configuration CLI tool
- `.env`: Environment variables (not in repo, copy from [.env.example](.env.example))

### Frontend
- [static/*.js](static/): Source JavaScript files
- [static/css/](static/css/): Stylesheets
- [static/dist/](static/dist/): Built assets (gitignored, generated by build)
- [templates/](templates/): Jinja2 HTML templates

### Data Directory (gitignored)
- `data/papers.db`: Paper metadata
- `data/dict.db`: User data (tags, keywords, reading list, etc.)
- `data/features.p`: Computed TF-IDF and embedding features
- `data/summary/`: Cached LLM summaries
- `data/pdfs/`, `data/mineru/`, `data/html_md/`: Intermediate caches
