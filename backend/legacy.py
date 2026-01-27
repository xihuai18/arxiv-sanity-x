"""
Core business logic for arxiv-sanity Flask application.

This module contains the main business logic including:
- Data caching and access (papers, metas, features)
- Search and ranking algorithms (TF-IDF, semantic, hybrid, SVM)
- Summary generation and caching
- Tag and keyword management (delegated to tag_service, keyword_service)
- User authentication and session handling (delegated to auth_service)
- Reading list functionality (delegated to readinglist_service)

Note: This module is being gradually refactored into smaller service modules.
See backend/services/ for the new modular structure.
"""

# Multi-core optimization configuration - Ubuntu system
import json
import logging
import os
import queue
import re
import secrets
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Type

import numpy as np
import requests
from flask import (  # global session-level object
    Response,
    abort,
    current_app,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from loguru import logger
from pydantic import BaseModel, ValidationError
from werkzeug.exceptions import HTTPException

from aslite.db import FEATURES_FILE, PAPERS_DB_FILE, get_summary_status_db

# Repository layer for cleaner data access patterns
from aslite.repositories import (
    MetaRepository,
    PaperRepository,
    SummaryStatusRepository,
    summary_status_key,
)
from config import settings
from tools.paper_summarizer import (
    normalize_summary_source,
    read_summary_meta,
    split_pid_version,
    summary_cache_paths,
    summary_source_matches,
)

# Service layer imports (consolidated from scattered imports)
from .schemas.readinglist import ReadingListPidRequest
from .schemas.summary import (
    SummaryClearModelRequest,
    SummaryGetRequest,
    SummaryPidRequest,
    SummaryStatusRequest,
    SummaryTriggerRequest,
)
from .schemas.tags import PaperTitlesRequest, TagFeedbackRequest
from .services.search_service import apply_limit as _apply_limit
from .services.summary_service import SummaryCacheMiss
from .services.summary_service import clear_model_summary as _clear_model_summary_impl
from .services.summary_service import clear_paper_cache as _clear_paper_cache_impl
from .services.summary_service import (
    get_summary_cache_stats as _get_summary_cache_stats,
)
from .services.summary_service import get_summary_status
from .utils.cache import LRUCacheTTL as _LRUCacheTTL
from .utils.sse import register_user_stream as _register_user_stream
from .utils.sse import unregister_user_stream as _unregister_user_stream

# -----------------------------------------------------------------------------
# Configuration constants (loaded from settings)
# -----------------------------------------------------------------------------
DATA_DIR = str(settings.data_dir)
SUMMARY_DIR = str(settings.summary_dir)
LLM_NAME = settings.llm.name
LLM_BASE_URL = settings.llm.base_url
LLM_API_KEY = settings.llm.api_key
EMBED_PORT = settings.embedding.port
MINERU_ENABLED = settings.mineru.enabled
MINERU_PORT = settings.mineru.port
SUMMARY_MARKDOWN_SOURCE = settings.summary.markdown_source
SUMMARY_DEFAULT_SEMANTIC_WEIGHT = settings.summary.default_semantic_weight
SVM_C = settings.svm.c

# Configure loguru early to reduce import-time noise
_DEFAULT_LOG_LEVEL = settings.log_level.upper()
logger.remove()
logger.add(sys.stdout, level=_DEFAULT_LOG_LEVEL)

# Optional task queue integration (Huey)
try:
    from tasks import SUMMARY_PRIORITY_HIGH, SUMMARY_PRIORITY_LOW, enqueue_summary_task

    _TASK_QUEUE_AVAILABLE = True
except Exception as _task_queue_exc:  # pragma: no cover - optional dependency
    SUMMARY_PRIORITY_HIGH = None
    SUMMARY_PRIORITY_LOW = None
    enqueue_summary_task = None
    _TASK_QUEUE_AVAILABLE = False
    logger.debug(f"Huey task queue unavailable: {_task_queue_exc}")

# Reduce noisy access logs unless explicitly enabled.
_ACCESS_LOG = settings.web.access_log
if not _ACCESS_LOG:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Set multi-threading environment variables
try:
    requested = int(settings.reco.num_threads)
except Exception:
    requested = 0
try:
    cap = int(settings.reco.max_threads)
except Exception:
    cap = 192

num_threads = min(cpu_count(), cap) if requested <= 0 else min(requested, cap)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

# Try to use Intel extensions (if available)
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    logger.debug(f"Intel scikit-learn extension enabled with {num_threads} threads")
except ImportError:
    logger.debug(f"Using standard sklearn with {num_threads} threads")

# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = settings.search.ret_num  # number of papers to return per page
MAX_RESULTS = settings.search.max_results  # Process at most 10 pages of results, avoid processing all data

# Features caching is handled by backend.services.data_service.
# Papers/metas/pids caching is handled by backend.services.data_service.


# Whether to cache the entire (large) papers table in RAM.
#
# NOTE: For large corpora (e.g., hundreds of thousands of papers), caching the full
# `papers` table can be extremely expensive because entries are zlib-compressed
# pickles (see CompressedSqliteDict). Under gunicorn, each worker process would
# pay this cost and hold its own copy.
#
# Therefore we default this to OFF. Metas/pids are always cached and `get_papers_bulk()`
# will batch-read from SQLite efficiently when this is disabled.
#
# Override at runtime with ARXIV_SANITY_CACHE_PAPERS=1 if you have ample RAM and want
# maximum per-request throughput.
CACHE_PAPERS_IN_MEMORY = settings.web.cache_papers


# set the secret key so we can cryptographically sign cookies and maintain sessions

# Cookie & request hardening (can be overridden via env vars)


def add_security_headers(resp):
    # Basic hardening headers (CSP intentionally not set due to inline scripts/CDNs)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # COOP requires a potentially trustworthy origin (https:// or localhost).
    # When served over plain HTTP on a LAN IP, browsers will ignore COOP and log noise.
    # Keep COOP for https/localhost, skip it otherwise.
    try:
        host = (request.host or "").split(":", 1)[0].lower()
        is_localhost = host in ("localhost", "127.0.0.1", "::1")
        if request.is_secure or is_localhost:
            resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    except Exception:
        pass
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    resp.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), interest-cohort=()",
    )
    return resp


def _get_or_set_csrf_token() -> str:
    from backend.utils.validation import get_or_set_csrf_token

    return get_or_set_csrf_token()


def _is_same_origin_request() -> bool:
    """Best-effort same-origin check for legacy GET mutation endpoints."""
    from backend.utils.validation import _is_same_origin_request as is_same_origin

    return is_same_origin()


def _csrf_protect():
    """CSRF protection for state-changing endpoints."""
    from backend.utils.validation import csrf_protect

    csrf_protect()


# -----------------------------------------------------------------------------
# Input normalization helpers


def _normalize_name(value: Optional[str]) -> str:
    from backend.services.api_helpers import normalize_name

    return normalize_name(value)


# -----------------------------------------------------------------------------
# API Response Helpers (reduce code duplication in API endpoints)


def _api_error(error: str, status: int = 400, **extra):
    """Return a standardized JSON error response."""
    from backend.services.api_helpers import api_error

    return api_error(error, status, **extra)


def _api_success(**data):
    """Return a standardized JSON success response."""
    from backend.services.api_helpers import api_success

    return api_success(**data)


def _parse_api_request(
    require_login: bool = False,
    require_csrf: bool = True,
    require_pid: bool = False,
    schema: Optional[Type[BaseModel]] = None,
):
    """
    Common API request parsing and validation.

    Returns (data, error_response) tuple. If error_response is not None, return it immediately.
    """
    data = request.get_json(silent=True)
    if not data:
        data = {}

    if require_login and g.user is None:
        # Allow internal service-to-service calls for endpoints that already disable CSRF.
        # This is used by scripts like tools/send_emails.py which do not have a browser session.
        header_key = (request.headers.get("X-ARXIV-SANITY-API-KEY") or request.headers.get("X-API-KEY") or "").strip()
        auth_header = (request.headers.get("Authorization") or "").strip()
        bearer_key = ""
        if auth_header.lower().startswith("bearer "):
            bearer_key = auth_header.split(" ", 1)[-1].strip()
        provided_key = header_key or bearer_key

        configured_key = str(getattr(settings.reco, "api_key", "") or "").strip()
        machine_authed = bool(configured_key and provided_key and secrets.compare_digest(provided_key, configured_key))

        if machine_authed and not require_csrf:
            requested_user = str(data.get("user") or "").strip()
            if not requested_user:
                return None, _api_error("User is required", 400)
            g.user = requested_user
        else:
            return None, _api_error("Not logged in", 401)

    if require_csrf:
        _csrf_protect()

    if not data:
        return None, _api_error("No JSON data provided", 400)

    raw_pid = None
    if require_pid:
        pid_value = data.get("pid")
        pid = str(pid_value).strip() if pid_value is not None else ""
        if not pid:
            return None, _api_error("Paper ID is required", 400)
        raw_pid, _ = split_pid_version(pid)
        if not paper_exists(raw_pid):
            return None, _api_error("Paper not found", 404)

    if schema is not None:
        try:
            validated = schema.model_validate(data)
            data = validated.model_dump()
        except ValidationError as exc:
            # Convert Pydantic errors to JSON-serializable format
            errors = []
            for err in exc.errors():
                errors.append(
                    {
                        "loc": err.get("loc", []),
                        "msg": err.get("msg", ""),
                        "type": err.get("type", ""),
                    }
                )
            return None, _api_error("Invalid request data", 400, details=errors)

    if raw_pid:
        data["_raw_pid"] = raw_pid

    return data, None


# Keep tokenization consistent with compute.py for query-side TF-IDF.
TFIDF_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b|\b[a-zA-Z]+\-[a-zA-Z]+\b"
TFIDF_STOP_WORDS = "english"

# Treat common separators (incl. CJK punctuation) as spaces for multi-keyword queries.
_QUERY_SEP_RE = re.compile(r"[,;\uFF0C\u3001\uFF1B\uFF1A:/\\|\(\)\[\]{}]+")
_BOOLEAN_TOKENS = {"and", "or", "not"}


def _validate_tag_name(tag: str) -> Optional[str]:
    from backend.utils.validation import validate_tag_name

    return validate_tag_name(tag)


def _validate_keyword_name(keyword: str) -> Optional[str]:
    from backend.utils.validation import validate_keyword_name

    return validate_keyword_name(keyword)


# -----------------------------------------------------------------------------
# Small in-memory caches (thread-safe, per-process)
# Note: Some caches are now in backend.services modules

# Cache for queue rank computation (TTL 3 seconds to balance freshness and performance)
_QUEUE_RANK_CACHE = _LRUCacheTTL(maxsize=128, ttl_s=3.0)

PAPER_CACHE = _LRUCacheTTL(maxsize=2048, ttl_s=300.0)


# globals that manage the (lazy) loading of various state for a request
def get_tags():
    """Get user tags using Repository layer with request-level caching."""
    from backend.services.user_service import get_tags as _get_tags

    return _get_tags()


def get_neg_tags():
    """Get user negative tags using Repository layer with request-level caching."""
    from backend.services.user_service import get_neg_tags as _get_neg_tags

    return _get_neg_tags()


def get_combined_tags():
    """Get user combined tags using Repository layer with request-level caching."""
    from backend.services.user_service import get_combined_tags as _get_combined_tags

    return _get_combined_tags()


def get_keys():
    """Get user keywords with caching."""
    from backend.services.user_service import get_keys as _get_keys

    return _get_keys()


def _build_user_tag_list():
    from backend.services.user_service import build_user_tag_list

    rtags = build_user_tag_list()
    return sorted(rtags, key=lambda item: item["name"])


def _build_user_key_list():
    from backend.services.user_service import build_user_key_list

    rkeys = build_user_key_list()
    rkeys.append({"name": "Artificial general intelligence"})
    return sorted(rkeys, key=lambda item: item["name"])


def _build_user_combined_tag_list():
    from backend.services.user_service import build_user_combined_tag_list

    rctags = build_user_combined_tag_list()
    return sorted(rctags, key=lambda item: item["name"])


# -----------------------------------------------------------------------------
# Intelligent unified data caching functionality


def get_data_cached():
    """Load papers/metas/pids caches (delegates to backend.services.data_service).

    Returns:
        Tuple (papers_cache_or_None, metas_dict, pids_list)
    """
    from backend.services.data_service import get_data_cached as _get_data_cached

    data = _get_data_cached()
    return data.get("papers"), data.get("metas") or {}, data.get("pids") or []


# -----------------------------------------------------------------------------
# Cached data access functions


def get_pids():
    """Get all paper ID list"""
    from backend.services.data_service import get_pids as _get_pids

    return _get_pids()


def get_papers():
    """Get papers database"""
    # Preserve legacy semantics: return None unless ARXIV_SANITY_CACHE_PAPERS=1.
    papers, _metas, _pids = get_data_cached()
    return papers


def paper_exists(pid: str) -> bool:
    """Fast existence check using metas (always cached)."""
    if not pid:
        return False

    # Check for upload pid
    from backend.utils.upload_utils import is_upload_pid

    if is_upload_pid(pid):
        from aslite.repositories import UploadedPaperRepository

        # Uploaded papers are private. Avoid leaking existence to anonymous users
        # or other users.
        if not g.user:
            return False
        record = UploadedPaperRepository.get(pid)
        if not record or record.get("owner") != g.user:
            return False
        return record.get("parse_status") == "ok"

    mdb = get_metas()
    return pid in mdb


def get_paper(pid: str):
    """
    Get one paper record by pid.

    - If full papers cache is enabled, read from in-memory dict.
    - Otherwise, read through SqliteDict and cache a small working set in RAM.
    """
    if not pid:
        return None

    if CACHE_PAPERS_IN_MEMORY:
        pdb = get_papers()
        if pdb is None:
            return None
        return pdb.get(pid)

    cached = PAPER_CACHE.get(pid)
    if cached is not None:
        return cached

    try:
        paper = PaperRepository.get_by_id(pid)
        if paper is None:
            return None
        PAPER_CACHE.set(pid, paper)
        return paper
    except Exception as e:
        logger.warning(f"Failed to read paper {pid} from db: {e}")
        return None


def get_papers_bulk(pids):
    """
    Fetch a batch of papers efficiently using Repository layer.

    Uses PaperRepository for optimized batch queries with automatic caching.
    Falls back to in-memory cache when ARXIV_SANITY_CACHE_PAPERS=1.
    """
    out = {}
    if not pids:
        return out

    if CACHE_PAPERS_IN_MEMORY:
        pdb = get_papers()
        if pdb is None:
            return out
        for pid in pids:
            paper = pdb.get(pid)
            if paper is not None:
                out[pid] = paper
        return out

    # Use Repository layer for batch fetch with caching
    try:
        out = PaperRepository.get_by_ids_with_cache(list(pids), PAPER_CACHE)
    except Exception as e:
        logger.warning(f"Failed to bulk-read papers from repository: {e}")
    return out


def get_metas():
    """Get metadata database"""
    from backend.services.data_service import get_metas as _get_metas

    return _get_metas()


def _warmup_data_cache():
    """Warm up data cache in background."""
    from backend.services.background import _warmup_data_cache as _warmup

    _warmup()


def _ensure_background_services_started():
    """Start background threads/schedulers lazily (per worker process)."""
    from backend.services.background import ensure_background_services_started

    ensure_background_services_started()


def before_request():
    from backend.services.user_service import before_request as _before_request

    _before_request()


def close_connection(_error=None):
    from backend.services.user_service import close_connection as _close_connection

    return _close_connection(_error)


# -----------------------------------------------------------------------------
# Intelligent feature caching functionality


def get_features_cached():
    """Load features with caching (delegates to backend.services.data_service)."""
    from backend.services.data_service import (
        get_features_cached as _get_features_cached,
    )

    return _get_features_cached()


# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests


def _get_thumb_url(pid: str) -> str:
    from backend.services.render_service import get_thumb_url

    return get_thumb_url(pid)


# TL;DR cache - now imported from services
# TLDR_CACHE = _LRUCacheTTL(maxsize=2000, ttl_s=600.0)  # Moved to summary_service


def render_pid(pid, pid_to_utags=None, pid_to_ntags=None, paper=None):
    """Render a single paper for the UI - delegates to render_service."""
    from backend.services.render_service import render_pid as _render_pid

    return _render_pid(
        pid,
        pid_to_utags=pid_to_utags,
        pid_to_ntags=pid_to_ntags,
        paper=paper,
        get_paper_fn=get_paper,
        get_tags_fn=get_tags,
        get_neg_tags_fn=get_neg_tags,
    )


# _apply_limit is imported from search_service


def random_rank(limit=None):
    pids_all = get_pids()
    from backend.services.search_service import random_rank as _random_rank

    return _random_rank(pids_all, limit)


def time_rank(limit=None):
    mdb = get_metas()
    from backend.services.search_service import time_rank as _time_rank

    return _time_rank(mdb, limit)


def _filter_by_time_with_tags(pids, time_filter, user_tagged_pids=None):
    """
    Smart time filtering: keep tagged papers (even if outside time window) and papers within time window
    Filter papers by time but keep tagged papers even if outside time window
    """
    if not time_filter:
        return pids, list(range(len(pids)))

    mdb = get_metas()
    from backend.services.search_service import filter_by_time

    return filter_by_time(pids, mdb, time_filter, user_tagged_pids)


def _filter_by_time(pids, time_filter):
    """Original time filtering function, maintain backward compatibility"""
    return _filter_by_time_with_tags(pids, time_filter, None)


def svm_rank(
    tags: str = "",
    s_pids: str = "",
    C: float = None,
    logic: str = "and",
    time_filter: str = "",
    limit=None,
):
    """SVM-based paper ranking - delegates to search_service.svm_rank."""
    from backend.services.search_service import svm_rank as _svm_rank

    # Get user from Flask context for cache key
    user = getattr(g, "user", None)

    return _svm_rank(
        tags=tags,
        s_pids=s_pids,
        C=C,
        logic=logic,
        time_filter=time_filter,
        limit=limit,
        user=user,
    )


_ARXIV_ID_RE = re.compile(
    r"(?:(?:arxiv:)?)(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v(?P<v>\d+))?",
    re.IGNORECASE,
)


def _paper_text_fields(p: dict) -> dict:
    """Build normalized text fields for scoring - delegates to render_service."""
    from backend.services.render_service import build_paper_text_fields

    return build_paper_text_fields(p)


def _is_title_like_query(parsed: dict) -> bool:
    from backend.services.search_service import is_title_like_query

    return is_title_like_query(parsed)


def _title_candidate_scan(
    parsed: dict,
    max_candidates: int = 500,
    max_scan: int = 120000,
    time_budget_s: float = 0.6,
):
    from backend.services.search_service import title_candidate_scan

    return title_candidate_scan(
        parsed,
        get_pids_fn=get_pids,
        get_papers_bulk_fn=get_papers_bulk,
        paper_text_fields_fn=_paper_text_fields,
        max_candidates=max_candidates,
        max_scan=max_scan,
        time_budget_s=time_budget_s,
    )


def _compute_paper_score_parsed(parsed: dict, p: dict, pid: str) -> float:
    from backend.services.search_service import compute_paper_score_parsed

    return compute_paper_score_parsed(
        parsed,
        p,
        pid,
        paper_text_fields_fn=_paper_text_fields,
        get_metas_fn=get_metas,
    )


def _lexical_rank_over_pids(pids: list, parsed: dict, limit: Optional[int] = None):
    from backend.services.search_service import lexical_rank_over_pids

    return lexical_rank_over_pids(
        list(pids or []),
        parsed,
        get_papers_bulk_fn=get_papers_bulk,
        paper_text_fields_fn=_paper_text_fields,
        get_metas_fn=get_metas,
        apply_limit_fn=_apply_limit,
        limit=limit,
    )


def _lexical_rank_fullscan(parsed: dict, limit: Optional[int] = None):
    from backend.services.search_service import lexical_rank_fullscan

    return lexical_rank_fullscan(
        parsed,
        get_pids_fn=get_pids,
        get_papers_fn=get_papers,
        get_papers_bulk_fn=get_papers_bulk,
        paper_text_fields_fn=_paper_text_fields,
        get_metas_fn=get_metas,
        max_results=MAX_RESULTS,
        limit=limit,
    )


def _compute_paper_score(q: str, qs: list, q_norm: str, qs_norm: list, p: dict, pid: str) -> float:
    from backend.services.search_service import compute_paper_score_simple

    return compute_paper_score_simple(q, list(qs or []), q_norm, list(qs_norm or []), p, pid)


def count_match(q, pid_start, n_pids):
    from backend.services.search_service import count_match as _count_match

    return _count_match(q, int(pid_start), int(n_pids))


def legacy_search_rank(q: str = "", limit=None):
    from backend.services.search_service import (
        legacy_search_rank as _legacy_search_rank,
    )

    return _legacy_search_rank(q, limit)


# Query-side TF-IDF vectorizer - delegates to search_service
def _get_query_vectorizer(features):
    """Reconstruct a query-side TF-IDF encoder from cached features."""
    from backend.services.search_service import get_query_vectorizer

    return get_query_vectorizer(features, FEATURES_FILE)


def search_rank(q: str = "", limit=None):
    """Fast keyword search - delegates to search_service.search_rank."""
    from backend.services.search_service import search_rank as _search_rank

    return _search_rank(
        q=q,
        limit=limit,
        get_features_fn=get_features_cached,
        get_pids_fn=get_pids,
        get_papers_bulk_fn=get_papers_bulk,
        get_metas_fn=get_metas,
        paper_exists_fn=paper_exists,
        paper_text_fields_fn=_paper_text_fields,
    )


def get_semantic_model():
    """Get semantic model instance (delegates to backend.services.semantic_service)."""
    from backend.services.semantic_service import (
        get_semantic_model as _get_semantic_model,
    )

    return _get_semantic_model()


def get_paper_embeddings():
    """Get paper embedding vectors (delegates to backend.services.semantic_service)."""
    from backend.services.semantic_service import (
        get_paper_embeddings as _get_paper_embeddings,
    )

    return _get_paper_embeddings()


def _get_query_embedding(q: str, embed_dim: int):
    """Backwards-compatible wrapper for query embedding caching."""
    from backend.services.semantic_service import (
        get_query_embedding as _get_query_embedding,
    )

    return _get_query_embedding(q, embed_dim)


def semantic_search_rank(q: str = "", limit=None):
    """Execute pure semantic search (delegates to backend.services.semantic_service)."""
    from backend.services.semantic_service import (
        semantic_search_rank as _semantic_search_rank,
    )

    return _semantic_search_rank(q, limit)


def hybrid_search_rank(q: str = "", limit=None, semantic_weight=SUMMARY_DEFAULT_SEMANTIC_WEIGHT):
    """Hybrid search - delegates to search_service.hybrid_search_rank."""
    from backend.services.search_service import (
        hybrid_search_rank as _hybrid_search_rank,
    )

    return _hybrid_search_rank(
        q=q,
        limit=limit,
        semantic_weight=semantic_weight,
        search_rank_fn=search_rank,
        semantic_search_fn=semantic_search_rank,
    )


def enhanced_search_rank(
    q: str = "",
    limit=None,
    search_mode="keyword",
    semantic_weight=SUMMARY_DEFAULT_SEMANTIC_WEIGHT,
):
    """Enhanced search - delegates to search_service.enhanced_search_rank."""
    from backend.services.search_service import (
        enhanced_search_rank as _enhanced_search_rank,
    )

    return _enhanced_search_rank(
        q=q,
        limit=limit,
        search_mode=search_mode,
        semantic_weight=semantic_weight,
    )


# -----------------------------------------------------------------------------
# primary application endpoints


def default_context():
    # any global context across all pages, e.g. related to the current user
    context = {}
    context["user"] = g.user if g.user is not None else ""
    context["csrf_token"] = _get_or_set_csrf_token()
    return context


def main():
    # default settings
    default_rank = "time"
    default_tags = ""
    default_time_filter = ""
    default_skip_have = "no"
    default_logic = "and"
    if not getattr(current_app, "_logger_configured", False):
        try:
            level = settings.log_level.upper()
        except Exception:
            level = "WARNING"
        logger.remove()
        logger.add(sys.stdout, level=level)
        setattr(current_app, "_logger_configured", True)

    # override variables with any provided options via the interface
    form_errors = []
    allowed_ranks = {"search", "tags", "pid", "time", "random"}
    allowed_logic = {"and", "or"}
    allowed_search_modes = {"keyword", "semantic", "hybrid"}
    allowed_skip_have = {"yes", "no"}

    # Check if rank was explicitly provided in the request
    rank_explicitly_set = "rank" in request.args
    opt_rank = _normalize_name(request.args.get("rank", default_rank)).lower()
    opt_q = _normalize_name(request.args.get("q", ""))
    opt_tags = _normalize_name(request.args.get("tags", default_tags))
    opt_pid = _normalize_name(request.args.get("pid", ""))
    opt_time_filter = request.args.get("time_filter", default_time_filter)  # number of days to filter by
    opt_skip_have = _normalize_name(request.args.get("skip_have", default_skip_have)).lower()
    opt_logic = _normalize_name(request.args.get("logic", default_logic)).lower()
    opt_svm_c = request.args.get("svm_c", "")  # svm C parameter
    opt_page_number = request.args.get("page_number", "1")  # page number for pagination
    opt_search_mode = _normalize_name(request.args.get("search_mode", "hybrid")).lower()
    opt_semantic_weight = request.args.get("semantic_weight", str(SUMMARY_DEFAULT_SEMANTIC_WEIGHT))

    if opt_rank not in allowed_ranks:
        form_errors.append(f"Unknown rank '{opt_rank}', using '{default_rank}'.")
        opt_rank = default_rank
    if opt_logic not in allowed_logic:
        form_errors.append("Logic must be 'and' or 'or'; using default.")
        opt_logic = default_logic
    if opt_skip_have not in allowed_skip_have:
        form_errors.append("Skip seen must be 'yes' or 'no'; using default.")
        opt_skip_have = default_skip_have
    if opt_search_mode not in allowed_search_modes:
        form_errors.append("Search mode must be keyword, semantic, or hybrid; using hybrid.")
        opt_search_mode = "hybrid"

    # if a query is given and rank was not explicitly set, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    # but respects explicit rank choices (e.g., user wants to search within time-sorted results)
    if opt_q and not rank_explicitly_set:
        opt_rank = "search"

    # Parse time filter (days)
    opt_time_filter = (opt_time_filter or "").strip()
    time_filter_provided = bool(opt_time_filter)
    time_filter_invalid = False
    if opt_time_filter:
        try:
            parsed_time_filter = float(opt_time_filter)
        except Exception:
            time_filter_invalid = True
            opt_time_filter = ""
        else:
            if parsed_time_filter <= 0:
                time_filter_invalid = True
                opt_time_filter = ""

    # if using svm_rank (tags or pid) and no time filter is specified, default to 365 days
    if opt_rank in ["tags", "pid"] and not opt_time_filter:
        if time_filter_provided:
            form_errors.append("Invalid time filter; using default 365 days for tag/pid ranking.")
        opt_time_filter = "365"
    elif time_filter_invalid:
        form_errors.append("Time filter must be a positive number; ignoring it.")

    # try to parse opt_svm_c into something sensible (a float)
    opt_svm_c = (opt_svm_c or "").strip()
    if opt_svm_c:
        try:
            C = float(opt_svm_c)
        except Exception:
            C = SVM_C
            form_errors.append(f"SVM C must be a positive number; using default {C}.")
        else:
            if C <= 0:
                C = SVM_C
                form_errors.append(f"SVM C must be a positive number; using default {C}.")
    else:
        C = SVM_C

    # parse semantic weight (0..1)
    semantic_weight = SUMMARY_DEFAULT_SEMANTIC_WEIGHT
    opt_semantic_weight = (opt_semantic_weight or "").strip()
    if opt_semantic_weight:
        try:
            semantic_weight = float(opt_semantic_weight)
        except Exception:
            form_errors.append("Semantic weight must be between 0 and 1; using default.")
            semantic_weight = SUMMARY_DEFAULT_SEMANTIC_WEIGHT
        else:
            if semantic_weight < 0 or semantic_weight > 1:
                form_errors.append("Semantic weight must be between 0 and 1; using default.")
                semantic_weight = SUMMARY_DEFAULT_SEMANTIC_WEIGHT
    opt_semantic_weight = str(semantic_weight)

    # Calculate required number of results (considering pagination and possible need for more results after filtering)
    try:
        page_number = max(1, int(opt_page_number))
    except ValueError:
        page_number = 1

    # Calculate how many results to get to support current page display
    # Basic requirement: results needed for current page = RET_NUM * page_number
    base_needed = RET_NUM * page_number

    # Consider that filtering conditions will reduce available results, so need to get more
    buffer_multiplier = 1
    if opt_time_filter:
        buffer_multiplier += 2  # Time filtering may remove many results, take 2x more
    if opt_skip_have == "yes":
        buffer_multiplier += 1  # skip_have will also remove some, take 1x more

    # Final limit = basic requirement * buffer multiplier, but not exceeding maximum
    dynamic_limit = min(MAX_RESULTS, base_needed * buffer_multiplier)

    # logger.trace(
    #     f"Page {page_number}, base_needed={base_needed}, buffer_multiplier={buffer_multiplier}, dynamic_limit={dynamic_limit}"
    # )

    # rank papers: by tags, by time, by random
    words = []  # only populated in the case of svm rank
    score_details = {}  # Save detailed score information, only used in hybrid mode

    if opt_rank == "search":
        t_s = time.time()
        # Use enhanced search function
        if opt_search_mode == "hybrid":
            pids, scores, score_details = enhanced_search_rank(
                q=opt_q,
                limit=dynamic_limit,
                search_mode=opt_search_mode,
                semantic_weight=semantic_weight,
            )
        else:
            pids, scores = enhanced_search_rank(
                q=opt_q,
                limit=dynamic_limit,
                search_mode=opt_search_mode,
                semantic_weight=semantic_weight,
            )
        logger.debug(
            f"User {g.user} {opt_search_mode} search '{opt_q}' weight={semantic_weight}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "tags":
        t_s = time.time()
        pids, scores, words = svm_rank(
            tags=opt_tags,
            s_pids=opt_pid,
            C=C,
            logic=opt_logic,
            time_filter=opt_time_filter,
            limit=dynamic_limit,
        )
        logger.debug(
            f"User {g.user} tags {opt_tags} pids {opt_pid} C {C} logic {opt_logic} time_filter {opt_time_filter}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "pid":
        t_s = time.time()
        pids, scores, words = svm_rank(
            tags=opt_tags,
            s_pids=opt_pid,
            C=C,
            logic=opt_logic,
            time_filter=opt_time_filter,
            limit=dynamic_limit,
        )
        logger.debug(
            f"User {g.user} tags {opt_tags} pids {opt_pid} C {C} logic {opt_logic} time_filter {opt_time_filter}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "time":
        t_s = time.time()
        if opt_q:
            # If there's a search query, first search then sort by time
            search_result = enhanced_search_rank(
                q=opt_q,
                limit=dynamic_limit * 2,
                search_mode=opt_search_mode,
                semantic_weight=semantic_weight,
            )
            # Handle both 2 and 3 return values (hybrid mode returns 3)
            if len(search_result) == 3:
                search_pids, _, _ = search_result
            else:
                search_pids, _ = search_result
            # Re-sort by time
            mdb = get_metas()
            tnow = time.time()
            pids_with_time = [(pid, (mdb.get(pid) or {}).get("_time", 0)) for pid in search_pids if pid in mdb]
            pids_with_time.sort(key=lambda x: x[1], reverse=True)
            pids = [p[0] for p in pids_with_time][:dynamic_limit]
            scores = [(tnow - (mdb.get(pid) or {}).get("_time", tnow)) / 60 / 60 / 24 for pid in pids]
            logger.debug(f"User {g.user} time rank with search '{opt_q}', time {time.time() - t_s:.3f}s")
        else:
            pids, scores = time_rank(limit=dynamic_limit)
            logger.debug(f"User {g.user} time rank, time {time.time() - t_s:.3f}s")
    elif opt_rank == "random":
        t_s = time.time()
        pids, scores = random_rank(limit=dynamic_limit)
        logger.debug(f"User {g.user} random rank, time {time.time() - t_s:.3f}s")
    else:
        raise ValueError(f"opt_rank {opt_rank} is not a thing")

    # For tag-based recommendations, exclude both positive and negative samples for the requested tags
    if opt_rank == "tags" and g.user:
        try:
            user_tags = get_tags()
            user_neg_tags = get_neg_tags()
            if opt_tags:
                if opt_tags == "all":
                    tag_list = list(set(user_tags.keys()) | set(user_neg_tags.keys()))
                else:
                    tag_list = [t.strip() for t in opt_tags.split(",") if t.strip()]
            else:
                tag_list = []
            if tag_list:
                tagged_set = set()
                for tag in tag_list:
                    tagged_set.update(user_tags.get(tag, set()))
                    tagged_set.update(user_neg_tags.get(tag, set()))
                if tagged_set:
                    keep = [i for i, pid in enumerate(pids) if pid not in tagged_set]
                    pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]
        except Exception:
            pass

    # filter by time (now handled within svm_rank for SVM-based rankings)
    if opt_time_filter and opt_rank not in ["tags", "pid"]:
        # Collect user tagged papers for intelligent time filtering
        user_tagged_pids = set()
        tags = get_tags()
        neg_tags = get_neg_tags()
        for tag_pids in tags.values():
            user_tagged_pids.update(tag_pids)
        for tag_pids in neg_tags.values():
            user_tagged_pids.update(tag_pids)

        # Use intelligent time filtering
        pids, time_valid_indices = _filter_by_time_with_tags(pids, opt_time_filter, user_tagged_pids)
        scores = [scores[i] for i in time_valid_indices]

    # optionally hide papers we already have
    if opt_skip_have == "yes":
        tags = get_tags()
        have = set().union(*tags.values()) if tags else set()
        neg_tags = get_neg_tags()
        if neg_tags:
            have.update(*neg_tags.values())
        keep = [i for i, pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    total_count = len(pids)
    max_pages = max(1, (total_count + RET_NUM - 1) // RET_NUM) if total_count else 1
    page_adjusted = False
    if page_number > max_pages:
        page_number = max_pages
        page_adjusted = True

    # Pagination processing (page_number already calculated above)
    start_index = (page_number - 1) * RET_NUM  # desired starting index
    end_index = min(start_index + RET_NUM, len(pids))  # desired ending index
    pids = pids[start_index:end_index]
    scores = scores[start_index:end_index]

    # logger.trace(f"Final pagination: start={start_index}, end={end_index}, got {len(pids)} papers")

    # render all papers to just the information we need for the UI
    pid_to_utags = None
    pid_to_ntags = None
    if g.user:
        try:
            user_tags = get_tags()
            user_neg_tags = get_neg_tags()
            pid_set = set(pids)
            pid_to_utags = {pid: [] for pid in pid_set}
            pid_to_ntags = {pid: [] for pid in pid_set}
            for tag, tag_pids in user_tags.items():
                for pid in pid_set.intersection(tag_pids):
                    pid_to_utags[pid].append(tag)
            for tag, tag_pids in user_neg_tags.items():
                for pid in pid_set.intersection(tag_pids):
                    pid_to_ntags[pid].append(tag)
        except Exception:
            pid_to_utags = None
            pid_to_ntags = None

    pid_to_paper = get_papers_bulk(pids)
    papers = [
        render_pid(
            pid,
            pid_to_utags=pid_to_utags,
            pid_to_ntags=pid_to_ntags,
            paper=pid_to_paper.get(pid),
        )
        for pid in pids
    ]
    for i, p in enumerate(papers):
        p["weight"] = float(scores[i])

        # If hybrid search mode, add detailed score information
        if opt_rank == "search" and opt_search_mode == "hybrid" and score_details:
            pid = p["id"]
            if pid in score_details:
                details = score_details[pid]
                kw_rank = details.get("keyword_rank")
                sem_rank = details.get("semantic_rank")
                # Format: K#rank · S#rank (compact display within badge)
                kw_part = f"K#{kw_rank}" if kw_rank is not None else "K-"
                sem_part = f"S#{sem_rank}" if sem_rank is not None else "S-"
                p["score_breakdown"] = f"{kw_part} · {sem_part}"

    # build the current tags/keys/combined-tags for the user
    rtags = _build_user_tag_list()
    rkeys = _build_user_key_list()
    rctags = _build_user_combined_tag_list()

    # build the page context information and render
    context = default_context()
    context["form_errors"] = form_errors
    context["papers"] = papers
    context["words"] = words

    context["tags"] = sorted(rtags, key=lambda item: item["name"])
    context["keys"] = sorted(rkeys, key=lambda item: item["name"])
    context["combined_tags"] = sorted(rctags, key=lambda item: item["name"])

    # test keys
    # context["keys"] = [
    #     {"name": "Reinforcement Learning", "pids": []},
    #     {"name": "Zero-shot Coordination", "pids": []},
    # ]

    context["words_desc"] = (
        "Here are the top 40 most positive and bottom 20 most negative weights of the SVM. If they don't look great then try tuning the regularization strength hyperparameter of the SVM, svm_c, above. Lower C is higher regularization."
    )
    context["gvars"] = {}
    context["gvars"]["rank"] = opt_rank
    context["gvars"]["tags"] = opt_tags
    context["gvars"]["pid"] = opt_pid
    context["gvars"]["time_filter"] = opt_time_filter
    context["gvars"]["skip_have"] = opt_skip_have
    context["gvars"]["logic"] = opt_logic
    context["gvars"]["search_query"] = opt_q
    context["gvars"]["svm_c"] = str(C)
    context["gvars"]["page_number"] = str(page_number)
    context["page_adjusted"] = page_adjusted
    context["max_pages"] = max_pages
    context["gvars"]["search_mode"] = opt_search_mode
    context["gvars"]["semantic_weight"] = opt_semantic_weight
    context["show_score_breakdown"] = opt_rank == "search" and opt_search_mode == "hybrid"
    context["default_summary_model"] = LLM_NAME or ""
    logger.trace(
        f"User: {context['user']}\ntags {context['tags']}\nkeys {context['keys']}\nctags {context['combined_tags']}"
    )
    return render_template("index.html", **context)


def api_user_state():
    if g.user is None:
        return _api_error("Not logged in", 401)
    try:
        return _api_success(
            tags=_build_user_tag_list(),
            keys=_build_user_key_list(),
            combined_tags=_build_user_combined_tag_list(),
        )
    except Exception as e:
        logger.error(f"Failed to build user state: {e}")
        return _api_error("Server error", 500)


def api_user_stream():
    if g.user is None:
        return _api_error("Not logged in", 401)

    # Capture user in request context to avoid issues in call_on_close callback
    user = g.user
    q = _register_user_stream(user)

    def gen():
        yield "event: init\ndata: {}\n\n"
        while True:
            try:
                payload = q.get(timeout=25)
                data = json.dumps(payload, ensure_ascii=False)
                yield f"data: {data}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"
            except GeneratorExit:
                break
            except Exception:
                yield ": keep-alive\n\n"

    resp = Response(stream_with_context(gen()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.call_on_close(lambda: _unregister_user_stream(user, q))
    return resp


def inspect():
    # fetch the paper of interest based on the pid
    pid = request.args.get("pid", "")

    # Check for upload pid first
    from backend.utils.upload_utils import is_upload_pid

    if is_upload_pid(pid):
        return _inspect_uploaded_paper(pid)

    if not paper_exists(pid):
        return f"<h1>Error</h1><p>Paper with ID '{pid}' not found in database.</p>", 404

    # Use intelligent cache to load features
    features = get_features_cached()  # Replace: features = load_features()

    # Use original TF-IDF matrix for inspect (if exists)
    x = features.get("x_tfidf", features["x"])
    idf = features["idf"]
    ivocab = {v: k for k, v in features["vocab"].items()}
    pid_to_index = features.get("pid_to_index")
    if isinstance(pid_to_index, dict) and pid in pid_to_index:
        pix = pid_to_index[pid]
    else:
        pix = features["pids"].index(pid)
    wixs = np.flatnonzero(np.asarray(x[pix].todense()))
    words = []
    for ix in wixs:
        words.append(
            {
                "word": ivocab[ix],
                "weight": float(x[pix, ix]),
                "idf": float(idf[ix]),
            }
        )
    words.sort(key=lambda w: w["weight"], reverse=True)

    # package everything up and render
    paper = render_pid(pid)
    context = default_context()
    context["paper"] = paper
    context["words"] = words
    context["words_desc"] = (
        "The following are the tokens and their (tfidf) weight in the paper vector. This is the actual summary that feeds into the SVM to power recommendations, so hopefully it is good and representative!"
    )
    # Add paper name to page title
    context["title"] = f"Paper Inspect - {paper['title']}"
    return render_template("inspect.html", **context)


def _inspect_uploaded_paper(pid: str):
    """Inspect TF-IDF features for an uploaded paper.

    Similar to the arXiv paper inspect, but uses computed upload features.
    Requires the user to be logged in and own the paper.
    """
    import scipy.sparse as sp

    from aslite.repositories import UploadedPaperRepository
    from backend.services.upload_similarity_service import (
        compute_upload_features,
        load_upload_features,
    )

    # Uploaded papers are private - require login
    if not g.user:
        return "<h1>Error</h1><p>Paper not found.</p>", 404

    # Check if paper exists and belongs to user
    record = UploadedPaperRepository.get(pid)
    if not record:
        return f"<h1>Error</h1><p>Uploaded paper with ID '{pid}' not found.</p>", 404

    if record.get("owner") != g.user:
        return "<h1>Error</h1><p>Paper not found.</p>", 404

    # Check if paper is parsed
    if record.get("parse_status") != "ok":
        return "<h1>Error</h1><p>Paper must be parsed before inspection. Please parse the PDF first.</p>", 400

    # Check if metadata has been extracted
    meta = record.get("meta_extracted", {})
    override = record.get("meta_override", {})
    title = override.get("title") or meta.get("title") or ""
    abstract = override.get("abstract") or meta.get("abstract") or ""
    if not title and not abstract:
        return (
            "<h1>Error</h1><p>Please extract metadata first (click 'Extract Info' button) before inspecting features.</p>",
            400,
        )

    try:
        # Load or compute features
        features = load_upload_features(pid)
        if features is None:
            # Try to compute features on-demand
            features = compute_upload_features(pid)

        if features is None:
            return (
                "<h1>Error</h1><p>Failed to compute features for this paper. Please ensure metadata has been extracted.</p>",
                500,
            )

        # Get TF-IDF vector (avoid using `or` with sparse matrices)
        tfidf_vec = features.get("x_tfidf")
        if tfidf_vec is None:
            tfidf_vec = features.get("tfidf")
        if tfidf_vec is None:
            return "<h1>Error</h1><p>No TF-IDF features available for this paper.</p>", 500

        # Load global features for vocab and idf
        global_features = get_features_cached()
        if not global_features:
            return "<h1>Error</h1><p>Global features not available.</p>", 500

        vocab = global_features.get("vocab", {})
        idf = global_features.get("idf")
        ivocab = {v: k for k, v in vocab.items()}

        # Extract words and weights from TF-IDF vector
        words = []
        if sp.issparse(tfidf_vec):
            tfidf_arr = np.asarray(tfidf_vec.todense()).flatten()
        else:
            tfidf_arr = np.asarray(tfidf_vec).flatten()

        wixs = np.flatnonzero(tfidf_arr)
        for ix in wixs:
            if ix in ivocab:
                words.append(
                    {
                        "word": ivocab[ix],
                        "weight": float(tfidf_arr[ix]),
                        "idf": float(idf[ix]) if idf is not None and ix < len(idf) else 0.0,
                    }
                )
        words.sort(key=lambda w: w["weight"], reverse=True)
    except Exception as e:
        logger.error(f"Failed to compute/load features for uploaded paper {pid}: {e}")
        return f"<h1>Error</h1><p>Failed to process features: {str(e)}</p>", 500

    # Build paper info for display
    authors = override.get("authors") or meta.get("authors") or []
    display_title = title or record.get("original_filename", pid)

    # Format upload time for display
    created_time = record.get("created_time", 0)
    if created_time:
        from datetime import datetime

        dt = datetime.fromtimestamp(created_time)
        time_str = f"Uploaded: {dt.strftime('%Y-%m-%d %H:%M')}"
    else:
        time_str = ""

    paper = {
        "id": pid,
        "title": display_title,
        "authors": ", ".join(authors) if isinstance(authors, list) else str(authors),
        "summary": abstract,
        "time": time_str,
        "tags": "",
        "kind": "upload",
    }

    context = default_context()
    context["paper"] = paper
    context["words"] = words
    context["words_desc"] = (
        "The following are the tokens and their (TF-IDF) weight in the uploaded paper vector. "
        "These features are used to find similar arXiv papers."
    )
    context["title"] = f"Paper Inspect - {display_title}"
    return render_template("inspect.html", **context)


def summary():
    """
    Display AI-generated markdown format summary of the paper.

    If a versioned PID is provided (e.g., 2512.21789v1), redirect to the
    unversioned URL (/summary?pid=2512.21789) since we only cache the latest version.
    """
    # Get paper ID
    pid = request.args.get("pid", "")
    logger.debug(f"show paper summary page for paper {pid}")

    # Check for upload pid first
    from backend.utils.upload_utils import is_upload_pid

    if is_upload_pid(pid):
        from backend.services.upload_service import get_upload_summary_context

        # Uploaded papers are private. Avoid leaking existence to anonymous users.
        if not g.user:
            return "<h1>Error</h1><p>Paper not found.</p>", 404

        context = get_upload_summary_context(pid, g.user)
        if context is None:
            return "<h1>Error</h1><p>Paper not found.</p>", 404

        # Add common context
        base_context = default_context()
        base_context.update(context)
        base_context["default_summary_model"] = LLM_NAME or ""
        base_context["tags"] = _build_user_tag_list() if g.user else []
        base_context["title"] = f"Paper Summary - {context['paper']['title']}"
        return render_template("summary.html", **base_context)

    raw_pid, version = split_pid_version(pid)

    # If versioned PID provided, redirect to unversioned URL
    if version is not None:
        return redirect(url_for("web.summary", pid=raw_pid), code=302)

    if not paper_exists(raw_pid):
        return f"<h1>Error</h1><p>Paper with ID '{pid}' not found in database.</p>", 404

    # Get basic paper information with user tags (positive + negative)
    pid_to_utags = None
    pid_to_ntags = None
    if g.user:
        try:
            user_tags = get_tags()
            user_neg_tags = get_neg_tags()
            pid_to_utags = {raw_pid: []}
            pid_to_ntags = {raw_pid: []}
            for tag, tag_pids in user_tags.items():
                if raw_pid in tag_pids:
                    pid_to_utags[raw_pid].append(tag)
            for tag, tag_pids in user_neg_tags.items():
                if raw_pid in tag_pids:
                    pid_to_ntags[raw_pid].append(tag)
        except Exception:
            pid_to_utags = None
            pid_to_ntags = None

    paper = render_pid(raw_pid, pid_to_utags=pid_to_utags, pid_to_ntags=pid_to_ntags)

    # Build the current tags for the user (same shape as main page: pos+neg, includes neg-only)
    rtags = _build_user_tag_list() if g.user else []

    # Build page context, don't call get_paper_summary here
    context = default_context()
    context["paper"] = paper
    context["pid"] = raw_pid  # Always use raw_pid (unversioned)
    context["default_summary_model"] = LLM_NAME or ""
    context["tags"] = sorted(rtags, key=lambda item: item["name"]) if rtags else []
    # Add paper name to page title
    context["title"] = f"Paper Summary - {paper['title']}"
    return render_template("summary.html", **context)


def api_get_paper_summary():
    """API endpoint: Get paper summary asynchronously"""
    try:
        data, err = _parse_api_request(require_pid=True, schema=SummaryGetRequest)
        if err:
            return err

        pid = data.get("pid", "").strip()
        model = (data.get("model") or LLM_NAME or "").strip()
        if not model:
            return _api_error("Model is required", 400)

        force_regen = bool(data.get("force", False) or data.get("force_regenerate", False))
        cache_only = bool(data.get("cache_only", False))

        logger.debug(f"Getting paper summary for: {pid}")

        summary_content, summary_meta = generate_paper_summary(
            pid, model=model, force_refresh=force_regen, cache_only=cache_only
        )

        return _api_success(
            pid=pid,
            summary_content=summary_content,
            summary_meta=_public_summary_meta(summary_meta),
        )

    except SummaryCacheMiss:
        # Return 200 with cached=false instead of 404 to avoid error handling in frontend
        return _api_success(
            pid=pid,
            summary_content=None,
            summary_meta=None,
            cached=False,
        )
    except HTTPException:
        raise  # Let Flask handle HTTP exceptions (e.g., CSRF 403)
    except Exception as e:
        logger.error(f"Paper summary API error: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def api_trigger_paper_summary():
    """Trigger summary generation without returning content."""
    try:
        data, err = _parse_api_request(require_pid=True, schema=SummaryTriggerRequest)
        if err:
            return err

        raw_pid = data["_raw_pid"]
        model = (data.get("model") or LLM_NAME or "").strip()
        if not model:
            return _api_error("Model is required", 400)

        status, _last_error = get_summary_status(raw_pid, model)
        if status == "ok":
            return _api_success(pid=raw_pid, status="ok")
        if status in ("running", "queued"):
            task_id = None
            try:
                info = SummaryStatusRepository.get_status(raw_pid, model)
                if isinstance(info, dict):
                    task_user = info.get("task_user")
                    if task_user is None or (g.user and task_user == g.user):
                        task_id = info.get("task_id")
            except Exception:
                task_id = None
            # Never leak task_id; if not owned, omit it.
            return _api_success(pid=raw_pid, status=status, last_error=None, task_id=task_id)

        _update_summary_status_db(raw_pid, model, "queued", None, task_user=g.user)
        if g.user:
            _update_readinglist_summary_status(g.user, raw_pid, "queued", None)
        priority = data.get("priority")
        try:
            priority = int(priority) if priority is not None else None
        except Exception:
            priority = None
        task_id = _trigger_summary_async(g.user, raw_pid, model=model, priority=priority)
        if task_id is None and not settings.huey.allow_thread_fallback:
            # Enqueue failed and we do not allow running summaries inside web workers.
            err_msg = None
            try:
                info = SummaryStatusRepository.get_status(raw_pid, model)
                if isinstance(info, dict):
                    err_msg = info.get("last_error")
            except Exception:
                err_msg = None
            err_msg = err_msg or "Failed to enqueue summary task"
            _update_summary_status_db(raw_pid, model, "failed", err_msg, task_user=g.user)
            if g.user:
                _update_readinglist_summary_status(g.user, raw_pid, "failed", err_msg)
            return _api_error(err_msg, 503, code="summary_queue_unavailable")
        if task_id:
            _update_summary_status_db(raw_pid, model, "queued", None, task_id=task_id, task_user=g.user)
            if g.user:
                _update_readinglist_summary_status(g.user, raw_pid, "queued", None, task_id=task_id)

        # If we couldn't enqueue because an active task exists (but we didn't return its id
        # due to ownership restrictions), still return queued without task_id.
        if task_id == "":
            task_id = None

        return _api_success(pid=raw_pid, status="queued", last_error=None, task_id=task_id)

    except HTTPException:
        raise  # Let Flask handle HTTP exceptions (e.g., CSRF 403)
    except Exception as e:
        logger.error(f"Trigger summary API error: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def api_task_status(task_id: str):
    """Return status for a queued task.

    Returns sanitized fields for security:
    - status, updated_time, queue_rank, queue_total are always returned
    - pid, model, error are only returned if user owns the task or is admin
    """
    if not task_id:
        return _api_error("Task ID is required", 400)
    try:
        key = f"task::{task_id}"
        info = SummaryStatusRepository.get_task_status(task_id)
        if not info:
            return _api_error("Task not found", 404)

        # Check ownership for sensitive fields
        task_user = info.get("user")
        is_owner = g.user and task_user and g.user == task_user

        # Always return these fields
        payload = {
            "status": info.get("status"),
            "updated_time": info.get("updated_time"),
        }

        # Only return sensitive fields to owner
        if is_owner:
            payload["pid"] = info.get("pid")
            payload["model"] = info.get("model")
            payload["error"] = info.get("error")
            payload["priority"] = info.get("priority")

        try:
            priority = info.get("priority")
            status = info.get("status")
            if (
                SUMMARY_PRIORITY_HIGH is not None
                and status == "queued"
                and priority is not None
                and int(priority) >= int(SUMMARY_PRIORITY_HIGH)
            ):
                # Try cache first
                cache_key = "queue_rank_list"
                cached_queued = _QUEUE_RANK_CACHE.get(cache_key)

                if cached_queued is None:
                    # Compute and cache
                    queued = []
                    for tkey, tinfo in SummaryStatusRepository.get_items_with_prefix("task::"):
                        if not isinstance(tinfo, dict):
                            continue
                        if tinfo.get("status") != "queued":
                            continue
                        tprio = tinfo.get("priority")
                        if tprio is None or int(tprio) < int(SUMMARY_PRIORITY_HIGH):
                            continue
                        ts = tinfo.get("updated_time") or 0
                        queued.append((ts, tkey))
                    queued.sort(key=lambda item: (item[0], item[1]))
                    _QUEUE_RANK_CACHE.set(cache_key, queued)
                else:
                    queued = cached_queued

                for idx, (_ts, tkey) in enumerate(queued):
                    if tkey == key:
                        payload["queue_rank"] = idx + 1
                        payload["queue_total"] = len(queued)
                        break
        except Exception as rank_exc:
            logger.warning(f"Failed to compute queue rank for {task_id}: {rank_exc}")
        return _api_success(task_id=task_id, **payload)
    except Exception as e:
        logger.error(f"Task status API error: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def api_queue_stats():
    """Return global queue statistics (queued count, running count)."""
    try:
        queued_count = 0
        running_count = 0
        task_entries = []

        for key, info in SummaryStatusRepository.get_items_with_prefix("task::"):
            if not isinstance(info, dict):
                continue
            task_entries.append((key, info))

        if task_entries:
            for _key, info in task_entries:
                status = info.get("status")
                if status == "queued":
                    queued_count += 1
                elif status == "running":
                    running_count += 1

        if not task_entries or (queued_count + running_count) == 0:
            summary_queued = 0
            summary_running = 0
            for key, info in SummaryStatusRepository.get_all_items():
                if not isinstance(info, dict):
                    continue
                if key.startswith("task::"):
                    continue
                status = info.get("status")
                if status == "queued":
                    summary_queued += 1
                elif status == "running":
                    summary_running += 1
            if (queued_count + running_count) == 0:
                queued_count = summary_queued
                running_count = summary_running

        return _api_success(queued=queued_count, running=running_count)
    except Exception as e:
        logger.error(f"Queue stats API error: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def api_summary_status():
    """Return summary generation status for one or more papers (optimized with batch queries)."""
    try:
        data, err = _parse_api_request(require_csrf=True, schema=SummaryStatusRequest)
        if err:
            return err

        pids = data.get("pids")
        if not pids:
            pid = (data.get("pid") or "").strip()
            if pid:
                pids = [pid]

        if isinstance(pids, str):
            pids = [pids]
        if not isinstance(pids, list) or not pids:
            return _api_error("Paper IDs are required", 400)

        model = (data.get("model") or LLM_NAME or "").strip()
        if not model:
            return _api_error("Model is required", 400)

        # Batch process: normalize pids and check existence
        raw_pids = []
        for pid in pids:
            raw_pid, _ = split_pid_version(str(pid))
            if raw_pid:
                raw_pids.append(raw_pid)

        # Batch check paper existence (if MetaRepository supports it)
        existing_pids = set()
        try:
            metas = MetaRepository.get_by_ids(raw_pids)
            existing_pids = set(metas.keys())
        except Exception:
            # Fallback to individual checks
            for raw_pid in raw_pids:
                if paper_exists(raw_pid):
                    existing_pids.add(raw_pid)

        # Batch get summary status (preserve cache/lock priority)
        statuses = {}
        pending_db = []
        summary_source = normalize_summary_source(SUMMARY_MARKDOWN_SOURCE)

        for raw_pid in raw_pids:
            if raw_pid not in existing_pids:
                statuses[raw_pid] = {
                    "status": "not_found",
                    "last_error": "Paper not found",
                }
                continue

            cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(
                raw_pid, model
            )

            if cache_file.exists() or legacy_cache.exists():
                meta = read_summary_meta(meta_file) if meta_file.exists() else read_summary_meta(legacy_meta)
                if summary_source_matches(meta, summary_source):
                    statuses[raw_pid] = {"status": "ok", "last_error": None}
                    continue

            if lock_file.exists() or legacy_lock.exists():
                statuses[raw_pid] = {"status": "running", "last_error": None}
                continue

            pending_db.append(raw_pid)

        def _allow_task_id(info: dict) -> bool:
            task_user = info.get("task_user")
            return task_user is None or (g.user and task_user == g.user)

        if pending_db:
            status_keys = [summary_status_key(raw_pid, model) for raw_pid in pending_db]
            try:
                with get_summary_status_db() as sdb:
                    status_data = sdb.get_many(status_keys)

                for raw_pid in pending_db:
                    key = summary_status_key(raw_pid, model)
                    info = status_data.get(key)
                    if isinstance(info, dict):
                        payload = {
                            "status": info.get("status") or "",
                            "last_error": info.get("last_error"),
                        }
                        if _allow_task_id(info):
                            task_id = info.get("task_id")
                            if task_id:
                                payload["task_id"] = str(task_id)
                        statuses[raw_pid] = payload
                    else:
                        statuses[raw_pid] = {"status": "", "last_error": None}
            except Exception as e:
                logger.warning(f"Batch status query failed, falling back: {e}")
                for raw_pid in pending_db:
                    status, last_error = get_summary_status(raw_pid, model)
                    statuses[raw_pid] = {"status": status, "last_error": last_error}

        return _api_success(statuses=statuses, model=model)

    except HTTPException:
        raise  # Let Flask handle HTTP exceptions (e.g., CSRF 403)
    except Exception as e:
        logger.error(f"Summary status API error: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def api_clear_model_summary():
    """API endpoint: Clear summary for a specific model only"""
    try:
        data, err = _parse_api_request(require_pid=True, schema=SummaryClearModelRequest)
        if err:
            return err

        pid = data.get("pid", "").strip()
        model = (data.get("model") or "").strip()
        if not model:
            return _api_error("Model name is required", 400)

        _clear_model_summary(pid, model)
        return _api_success(pid=pid, model=model)

    except HTTPException:
        raise  # Let Flask handle HTTP exceptions (e.g., CSRF 403)
    except Exception as e:
        logger.error(f"Failed to clear model summary: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def api_clear_paper_cache():
    """API endpoint: Clear all caches for a paper (all models, HTML, MinerU, etc.)"""
    try:
        data, err = _parse_api_request(require_pid=True, schema=SummaryPidRequest)
        if err:
            return err

        pid = data.get("pid", "").strip()
        _clear_paper_cache(pid)
        return _api_success(pid=pid)

    except HTTPException:
        raise  # Let Flask handle HTTP exceptions (e.g., CSRF 403)
    except Exception as e:
        logger.error(f"Failed to clear paper cache: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


# -----------------------------------------------------------------------------
# Image serving helpers (reduce duplication between paper_image and mineru_image)


def _serve_paper_image(pid: str, filename: str, base_dir: Path, search_subdirs: list = None):
    """Common logic for serving paper images from cache directories."""
    from backend.services.render_service import serve_paper_image

    return serve_paper_image(pid, filename, base_dir, search_subdirs)


def api_paper_image(pid: str, filename: str):
    """Serve paper images from HTML markdown cache."""
    try:
        return _serve_paper_image(pid, filename, Path(DATA_DIR) / "html_md")
    except Exception as e:
        if hasattr(e, "code"):
            raise
        logger.error(f"Failed to serve paper image: {e}")
        abort(500, f"Server error: {str(e)}")


def api_mineru_image(pid: str, filename: str):
    """Serve paper images from MinerU parsed cache."""
    try:
        # Check upload pid permission
        from backend.utils.upload_utils import is_upload_pid

        if is_upload_pid(pid):
            from aslite.repositories import UploadedPaperRepository

            record = UploadedPaperRepository.get(pid)
            if not record or record.get("owner") != g.user:
                abort(404)

        return _serve_paper_image(pid, filename, Path(DATA_DIR) / "mineru", ["auto", "vlm", "api"])
    except Exception as e:
        if hasattr(e, "code"):
            raise
        logger.error(f"Failed to serve MinerU image: {e}")
        abort(500, f"Server error: {str(e)}")


def api_llm_models():
    try:
        base_url = (LLM_BASE_URL or "").rstrip("/")
        if not base_url:
            models = [{"id": LLM_NAME}] if LLM_NAME else []
            # Keep response usable for clients: if we can provide a fallback model,
            # return success=True with a warning instead of HTTP-200+success=False.
            if models:
                return _api_success(models=models, warning="LLM base URL is not configured")
            return _api_error("LLM base URL is not configured", 500, models=[])

        headers = {}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        candidate_urls = []
        if base_url.endswith("/v1"):
            candidate_urls.append(f"{base_url}/models")
        else:
            candidate_urls.append(f"{base_url}/models")
            candidate_urls.append(f"{base_url}/v1/models")

        payload = None
        last_error = None
        for url in candidate_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 404 and url != candidate_urls[-1]:
                    last_error = f"HTTP 404 for {url}"
                    continue
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as e:
                last_error = e
                continue

        if payload is None:
            raise RuntimeError(f"Failed to fetch models from {candidate_urls}. Last error: {last_error}")
        models = []
        for item in payload.get("data", []):
            mid = item.get("id")
            if mid:
                models.append(item)
        if not models and LLM_NAME:
            models = [{"id": LLM_NAME}]
        return _api_success(models=models)
    except Exception as e:
        logger.error(f"Failed to fetch LLM models: {e}")
        models = [{"id": LLM_NAME}] if LLM_NAME else []
        if models:
            return _api_success(models=models, warning="Failed to fetch model list from LLM endpoint")
        return _api_error("Failed to load model list", 502, models=[])


def api_check_paper_summaries():
    """
    API endpoint: Check which models have existing summaries for a paper
    Returns a list of model names that have cached summaries
    """
    try:
        pid = request.args.get("pid", "").strip()
        if not pid:
            return _api_error("Paper ID is required", 400)

        raw_pid, _ = split_pid_version(pid)
        if not paper_exists(raw_pid):
            return _api_error("Paper not found", 404)

        # Check summary directory for this paper
        summary_dir = Path(SUMMARY_DIR) / raw_pid
        available_models = []

        if summary_dir.exists() and summary_dir.is_dir():
            # List all .md files in the directory (excluding legacy .md file)
            for md_file in summary_dir.glob("*.md"):
                # Extract model name from filename (e.g., "gpt-4.md" -> "gpt-4")
                model_name = md_file.stem
                # Check if corresponding meta file exists and is valid
                meta_file = summary_dir / f"{model_name}.meta.json"
                if meta_file.exists() and md_file.stat().st_size > 0:
                    available_models.append(model_name)

        return _api_success(available_models=available_models)

    except Exception as e:
        logger.error(f"Check paper summaries API error: {e}")
        return _api_error(f"Server error: {str(e)}", 500)


def _write_summary_meta(meta_path: Path, data: dict) -> None:
    from backend.services.summary_service import write_summary_meta

    write_summary_meta(meta_path, data)


def _public_summary_meta(meta: dict) -> dict:
    """Filter summary metadata for client responses."""
    from backend.services.summary_service import public_summary_meta

    return public_summary_meta(meta)


def _sanitize_summary_meta(meta: dict) -> dict:
    from backend.services.summary_service import sanitize_summary_meta

    return sanitize_summary_meta(meta)


def generate_paper_summary(
    pid: str,
    model: Optional[str] = None,
    force_refresh: bool = False,
    cache_only: bool = False,
):
    """Generate paper summary - delegates to summary_service.generate_paper_summary."""
    from backend.services.summary_service import (
        generate_paper_summary as _generate_paper_summary,
    )

    return _generate_paper_summary(
        pid=pid,
        model=model,
        force_refresh=force_refresh,
        cache_only=cache_only,
        metas_getter=get_metas,
        paper_exists_fn=paper_exists,
    )


def _clear_model_summary(pid: str, model: str):
    """Clear summary cache for a specific model only."""
    _clear_model_summary_impl(pid, model, metas_getter=get_metas)


def _clear_paper_cache(pid: str):
    """Clear all caches for a paper."""
    _clear_paper_cache_impl(pid, metas_getter=get_metas)


def profile():
    """Render profile page - delegates to auth_service for email."""
    from backend.services.auth_service import get_user_email

    context = default_context()
    email = get_user_email(g.user)
    context["email"] = email
    return render_template("profile.html", **context)


def stats():
    context = default_context()
    mdb = get_metas()
    times = [v["_time"] for v in mdb.values()]

    def tstr(t):
        return time.strftime("%b %d %Y", time.localtime(t))

    context["num_papers"] = len(mdb)
    if len(mdb) > 0:
        context["earliest_paper"] = tstr(min(times))
        context["latest_paper"] = tstr(max(times))
    else:
        context["earliest_paper"] = "N/A"
        context["latest_paper"] = "N/A"

    # count number of papers from various time deltas to now
    tnow = time.time()
    for thr in [1, 6, 12, 24, 48, 72, 96]:
        context["thr_%d" % thr] = len([t for t in times if t > tnow - thr * 60 * 60])

    context["thr_week"] = len([t for t in times if t > tnow - 7 * 24 * 60 * 60])
    context["thr_month"] = len([t for t in times if t > tnow - 30.25 * 24 * 60 * 60])
    context["thr_quarter"] = len([t for t in times if t > tnow - 91.5 * 24 * 60 * 60])
    context["thr_semiannual"] = len([t for t in times if t > tnow - 182.5 * 24 * 60 * 60])
    context["thr_year"] = len([t for t in times if t > tnow - 365 * 24 * 60 * 60])

    summary_status_map = {}
    summary_model_counts = []
    summary_paper_count = 0
    summary_total_ok = 0
    summary_cache_total = 0
    summary_cache_paper_count = 0
    summary_cache_model_counts = []
    queue_status_map = {}
    queue_tasks = []
    queue_priority_map = {}
    queue_active_map = {}
    queue_history_map = {}
    queue_active_total = 0
    queue_history_total = 0

    def _format_age(ts: Optional[float]) -> str:
        if not ts:
            return "n/a"
        delta = max(0.0, time.time() - float(ts))
        if delta < 60:
            return f"{int(delta)}s"
        if delta < 3600:
            return f"{int(delta // 60)}m"
        if delta < 86400:
            return f"{int(delta // 3600)}h"
        return f"{int(delta // 86400)}d"

    try:
        summary_status_counts = defaultdict(int)
        model_counts = defaultdict(int)
        paper_ids = set()

        for key, info in SummaryStatusRepository.get_all_items():
            if not isinstance(info, dict):
                continue
            if key.startswith("task::"):
                continue
            if "::" not in key:
                continue
            pid, model = key.split("::", 1)
            status = info.get("status") or "unknown"
            summary_status_counts[str(status)] += 1
            if status == "ok":
                summary_total_ok += 1
                if pid:
                    paper_ids.add(pid)
                if model:
                    model_counts[model] += 1

        summary_paper_count = len(paper_ids)
        summary_status_map = dict(summary_status_counts)
        summary_model_counts = [
            {"model": model, "count": count}
            for model, count in sorted(model_counts.items(), key=lambda x: (-x[1], x[0]))
        ]

        task_counts = defaultdict(int)
        priority_counts = defaultdict(lambda: defaultdict(int))
        task_rows = []
        for key, info in SummaryStatusRepository.get_items_with_prefix("task::"):
            if not isinstance(info, dict):
                continue
            task_id = key.split("task::", 1)[-1]
            status = info.get("status") or "unknown"
            task_counts[str(status)] += 1
            raw_priority = info.get("priority")
            bucket = "unknown"
            if raw_priority is not None:
                try:
                    prio_val = int(raw_priority)
                    if SUMMARY_PRIORITY_HIGH is not None and prio_val >= int(SUMMARY_PRIORITY_HIGH):
                        bucket = "high"
                    elif SUMMARY_PRIORITY_LOW is not None and prio_val <= int(SUMMARY_PRIORITY_LOW):
                        bucket = "low"
                    else:
                        bucket = "normal"
                except Exception:
                    bucket = "unknown"
            priority_counts[bucket][str(status)] += 1
            updated_time = info.get("updated_time")
            task_rows.append(
                {
                    "task_id": task_id,
                    "pid": info.get("pid") or "",
                    "model": info.get("model") or "",
                    "status": status,
                    "priority": info.get("priority"),
                    "updated_time": updated_time,
                    "updated_ago": _format_age(updated_time),
                    "error": info.get("error") or "",
                }
            )

        task_rows.sort(key=lambda row: row.get("updated_time") or 0, reverse=True)
        queue_tasks = task_rows[:50]
        queue_status_map = dict(task_counts)
        queue_priority_map = {bucket: dict(counts) for bucket, counts in priority_counts.items()}
    except Exception as e:
        logger.warning(f"Failed to compute summary/queue stats: {e}")

    try:
        cache_stats = _get_summary_cache_stats()
        cache_data = cache_stats.get("data") or {}
        summary_cache_total = cache_data.get("summary_cache_total", 0)
        summary_cache_paper_count = cache_data.get("summary_cache_paper_count", 0)
        summary_cache_model_counts = cache_data.get("summary_cache_model_counts", [])
        context["summary_cache_scan_active"] = cache_stats.get("in_progress", False)
        context["summary_cache_updated_time"] = cache_stats.get("updated_time", 0.0)
        context["summary_cache_ttl"] = cache_stats.get("ttl", 0)
        context["summary_cache_scan_duration"] = cache_stats.get("duration", 0.0)
        updated_time = context["summary_cache_updated_time"]
        context["summary_cache_updated_ago"] = _format_age(updated_time) if updated_time else "n/a"
    except Exception as e:
        logger.warning(f"Failed to compute summary cache stats: {e}")

    queue_active_map = {
        "queued": queue_status_map.get("queued", 0),
        "running": queue_status_map.get("running", 0),
    }
    queue_active_total = queue_active_map["queued"] + queue_active_map["running"]
    queue_history_map = {k: v for k, v in queue_status_map.items() if k not in {"queued", "running"}}
    queue_history_total = sum(queue_history_map.values()) if queue_history_map else 0

    context["summary_status_map"] = summary_status_map
    context["summary_model_counts"] = summary_model_counts
    context["summary_paper_count"] = summary_paper_count
    context["summary_total_ok"] = summary_total_ok
    context["summary_cache_total"] = summary_cache_total
    context["summary_cache_paper_count"] = summary_cache_paper_count
    context["summary_cache_model_counts"] = summary_cache_model_counts
    context["queue_status_map"] = queue_status_map
    context["queue_tasks"] = queue_tasks
    context["queue_total"] = sum(queue_status_map.values()) if queue_status_map else 0
    context["queue_priority_map"] = queue_priority_map
    context["queue_active_map"] = queue_active_map
    context["queue_history_map"] = queue_history_map
    context["queue_active_total"] = queue_active_total
    context["queue_history_total"] = queue_history_total
    context["queue_auto_refresh"] = True
    context["queue_refresh_seconds"] = 15

    return render_template("stats.html", **context)


def about():
    context = default_context()
    try:
        from tools.arxiv_daemon import ALL_TAGS

        context["indexed_categories"] = ALL_TAGS
    except Exception:
        context["indexed_categories"] = []
    return render_template("about.html", **context)


# -----------------------------------------------------------------------------
# Helper for API endpoints
# -----------------------------------------------------------------------------


def _resolve_time_delta(time_delta, time_filter):
    """Resolve time_delta from time_filter string or return provided time_delta.

    Args:
        time_delta: Explicit time delta in days (float or None)
        time_filter: String filter like 'day', 'week', 'month', 'year', 'all'

    Returns:
        float: Resolved time delta in days (defaults to 3.0 if neither provided)
    """
    if time_delta is not None:
        return time_delta

    if time_filter:
        mapping = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0, "all": 0.0}
        if time_filter in mapping:
            return mapping[time_filter]
        # If time_filter is not in mapping, try to use it directly
        # (for backward compatibility with numeric string values)
        return time_filter

    # Default to 3 days
    return 3.0


@contextmanager
def _temporary_user_context(user):
    """Context manager to temporarily set g.user and g._tags for API calls."""
    from backend.services.user_service import temporary_user_context

    with temporary_user_context(user) as user_tags:
        yield user_tags


# -----------------------------------------------------------------------------
# API endpoints for email recommendation system
# -----------------------------------------------------------------------------


def api_keyword_search():
    """API interface: single keyword search"""
    try:
        data = request.get_json()
        # logger.info(f"API keyword search data: {data}")
        if not data:
            return _api_error("No JSON data provided", 400)

        keyword = (data.get("keyword") or data.get("q") or data.get("search_query") or "").strip()
        time_delta = data.get("time_delta", None)  # days
        time_filter = str(data.get("time_filter") or "").strip().lower()
        limit = data.get("limit", 50)
        skip_num = data.get("skip_num", 0)

        if not keyword:
            return _api_error("Keyword is required", 400)

        try:
            limit = min(int(limit), MAX_RESULTS)
        except Exception:
            # Backward compatible: fall back to default when limit is invalid
            limit = 50
        if limit <= 0:
            return _api_success(pids=[], scores=[], total_count=0)
        try:
            skip_num = max(0, int(skip_num))
        except Exception:
            skip_num = 0

        time_delta = _resolve_time_delta(time_delta, time_filter)
        try:
            time_delta = float(time_delta)
        except Exception:
            return _api_error("time_delta must be a number", 400)

        # Use enhanced search
        search_limit = min(limit * 5, MAX_RESULTS)  # Get more because time filtering is needed
        pids, scores = enhanced_search_rank(q=keyword, limit=search_limit, search_mode="keyword")

        # Apply time filtering
        if time_delta:
            mdb = get_metas()
            tnow = time.time()
            deltat = time_delta * 60 * 60 * 24
            keep = [
                i
                for i, pid in enumerate(pids)
                if (meta := mdb.get(pid)) is not None and (tnow - meta["_time"]) < deltat
            ]
            pids = [pids[i] for i in keep]
            scores = [scores[i] for i in keep]

        if skip_num:
            pids = pids[skip_num:]
            scores = scores[skip_num:]

        # Limit final result count
        if len(pids) > limit:
            pids = pids[:limit]
            scores = scores[:limit]
        # logger.trace(f"API keyword search results: {len(pids)} papers found")
        return _api_success(pids=pids, scores=scores, total_count=len(pids))

    except Exception as e:
        logger.error(f"API keyword search error: {e}")
        return _api_error(str(e), 500)


def api_tag_search():
    """API interface: single tag recommendation"""
    try:
        data, err = _parse_api_request(require_login=True, require_csrf=False)
        if err:
            return err

        logger.trace(f"API tag search data: {data}")

        tag_name = (data.get("tag_name") or data.get("tag") or "").strip()  # tag name
        body_user = data.get("user", "")  # backward-compatible field
        time_delta = data.get("time_delta", None)  # days
        time_filter = str(data.get("time_filter") or "").strip().lower()
        limit = data.get("limit", 50)
        skip_num = data.get("skip_num", 0)
        C = data.get("C", 0.1)

        if not tag_name:
            return _api_error("tag_name is required", 400)
        if body_user and body_user != g.user:
            logger.warning(f"API tag search user mismatch: body={body_user}, session={g.user}")
            return _api_error("User mismatch", 403)
        try:
            limit = min(int(limit), MAX_RESULTS)
        except Exception:
            return _api_error("limit must be an integer", 400)
        if limit <= 0:
            return _api_success(pids=[], scores=[], total_count=0)
        try:
            skip_num = max(0, int(skip_num))
        except Exception:
            skip_num = 0

        time_delta = _resolve_time_delta(time_delta, time_filter)
        try:
            time_delta = float(time_delta)
        except Exception:
            return _api_error("time_delta must be a number", 400)

        with _temporary_user_context(g.user) as user_tags:
            # Check if user has this tag
            if tag_name not in user_tags or len(user_tags[tag_name]) == 0:
                logger.warning(f"User {g.user} has no papers tagged with '{tag_name}'")
                return _api_success(pids=[], scores=[], total_count=0)

            logger.trace(f"User {g.user} has {len(user_tags[tag_name])} papers tagged with '{tag_name}'")

            # Use tag name for recommendation
            try:
                svm_limit = min(int(limit) * 5, MAX_RESULTS)
            except Exception:
                svm_limit = MAX_RESULTS

            rec_pids, rec_scores, _ = svm_rank(
                tags=tag_name,
                s_pids="",
                C=C,
                logic="and",
                time_filter=str(time_delta),
                limit=svm_limit,
            )

            logger.trace(f"svm_rank returned {len(rec_pids)} results before filtering")

            tagged_set = set(user_tags.get(tag_name, set()))
            neg_tagged_set = set(get_neg_tags().get(tag_name, set()))
            exclude_set = tagged_set.union(neg_tagged_set)
            if exclude_set:
                keep = [i for i, pid in enumerate(rec_pids) if pid not in exclude_set]
                rec_pids = [rec_pids[i] for i in keep]
                rec_scores = [rec_scores[i] for i in keep]

        if skip_num:
            rec_pids = rec_pids[skip_num:]
            rec_scores = rec_scores[skip_num:]

        # Limit result count
        if len(rec_pids) > limit:
            rec_pids = rec_pids[:limit]
            rec_scores = rec_scores[:limit]
        logger.trace(f"API tag search results: {len(rec_pids)} papers found")
        return _api_success(pids=rec_pids, scores=rec_scores, total_count=len(rec_pids))

    except Exception as e:
        logger.error(f"API tag search error: {e}")
        return _api_error(str(e), 500)


def api_tags_search():
    """API interface: joint tag recommendation"""
    try:
        data, err = _parse_api_request(require_login=True, require_csrf=False)
        if err:
            return err

        logger.trace(f"API combined tags search data: {data}")

        tags_list = data.get("tags", [])  # List[tag_name]
        body_user = data.get("user", "")  # backward-compatible field
        logic = data.get("logic", "and")  # and|or
        time_delta = data.get("time_delta", None)  # days
        time_filter = str(data.get("time_filter") or "").strip().lower()
        limit = data.get("limit", 50)
        skip_num = data.get("skip_num", 0)
        C = data.get("C", 0.1)

        if isinstance(tags_list, str):
            tags_list = [t.strip() for t in tags_list.split(",") if t.strip()]

        if not tags_list:
            return _api_error("Tags list is required", 400)
        if body_user and body_user != g.user:
            logger.warning(f"API tags search user mismatch: body={body_user}, session={g.user}")
            return _api_error("User mismatch", 403)
        try:
            limit = min(int(limit), MAX_RESULTS)
        except Exception:
            return _api_error("limit must be an integer", 400)
        if limit <= 0:
            return _api_success(pids=[], scores=[], total_count=0)
        try:
            skip_num = max(0, int(skip_num))
        except Exception:
            skip_num = 0

        time_delta = _resolve_time_delta(time_delta, time_filter)
        try:
            time_delta = float(time_delta)
        except Exception:
            return _api_error("time_delta must be a number", 400)

        with _temporary_user_context(g.user) as user_tags:
            # Check if user has any of the tags
            valid_tags = [tag for tag in tags_list if tag in user_tags and len(user_tags[tag]) > 0]
            if not valid_tags:
                logger.warning(f"User {g.user} has no papers tagged with any of {tags_list}")
                return _api_success(pids=[], scores=[], total_count=0)

            # Convert tag list to comma-separated string
            tags_str = ",".join(valid_tags)

            try:
                svm_limit = min(int(limit) * 5, MAX_RESULTS)
            except Exception:
                svm_limit = MAX_RESULTS

            rec_pids, rec_scores, _ = svm_rank(
                tags=tags_str,
                s_pids="",
                C=C,
                logic=logic,
                time_filter=str(time_delta),
                limit=svm_limit,
            )

            # Exclude papers already tagged by the user under any of the involved tags (positive or negative)
            all_tagged = set()
            neg_tags = get_neg_tags()
            for tag in valid_tags:
                all_tagged.update(user_tags.get(tag, set()))
                all_tagged.update(neg_tags.get(tag, set()))

            keep = [i for i, pid in enumerate(rec_pids) if pid not in all_tagged]
            rec_pids = [rec_pids[i] for i in keep]
            rec_scores = [rec_scores[i] for i in keep]

        if skip_num:
            rec_pids = rec_pids[skip_num:]
            rec_scores = rec_scores[skip_num:]

        # Limit result count
        if len(rec_pids) > limit:
            rec_pids = rec_pids[:limit]
            rec_scores = rec_scores[:limit]

        logger.trace(f"API combined tags search results: {len(rec_pids)} papers found")

        return _api_success(pids=rec_pids, scores=rec_scores, total_count=len(rec_pids))

    except Exception as e:
        logger.error(f"API tags search error: {e}")
        return _api_error(str(e), 500)


def cache_status():
    """Debug endpoint to display cache status"""
    if not settings.web.enable_cache_status:
        abort(404)
    if not g.user:
        return "Access denied"

    from backend.services import data_service

    features_cache = data_service.peek_features_cache()
    features_cache_time = float(data_service.get_features_cache_time() or 0.0)
    features_file_mtime = float(data_service.get_features_file_mtime() or 0.0)

    try:
        data = data_service.get_data_cached()
    except Exception as e:
        data = {"pids": [], "papers": None, "metas": {}}
        logger.warning(f"Failed to read data cache status: {e}")

    papers_cache = data.get("papers")
    metas_cache = data.get("metas") or {}
    pids_cache = data.get("pids") or []

    papers_db_cache_time = float(data_service.get_papers_db_cache_time() or 0.0)
    papers_db_file_mtime = float(data_service.get_papers_db_file_mtime() or 0.0)

    # Check backend service status
    def check_http_service(port, path, service_name):
        """Check if a local HTTP service is available"""
        try:
            logger.trace(f"Checking {service_name} at port {port}...")
            sess = requests.Session()
            sess.trust_env = False
            response = sess.get(f"http://localhost:{port}{path}", timeout=2)
            is_available = response.status_code == 200
            if is_available:
                logger.debug(f"{service_name} is available (status: {response.status_code})")
            else:
                logger.warning(f"{service_name} returned non-200 status: {response.status_code}")
            return {
                "available": is_available,
                "status_code": response.status_code,
                "service_name": service_name,
            }
        except Exception as e:
            logger.warning(f"{service_name} is not available: {str(e)}")
            return {
                "available": False,
                "error": str(e),
                "service_name": service_name,
            }

    backend_services = {
        "embedding_service": check_http_service(
            EMBED_PORT, "/api/version", f"Ollama Embedding Service (port {EMBED_PORT})"
        ),
    }

    # Only check MinerU service if it's enabled
    if MINERU_ENABLED:
        backend_services["mineru_service"] = check_http_service(
            MINERU_PORT, "/health", f"MinerU VLM Service (port {MINERU_PORT})"
        )
    else:
        backend_services["mineru_service"] = {
            "status": "disabled",
            "message": "MinerU service is disabled (ARXIV_SANITY_MINERU_ENABLED=false)",
        }

    status = {
        "current_time": time.time(),
        "features": {
            "cached": features_cache is not None,
            "cache_time": features_cache_time,
            "file_mtime": features_file_mtime,
        },
        "papers_and_metas": {
            "papers_cached": papers_cache is not None,
            "metas_cached": bool(metas_cache),
            "cache_time": papers_db_cache_time,
            "db_file_mtime": papers_db_file_mtime,
        },
        "backend_services": backend_services,
    }

    # Features details
    if features_cache:
        status["features"].update(
            {
                "cache_age_seconds": time.time() - features_cache_time if features_cache_time else None,
                "feature_shape": str(features_cache.get("x").shape) if features_cache.get("x") is not None else None,
                "num_papers": len(features_cache.get("pids") or []),
                "vocab_size": len(features_cache.get("vocab") or {}),
            }
        )

    if os.path.exists(FEATURES_FILE):
        current_mtime = os.path.getmtime(FEATURES_FILE)
        status["features"].update(
            {
                "file_exists": True,
                "current_file_mtime": current_mtime,
                "file_is_newer": current_mtime > features_file_mtime,
            }
        )
    else:
        status["features"]["file_exists"] = False

    # Papers and metas details (unified since they share the same db file)
    if papers_cache and metas_cache:
        status["papers_and_metas"].update(
            {
                "cache_age_seconds": time.time() - papers_db_cache_time if papers_db_cache_time else None,
                "num_papers": len(papers_cache) if isinstance(papers_cache, dict) else 0,
                "num_metas": len(metas_cache),
                "num_pids": len(pids_cache),
            }
        )

    # Database file details
    if os.path.exists(PAPERS_DB_FILE):
        current_db_mtime = os.path.getmtime(PAPERS_DB_FILE)
        status["papers_and_metas"].update(
            {
                "db_file_exists": True,
                "current_db_file_mtime": current_db_mtime,
                "db_file_is_newer": current_db_mtime > papers_db_file_mtime,
            }
        )
    else:
        status["papers_and_metas"]["db_file_exists"] = False

    return jsonify(status)


# -----------------------------------------------------------------------------
# tag related endpoints: add, delete tags for any paper


def api_tag_feedback():
    """Submit tag feedback - delegates to tag_service."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    try:
        data, err = _parse_api_request(require_csrf=True, schema=TagFeedbackRequest)
        if err:
            return err

        pid = data["pid"]
        tag = data["tag"]
        label = data["label"]

        err = _validate_tag_name(tag)
        if err:
            return _api_error(err, 400)

        from backend.services.tag_service import set_tag_feedback

        set_tag_feedback(pid, tag, label)
        return _api_success()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tag feedback error: {e}")
        return _api_error(str(e), 500)


def api_tag_members():
    """List papers under a tag - delegates to tag_service."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    tag = _normalize_name(request.args.get("tag", ""))
    label = _normalize_name(request.args.get("label", "all")).lower()
    search = _normalize_name(request.args.get("search", "")).lower().strip()
    if not tag:
        return _api_error("tag is required", 400)
    if label not in {"all", "pos", "neg"}:
        label = "all"

    try:
        page_number = max(1, int(request.args.get("page_number", "1")))
    except Exception:
        page_number = 1
    try:
        page_size = int(request.args.get("page_size", "20"))
    except Exception:
        page_size = 20
    page_size = max(1, min(200, page_size))

    try:
        from backend.services.tag_service import get_tag_members

        result = get_tag_members(
            tag=tag,
            label=label,
            search=search,
            page_number=page_number,
            page_size=page_size,
            get_metas_fn=get_metas,
            get_papers_bulk_fn=get_papers_bulk,
        )
        return _api_success(**result)
    except Exception as e:
        logger.error(f"Failed to list tag members for user {g.user}, tag={tag}: {e}")
        return _api_error("Server error", 500)


def api_paper_titles():
    """Resolve paper titles - delegates to tag_service."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    try:
        data, err = _parse_api_request(require_csrf=True, schema=PaperTitlesRequest)
        if err:
            return err

        raw = data.get("pids", [])
        if isinstance(raw, str):
            raw_list = [raw]
        elif isinstance(raw, list):
            raw_list = raw
        else:
            raw_list = []

        # Normalize + de-dup while keeping order
        pids = []
        seen = set()
        for v in raw_list:
            pid = _normalize_name(v)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            pids.append(pid)
            if len(pids) >= 200:
                break

        from backend.services.tag_service import resolve_paper_titles

        items = resolve_paper_titles(pids, get_papers_bulk_fn=get_papers_bulk)
        return _api_success(items=items)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper titles error: {e}")
        return _api_error(str(e), 500)


def add_tag(tag=None):
    """Add an empty tag - delegates to tag_service."""
    from backend.services.tag_service import create_empty_tag

    _csrf_protect()
    tag = _normalize_name(tag)
    return create_empty_tag(tag)


def add(pid=None, tag=None):
    """Add paper to tag - delegates to tag_service."""
    from backend.services.tag_service import add_paper_to_tag

    _csrf_protect()
    pid = _normalize_name(pid)
    tag = _normalize_name(tag)
    return add_paper_to_tag(pid, tag)


def sub(pid=None, tag=None):
    """Remove paper from tag - delegates to tag_service."""
    from backend.services.tag_service import remove_paper_from_tag

    _csrf_protect()
    pid = _normalize_name(pid)
    tag = _normalize_name(tag)
    return remove_paper_from_tag(pid, tag)


def delete_tag(tag=None):
    """Delete a tag - delegates to tag_service."""
    from backend.services.tag_service import delete_tag as _delete_tag

    _csrf_protect()
    tag = _normalize_name(tag)
    return _delete_tag(tag)


def rename_tag(otag=None, ntag=None):
    """Rename a tag - delegates to tag_service."""
    from backend.services.tag_service import rename_tag as _rename_tag

    _csrf_protect()
    otag = _normalize_name(otag)
    ntag = _normalize_name(ntag)
    return _rename_tag(otag, ntag)


def add_ctag(ctag=None):
    """Add a combined tag - delegates to tag_service."""
    from backend.services.tag_service import create_combined_tag

    _csrf_protect()
    ctag = _normalize_name(ctag)
    return create_combined_tag(ctag)


def delete_ctag(ctag=None):
    """Delete a combined tag - delegates to tag_service."""
    from backend.services.tag_service import delete_combined_tag

    _csrf_protect()
    ctag = _normalize_name(ctag)
    return delete_combined_tag(ctag)


def rename_ctag(otag=None, ntag=None):
    """Rename a combined tag - delegates to tag_service."""
    from backend.services.tag_service import rename_combined_tag

    _csrf_protect()
    return rename_combined_tag(otag, ntag)


def add_key(keyword=None):
    """Add keyword - delegates to keyword_service."""
    from backend.services.keyword_service import add_keyword

    _csrf_protect()
    keyword = _normalize_name(keyword)
    return add_keyword(keyword)


def delete_key(keyword=None):
    """Delete keyword - delegates to keyword_service."""
    from backend.services.keyword_service import delete_keyword

    _csrf_protect()
    keyword = _normalize_name(keyword)
    return delete_keyword(keyword)


def rename_key(okey=None, nkey=None):
    """Rename keyword - delegates to keyword_service."""
    from backend.services.keyword_service import rename_keyword

    _csrf_protect()
    okey = _normalize_name(okey)
    nkey = _normalize_name(nkey)
    return rename_keyword(okey, nkey)


# -----------------------------------------------------------------------------
# endpoints to log in and out


def login():
    """Log in user - delegates to auth_service."""
    from backend.services.auth_service import login_user

    _csrf_protect()
    username = (request.form.get("username") or "").strip()
    return redirect(login_user(username))


def logout():
    """Log out user - delegates to auth_service."""
    from backend.services.auth_service import logout_user

    _csrf_protect()
    return redirect(logout_user())


# -----------------------------------------------------------------------------
# user settings and configurations


def register_email():
    """Register email - delegates to auth_service."""
    from backend.services.auth_service import register_user_email

    _csrf_protect()
    email = (request.form.get("email") or "").strip()
    return redirect(register_user_email(email))


# -----------------------------------------------------------------------------
# Reading List functionality


def get_readinglist():
    """Get reading list - delegates to readinglist_service."""
    from backend.services.readinglist_service import get_user_readinglist

    return get_user_readinglist()


def compute_top_tags_for_paper(pid: str, user_tags: dict, max_tags: int = 3, threshold: float = 0.3) -> list:
    """Delegates to backend.services.semantic_service.compute_top_tags_for_paper."""
    from backend.services.semantic_service import (
        compute_top_tags_for_paper as _compute_top_tags,
    )

    return _compute_top_tags(pid, user_tags, max_tags=max_tags, threshold=threshold)


def _update_readinglist_summary_status(
    user: str,
    pid: str,
    status: str,
    error: Optional[str] = None,
    task_id: Optional[str] = None,
) -> None:
    """Update summary status in reading list - delegates to readinglist_service."""
    from backend.services.readinglist_service import update_summary_status

    update_summary_status(user, pid, status, error, task_id)


def _update_summary_status_db(
    pid: str,
    model: Optional[str],
    status: str,
    error: Optional[str] = None,
    task_id: Optional[str] = None,
    task_user: Optional[str] = None,
) -> None:
    """Persist summary status - delegates to readinglist_service."""
    from backend.services.readinglist_service import update_summary_status_db

    update_summary_status_db(pid, model, status, error, task_id, task_user, default_model=LLM_NAME)


def _trigger_summary_async(
    user: Optional[str],
    pid: str,
    model: Optional[str] = None,
    priority: Optional[int] = None,
) -> Optional[str]:
    """Trigger summary generation - delegates to readinglist_service."""
    from backend.services.readinglist_service import trigger_summary_async

    return trigger_summary_async(
        user=user,
        pid=pid,
        model=model,
        priority=priority,
        generate_summary_fn=generate_paper_summary,
        update_readinglist_fn=_update_readinglist_summary_status,
        update_db_fn=_update_summary_status_db,
        default_model=LLM_NAME,
    )


def readinglist_page():
    """Display reading list page"""
    context = default_context()

    if not g.user:
        context["papers"] = []
        context["tags"] = []
        context["message"] = "Please log in to use reading list."
        context["default_summary_model"] = LLM_NAME or ""
        return render_template("readinglist.html", **context)

    readinglist = get_readinglist()

    # Sort by added_time descending
    sorted_items = sorted(readinglist.items(), key=lambda x: x[1].get("added_time", 0), reverse=True)

    # Render papers
    papers = []
    pids = [pid for pid, _ in sorted_items]
    pid_to_paper = get_papers_bulk(pids)

    for pid, info in sorted_items:
        paper = pid_to_paper.get(pid)
        if not paper:
            continue

        rendered = render_pid(pid, paper=paper)
        rendered["added_time"] = info.get("added_time", 0)
        rendered["top_tags"] = info.get("top_tags", [])
        rendered["summary_status"] = info.get("summary_status")
        rendered["summary_last_error"] = info.get("summary_last_error")
        rendered["summary_updated_time"] = info.get("summary_updated_time")
        rendered["summary_task_id"] = info.get("summary_task_id")
        rendered["in_readinglist"] = True
        papers.append(rendered)

    context["papers"] = papers
    context["default_summary_model"] = LLM_NAME or ""
    # Provide tag list for tag dropdown (same shape as main page)
    if g.user:
        tags_db = get_tags()
        neg_tags_db = get_neg_tags()
        rtags = []
        for t in set(tags_db.keys()) | set(neg_tags_db.keys()):
            pos_n = len(tags_db.get(t, set()))
            neg_n = len(neg_tags_db.get(t, set()))
            rtags.append(
                {
                    "name": t,
                    "n": pos_n + neg_n,
                    "pos_n": pos_n,
                    "neg_n": neg_n,
                    "neg_only": pos_n == 0 and neg_n > 0,
                }
            )
        if rtags:
            rtags.append({"name": "all"})
        context["tags"] = sorted(rtags, key=lambda item: item["name"]) if rtags else []
    else:
        context["tags"] = []
    return render_template("readinglist.html", **context)


def api_readinglist_add():
    """Add paper to reading list - delegates to readinglist_service."""
    try:
        data, err = _parse_api_request(require_login=True, require_pid=True, schema=ReadingListPidRequest)
        if err:
            return err

        raw_pid = data["_raw_pid"]

        from backend.services.readinglist_service import add_to_readinglist

        def _trigger_fn(user, pid):
            task_id = _trigger_summary_async(user, pid)
            if task_id is None and not settings.huey.allow_thread_fallback:
                err_msg = None
                try:
                    info = SummaryStatusRepository.get_status(pid, (LLM_NAME or "").strip())
                    if isinstance(info, dict):
                        err_msg = info.get("last_error")
                except Exception:
                    err_msg = None
                err_msg = err_msg or "Failed to enqueue summary task"
                _update_summary_status_db(pid, None, "failed", err_msg, task_user=user)
                _update_readinglist_summary_status(user, pid, "failed", err_msg)
                return None
            if task_id:
                _update_summary_status_db(pid, None, "queued", None, task_id=task_id, task_user=user)
                _update_readinglist_summary_status(user, pid, "queued", None, task_id=task_id)
            else:
                _update_summary_status_db(pid, None, "queued", None, task_user=user)
            return task_id

        result = add_to_readinglist(
            pid=raw_pid,
            compute_top_tags_fn=compute_top_tags_for_paper,
            get_tags_fn=get_tags,
            trigger_summary_fn=_trigger_fn,
        )

        if "error" in result:
            return _api_error(result["error"], 401 if result["error"] == "Not logged in" else 500)

        return _api_success(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add to reading list: {e}")
        return _api_error(str(e), 500)


def api_readinglist_remove():
    """Remove paper from reading list - delegates to readinglist_service."""
    try:
        data, err = _parse_api_request(require_login=True, require_csrf=True, schema=ReadingListPidRequest)
        if err:
            return err

        raw_pid, _ = split_pid_version(data["pid"])

        from backend.services.readinglist_service import remove_from_readinglist

        result = remove_from_readinglist(raw_pid)

        if "error" in result:
            status = 401 if result["error"] == "Not logged in" else 404
            return _api_error(result["error"], status)

        return _api_success(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove from reading list: {e}")
        return _api_error(str(e), 500)


def api_readinglist_list():
    """Get reading list - delegates to readinglist_service."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    from backend.services.readinglist_service import list_readinglist

    items = list_readinglist()
    return _api_success(items=items)
