"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

# Multi-core optimization configuration - Ubuntu system
import os
import secrets
import shutil
import threading
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from typing import Optional
from urllib.parse import urlparse

import requests
from loguru import logger

# Set multi-threading environment variables
num_threads = min(cpu_count(), 192)  # Reasonable thread limit
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

# Try to use Intel extensions (if available)
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    logger.info(f"Intel scikit-learn extension enabled with {num_threads} threads")
    USE_INTEL_EXT = True
except ImportError:
    logger.info(f"Using standard sklearn with {num_threads} threads")
    USE_INTEL_EXT = False

import re
import sys
import time
from pathlib import Path
from random import shuffle

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from flask import (  # global session-level object
    Flask,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from sklearn import svm
from tqdm import tqdm

from aslite.db import (
    DICT_DB_FILE,
    FEATURES_FILE,
    get_combined_tags_db,
    get_email_db,
    get_keywords_db,
    get_last_active_db,
    get_metas_db,
    get_papers_db,
    get_tags_db,
    load_features,
)
from compute import Qwen3EmbeddingVllm
from paper_summarizer import acquire_summary_lock, atomic_write_json, atomic_write_text
from paper_summarizer import (
    generate_paper_summary as generate_paper_summary_from_module,
)
from paper_summarizer import (
    normalize_summary_result,
    normalize_summary_source,
    read_summary_meta,
    release_summary_lock,
    resolve_cache_pid,
    split_pid_version,
    summary_cache_paths,
    summary_source_matches,
)
from vars import (
    EMBED_API_BASE,
    EMBED_API_KEY,
    EMBED_MODEL_NAME,
    EMBED_PORT,
    EMBED_USE_LLM_API,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_NAME,
    MINERU_ENABLED,
    MINERU_PORT,
    SUMMARY_DEFAULT_SEMANTIC_WEIGHT,
    SUMMARY_DIR,
    SUMMARY_MARKDOWN_SOURCE,
    SVM_C,
    SVM_MAX_ITER,
    SVM_TOL,
)

# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = 100  # number of papers to return per page
MAX_RESULTS = RET_NUM * 10  # Process at most 10 pages of results, avoid processing all data

# Feature cache related global variables
FEATURES_CACHE = None
FEATURES_FILE_MTIME = 0  # Feature file modification time
FEATURES_CACHE_TIME = 0  # Cache creation time
FEATURES_CACHE_LOCK = threading.Lock()

# Papers and metas cache related global variables (stored in the same database file)
PAPERS_CACHE = None
METAS_CACHE = None
PIDS_CACHE = None
PAPERS_DB_FILE_MTIME = 0  # Papers database file modification time
PAPERS_DB_CACHE_TIME = 0  # Database cache creation time
PAPERS_DB_CACHE_LOCK = threading.Lock()

# Database file path
from aslite.db import PAPERS_DB_FILE as PAPERS_DB_PATH

app = Flask(__name__)

# Whether to cache the entire (large) papers table in RAM.
# Default on when memory is sufficient; metas/pids remain cached.
CACHE_PAPERS_IN_MEMORY = os.environ.get("ARXIV_SANITY_CACHE_PAPERS", "1").lower() in ("1", "true", "yes")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "y", "on")


def _load_secret_key() -> str:
    """
    Prefer env var, then secret_key.txt. If neither exists, generate a random key.

    This avoids shipping an insecure default and reduces accidental session forgery.
    """
    sk = (os.environ.get("ARXIV_SANITY_SECRET_KEY") or "").strip()
    if sk:
        return sk

    if os.path.isfile("secret_key.txt"):
        try:
            with open("secret_key.txt", encoding="utf-8") as f:
                sk = f.read().strip()
            if sk:
                return sk
        except Exception as e:
            logger.warning(f"Failed to read secret_key.txt: {e}")

    logger.warning(
        "No secret key found (ARXIV_SANITY_SECRET_KEY/secret_key.txt); generating a random key (sessions reset on restart)"
    )
    return secrets.token_urlsafe(32)


# set the secret key so we can cryptographically sign cookies and maintain sessions
app.secret_key = _load_secret_key()

# Cookie & request hardening (can be overridden via env vars)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=os.environ.get("ARXIV_SANITY_COOKIE_SAMESITE", "Lax"),
    SESSION_COOKIE_SECURE=_truthy_env("ARXIV_SANITY_COOKIE_SECURE", "0"),
    MAX_CONTENT_LENGTH=int(os.environ.get("ARXIV_SANITY_MAX_CONTENT_LENGTH", str(1 * 1024 * 1024))),  # 1 MiB
)


@app.after_request
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
    tok = session.get("_csrf_token")
    if not tok:
        tok = secrets.token_urlsafe(32)
        session["_csrf_token"] = tok
    return tok


def _is_same_origin_request() -> bool:
    """
    Best-effort same-origin check for legacy GET mutation endpoints.

    - For modern browsers, `Sec-Fetch-Site: cross-site` is a reliable CSRF signal.
    - Otherwise fall back to Origin/Referer checks.
    """
    sfs = (request.headers.get("Sec-Fetch-Site") or "").lower().strip()
    if sfs:
        return sfs in ("same-origin", "same-site", "none")

    origin = (request.headers.get("Origin") or "").strip()
    if origin:
        return origin.rstrip("/") == request.host_url.rstrip("/")

    referer = (request.headers.get("Referer") or "").strip()
    if referer:
        try:
            return urlparse(referer).netloc == request.host
        except Exception:
            return False

    return False


def _csrf_protect():
    """
    CSRF protection for state-changing endpoints.

    - POST: require session token via header/form/json
    - GET (legacy): require same-origin *or* explicit query token
    """
    tok = _get_or_set_csrf_token()

    if request.method == "GET":
        if request.args.get("csrf_token") == tok:
            return
        if _is_same_origin_request():
            return
        abort(403, description="CSRF blocked (cross-site GET)")

    # Unsafe methods: token required
    token = (request.headers.get("X-CSRF-Token") or "").strip()
    if not token:
        token = (request.form.get("csrf_token") or "").strip()
    if not token and request.is_json:
        data = request.get_json(silent=True) or {}
        token = (data.get("csrf_token") or "").strip()

    if not token or token != tok:
        abort(403, description="CSRF token missing/invalid")


# -----------------------------------------------------------------------------
# Input normalization helpers


def _normalize_name(value: Optional[str]) -> str:
    return (value or "").strip()


# Keep tokenization consistent with compute.py for query-side TF-IDF.
TFIDF_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b|\b[a-zA-Z]+\-[a-zA-Z]+\b"
TFIDF_STOP_WORDS = "english"

# Treat common separators (incl. CJK punctuation) as spaces for multi-keyword queries.
_QUERY_SEP_RE = re.compile(r"[,;\uFF0C\u3001\uFF1B\uFF1A:/\\|\(\)\[\]{}]+")
_BOOLEAN_TOKENS = {"and", "or", "not"}


def _validate_tag_name(tag: str) -> Optional[str]:
    if not tag:
        return "error, tag is required"
    if tag in ("all", "null"):
        return f"error, cannot use the protected tag '{tag}'"
    return None


def _validate_keyword_name(keyword: str) -> Optional[str]:
    if not keyword:
        return "error, keyword is required"
    if keyword == "null":
        return "error, cannot use the protected keyword 'null'"
    return None


# -----------------------------------------------------------------------------
# Small in-memory caches (thread-safe, per-process)


class _LRUCacheTTL:
    def __init__(self, maxsize: int = 256, ttl_s: float = 120.0):
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if now - ts > self._ttl_s:
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value):
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


SVM_RANK_CACHE = _LRUCacheTTL(maxsize=128, ttl_s=180.0)
SEARCH_RANK_CACHE = _LRUCacheTTL(maxsize=256, ttl_s=120.0)
PAPER_CACHE = _LRUCacheTTL(maxsize=2048, ttl_s=300.0)
THUMB_CACHE = _LRUCacheTTL(maxsize=4096, ttl_s=600.0)
QUERY_EMBED_CACHE = _LRUCacheTTL(maxsize=256, ttl_s=600.0)


# -----------------------------------------------------------------------------
# Helper function to reduce code duplication
def _get_user_data(db_func, attr_name):
    """Generic function to get user data from database"""
    if g.user is None:
        return {}
    if not hasattr(g, attr_name):
        with db_func() as db:
            data = db[g.user] if g.user in db else {}
        setattr(g, attr_name, data)
    return getattr(g, attr_name)


# globals that manage the (lazy) loading of various state for a request
def get_tags():
    return _get_user_data(get_tags_db, "_tags")


def get_combined_tags():
    return _get_user_data(get_combined_tags_db, "_combined_tags")


def get_keys():
    return _get_user_data(get_keywords_db, "_keys")


# -----------------------------------------------------------------------------
# Intelligent unified data caching functionality


def get_data_cached():
    """
    Intelligent unified data cache loading function
    - Unified management of papers and metas data loading
    - Detect database file updates and automatically reload
    - Record detailed cache status logs
    """
    global PAPERS_CACHE, METAS_CACHE, PIDS_CACHE
    global PAPERS_DB_FILE_MTIME, PAPERS_DB_CACHE_TIME

    current_time = time.time()

    # Check if database file exists
    if not os.path.exists(PAPERS_DB_PATH):
        logger.error(f"Papers database file not found: {PAPERS_DB_PATH}")
        raise FileNotFoundError(f"Papers database file not found: {PAPERS_DB_PATH}")

    # Get current database file modification time
    current_file_mtime = os.path.getmtime(PAPERS_DB_PATH)

    # We always cache metas/pids (they are small enough and heavily used for ranking/time filtering).
    # Papers cache is optional because the decoded objects can be very large.
    need_reload_metas = METAS_CACHE is None or PIDS_CACHE is None or current_file_mtime > PAPERS_DB_FILE_MTIME
    need_reload_papers = CACHE_PAPERS_IN_MEMORY and (PAPERS_CACHE is None or current_file_mtime > PAPERS_DB_FILE_MTIME)
    need_reload = need_reload_metas or need_reload_papers

    if need_reload:
        with PAPERS_DB_CACHE_LOCK:
            # Re-check inside lock to avoid duplicate reloads in multi-thread servers
            try:
                current_file_mtime = os.path.getmtime(PAPERS_DB_PATH)
            except Exception:
                current_file_mtime = 0.0
            need_reload_metas = METAS_CACHE is None or PIDS_CACHE is None or current_file_mtime > PAPERS_DB_FILE_MTIME
            need_reload_papers = CACHE_PAPERS_IN_MEMORY and (
                PAPERS_CACHE is None or current_file_mtime > PAPERS_DB_FILE_MTIME
            )
            need_reload = need_reload_metas or need_reload_papers
            if need_reload:
                logger.info("Loading metas%s from database..." % (" and papers" if CACHE_PAPERS_IN_MEMORY else ""))
                if PAPERS_CACHE is not None:
                    logger.trace(
                        f"Database file updated (old mtime: {PAPERS_DB_FILE_MTIME}, new mtime: {current_file_mtime})"
                    )

                start_time = time.time()

                try:
                    if CACHE_PAPERS_IN_MEMORY:
                        with get_papers_db() as papers_db:
                            PAPERS_CACHE = {
                                k: v
                                for k, v in tqdm(
                                    papers_db.items(), desc="loading papers db", ncols=100, leave=True, file=sys.stderr
                                )
                            }
                    else:
                        PAPERS_CACHE = None

                    with get_metas_db() as metas_db:
                        METAS_CACHE = {
                            k: v
                            for k, v in tqdm(
                                metas_db.items(), desc="loading metas db", ncols=100, leave=True, file=sys.stderr
                            )
                        }
                        PIDS_CACHE = list(METAS_CACHE.keys())

                    PAPERS_DB_FILE_MTIME = current_file_mtime
                    PAPERS_DB_CACHE_TIME = current_time

                    load_time = time.time() - start_time
                    logger.info(f"Data loaded successfully in {load_time:.3f}s")
                    if PAPERS_CACHE is not None:
                        logger.trace(f"Number of papers: {len(PAPERS_CACHE)}")
                    logger.trace(f"Number of metas: {len(METAS_CACHE)}")
                    logger.trace(f"Number of pids: {len(PIDS_CACHE)}")

                except Exception as e:
                    logger.error(f"Failed to load papers and metas: {e}")
                    raise
    else:
        # Use cache
        pass

    return PAPERS_CACHE, METAS_CACHE, PIDS_CACHE


# -----------------------------------------------------------------------------
# Cached data access functions


def get_pids():
    """Get all paper ID list"""
    get_data_cached()  # Ensure cache is loaded
    return PIDS_CACHE


def get_papers():
    """Get papers database"""
    get_data_cached()  # Ensure cache is loaded
    return PAPERS_CACHE


def paper_exists(pid: str) -> bool:
    """Fast existence check using metas (always cached)."""
    if not pid:
        return False
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
        with get_papers_db() as pdb:
            paper = pdb.get(pid)
            if paper is None:
                return None
        PAPER_CACHE.set(pid, paper)
        return paper
    except Exception as e:
        logger.warning(f"Failed to read paper {pid} from db: {e}")
        return None


def get_papers_bulk(pids):
    """
    Fetch a batch of papers efficiently.

    When `ARXIV_SANITY_CACHE_PAPERS=0`, this avoids opening the SqliteDict once per paper.
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

    try:
        with get_papers_db() as pdb:
            for pid in pids:
                paper = pdb.get(pid)
                if paper is None:
                    continue
                out[pid] = paper
                try:
                    PAPER_CACHE.set(pid, paper)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Failed to bulk-read papers from db: {e}")
    return out


def get_metas():
    """Get metadata database"""
    get_data_cached()  # Ensure cache is loaded
    return METAS_CACHE


_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_STARTED = False
_SCHEDULER = None


def _warmup_data_cache():
    try:
        logger.info("Warming metas/pids cache in background...")
        get_data_cached()
    except Exception as e:
        logger.warning(f"Data cache warmup failed: {e}")


def _ensure_background_services_started():
    """
    Start background threads/schedulers lazily (per worker process).

    This avoids starting threads at import-time, which is important when running
    under gunicorn with `--preload` (forking after threads start is unsafe).
    """
    global _BACKGROUND_STARTED, _SCHEDULER
    if _BACKGROUND_STARTED:
        return
    with _BACKGROUND_LOCK:
        if _BACKGROUND_STARTED:
            return

        if os.environ.get("ARXIV_SANITY_WARMUP_DATA", "1").lower() in ("1", "true", "yes"):
            threading.Thread(target=_warmup_data_cache, daemon=True).start()

        if os.environ.get("ARXIV_SANITY_WARMUP_ML", "1").lower() in ("1", "true", "yes"):
            threading.Thread(target=_warmup_ml_cache, daemon=True).start()

        if os.environ.get("ARXIV_SANITY_ENABLE_SCHEDULER", "1").lower() in ("1", "true", "yes"):
            try:
                _SCHEDULER = BackgroundScheduler(timezone="Asia/Shanghai")
                _SCHEDULER.add_job(get_data_cached, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
                _SCHEDULER.start()
            except Exception as e:
                logger.warning(f"Failed to start scheduler: {e}")

        _BACKGROUND_STARTED = True


@app.before_request
def before_request():
    _ensure_background_services_started()
    g.user = session.get("user", None)

    # record activity on this user so we can reserve periodic
    # recommendations heavy compute only for active users
    if g.user:
        with get_last_active_db(flag="c") as last_active_db:
            last_active_db[g.user] = int(time.time())


@app.teardown_request
def close_connection(error=None):
    # papers and metas now use global cache, no cleanup needed
    # All data access uses global variables directly
    pass


# -----------------------------------------------------------------------------
# Intelligent feature caching functionality


def get_features_cached():
    """
    Intelligent feature cache loading function
    - Detect if feature file is updated
    - Automatically reload updated features
    - Record detailed cache status logs
    """
    global FEATURES_CACHE, FEATURES_FILE_MTIME, FEATURES_CACHE_TIME

    current_time = time.time()

    # Check if feature file exists
    if not os.path.exists(FEATURES_FILE):
        logger.error(f"Features file not found: {FEATURES_FILE}")
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    # Get current modification time of feature file
    current_file_mtime = os.path.getmtime(FEATURES_FILE)

    # Determine if features need to be reloaded
    need_reload = FEATURES_CACHE is None or current_file_mtime > FEATURES_FILE_MTIME  # First load  # File updated

    if need_reload:
        with FEATURES_CACHE_LOCK:
            # Re-check inside lock to avoid duplicate reloads in multi-thread servers
            try:
                current_file_mtime = os.path.getmtime(FEATURES_FILE)
            except Exception:
                current_file_mtime = 0.0
            need_reload = FEATURES_CACHE is None or current_file_mtime > FEATURES_FILE_MTIME

            if need_reload:
                logger.info("Loading features from disk...")
                if FEATURES_CACHE is not None:
                    logger.trace(
                        f"Features file updated (old mtime: {FEATURES_FILE_MTIME}, new mtime: {current_file_mtime})"
                    )

                start_time = time.time()

                try:
                    FEATURES_CACHE = load_features()
                    FEATURES_FILE_MTIME = current_file_mtime
                    FEATURES_CACHE_TIME = current_time

                    # Build pid -> index map for fast lookups (inspect, etc.)
                    try:
                        if (
                            isinstance(FEATURES_CACHE, dict)
                            and "pids" in FEATURES_CACHE
                            and "pid_to_index" not in FEATURES_CACHE
                        ):
                            FEATURES_CACHE["pid_to_index"] = {pid: i for i, pid in enumerate(FEATURES_CACHE["pids"])}
                    except Exception as e:
                        logger.warning(f"Failed to build pid_to_index map: {e}")

                    load_time = time.time() - start_time
                    logger.info(f"Features loaded successfully in {load_time:.3f}s")
                    logger.trace(f"Feature matrix shape: {FEATURES_CACHE['x'].shape}")
                    logger.trace(f"Number of papers: {len(FEATURES_CACHE['pids'])}")
                    logger.trace(f"Vocabulary size: {len(FEATURES_CACHE['vocab'])}")

                except Exception as e:
                    logger.error(f"Failed to load features: {e}")
                    raise
    else:
        # Use cache
        pass
        # cache_age = current_time - FEATURES_CACHE_TIME
        # logger.trace(f"Using cached features (age: {cache_age:.1f}s)")

    return FEATURES_CACHE


# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests


def _get_thumb_url(pid: str) -> str:
    cached = THUMB_CACHE.get(pid)
    if cached is not None:
        return cached
    thumb_path = Path("static/thumb") / f"{pid}.jpg"
    thumb_url = f"static/thumb/{pid}.jpg" if thumb_path.is_file() else ""
    THUMB_CACHE.set(pid, thumb_url)
    return thumb_url


# TL;DR cache to avoid repeated file reads (TTL allows new summaries to be picked up)
TLDR_CACHE = _LRUCacheTTL(maxsize=2000, ttl_s=600.0)  # 10 minutes TTL


def _extract_tldr_from_summary(pid: str) -> str:
    """
    Extract TL;DR from cached summary file if it exists.

    Args:
        pid: Paper ID

    Returns:
        TL;DR content string, or empty string if not found
    """
    from summary_utils import read_tldr_from_summary_file

    cached = TLDR_CACHE.get(pid)
    if cached is not None:
        return cached

    tldr = read_tldr_from_summary_file(pid)

    # Only cache positive results to allow new summaries to be picked up
    if tldr:
        TLDR_CACHE.set(pid, tldr)

    return tldr


def render_pid(pid, pid_to_utags=None, paper=None):
    # render a single paper with just the information we need for the UI
    thumb_url = _get_thumb_url(pid)
    tldr = _extract_tldr_from_summary(pid)
    d = paper if paper is not None else get_paper(pid)
    if d is None:
        # Extremely defensive: return a minimal stub instead of crashing the request.
        return dict(
            weight=0.0,
            id=pid,
            title="(missing paper)",
            time="",
            authors="",
            tags="",
            utags=pid_to_utags.get(pid, []) if pid_to_utags else [],
            summary="",
            tldr="",
            thumb_url=thumb_url,
        )
    if pid_to_utags is not None:
        utags = pid_to_utags.get(pid, [])
    else:
        tags = get_tags()
        utags = [t for t, tpids in tags.items() if pid in tpids]
    return dict(
        weight=0.0,
        id=d["_id"],
        title=d["title"],
        time=d["_time_str"],
        authors=", ".join(a["name"] for a in d["authors"]),
        tags=", ".join(t["term"] for t in d["tags"]),
        utags=utags,
        summary=d["summary"],
        tldr=tldr,
        thumb_url=thumb_url,
    )


def _apply_limit(pids, scores, limit):
    """Apply limit to results if specified"""
    if limit is not None and len(pids) > limit:
        return pids[:limit], scores[:limit]
    return pids, scores


def random_rank(limit=None):
    pids_all = get_pids()
    if limit is not None:
        try:
            k = min(int(limit), len(pids_all))
        except Exception:
            k = len(pids_all)
        if k <= 0:
            return [], []
        # sample without shuffling the full list (much faster on large corpora)
        import random

        pids = random.sample(pids_all, k)
        scores = [0.0 for _ in pids]
        return pids, scores

    pids = list(pids_all)
    shuffle(pids)
    scores = [0.0 for _ in pids]
    return pids, scores


def time_rank(limit=None):
    mdb = get_metas()
    if limit is not None:
        import heapq

        try:
            k = min(int(limit), len(mdb))
        except Exception:
            k = len(mdb)
        if k <= 0:
            return [], []
        ms = heapq.nlargest(k, mdb.items(), key=lambda kv: kv[1]["_time"])
    else:
        ms = sorted(mdb.items(), key=lambda kv: kv[1]["_time"], reverse=True)

    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v["_time"]) / 60 / 60 / 24 for k, v in ms]  # time delta in days
    return pids, scores


def _filter_by_time_with_tags(pids, time_filter, user_tagged_pids=None):
    """
    Smart time filtering: keep tagged papers (even if outside time window) and papers within time window
    Filter papers by time but keep tagged papers even if outside time window
    """
    if not time_filter:
        return pids, list(range(len(pids)))

    mdb = get_metas()
    tnow = time.time()
    try:
        deltat = float(time_filter) * 60 * 60 * 24
    except Exception:
        logger.warning(f"Invalid time_filter '{time_filter}', skipping time filtering")
        return pids, list(range(len(pids)))

    # Get set of all user-tagged paper IDs
    tagged_set = set()
    if user_tagged_pids:
        tagged_set = user_tagged_pids

    # Find indices of papers that meet criteria:
    # 1. Papers within time window, or
    # 2. User-tagged papers (even if outside time window)
    valid_indices = []
    valid_pids = []
    for i, pid in enumerate(pids):
        meta = mdb.get(pid)
        if not meta:
            continue
        # Check if within time window
        in_time_window = (tnow - meta["_time"]) < deltat
        # Check if tagged by user
        is_tagged = pid in tagged_set

        # Keep if either condition is met
        if in_time_window or is_tagged:
            valid_indices.append(i)
            valid_pids.append(pid)

    return valid_pids, valid_indices


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
    # Use default value
    if C is None:
        C = SVM_C
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or s_pids):
        return [], [], []

    # Fast path: cache (safe because we key by dict.db/features.p mtime)
    try:
        dict_mtime = os.path.getmtime(DICT_DB_FILE) if os.path.exists(DICT_DB_FILE) else 0.0
        feat_mtime = os.path.getmtime(FEATURES_FILE) if os.path.exists(FEATURES_FILE) else 0.0
        cache_key = (
            "svm",
            getattr(g, "user", None),
            tags,
            s_pids,
            float(C),
            logic,
            time_filter,
            int(limit) if limit is not None else None,
            dict_mtime,
            feat_mtime,
        )
        cached = SVM_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    # Use intelligent cache to load features
    s_time = time.time()
    features = get_features_cached()  # Replace: features = load_features()
    x, pids = features["x"], features["pids"]

    # Collect all user-tagged paper IDs for smart time filtering
    user_tagged_pids = set()
    if tags:
        tags_db = get_tags()
        tags_filter_to = tags_db.keys() if tags == "all" else set(map(str.strip, tags.split(",")))
        for tag in tags_filter_to:
            if tag in tags_db:
                user_tagged_pids.update(tags_db[tag])

    if s_pids:
        user_tagged_pids.update(map(str.strip, s_pids.split(",")))

    # Apply smart time filtering: keep tagged papers and papers within time window
    if time_filter:
        pids, time_valid_indices = _filter_by_time_with_tags(pids, time_filter, user_tagged_pids)
        if not pids:
            return [], [], []
        x = x[time_valid_indices]
        logger.trace(
            f"After intelligent time filtering, kept {len(pids)} papers (including tagged papers and those within {time_filter} days)"
        )

    n, _d = x.shape

    # Construct the positive set (avoid building pid->idx for all papers)
    pos_weights = {}
    tags_filter_to = []

    if tags:
        tags_db = get_tags()
        tags_filter_to = list(tags_db.keys()) if tags == "all" else [t.strip() for t in tags.split(",") if t.strip()]
        if logic == "and":
            tag_counts = {}
            for tag in tags_filter_to:
                if tag not in tags_db:
                    continue
                for pid in tags_db[tag]:
                    tag_counts[pid] = tag_counts.get(pid, 0) + 1
            for pid, count in tag_counts.items():
                pos_weights[pid] = max(pos_weights.get(pid, 0.0), float(count))
        else:
            for tag in tags_filter_to:
                if tag not in tags_db:
                    continue
                for pid in tags_db[tag]:
                    pos_weights[pid] = max(pos_weights.get(pid, 0.0), 1.0)

    if s_pids:
        pid_list = [p.strip() for p in s_pids.split(",") if p.strip()]
        seed_weight = 1.0
        if logic == "and" and tags_filter_to:
            seed_weight = float(len(tags_filter_to))
        for pid in pid_list:
            pos_weights[pid] = max(pos_weights.get(pid, 0.0), seed_weight)

    y = np.zeros(n, dtype=np.int8)
    sample_weight = np.ones(n, dtype=np.float32)
    found = 0
    if pos_weights:
        for i, pid in enumerate(pids):
            w = pos_weights.get(pid)
            if w:
                y[i] = 1
                sample_weight[i] = float(w)
                found += 1
        logger.trace(f"Found {found} positive papers in current feature slice")
    e_time = time.time()

    logger.trace(f"feature loading/caching for {e_time - s_time:.5f}s")

    if found == 0:
        return [], [], []  # there are no positives?

    if found == n:
        scores = sample_weight
        if limit is not None:
            k = min(int(limit), len(scores))
            if k <= 0:
                return [], [], []
            top_ix = np.argpartition(-scores, k - 1)[:k]
            top_ix = top_ix[np.argsort(-scores[top_ix])]
        else:
            top_ix = np.argsort(-scores)
        pids_out = [pids[int(ix)] for ix in top_ix]
        scores_out = [100 * float(scores[int(ix)]) for ix in top_ix]
        return pids_out, scores_out, []

    s_time = time.time()

    # classify - optimize parameters to accelerate training while maintaining accuracy
    clf = svm.LinearSVC(
        class_weight="balanced",
        verbose=0,
        max_iter=SVM_MAX_ITER,
        tol=SVM_TOL,
        C=C,
        dual=False,  # Faster for high-dimensional features
        random_state=42,  # Ensure reproducible results
        fit_intercept=True,  # Usually improves accuracy
        multi_class="ovr",  # One-vs-rest, faster for binary classification
    )
    # feature_map_nystroem = Nystroem(
    #     random_state=0, n_components=100, n_jobs=-1
    # )
    # x = feature_map_nystroem.fit_transform(x)
    # rbf_feature = RBFSampler(gamma=1, random_state=0, n_components=200)
    # x = rbf_feature.fit_transform(x)
    # e_time = time.time()
    # logger.trace(f"Dimension reduction for {e_time - s_time:.5f}s")

    clf.fit(x, y, sample_weight=sample_weight)
    e_time = time.time()
    logger.trace(f"SVM fitting data for {e_time - s_time:.5f}s")

    s = clf.decision_function(x)
    if getattr(s, "ndim", 1) > 1:
        s = s.reshape(-1)
    e_time = time.time()
    logger.trace(f"SVM decsion function for {e_time - s_time:.5f}s")

    # Top-k selection without full sort (limit is already bounded by MAX_RESULTS in callers)
    if limit is not None:
        k = min(int(limit), len(s))
        if k <= 0:
            return [], [], []
        top_ix = np.argpartition(-s, k - 1)[:k]
        top_ix = top_ix[np.argsort(-s[top_ix])]
    else:
        top_ix = np.argsort(-s)

    pids_out = [pids[int(ix)] for ix in top_ix]
    scores_out = [100 * float(s[int(ix)]) for ix in top_ix]

    # get the words that score most positively and most negatively for the svm
    ivocab = {v: k for k, v in features["vocab"].items()}  # index to word mapping
    weights = clf.coef_[0] if getattr(clf.coef_, "ndim", 1) > 1 else clf.coef_

    # Only analyze TF-IDF part weights (vocab size), ignore embedding dimensions
    vocab_size = len(ivocab)
    tfidf_weights = weights[:vocab_size]  # Only TF-IDF weights

    sortix = np.argsort(-tfidf_weights)
    e_time = time.time()
    logger.trace(f"rank calculation for {e_time - s_time:.5f}s")
    # logger.trace(f"Total features: {len(weights)}, TF-IDF features/: {vocab_size}")

    words = []
    for ix in list(sortix[:40]) + list(sortix[-20:]):
        words.append(
            {
                "word": ivocab[ix],
                "weight": float(tfidf_weights[ix]),
            }
        )

    result = (pids_out, scores_out, words)
    if cache_key is not None:
        try:
            SVM_RANK_CACHE.set(cache_key, result)
        except Exception:
            pass
    return result


def _normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, handle hyphens, extra spaces."""
    text = (text or "").lower().strip()
    if not text:
        return ""
    # Treat hyphens as spaces for matching (self-attention ≈ self attention)
    text = text.replace("-", " ")
    # Split on common separators (commas, slashes, colons, brackets, etc.)
    text = _QUERY_SEP_RE.sub(" ", text)
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


def _normalize_text_loose(text: str) -> str:
    """Looser normalization for title-like matching.

    - Lowercase
    - Treat hyphens and common punctuation as spaces
    - Collapse whitespace

    This helps matching copied paper titles that include punctuation like ':' ',' '.' '()' etc.
    """
    s = (text or "").lower().strip()
    if not s:
        return ""
    # Replace hyphens and most punctuation with spaces
    s = re.sub(r"[\-\u2010\u2011\u2012\u2013\u2014]", " ", s)
    s = re.sub(r"[\:\;\,\.!\?\(\)\[\]\{\}\/\\\|\"\'\`\~\+\=\*\#\$\%\^\&\@]", " ", s)
    s = " ".join(s.split())
    return s


_ARXIV_ID_RE = re.compile(
    r"(?:(?:arxiv:)?)(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v(?P<v>\d+))?",
    re.IGNORECASE,
)


def _looks_like_cjk_query(q: str) -> bool:
    """Heuristic: query contains mostly CJK and little ASCII."""
    if not q:
        return False
    s = q.strip()
    if not s:
        return False
    cjk = 0
    ascii_alnum = 0
    for ch in s:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch in (".", "-", "/")):
            ascii_alnum += 1
    # Treat as CJK query if it has any CJK and very few ASCII tokens.
    return cjk >= 2 and ascii_alnum <= max(2, len(s) // 12)


def _parse_search_query(q: str) -> dict:
    """Parse a paper-search query.

    Supported syntax (no UI changes; purely query-string semantics):
    - Field filters: ti:/title:, au:/author:, abs:/abstract:, cat:/category:, id:
      Examples: ti:"diffusion model" au:goodfellow cat:cs.LG id:2312.12345
    - Quoted phrases: "graph neural network"
    - Negation tokens: -term or !term (applies to any searched field)

    Returns a dict with normalized tokens and extracted filters.
    """
    raw = (q or "").strip()
    parsed = {
        "raw": raw,
        "raw_lower": raw.lower(),
        "norm": _normalize_text(raw),
        "terms": [],
        "phrases": [],
        "neg_terms": [],
        "filters": {"ti": [], "au": [], "abs": [], "cat": [], "id": []},
        "filters_phrases": {"ti": [], "au": [], "abs": [], "cat": [], "id": []},
    }
    if not raw:
        return parsed

    # 1) Extract key:"..." filters first
    work = raw
    for key, canon in (
        ("title", "ti"),
        ("ti", "ti"),
        ("author", "au"),
        ("au", "au"),
        ("abstract", "abs"),
        ("abs", "abs"),
        ("category", "cat"),
        ("cat", "cat"),
        ("id", "id"),
    ):
        pat = re.compile(rf"(?i)(?:^|\s){re.escape(key)}:\"([^\"]+)\"(?=\s|$)")
        while True:
            m = pat.search(work)
            if not m:
                break
            parsed["filters"][canon].append(m.group(1).strip())
            parsed["filters_phrases"][canon].append(m.group(1).strip())
            work = (work[: m.start()] + " " + work[m.end() :]).strip()

    # 2) Extract free quoted phrases
    for m in re.finditer(r"\"([^\"]+)\"", work):
        phr = m.group(1).strip()
        if phr:
            parsed["phrases"].append(phr)
    work = re.sub(r"\"([^\"]+)\"", " ", work)

    # 3) Tokenize remaining by whitespace and pick up key:value filters
    tokens = [t for t in re.split(r"\s+", work) if t]
    for tok in tokens:
        # Negation
        if tok.startswith("-") or tok.startswith("!"):
            val = tok[1:].strip()
            if val:
                parsed["neg_terms"].append(val)
            continue

        m = re.match(r"(?i)^(title|ti|author|au|abstract|abs|category|cat|id):(.+)$", tok)
        if m:
            k = m.group(1).lower()
            v = m.group(2).strip()
            canon = {
                "title": "ti",
                "ti": "ti",
                "author": "au",
                "au": "au",
                "abstract": "abs",
                "abs": "abs",
                "category": "cat",
                "cat": "cat",
                "id": "id",
            }[k]
            if v:
                parsed["filters"][canon].append(v)
            continue

        parsed["terms"].append(tok)

    # Normalize filter values by splitting into terms (keep full string too)
    def _split_terms(values):
        out = []
        for v in values:
            vs = _normalize_text(v)
            if not vs:
                continue
            out.extend([p for p in vs.split() if p])
        return out

    parsed["filters_terms"] = {
        "ti": _split_terms(parsed["filters"]["ti"]),
        "au": _split_terms(parsed["filters"]["au"]),
        "abs": _split_terms(parsed["filters"]["abs"]),
        "cat": _split_terms(parsed["filters"]["cat"]),
        "id": _split_terms(parsed["filters"]["id"]),
    }

    parsed["filters_phrases_norm"] = {
        "ti": [_normalize_text(p) for p in parsed["filters_phrases"]["ti"] if p.strip()],
        "au": [_normalize_text(p) for p in parsed["filters_phrases"]["au"] if p.strip()],
        "abs": [_normalize_text(p) for p in parsed["filters_phrases"]["abs"] if p.strip()],
        "cat": [_normalize_text(p) for p in parsed["filters_phrases"]["cat"] if p.strip()],
        "id": [_normalize_text(p) for p in parsed["filters_phrases"]["id"] if p.strip()],
    }

    # Normalize terms/neg_terms too
    parsed["terms_norm"] = [
        t for t in _normalize_text(" ".join(parsed["terms"])).split() if t and t not in _BOOLEAN_TOKENS
    ]
    parsed["neg_terms_norm"] = [t for t in _normalize_text(" ".join(parsed["neg_terms"])).split() if t]
    parsed["phrases_norm"] = [_normalize_text(p) for p in parsed["phrases"] if p.strip()]
    return parsed


def _extract_arxiv_ids(q: str) -> list[str]:
    """Return normalized arXiv IDs mentioned in the query."""
    if not q:
        return []
    ids = []
    for m in _ARXIV_ID_RE.finditer(q.strip()):
        pid = (m.group("id") or "").strip().lower()
        ver = (m.group("v") or "").strip()
        if not pid:
            continue
        ids.append(f"{pid}v{ver}" if ver else pid)
    # de-dup preserve order
    seen = set()
    out = []
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _paper_text_fields(p: dict) -> dict:
    """Build normalized text fields for scoring."""
    title = p.get("title") or ""
    authors = p.get("authors") or []
    authors_str = " ".join([a.get("name", "") for a in authors if isinstance(a, dict)])
    summary = p.get("summary") or ""
    tags = p.get("tags") or []
    tags_str = " ".join([t.get("term", "") for t in tags if isinstance(t, dict)])

    return {
        "title": title,
        "title_lower": title.lower(),
        "title_norm": _normalize_text(title),
        "title_norm_loose": _normalize_text_loose(title),
        "authors": authors_str,
        "authors_lower": authors_str.lower(),
        "authors_norm": _normalize_text(authors_str),
        "summary": summary,
        "summary_lower": summary.lower(),
        "summary_norm": _normalize_text(summary),
        "summary_norm_loose": _normalize_text_loose(summary),
        "tags": tags_str,
        "tags_lower": tags_str.lower(),
        "tags_norm": _normalize_text(tags_str),
    }


def _is_title_like_query(parsed: dict) -> bool:
    """Heuristic: user pasted a paper title."""
    if not parsed:
        return False
    raw = (parsed.get("raw") or "").strip()
    if not raw:
        return False
    # Longer, multi-token queries are more likely titles.
    terms = parsed.get("terms_norm") or []
    if len(raw) >= 22 and len(terms) >= 3:
        return True
    # Punctuation typical in titles.
    if ":" in raw and len(raw) >= 16:
        return True
    return False


def _title_candidate_scan(parsed: dict, max_candidates: int = 500, max_scan: int = 120000, time_budget_s: float = 0.6):
    """Bounded scan over titles to pull in strong title matches.

    This is a safety net for "paste full title" searches when TF-IDF recall misses
    due to tokenization/vocab/min_df effects.
    """
    q_norm = (parsed.get("norm") or "").strip()
    q_loose = _normalize_text_loose(parsed.get("raw") or "")
    if not q_norm and not q_loose:
        return []

    # Prefer loose form for punctuation-insensitive matching.
    q_use = q_loose or q_norm
    if not q_use:
        return []

    import heapq

    start = time.time()
    heap = []
    all_pids = get_pids()
    if not all_pids:
        return []

    scan_n = min(len(all_pids), int(max_scan))
    chunk = 2000
    for i in range(0, scan_n, chunk):
        if time.time() - start > time_budget_s:
            break
        part = all_pids[i : i + chunk]
        pid_to_paper = get_papers_bulk(part)
        for pid in part:
            p = pid_to_paper.get(pid)
            if p is None:
                continue
            fields = _paper_text_fields(p)
            title_loose = fields.get("title_norm_loose") or ""
            title_norm = fields.get("title_norm") or ""
            if not title_loose and not title_norm:
                continue

            s = 0.0
            # Exact loose match is very strong.
            if q_use and title_loose and q_use == title_loose:
                s = 2000.0
            elif q_use and title_loose and q_use in title_loose:
                # Substring match on loose title.
                s = 900.0 + min(200.0, 10.0 * (len(q_use) / max(1.0, len(title_loose))))
            elif q_norm and title_norm and q_norm in title_norm and len(q_norm) >= 18:
                s = 750.0
            else:
                continue

            if len(heap) < max_candidates:
                heapq.heappush(heap, (s, pid))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, pid))

    heap.sort(reverse=True)
    return [pid for _, pid in heap]


def _compute_paper_score_parsed(parsed: dict, p: dict, pid: str) -> float:
    """Paper-oriented lexical scoring with optional field filters.

    This is intentionally simple and fast; it boosts:
    - ID exact/partial matches
    - Title/Author matches (stronger)
    - Category(tag) matches (medium)
    - Abstract matches (weaker)
    It also supports negation terms for coarse exclusion.
    """
    if not parsed or p is None:
        return 0.0

    pid_lower = (pid or "").lower()
    fields = _paper_text_fields(p)

    # If the user input looks like a full title, strongly boost exact/substring title matches.
    # This makes "search by paper title" behave like users expect.
    q_raw_lower = (parsed.get("raw_lower") or "").strip()
    q_norm_full = (parsed.get("norm") or "").strip()
    q_loose_full = _normalize_text_loose(parsed.get("raw") or "")
    if q_norm_full and len(q_norm_full) >= 18:
        if q_norm_full == fields["title_norm"] or (q_loose_full and q_loose_full == fields.get("title_norm_loose", "")):
            return 1800.0
        if (q_norm_full in fields["title_norm"]) or (
            q_loose_full and q_loose_full in (fields.get("title_norm_loose", "") or "")
        ):
            # Near-exact title phrase match
            # (punctuation/case normalized; hyphens treated as spaces)
            # Keep below ID match fast-path.
            # Large enough to dominate common-term TF-IDF noise.
            title_phrase_bonus = 700.0
            # Extra small bonus if the raw string also appears (helps for shorter normalization collisions)
            if q_raw_lower and q_raw_lower in fields["title_lower"]:
                title_phrase_bonus += 50.0
            score = title_phrase_bonus
        else:
            score = 0.0
    else:
        score = 0.0

    # Exact ID fast path
    for want in _extract_arxiv_ids(parsed.get("raw", "")) + parsed.get("filters").get("id", []):
        want_l = want.strip().lower()
        if not want_l:
            continue
        if want_l == pid_lower:
            return 2000.0
        if want_l in pid_lower or pid_lower in want_l:
            return 1200.0

    # Negation: if any neg term appears anywhere significant, penalize hard
    neg = parsed.get("neg_terms_norm") or []
    if neg:
        hay = " ".join([fields["title_norm"], fields["authors_norm"], fields["tags_norm"], fields["summary_norm"]])
        for t in neg:
            if t and t in hay:
                return -50.0

    # Decide which terms apply to which field
    f_terms = parsed.get("filters_terms") or {}
    f_phrases = parsed.get("filters_phrases_norm") or {}
    general = parsed.get("terms_norm") or []
    phrases = parsed.get("phrases_norm") or []

    # If the user specified any field filters, do not implicitly search all fields equally.
    has_field_filters = any(len(parsed.get("filters", {}).get(k, []) or []) > 0 for k in ("ti", "au", "abs", "cat"))

    title_terms = (f_terms.get("ti") or []) + ([] if has_field_filters else general)
    author_terms = (f_terms.get("au") or []) + ([] if has_field_filters else general)
    abs_terms = (f_terms.get("abs") or []) + ([] if has_field_filters else general)
    cat_terms = (f_terms.get("cat") or []) + ([] if has_field_filters else general)

    # Enforce explicit field filters: require a match in each filtered field.
    if f_phrases.get("ti"):
        if any(ph and ph in fields["title_norm"] for ph in f_phrases["ti"]):
            score += 180.0
        elif f_terms.get("ti"):
            if not any(t and t in fields["title_norm"] for t in f_terms["ti"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("ti"):
        if not any(t and t in fields["title_norm"] for t in f_terms["ti"]):
            return 0.0

    if f_phrases.get("au"):
        if any(ph and ph in fields["authors_norm"] for ph in f_phrases["au"]):
            score += 120.0
        elif f_terms.get("au"):
            if not any(t and t in fields["authors_norm"] for t in f_terms["au"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("au"):
        if not any(t and t in fields["authors_norm"] for t in f_terms["au"]):
            return 0.0

    if f_phrases.get("abs"):
        if any(ph and ph in fields["summary_norm"] for ph in f_phrases["abs"]):
            score += 60.0
        elif f_terms.get("abs"):
            if not any(t and t in fields["summary_norm"] for t in f_terms["abs"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("abs"):
        if not any(t and t in fields["summary_norm"] for t in f_terms["abs"]):
            return 0.0

    if f_phrases.get("cat"):
        if any(ph and ph in fields["tags_norm"] for ph in f_phrases["cat"]):
            score += 45.0
        elif f_terms.get("cat"):
            if not any(t and t in fields["tags_norm"] for t in f_terms["cat"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("cat"):
        if not any(t and t in fields["tags_norm"] for t in f_terms["cat"]):
            return 0.0

    # Phrases: strongest in title, then abstract
    for ph in phrases:
        if not ph:
            continue
        if ph in fields["title_norm"]:
            score += 140.0
        elif ph in fields["summary_norm"]:
            score += 25.0

    # Implicit phrase boost for short multi-keyword queries (technical terms, short titles).
    if not phrases and general and len(general) >= 2 and not has_field_filters:
        general_phrase = " ".join(general)
        if general_phrase and general_phrase in fields["title_norm"]:
            score += 45.0
        elif general_phrase and general_phrase in fields["summary_norm"]:
            score += 12.0

    # Title scoring
    if title_terms:
        hits = sum(1 for t in title_terms if t and (t in fields["title_norm"]))
        if hits:
            score += 70.0 * (hits / max(1, len(title_terms)))
            # Frequency cap
            freq = sum(min(2, fields["title_lower"].count(t)) for t in title_terms if t) / max(1, len(title_terms))
            score += 12.0 * freq
            if hits == len(title_terms) and len(title_terms) >= 2:
                score += 40.0

    # Author scoring
    if author_terms:
        hits = sum(1 for t in author_terms if t and (t in fields["authors_norm"]))
        if hits:
            score += 55.0 * (hits / max(1, len(author_terms)))

    # Category/tag scoring
    if cat_terms:
        hits = sum(1 for t in cat_terms if t and (t in fields["tags_norm"]))
        if hits:
            score += 35.0 * (hits / max(1, len(cat_terms)))

    # Abstract scoring (lower weight)
    if abs_terms:
        hits = sum(1 for t in abs_terms if t and (t in fields["summary_norm"]))
        if hits:
            score += 12.0 * (hits / max(1, len(abs_terms)))

    # Tiny recency tie-breaker (do not overpower lexical relevance)
    try:
        meta = get_metas().get(pid)
        if meta and meta.get("_time"):
            age_days = max(0.0, (time.time() - float(meta["_time"])) / 86400.0)
            score += 0.35 * (1.0 / (1.0 + age_days / 365.0))
    except Exception:
        pass

    # Boost coverage for multi-keyword queries (prefer papers matching more terms).
    if general and len(general) > 1 and not has_field_filters:
        hay = " ".join([fields["title_norm"], fields["authors_norm"], fields["tags_norm"], fields["summary_norm"]])
        matched = {t for t in general if t and t in hay}
        if matched:
            coverage = len(matched) / max(1, len(general))
            score += 18.0 * coverage
            if coverage == 1.0:
                score += 24.0

    return score


def _lexical_rank_over_pids(pids: list, parsed: dict, limit: Optional[int] = None):
    """Score a provided candidate pid list using lexical scoring, returning ranked pids/scores."""
    if not pids:
        return [], []
    try:
        k = int(limit) if limit is not None else None
    except Exception:
        k = None

    # Bulk read for speed
    pid_to_paper = get_papers_bulk(pids)
    pairs = []
    for pid in pids:
        p = pid_to_paper.get(pid)
        if p is None:
            continue
        s = _compute_paper_score_parsed(parsed, p, pid)
        if s > 0:
            pairs.append((s, pid))
    pairs.sort(reverse=True)
    out_pids = [pid for _, pid in pairs]
    out_scores = [float(s) for s, _ in pairs]
    if k is not None:
        out_pids, out_scores = _apply_limit(out_pids, out_scores, k)
    return out_pids, out_scores


def _lexical_rank_fullscan(parsed: dict, limit: Optional[int] = None):
    """Lexical ranking by scanning the whole corpus.

    Used only as a fallback when TF-IDF candidates are insufficient or when the query
    is explicitly fielded (e.g., author/title/category filters).
    """
    all_pids = get_pids()
    if not all_pids:
        return [], []

    try:
        k = int(limit) if limit is not None else 200
    except Exception:
        k = 200
    k = max(1, min(k, MAX_RESULTS))

    import heapq

    # Keep a min-heap of top-k
    heap = []
    if CACHE_PAPERS_IN_MEMORY:
        pdb = get_papers() or {}
        for pid in all_pids:
            p = pdb.get(pid)
            if p is None:
                continue
            s = _compute_paper_score_parsed(parsed, p, pid)
            if s <= 0:
                continue
            if len(heap) < k:
                heapq.heappush(heap, (s, pid))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, pid))
    else:
        # Fallback: bulk-read in chunks
        chunk = 2000
        for i in range(0, len(all_pids), chunk):
            part = all_pids[i : i + chunk]
            pid_to_paper = get_papers_bulk(part)
            for pid in part:
                p = pid_to_paper.get(pid)
                if p is None:
                    continue
                s = _compute_paper_score_parsed(parsed, p, pid)
                if s <= 0:
                    continue
                if len(heap) < k:
                    heapq.heappush(heap, (s, pid))
                else:
                    if s > heap[0][0]:
                        heapq.heapreplace(heap, (s, pid))

    heap.sort(reverse=True)
    pids = [pid for _, pid in heap]
    scores = [float(s) for s, _ in heap]
    return pids, scores


def _compute_paper_score(q: str, qs: list, q_norm: str, qs_norm: list, p: dict, pid: str) -> float:
    """
    Compute relevance score for a single paper.

    Scoring philosophy for academic paper search:
    1. Paper ID match: Highest priority (user knows exactly what they want)
    2. Title match: Very high priority (titles are precise and informative)
    3. Author match: High priority (common search pattern)
    4. Abstract match: Lower priority (used for concept search)

    Match types (in order of importance):
    - Exact phrase match: Query appears as-is
    - All words match: All query words appear (any order)
    - Partial match: Some query words appear
    - Word frequency: How often query words appear
    """
    score = 0.0

    # === 1. Paper ID matching (highest priority) ===
    pid_lower = pid.lower()
    if q == pid_lower:
        # Exact ID match - user knows exactly what they want
        return 1000.0
    if q in pid_lower or pid_lower in q:
        # Partial ID match (e.g., "2312.12345" matches "2312.12345v2")
        score += 500.0

    # === 2. Title matching (very high priority) ===
    title = p.get("title", "")
    title_lower = title.lower()
    title_norm = _normalize_text(title)

    # Exact phrase match in title
    if q in title_lower:
        score += 100.0
    elif q_norm in title_norm:
        score += 90.0  # Normalized match (handles hyphens)

    # All query words appear in title
    if len(qs) > 1:
        if all(qp in title_lower for qp in qs):
            score += 60.0
        elif all(qp in title_norm for qp in qs_norm):
            score += 50.0

    # Partial word match in title (what fraction of query words appear)
    words_in_title = sum(1 for qp in qs if qp in title_lower)
    words_in_title_norm = sum(1 for qp in qs_norm if qp in title_norm)
    best_partial = max(words_in_title, words_in_title_norm)
    if best_partial > 0:
        partial_ratio = best_partial / len(qs)
        score += 30.0 * partial_ratio

        # Bonus for word frequency in title (indicates strong relevance)
        freq_score = sum(min(2, title_lower.count(qp)) for qp in qs) / len(qs)
        score += 10.0 * freq_score

    # === 3. Author matching (high priority) ===
    authors = p.get("authors", [])
    authors_str = " ".join([a.get("name", "") for a in authors]).lower()

    # Exact author name match
    if q in authors_str:
        score += 80.0
    # All query words in authors (useful for "first last" name search)
    elif len(qs) > 1 and all(qp in authors_str for qp in qs):
        score += 50.0
    # Partial author match
    else:
        author_matches = sum(1 for qp in qs if qp in authors_str)
        if author_matches > 0:
            score += 20.0 * (author_matches / len(qs))

    # === 4. Abstract/summary matching (lower priority, for concept search) ===
    summary = p.get("summary", "")
    summary_lower = summary.lower()
    summary_norm = _normalize_text(summary)

    # Exact phrase in abstract
    if q in summary_lower:
        score += 15.0
    elif q_norm in summary_norm:
        score += 12.0

    # All words in abstract
    if len(qs) > 1:
        if all(qp in summary_lower for qp in qs):
            score += 8.0
        elif all(qp in summary_norm for qp in qs_norm):
            score += 6.0

    # Partial match in abstract
    words_in_summary = sum(1 for qp in qs if qp in summary_lower)
    if words_in_summary > 0:
        score += 3.0 * (words_in_summary / len(qs))
        # Frequency bonus for abstract (less weight than title)
        freq_score = sum(min(3, summary_lower.count(qp)) for qp in qs) / len(qs)
        score += 1.0 * freq_score

    return score


def count_match(q, pid_start, n_pids):
    """
    Count-based matching for legacy search.
    Optimized for academic paper search patterns.
    """
    q = q.strip()
    if not q:
        return []

    q_lower = q.lower()
    qs = q_lower.split()  # Split query by spaces
    q_norm = _normalize_text(q)
    qs_norm = q_norm.split()

    sub_pairs = []

    with get_papers_db() as pdb:
        for pid in get_pids()[pid_start : pid_start + n_pids]:
            p = pdb.get(pid)
            if p is None:
                continue

            score = _compute_paper_score(q_lower, qs, q_norm, qs_norm, p, pid)

            if score > 0:
                sub_pairs.append((score, pid))

    return sub_pairs


def legacy_search_rank(q: str = "", limit=None):
    if not q:
        return [], []  # no query? no results
    n_pids = len(get_pids())
    chunk_size = 20000
    n_process = min(cpu_count() // 2, max(1, n_pids // chunk_size))
    n_process = max(1, n_process)
    with Pool(n_process) as pool:
        sub_pairs_list = pool.starmap(
            count_match,
            [(q, pid_start, chunk_size) for pid_start in range(0, n_pids, chunk_size)],
        )
        pairs = sum(sub_pairs_list, [])

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return _apply_limit(pids, scores, limit)


def _get_query_vectorizer(features):
    """Reconstruct a query-side TF-IDF encoder from cached features."""
    global _tfidf_query_vectorizer, _tfidf_query_vectorizer_mtime

    try:
        current_mtime = os.path.getmtime(FEATURES_FILE)
    except Exception:
        current_mtime = 0.0

    if _tfidf_query_vectorizer is not None and _tfidf_query_vectorizer_mtime == current_mtime:
        return _tfidf_query_vectorizer

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    tfidf_params = features.get("tfidf_params", {})
    ngram_max = int(tfidf_params.get("ngram_max", 1))
    token_pattern = tfidf_params.get("token_pattern") or TFIDF_TOKEN_PATTERN
    stop_words = tfidf_params.get("stop_words", TFIDF_STOP_WORDS)
    if isinstance(stop_words, str) and stop_words.lower() in ("", "none", "null"):
        stop_words = None
    vocab = features.get("vocab")
    idf = features.get("idf")

    if vocab is None or idf is None:
        return None

    try:
        import scipy.sparse as sp
    except Exception as e:
        logger.warning(f"scipy is required for TF-IDF keyword search, fallback to legacy: {e}")
        return None

    class _QueryTfidf:
        def __init__(self, vocabulary, idf_values, ngram_max_val):
            cv_kwargs = {
                "vocabulary": vocabulary,
                "ngram_range": (1, ngram_max_val),
                "dtype": np.int32,
                "lowercase": True,
                "strip_accents": "unicode",
            }
            if token_pattern:
                cv_kwargs["token_pattern"] = token_pattern
            if stop_words:
                cv_kwargs["stop_words"] = stop_words
            self._cv = CountVectorizer(**cv_kwargs)
            self._tfidf = TfidfTransformer(
                norm="l2",
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
            )
            idf_arr = np.asarray(idf_values, dtype=np.float32)
            self._tfidf.idf_ = idf_arr
            self._tfidf._idf_diag = sp.diags(
                idf_arr,
                offsets=0,
                shape=(idf_arr.shape[0], idf_arr.shape[0]),
                format="csr",
            )

        def transform(self, texts):
            counts = self._cv.transform(texts)
            return self._tfidf.transform(counts).astype(np.float32, copy=False)

    _tfidf_query_vectorizer = _QueryTfidf(vocab, idf, ngram_max)
    _tfidf_query_vectorizer_mtime = current_mtime
    return _tfidf_query_vectorizer


def search_rank(q: str = "", limit=None):
    """
    Fast keyword search using TF-IDF dot-product against the precomputed matrix.
    Also runs legacy substring matching and merges results to catch papers
    not in the TF-IDF index (e.g., newly added papers).
    """
    q = (q or "").strip()
    if not q:
        return [], []

    parsed = _parse_search_query(q)

    # If the query contains an explicit arXiv id, return it first (if present).
    # This is the most common and highest-precision paper search behavior.
    mentioned_ids = _extract_arxiv_ids(parsed.get("raw", ""))
    if mentioned_ids:
        for mid in mentioned_ids:
            raw_pid, _ = split_pid_version(mid)
            if paper_exists(raw_pid):
                pid = raw_pid
                # If the user asked for a versioned pid and it exists in db keys, prefer it.
                if mid != raw_pid and paper_exists(mid):
                    pid = mid
                return [pid], [2000.0]

    # Fielded queries (ti:/au:/abs:/cat:/id:) or phrases typically need lexical scoring.
    has_field_filters = any(
        len(parsed.get("filters", {}).get(k, []) or []) > 0 for k in ("ti", "au", "abs", "cat", "id")
    )
    has_phrases = bool(parsed.get("phrases"))
    is_title_like = _is_title_like_query(parsed)

    features = None
    if not has_field_filters:
        try:
            features = get_features_cached()
        except Exception:
            features = None

    cache_key = None
    try:
        cache_key = ("kw_merged", q.lower(), int(limit) if limit is not None else None, FEATURES_FILE_MTIME)
        cached = SEARCH_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    # If query is explicitly fielded, prefer lexical scan (more like academic search)
    if has_field_filters or has_phrases:
        pids, scores = _lexical_rank_fullscan(parsed, limit=limit)
        if cache_key is not None:
            SEARCH_RANK_CACHE.set(cache_key, (pids, scores))
        return pids, scores

    tfidf_pids = []

    # Try TF-IDF candidate generation first (fast, good recall on abstracts)
    if features:
        x_tfidf = features.get("x_tfidf")
        v = _get_query_vectorizer(features) if x_tfidf is not None else None

        if x_tfidf is not None and v is not None:
            try:
                q_vec = v.transform([q])
                if q_vec.nnz > 0:
                    scores_sparse = (x_tfidf @ q_vec.T).tocoo()
                    if scores_sparse.nnz > 0:
                        rows = scores_sparse.row
                        vals = scores_sparse.data
                        # Candidate limit to avoid sorting huge nnz arrays
                        # (keep enough for reranking + pagination buffers)
                        cand_k = 2000
                        if limit is not None:
                            try:
                                cand_k = max(cand_k, int(limit) * 20)
                            except Exception:
                                pass
                        cand_k = min(cand_k, MAX_RESULTS)

                        if scores_sparse.nnz > cand_k:
                            top_ix = np.argpartition(-vals, cand_k - 1)[:cand_k]
                            top_ix = top_ix[np.argsort(-vals[top_ix])]
                        else:
                            top_ix = np.argsort(-vals)

                        tfidf_pids = [features["pids"][int(rows[i])] for i in top_ix]
            except Exception as e:
                logger.warning(f"TF-IDF keyword search failed: {e}")

    # If TF-IDF found too little (or query is CJK-heavy), lexical scan fallback.
    # This avoids "no results" for title/author/category queries when TF-IDF corpus is abstract-only.
    if (not tfidf_pids or len(tfidf_pids) < 30) or _looks_like_cjk_query(parsed.get("raw", "")):
        # Prefer semantic mode in UI for CJK; but for keyword search we fallback to lexical.
        pids, scores = _lexical_rank_fullscan(parsed, limit=limit)
        if cache_key is not None:
            SEARCH_RANK_CACHE.set(cache_key, (pids, scores))
        return pids, scores

    # Safety net for pasted full titles: union in strong title matches from a bounded title scan.
    if is_title_like:
        extra = _title_candidate_scan(parsed, max_candidates=400)
        if extra:
            # Keep TF-IDF order first for stability, then add extras.
            seen = set(tfidf_pids)
            for pid in extra:
                if pid not in seen:
                    tfidf_pids.append(pid)
                    seen.add(pid)

    # Rerank TF-IDF candidates using paper-oriented lexical scoring (title/authors/tags boosted)
    lex_pids, lex_scores = _lexical_rank_over_pids(tfidf_pids, parsed, limit=None)

    # Rank fusion (RRF) is robust across heterogeneous score scales.
    rrf_k = 60
    tfidf_rank = {pid: r for r, pid in enumerate(tfidf_pids)}
    lex_rank = {pid: r for r, pid in enumerate(lex_pids)}

    # Weight lexical slightly higher because it better reflects academic intent.
    w_tfidf = 0.45
    w_lex = 0.55
    combined_scores = {}
    all_ids = set(tfidf_rank.keys()) | set(lex_rank.keys())
    for pid in all_ids:
        tr = tfidf_rank.get(pid)
        lr = lex_rank.get(pid)
        s = 0.0
        if tr is not None:
            s += w_tfidf / (rrf_k + tr)
        if lr is not None:
            s += w_lex / (rrf_k + lr)
        combined_scores[pid] = s

    merged = sorted(all_ids, key=lambda pid: combined_scores.get(pid, 0.0), reverse=True)
    pids = merged
    scores = [combined_scores[pid] * 100.0 for pid in pids]

    pids, scores = _apply_limit(pids, scores, limit)

    if cache_key is not None:
        SEARCH_RANK_CACHE.set(cache_key, (pids, scores))

    return pids, scores


# Global semantic search related variables
_semantic_model = None
_cached_embeddings = None
_cached_embeddings_mtime = 0.0

# Query-side TF-IDF vectorizer cache (reconstructed from features.p)
_tfidf_query_vectorizer = None
_tfidf_query_vectorizer_mtime = 0.0


def get_semantic_model():
    """Get semantic model instance (via API call)"""
    global _semantic_model
    if _semantic_model is None:
        try:
            # Determine API base URL based on configuration
            if EMBED_USE_LLM_API:
                api_base = EMBED_API_BASE if EMBED_API_BASE else LLM_BASE_URL
                api_key = EMBED_API_KEY if EMBED_API_KEY else LLM_API_KEY
            else:
                api_base = f"http://localhost:{EMBED_PORT}"
                api_key = None
            
            api_type = "OpenAI-compatible" if EMBED_USE_LLM_API else "Ollama"
            logger.info(f"Initializing semantic model {api_type} API client for query encoding...")
            _semantic_model = Qwen3EmbeddingVllm(
                model_name_or_path=EMBED_MODEL_NAME,
                instruction="Extract key concepts from this query to search computer science and AI paper",
                api_base=api_base,
                api_key=api_key,
                use_openai_api=EMBED_USE_LLM_API,
            )
            if not _semantic_model.initialize():
                logger.error("Failed to initialize semantic model API client")
                _semantic_model = None
        except Exception as e:
            logger.error(f"Error initializing semantic model API client: {e}")
            _semantic_model = None
    return _semantic_model


def get_paper_embeddings():
    """Get paper embedding vectors (loaded from features file)"""
    global _cached_embeddings, _cached_embeddings_mtime
    # Refresh if features file changed
    try:
        current_mtime = os.path.getmtime(FEATURES_FILE)
    except Exception:
        current_mtime = 0.0
    if _cached_embeddings is not None and _cached_embeddings_mtime == current_mtime:
        return _cached_embeddings

    try:
        features = get_features_cached()
        if features.get("feature_type") == "hybrid_sparse_dense" and "x_embeddings" in features:
            logger.trace("Using pre-computed embeddings from features file")
            embeddings = features["x_embeddings"]
            # Normalize once for fast cosine similarity via dot product
            try:
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32, copy=False)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32, copy=False)
                norms[norms == 0] = 1.0
                embeddings /= norms
            except Exception as e:
                logger.warning(f"Failed to normalize embeddings, fallback to raw vectors: {e}")
            _cached_embeddings = {"embeddings": embeddings, "pids": features["pids"]}
            _cached_embeddings_mtime = current_mtime
            return _cached_embeddings
        else:
            logger.warning("No embeddings found in features file")
            return None
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None


def _get_query_embedding(q: str, embed_dim: int):
    """Get cached query embedding to avoid repeated API calls."""
    key = (q.lower(), int(embed_dim), _cached_embeddings_mtime)
    cached = QUERY_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    model = get_semantic_model()
    if model is None:
        return None

    query_embedding = model.encode([q], dim=embed_dim)
    if query_embedding is None:
        logger.error("Failed to encode query")
        return None

    try:
        if hasattr(query_embedding, "cpu"):
            query_embedding = query_embedding.cpu().numpy()
        else:
            query_embedding = np.asarray(query_embedding)
    except Exception as e:
        logger.error(f"Failed to convert query embedding: {e}")
        return None

    query_vec = query_embedding.reshape(-1).astype(np.float32, copy=False)
    qn = float(np.linalg.norm(query_vec))
    if qn > 0:
        query_vec /= qn

    QUERY_EMBED_CACHE.set(key, query_vec)
    return query_vec


def _warmup_ml_cache():
    try:
        logger.info("Warming features/embeddings/model in background...")
        get_features_cached()
        get_paper_embeddings()
        get_semantic_model()
    except Exception as e:
        logger.warning(f"ML cache warmup failed: {e}")


def semantic_search_rank(q: str = "", limit=None):
    """Execute pure semantic search"""
    if not q:
        return [], []

    q = q.strip()

    # Get paper embeddings
    paper_data = get_paper_embeddings()
    if paper_data is None:
        logger.error("No paper embeddings available")
        return [], []

    try:
        cache_key = ("sem", q.lower(), int(limit) if limit is not None else None, _cached_embeddings_mtime)
        cached = SEARCH_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    try:
        embed_dim = 512
        try:
            embeddings = paper_data.get("embeddings")
            if hasattr(embeddings, "shape") and len(embeddings.shape) >= 2:
                embed_dim = int(embeddings.shape[1])
        except Exception:
            embed_dim = 512
        query_vec = _get_query_embedding(q, embed_dim)
        if query_vec is None:
            logger.error("Semantic query embedding unavailable")
            return [], []

        similarities = paper_data["embeddings"] @ query_vec  # (n_papers,)

        # Get top-k results
        if limit:
            k = min(int(limit), len(similarities))
            top_indices = np.argpartition(-similarities, min(k, len(similarities) - 1))[:k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)

        pids = [paper_data["pids"][i] for i in top_indices]
        scores = [float(similarities[i]) * 100 for i in top_indices]

        if cache_key is not None:
            SEARCH_RANK_CACHE.set(cache_key, (pids, scores))

        return pids, scores

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return [], []


def hybrid_search_rank(q: str = "", limit=None, semantic_weight=SUMMARY_DEFAULT_SEMANTIC_WEIGHT):
    """
    Hybrid search: weighted Reciprocal Rank Fusion (RRF) of keyword + semantic rankings.

    - More robust than raw score normalization across heterogeneous scoring functions.
    - semantic_weight (0..1) controls the relative contribution of semantic vs keyword ranks.
    """
    q = (q or "").strip()
    if not q:
        return [], [], {}

    try:
        cache_key = (
            "hyb",
            q.lower(),
            int(limit) if limit is not None else None,
            float(semantic_weight),
            FEATURES_FILE_MTIME,
            _cached_embeddings_mtime,
        )
        cached = SEARCH_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    candidate_k = None
    if limit is not None:
        candidate_k = min(int(limit) * 2, MAX_RESULTS)

    keyword_pids, keyword_scores = search_rank(q, limit=candidate_k)
    semantic_pids, semantic_scores = semantic_search_rank(q, limit=candidate_k)

    if not keyword_pids and not semantic_pids:
        return [], [], {}
    if not semantic_pids:
        return keyword_pids[:limit] if limit else keyword_pids, keyword_scores[:limit] if limit else keyword_scores, {}
    if not keyword_pids:
        return (
            semantic_pids[:limit] if limit else semantic_pids,
            semantic_scores[:limit] if limit else semantic_scores,
            {},
        )

    keyword_rank = {pid: i + 1 for i, pid in enumerate(keyword_pids)}
    semantic_rank = {pid: i + 1 for i, pid in enumerate(semantic_pids)}

    keyword_score_map = {pid: score for pid, score in zip(keyword_pids, keyword_scores)}
    semantic_score_map = {pid: score for pid, score in zip(semantic_pids, semantic_scores)}

    rrf_k = 60  # standard RRF constant
    w_sem = float(np.clip(semantic_weight, 0.0, 1.0))
    w_kw = 1.0 - w_sem

    combined_scores = {}
    all_pids = set(keyword_rank.keys()) | set(semantic_rank.keys())

    for pid in all_pids:
        kr = keyword_rank.get(pid)
        sr = semantic_rank.get(pid)
        kw_rrf = (w_kw / (rrf_k + kr)) if kr is not None else 0.0
        sem_rrf = (w_sem / (rrf_k + sr)) if sr is not None else 0.0
        final = kw_rrf + sem_rrf
        combined_scores[pid] = final

    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    if limit:
        sorted_results = sorted_results[: int(limit)]

    pids = [pid for pid, _ in sorted_results]
    scores = [combined_scores[pid] * 100.0 for pid in pids]

    score_details = {}
    for pid in pids:
        kr = keyword_rank.get(pid)
        sr = semantic_rank.get(pid)
        kw_rrf = (w_kw / (rrf_k + kr)) if kr is not None else 0.0
        sem_rrf = (w_sem / (rrf_k + sr)) if sr is not None else 0.0
        final = kw_rrf + sem_rrf
        score_details[pid] = {
            "keyword_score": float(keyword_score_map.get(pid, 0.0)),
            "semantic_score": float(semantic_score_map.get(pid, 0.0)),
            "keyword_rank": int(kr) if kr is not None else None,
            "semantic_rank": int(sr) if sr is not None else None,
            "keyword_rrf": float(kw_rrf),
            "semantic_rrf": float(sem_rrf),
            "keyword_weight": w_kw,
            "semantic_weight": w_sem,
            "final_score": float(final * 100.0),
        }

    result = (pids, scores, score_details)
    if cache_key is not None:
        SEARCH_RANK_CACHE.set(cache_key, result)
    return result


def enhanced_search_rank(
    q: str = "", limit=None, search_mode="keyword", semantic_weight=SUMMARY_DEFAULT_SEMANTIC_WEIGHT
):
    """
    Enhanced search function supporting multiple search modes

    Args:
        q: Search query
        limit: Result count limit
        search_mode: Search mode ('keyword', 'semantic', 'hybrid')
        semantic_weight: Semantic search weight (0-1), only used in hybrid mode

    Returns:
        (paper_ids, scores) tuple
    """
    if not q:
        return [], []

    if search_mode == "keyword":
        # Use existing keyword search
        return search_rank(q, limit)

    elif search_mode == "semantic":
        # Pure semantic search
        return semantic_search_rank(q, limit)

    elif search_mode == "hybrid":
        # Hybrid search
        return hybrid_search_rank(q, limit, semantic_weight)

    else:
        raise ValueError(f"Unknown search mode: {search_mode}")


# -----------------------------------------------------------------------------
# primary application endpoints


def default_context():
    # any global context across all pages, e.g. related to the current user
    context = {}
    context["user"] = g.user if g.user is not None else ""
    context["csrf_token"] = _get_or_set_csrf_token()
    return context


@app.route("/", methods=["GET"])
def main():
    # default settings
    default_rank = "time"
    default_tags = ""
    default_time_filter = ""
    default_skip_have = "no"
    default_logic = "and"
    if not getattr(app, "_logger_configured", False):
        try:
            level = os.environ.get("ARXIV_SANITY_LOG_LEVEL", "INFO").upper()
        except Exception:
            level = "INFO"
        logger.remove()
        logger.add(sys.stdout, level=level)
        setattr(app, "_logger_configured", True)

    # override variables with any provided options via the interface
    form_errors = []
    allowed_ranks = {"search", "tags", "pid", "time", "random"}
    allowed_logic = {"and", "or"}
    allowed_search_modes = {"keyword", "semantic", "hybrid"}
    allowed_skip_have = {"yes", "no"}

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

    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if opt_q:
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
                q=opt_q, limit=dynamic_limit, search_mode=opt_search_mode, semantic_weight=semantic_weight
            )
        else:
            pids, scores = enhanced_search_rank(
                q=opt_q, limit=dynamic_limit, search_mode=opt_search_mode, semantic_weight=semantic_weight
            )
        logger.info(
            f"User {g.user} {opt_search_mode} search '{opt_q}' weight={semantic_weight}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "tags":
        t_s = time.time()
        pids, scores, words = svm_rank(
            tags=opt_tags, C=C, logic=opt_logic, time_filter=opt_time_filter, limit=dynamic_limit
        )
        logger.info(
            f"User {g.user} tags {opt_tags} C {C} logic {opt_logic} time_filter {opt_time_filter}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "pid":
        t_s = time.time()
        pids, scores, words = svm_rank(
            s_pids=opt_pid, C=C, logic=opt_logic, time_filter=opt_time_filter, limit=dynamic_limit
        )
        logger.info(
            f"User {g.user} pid {opt_pid} C {C} logic {opt_logic} time_filter {opt_time_filter}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "time":
        t_s = time.time()
        pids, scores = time_rank(limit=dynamic_limit)
        logger.info(f"User {g.user} time rank, time {time.time() - t_s:.3f}s")
    elif opt_rank == "random":
        t_s = time.time()
        pids, scores = random_rank(limit=dynamic_limit)
        logger.info(f"User {g.user} random rank, time {time.time() - t_s:.3f}s")
    else:
        raise ValueError(f"opt_rank {opt_rank} is not a thing")

    # filter by time (now handled within svm_rank for SVM-based rankings)
    if opt_time_filter and opt_rank not in ["tags", "pid"]:
        # Collect user tagged papers for intelligent time filtering
        user_tagged_pids = set()
        tags = get_tags()
        for tag_pids in tags.values():
            user_tagged_pids.update(tag_pids)

        # Use intelligent time filtering
        pids, time_valid_indices = _filter_by_time_with_tags(pids, opt_time_filter, user_tagged_pids)
        scores = [scores[i] for i in time_valid_indices]

    # optionally hide papers we already have
    if opt_skip_have == "yes":
        tags = get_tags()
        have = set().union(*tags.values())
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
    if g.user:
        try:
            user_tags = get_tags()
            pid_set = set(pids)
            pid_to_utags = {pid: [] for pid in pid_set}
            for tag, tag_pids in user_tags.items():
                for pid in pid_set.intersection(tag_pids):
                    pid_to_utags[pid].append(tag)
        except Exception:
            pid_to_utags = None

    pid_to_paper = get_papers_bulk(pids)
    papers = [render_pid(pid, pid_to_utags=pid_to_utags, paper=pid_to_paper.get(pid)) for pid in pids]
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

    # build the current tags for the user, and append the special 'all' tag
    tags = get_tags()
    rtags = [{"name": t, "n": len(pids)} for t, pids in tags.items()]
    if rtags:
        rtags.append({"name": "all"})

    keys = get_keys()
    rkeys = [{"name": k} for k, pids in keys.items()]
    rkeys.append({"name": "Artificial general intelligence"})

    combined_tags = get_combined_tags()
    rctags = [{"name": k} for k in combined_tags]

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
    logger.trace(
        f'User: {context["user"]}\ntags {context["tags"]}\nkeys {context["keys"]}\nctags {context["combined_tags"]}'
    )
    return render_template("index.html", **context)


@app.route("/inspect", methods=["GET"])
def inspect():
    # fetch the paper of interest based on the pid
    pid = request.args.get("pid", "")
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


@app.route("/summary", methods=["GET"])
def summary():
    """
    Display AI-generated markdown format summary of the paper.

    If a versioned PID is provided (e.g., 2512.21789v1), redirect to the
    unversioned URL (/summary?pid=2512.21789) since we only cache the latest version.
    """
    # Get paper ID
    pid = request.args.get("pid", "")
    logger.info(f"show paper summary page for paper {pid}")
    raw_pid, version = split_pid_version(pid)

    # If versioned PID provided, redirect to unversioned URL
    if version is not None:
        return redirect(url_for("summary", pid=raw_pid), code=302)

    if not paper_exists(raw_pid):
        return f"<h1>Error</h1><p>Paper with ID '{pid}' not found in database.</p>", 404

    # Get basic paper information with user tags
    pid_to_utags = None
    if g.user:
        try:
            user_tags = get_tags()
            pid_to_utags = {raw_pid: []}
            for tag, tag_pids in user_tags.items():
                if raw_pid in tag_pids:
                    pid_to_utags[raw_pid].append(tag)
        except Exception:
            pid_to_utags = None

    paper = render_pid(raw_pid, pid_to_utags=pid_to_utags)

    # Build the current tags for the user (same as main page)
    tags = get_tags() if g.user else {}
    rtags = [{"name": t, "n": len(pids)} for t, pids in tags.items()]
    if rtags:
        rtags.append({"name": "all"})

    # Build page context, don't call get_paper_summary here
    context = default_context()
    context["paper"] = paper
    context["pid"] = raw_pid  # Always use raw_pid (unversioned)
    context["default_summary_model"] = LLM_NAME or ""
    context["tags"] = sorted(rtags, key=lambda item: item["name"]) if rtags else []
    # Add paper name to page title
    context["title"] = f"Paper Summary - {paper['title']}"
    return render_template("summary.html", **context)


@app.route("/api/get_paper_summary", methods=["POST"])
def api_get_paper_summary():
    """
    API endpoint: Get paper summary asynchronously
    """
    try:
        _csrf_protect()
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        pid = data.get("pid", "").strip()
        model = (data.get("model") or "").strip()
        if not model:
            model = (LLM_NAME or "").strip()
        if not model:
            return jsonify({"success": False, "error": "Model is required"}), 400
        force_regen = bool(data.get("force", False) or data.get("force_regenerate", False))
        cache_only = bool(data.get("cache_only", False))
        if not pid:
            return jsonify({"success": False, "error": "Paper ID is required"}), 400

        logger.info(f"Getting paper summary for: {pid}")

        # Check if paper exists
        raw_pid, _ = split_pid_version(pid)
        if not paper_exists(raw_pid):
            return jsonify({"success": False, "error": "Paper not found"}), 404

        # Generate paper summary and meta
        summary_content, summary_meta = generate_paper_summary(
            pid, model=model, force_refresh=force_regen, cache_only=cache_only
        )

        return jsonify(
            {
                "success": True,
                "pid": pid,
                "summary_content": summary_content,
                "summary_meta": _public_summary_meta(summary_meta),
            }
        )

    except SummaryCacheMiss as e:
        return jsonify({"success": False, "error": str(e), "code": "summary_cache_miss"}), 404
    except Exception as e:
        logger.error(f"Paper summary API error: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


@app.route("/api/clear_model_summary", methods=["POST"])
def api_clear_model_summary():
    """
    API endpoint: Clear summary for a specific model only
    """
    try:
        _csrf_protect()
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        pid = data.get("pid", "").strip()
        model = data.get("model", "").strip()

        if not pid:
            return jsonify({"success": False, "error": "Paper ID is required"}), 400
        if not model:
            return jsonify({"success": False, "error": "Model name is required"}), 400

        raw_pid, _ = split_pid_version(pid)
        if not paper_exists(raw_pid):
            return jsonify({"success": False, "error": "Paper not found"}), 404

        _clear_model_summary(pid, model)
        return jsonify({"success": True, "pid": pid, "model": model})

    except Exception as e:
        logger.error(f"Failed to clear model summary: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


@app.route("/api/clear_paper_cache", methods=["POST"])
def api_clear_paper_cache():
    """
    API endpoint: Clear all caches for a paper (all models, HTML, MinerU, etc.)
    """
    try:
        _csrf_protect()
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        pid = data.get("pid", "").strip()
        if not pid:
            return jsonify({"success": False, "error": "Paper ID is required"}), 400

        raw_pid, _ = split_pid_version(pid)
        if not paper_exists(raw_pid):
            return jsonify({"success": False, "error": "Paper not found"}), 404

        _clear_paper_cache(pid)
        return jsonify({"success": True, "pid": pid})

    except Exception as e:
        logger.error(f"Failed to clear paper cache: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


@app.route("/api/paper_image/<pid>/<filename>", methods=["GET"])
def api_paper_image(pid: str, filename: str):
    """
    Serve paper images from HTML markdown cache.

    Args:
        pid: Paper ID (e.g., "2301.00001" or "2301.00001v1" - version stripped)
        filename: Image filename

    Returns:
        Image file or 404 error
    """
    try:
        # Sanitize inputs to prevent path traversal
        pid = pid.strip()
        filename = filename.strip()

        if not pid or not filename:
            abort(400, "Invalid paper ID or filename")

        # Prevent path traversal attacks
        if ".." in pid or ".." in filename or "/" in filename or "\\" in filename:
            abort(400, "Invalid path")

        # Use raw PID only (strip version if present)
        raw_pid, _ = split_pid_version(pid)

        # Construct image path directly
        image_path = Path("data/html_md") / raw_pid / "images" / filename

        if not image_path.exists():
            abort(404, f"Image not found: {filename}")

        # Determine MIME type based on extension
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
        }
        mimetype = mime_types.get(suffix, "application/octet-stream")

        return send_file(image_path, mimetype=mimetype)

    except Exception as e:
        if hasattr(e, "code"):  # Flask abort exception
            raise
        logger.error(f"Failed to serve paper image: {e}")
        abort(500, f"Server error: {str(e)}")


@app.route("/api/mineru_image/<pid>/<filename>", methods=["GET"])
def api_mineru_image(pid: str, filename: str):
    """
    Serve paper images from MinerU parsed cache.

    Args:
        pid: Paper ID (e.g., "2301.00001" or "2301.00001v1" - version stripped)
        filename: Image filename

    Returns:
        Image file or 404 error
    """
    try:
        # Sanitize inputs to prevent path traversal
        pid = pid.strip()
        filename = filename.strip()

        if not pid or not filename:
            abort(400, "Invalid paper ID or filename")

        # Prevent path traversal attacks
        if ".." in pid or ".." in filename or "/" in filename or "\\" in filename:
            abort(400, "Invalid path")

        # Use raw PID only (strip version if present)
        raw_pid, _ = split_pid_version(pid)

        # Construct image path - MinerU stores images in {pid}/auto/images/, {pid}/vlm/images/, or {pid}/api/images/
        # Try all backends
        mineru_base = Path("data/mineru") / raw_pid
        image_path = None
        for backend_dir in ["auto", "vlm", "api"]:
            candidate = mineru_base / backend_dir / "images" / filename
            if candidate.exists():
                image_path = candidate
                break

        if not image_path:
            abort(404, f"Image not found: {filename}")

        # Determine MIME type based on extension
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
        }
        mimetype = mime_types.get(suffix, "application/octet-stream")

        return send_file(image_path, mimetype=mimetype)

    except Exception as e:
        if hasattr(e, "code"):  # Flask abort exception
            raise
        logger.error(f"Failed to serve MinerU image: {e}")
        abort(500, f"Server error: {str(e)}")


@app.route("/api/llm_models", methods=["GET"])
def api_llm_models():
    try:
        base_url = (LLM_BASE_URL or "").rstrip("/")
        if not base_url:
            models = [{"id": LLM_NAME}] if LLM_NAME else []
            return jsonify(
                {
                    "success": False,
                    "error": "LLM base URL is not configured",
                    "models": models,
                }
            )

        headers = {}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        models = []
        for item in payload.get("data", []):
            mid = item.get("id")
            if mid:
                models.append(item)
        if not models and LLM_NAME:
            models = [{"id": LLM_NAME}]
        return jsonify({"success": True, "models": models})
    except Exception as e:
        logger.error(f"Failed to fetch LLM models: {e}")
        models = [{"id": LLM_NAME}] if LLM_NAME else []
        return jsonify({"success": False, "error": "Failed to load model list", "models": models})


@app.route("/api/check_paper_summaries", methods=["GET"])
def api_check_paper_summaries():
    """
    API endpoint: Check which models have existing summaries for a paper
    Returns a list of model names that have cached summaries
    """
    try:
        pid = request.args.get("pid", "").strip()
        if not pid:
            return jsonify({"success": False, "error": "Paper ID is required"}), 400

        raw_pid, _ = split_pid_version(pid)
        if not paper_exists(raw_pid):
            return jsonify({"success": False, "error": "Paper not found"}), 404

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

        return jsonify({"success": True, "available_models": available_models})

    except Exception as e:
        logger.error(f"Check paper summaries API error: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


def _write_summary_meta(meta_path: Path, data: dict) -> None:
    try:
        atomic_write_json(meta_path, data)
    except Exception as e:
        logger.warning(f"Failed to write summary meta: {e}")


def _public_summary_meta(meta: dict) -> dict:
    """Filter summary metadata for client responses."""
    if not isinstance(meta, dict):
        return {}
    allowed = ("generated_at", "source", "llm")
    return {key: meta.get(key) for key in allowed if meta.get(key) is not None}


def _sanitize_summary_meta(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {}
    clean = dict(meta)
    clean.pop("prompt", None)
    clean.pop("updated_at", None)
    clean.pop("quality", None)
    clean.pop("chinese_ratio", None)
    clean.pop("model", None)
    return clean


class SummaryCacheMiss(Exception):
    pass


def generate_paper_summary(
    pid: str, model: Optional[str] = None, force_refresh: bool = False, cache_only: bool = False
):
    """
    Generate paper summary with intelligent caching mechanism
    1. Check if SUMMARY_DIR/{pid}.md exists
    2. If exists, return cached summary (unless force_refresh=True)
    3. If not exists, call paper_summarizer to generate and cache the summary
    """
    try:
        # Use shared resolve_cache_pid with local meta lookup
        meta = get_metas().get(pid.split("v")[0] if "v" in pid else pid) if pid else None
        cache_pid, raw_pid, has_explicit_version = resolve_cache_pid(pid, meta)
        if not paper_exists(raw_pid):
            return "# Error\n\nPaper not found.", {}

        summary_source = normalize_summary_source(SUMMARY_MARKDOWN_SOURCE)

        # Define cache file path
        cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(cache_pid, model)
        if legacy_cache.exists() and not cache_file.exists():
            lock_file = legacy_lock
        # Quality/chinese_ratio based retry removed; always reuse cache when available.

        # Ensure cache directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        def _read_from_paths(body_path: Path, meta_path: Path, inject_model: bool = True):
            if not body_path.exists():
                return None, {}
            try:
                with open(body_path, encoding="utf-8") as f:
                    cached = f.read()
                cached = cached if cached.strip() else None
                meta = read_summary_meta(meta_path)
                # Backfill generated_at for old caches without mutating meaning.
                if "generated_at" not in meta:
                    # Prefer legacy updated_at if present
                    ga = meta.get("updated_at")
                    if ga is None:
                        try:
                            ga = meta_path.stat().st_mtime
                        except Exception:
                            try:
                                ga = body_path.stat().st_mtime
                            except Exception:
                                ga = None
                    if ga is not None:
                        meta["generated_at"] = ga
                if "source" not in meta:
                    meta["source"] = summary_source
                if not cached:
                    return None, {}
                return cached, meta
            except Exception as e:
                logger.error(f"Failed to read cached summary: {e}")
                return None, {}

        def _read_cached_summary():
            cached, meta = _read_from_paths(cache_file, meta_file)
            if cached:
                return cached, meta

            legacy_cached, legacy_meta_data = _read_from_paths(legacy_cache, legacy_meta, inject_model=False)
            if not legacy_cached:
                return None, {}

            legacy_model = (legacy_meta_data.get("model") or "").strip()
            if model and (not legacy_model or legacy_model != model):
                return None, {}
            return legacy_cached, legacy_meta_data

        # Check if cached summary exists (must match current markdown source)
        cached_summary, cached_meta = _read_cached_summary()
        if cached_summary and not force_refresh:
            if not summary_source_matches(cached_meta, summary_source):
                cached_summary = None
            else:
                logger.info(f"Using cached paper summary: {pid}")
                return cached_summary, _sanitize_summary_meta(cached_meta)

        if cache_only:
            # If another request is generating the same cache (lock held), tell client.
            # This lets the UI show a non-blocking "generating" state when switching models.
            probe_fd = None
            try:
                probe_fd = acquire_summary_lock(lock_file, timeout_s=1)
            except Exception:
                probe_fd = None
            if probe_fd is None:
                return "# Error\n\nSummary is being generated, please retry shortly.", {}
            try:
                release_summary_lock(probe_fd, lock_file)
            except Exception:
                pass
            raise SummaryCacheMiss("Summary cache not found")

        lock_fd = acquire_summary_lock(lock_file, timeout_s=300)
        if lock_fd is None:
            if cached_summary and not force_refresh:
                return cached_summary, _sanitize_summary_meta(cached_meta)
            return "# Error\n\nSummary is being generated, please retry shortly.", {}

        try:
            cached_summary, cached_meta = _read_cached_summary()
            if cached_summary and not force_refresh:
                if not summary_source_matches(cached_meta, summary_source):
                    cached_summary = None
                else:
                    logger.info(f"Using cached paper summary after lock: {pid}")
                    return cached_summary, _sanitize_summary_meta(cached_meta)

            if cache_only:
                raise SummaryCacheMiss("Summary cache not found")

            # Generate new summary using paper_summarizer module
            logger.info(f"Generating new paper summary: {pid}")
            if summary_source == "html":
                pid_for_summary = pid if has_explicit_version else cache_pid
            else:
                pid_for_summary = pid if has_explicit_version else raw_pid

            summary_result = generate_paper_summary_from_module(pid_for_summary, source=summary_source, model=model)
            summary_content, summary_meta = normalize_summary_result(summary_result)
            summary_meta = summary_meta if isinstance(summary_meta, dict) else {}
            response_meta = _sanitize_summary_meta(summary_meta)

            # Only cache successful summaries (not error messages)
            # Check for various error message formats
            is_error = summary_content.startswith("# Error") or summary_content.startswith(
                "# PDF Parsing Service Unavailable"
            )

            if not is_error:
                try:
                    atomic_write_text(cache_file, summary_content)
                    meta = {}
                    meta.update(summary_meta)
                    # Ensure source exists for cache checks; keep detailed value if provided
                    meta.setdefault("source", summary_source)
                    meta.setdefault("generated_at", time.time())
                    _write_summary_meta(meta_file, meta)
                    logger.info(f"Paper summary cached to: {cache_file}")
                    response_meta = _sanitize_summary_meta(meta)
                except Exception as e:
                    logger.error(f"Failed to cache paper summary: {e}")
            else:
                logger.warning(f"Summary generation failed, not caching: {pid}")

            return summary_content, response_meta
        finally:
            release_summary_lock(lock_fd, lock_file)

    except SummaryCacheMiss:
        raise
    except Exception as e:
        logger.error(f"Error occurred while generating paper summary: {e}")
        return f"# Error\n\nFailed to generate summary: {str(e)}", {}


def _clear_model_summary(pid: str, model: str):
    """
    Clear summary cache for a specific model only.

    Args:
        pid: Paper ID (with or without version)
        model: Model name to clear summary for
    """
    meta = get_metas().get(pid.split("v")[0] if "v" in pid else pid) if pid else None
    cache_pid, raw_pid, _ = resolve_cache_pid(pid, meta)
    ids_to_clear = {cache_pid, raw_pid}

    model = (model or "").strip()
    cleared = False
    removed_paths = []

    for paper_id in ids_to_clear:
        cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(paper_id, model)

        # Primary per-model cache (dir-based)
        for file_path in (cache_file, meta_file, lock_file):
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_paths.append(str(file_path))
                    cleared = True
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")

        # Legacy flat cache (single file) only applies if it matches the requested model
        if legacy_meta.exists() and legacy_cache.exists():
            try:
                legacy_meta_data = read_summary_meta(legacy_meta)
            except Exception:
                legacy_meta_data = {}

            legacy_model = (legacy_meta_data.get("model") or "").strip()
            if legacy_model and legacy_model == model:
                for file_path in (legacy_cache, legacy_meta, legacy_lock):
                    if file_path.exists():
                        try:
                            file_path.unlink()
                            removed_paths.append(str(file_path))
                            cleared = True
                        except Exception as e:
                            logger.warning(f"Failed to remove legacy file {file_path}: {e}")

    if cleared:
        logger.info(f"Cleared summary for model '{model}' for paper {pid}")
        for p in removed_paths:
            logger.trace(f"Removed: {p}")
    else:
        logger.info(f"No summary cache found for model '{model}' for paper {pid}")


def _clear_paper_cache(pid: str):
    meta = get_metas().get(pid.split("v")[0] if "v" in pid else pid) if pid else None
    cache_pid, raw_pid, _ = resolve_cache_pid(pid, meta)
    ids_to_clear = {cache_pid, raw_pid}

    # Clear summary caches (all models) including legacy flat files
    for paper_id in ids_to_clear:
        summary_dir = Path(SUMMARY_DIR) / paper_id
        if summary_dir.exists():
            try:
                shutil.rmtree(summary_dir)
            except Exception as e:
                logger.warning(f"Failed to remove summary cache dir {summary_dir}: {e}")

        legacy_md = Path(SUMMARY_DIR) / f"{paper_id}.md"
        legacy_meta = Path(SUMMARY_DIR) / f"{paper_id}.meta.json"
        legacy_lock = Path(SUMMARY_DIR) / f".{paper_id}.lock"
        for path in (legacy_md, legacy_meta, legacy_lock):
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove legacy file {path}: {e}")

    # Clear HTML markdown caches
    for paper_id in ids_to_clear:
        html_dir = Path("data/html_md") / paper_id
        if html_dir.exists():
            try:
                shutil.rmtree(html_dir)
                logger.trace(f"Cleared HTML cache: {html_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove HTML cache {html_dir}: {e}")

    # Clear MinerU caches
    for paper_id in ids_to_clear:
        mineru_dir = Path("data/mineru") / paper_id
        if mineru_dir.exists():
            try:
                shutil.rmtree(mineru_dir)
                logger.trace(f"Cleared MinerU cache: {mineru_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove MinerU cache {mineru_dir}: {e}")


def _find_actual_cache_path(base_dir: Path, pid: str, meta: Optional[dict] = None) -> Optional[Path]:
    """
    Find the cache directory for a paper ID using raw PID.

    Simplified approach: arXiv/ar5iv automatically return latest version,
    so we always store with raw PID only.

    Args:
        base_dir: Base directory to search in (e.g., Path("data/html_md"))
        pid: Paper ID (version suffix will be stripped if present)
        meta: Unused, kept for backward compatibility

    Returns:
        Path to cache directory if found, None otherwise
    """
    raw_pid, _ = split_pid_version(pid)
    cache_dir = base_dir / raw_pid
    return cache_dir if cache_dir.exists() else None


@app.route("/profile")
def profile():
    context = default_context()
    with get_email_db() as edb:
        email = edb.get(g.user, "")
        context["email"] = email
    return render_template("profile.html", **context)


@app.route("/stats")
def stats():
    context = default_context()
    mdb = get_metas()
    times = [v["_time"] for v in mdb.values()]
    tstr = lambda t: time.strftime("%b %d %Y", time.localtime(t))

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

    return render_template("stats.html", **context)


@app.route("/about")
def about():
    context = default_context()
    return render_template("about.html", **context)


# -----------------------------------------------------------------------------
# Helper for API endpoints
# -----------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _temporary_user_context(user):
    """Context manager to temporarily set g.user and g._tags for API calls"""
    original_user = getattr(g, "user", None)
    original_tags = getattr(g, "_tags", None)

    try:
        # Get user tags
        with get_tags_db() as tags_db:
            user_tags = tags_db.get(user, {})

        # Set temporary context
        g.user = user
        g._tags = user_tags

        yield user_tags

    finally:
        # Restore original context
        if original_user is not None:
            g.user = original_user
        else:
            if hasattr(g, "user"):
                delattr(g, "user")
        if original_tags is not None:
            g._tags = original_tags
        else:
            if hasattr(g, "_tags"):
                delattr(g, "_tags")


# -----------------------------------------------------------------------------
# API endpoints for email recommendation system
# -----------------------------------------------------------------------------


@app.route("/api/keyword_search", methods=["POST"])
def api_keyword_search():
    """API interface: single keyword search"""
    try:
        data = request.get_json()
        # logger.info(f"API keyword search data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        keyword = data.get("keyword", "")
        time_delta = data.get("time_delta", 3)  # days
        limit = data.get("limit", 50)

        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400

        try:
            limit = min(int(limit), MAX_RESULTS)
        except Exception:
            limit = 50
        if limit <= 0:
            return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})
        try:
            time_delta = float(time_delta)
        except Exception:
            return jsonify({"error": "time_delta must be a number"}), 400

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

        # Limit final result count
        if len(pids) > limit:
            pids = pids[:limit]
            scores = scores[:limit]
        # logger.trace(f"API keyword search results: {len(pids)} papers found")
        return jsonify({"success": True, "pids": pids, "scores": scores, "total_count": len(pids)})

    except Exception as e:
        logger.error(f"API keyword search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tag_search", methods=["POST"])
def api_tag_search():
    """API interface: single tag recommendation"""
    try:
        data = request.get_json()
        logger.trace(f"API tag search data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        tag_name = data.get("tag_name", "")  # tag name
        user = data.get("user", "")  # username
        time_delta = data.get("time_delta", 3)  # days
        limit = data.get("limit", 50)
        C = data.get("C", 0.1)

        if not tag_name:
            return jsonify({"error": "tag_name is required"}), 400
        if not user:
            return jsonify({"error": "user is required"}), 400
        try:
            limit = min(int(limit), MAX_RESULTS)
        except Exception:
            return jsonify({"error": "limit must be an integer"}), 400
        if limit <= 0:
            return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})
        try:
            time_delta = float(time_delta)
        except Exception:
            return jsonify({"error": "time_delta must be a number"}), 400

        with _temporary_user_context(user) as user_tags:
            # Check if user has this tag
            if tag_name not in user_tags or len(user_tags[tag_name]) == 0:
                logger.warning(f"User {user} has no papers tagged with '{tag_name}'")
                return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})

            logger.trace(f"User {user} has {len(user_tags[tag_name])} papers tagged with '{tag_name}'")

            # Use tag name for recommendation
            try:
                svm_limit = min(int(limit) * 5, MAX_RESULTS)
            except Exception:
                svm_limit = MAX_RESULTS

            rec_pids, rec_scores, words = svm_rank(
                tags=tag_name, s_pids="", C=C, logic="and", time_filter=str(time_delta), limit=svm_limit
            )

            logger.trace(f"svm_rank returned {len(rec_pids)} results before filtering")

            tagged_set = set(user_tags.get(tag_name, set()))
            if tagged_set:
                keep = [i for i, pid in enumerate(rec_pids) if pid not in tagged_set]
                rec_pids = [rec_pids[i] for i in keep]
                rec_scores = [rec_scores[i] for i in keep]

        # Limit result count
        if len(rec_pids) > limit:
            rec_pids = rec_pids[:limit]
            rec_scores = rec_scores[:limit]
        logger.trace(f"API tag search results: {len(rec_pids)} papers found")
        return jsonify({"success": True, "pids": rec_pids, "scores": rec_scores, "total_count": len(rec_pids)})

    except Exception as e:
        logger.error(f"API tag search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tags_search", methods=["POST"])
def api_tags_search():
    """API interface: joint tag recommendation"""
    try:
        data = request.get_json()
        logger.trace(f"API combined tags search data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        tags_list = data.get("tags", [])  # List[tag_name]
        user = data.get("user", "")  # username
        logic = data.get("logic", "and")  # and|or
        time_delta = data.get("time_delta", 3)  # days
        limit = data.get("limit", 50)
        C = data.get("C", 0.1)

        if not tags_list:
            return jsonify({"error": "Tags list is required"}), 400
        if not user:
            return jsonify({"error": "user is required"}), 400
        try:
            limit = min(int(limit), MAX_RESULTS)
        except Exception:
            return jsonify({"error": "limit must be an integer"}), 400
        if limit <= 0:
            return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})
        try:
            time_delta = float(time_delta)
        except Exception:
            return jsonify({"error": "time_delta must be a number"}), 400

        with _temporary_user_context(user) as user_tags:
            # Check if user has any of the tags
            valid_tags = [tag for tag in tags_list if tag in user_tags and len(user_tags[tag]) > 0]
            if not valid_tags:
                logger.warning(f"User {user} has no papers tagged with any of {tags_list}")
                return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})

            # Convert tag list to comma-separated string
            tags_str = ",".join(valid_tags)

            try:
                svm_limit = min(int(limit) * 5, MAX_RESULTS)
            except Exception:
                svm_limit = MAX_RESULTS

            rec_pids, rec_scores, words = svm_rank(
                tags=tags_str, s_pids="", C=C, logic=logic, time_filter=str(time_delta), limit=svm_limit
            )

            # Exclude papers already tagged by the user under any of the involved tags
            all_tagged = set()
            for tag in valid_tags:
                all_tagged.update(user_tags.get(tag, set()))

            keep = [i for i, pid in enumerate(rec_pids) if pid not in all_tagged]
            rec_pids = [rec_pids[i] for i in keep]
            rec_scores = [rec_scores[i] for i in keep]

        # Limit result count
        if len(rec_pids) > limit:
            rec_pids = rec_pids[:limit]
            rec_scores = rec_scores[:limit]

        logger.trace(f"API combined tags search results: {len(rec_pids)} papers found")

        return jsonify({"success": True, "pids": rec_pids, "scores": rec_scores, "total_count": len(rec_pids)})

    except Exception as e:
        logger.error(f"API tags search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/cache_status")
def cache_status():
    """Debug endpoint to display cache status"""
    if not _truthy_env("ARXIV_SANITY_ENABLE_CACHE_STATUS", "0"):
        abort(404)
    if not g.user:
        return "Access denied"

    global FEATURES_CACHE, FEATURES_FILE_MTIME, FEATURES_CACHE_TIME
    global PAPERS_CACHE, METAS_CACHE, PIDS_CACHE, PAPERS_DB_FILE_MTIME, PAPERS_DB_CACHE_TIME

    # Check backend service status
    def check_http_service(port, path, service_name):
        """Check if a local HTTP service is available"""
        try:
            import requests

            logger.trace(f"Checking {service_name} at port {port}...")
            sess = requests.Session()
            sess.trust_env = False
            response = sess.get(f"http://localhost:{port}{path}", timeout=2)
            is_available = response.status_code == 200
            if is_available:
                logger.success(f"{service_name} is available (status: {response.status_code})")
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
            "cached": FEATURES_CACHE is not None,
            "cache_time": FEATURES_CACHE_TIME,
            "file_mtime": FEATURES_FILE_MTIME,
        },
        "papers_and_metas": {
            "papers_cached": PAPERS_CACHE is not None,
            "metas_cached": METAS_CACHE is not None,
            "cache_time": PAPERS_DB_CACHE_TIME,
            "db_file_mtime": PAPERS_DB_FILE_MTIME,
        },
        "backend_services": backend_services,
    }

    # Features details
    if FEATURES_CACHE:
        status["features"].update(
            {
                "cache_age_seconds": time.time() - FEATURES_CACHE_TIME,
                "feature_shape": str(FEATURES_CACHE["x"].shape),
                "num_papers": len(FEATURES_CACHE["pids"]),
                "vocab_size": len(FEATURES_CACHE["vocab"]),
            }
        )

    if os.path.exists(FEATURES_FILE):
        current_mtime = os.path.getmtime(FEATURES_FILE)
        status["features"].update(
            {
                "file_exists": True,
                "current_file_mtime": current_mtime,
                "file_is_newer": current_mtime > FEATURES_FILE_MTIME,
            }
        )
    else:
        status["features"]["file_exists"] = False

    # Papers and metas details (unified since they share the same db file)
    if PAPERS_CACHE and METAS_CACHE:
        status["papers_and_metas"].update(
            {
                "cache_age_seconds": time.time() - PAPERS_DB_CACHE_TIME,
                "num_papers": len(PAPERS_CACHE),
                "num_metas": len(METAS_CACHE),
                "num_pids": len(PIDS_CACHE) if PIDS_CACHE else 0,
            }
        )

    # Database file details
    if os.path.exists(PAPERS_DB_PATH):
        current_db_mtime = os.path.getmtime(PAPERS_DB_PATH)
        status["papers_and_metas"].update(
            {
                "db_file_exists": True,
                "current_db_file_mtime": current_db_mtime,
                "db_file_is_newer": current_db_mtime > PAPERS_DB_FILE_MTIME,
            }
        )
    else:
        status["papers_and_metas"]["db_file_exists"] = False

    return jsonify(status)


# -----------------------------------------------------------------------------
# tag related endpoints: add, delete tags for any paper


@app.route("/add_tag/<tag>", methods=["POST"])
def add_tag(tag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    tag = _normalize_name(tag)
    err = _validate_tag_name(tag)
    if err:
        return err

    with get_tags_db(flag="c") as tags_db:
        if g.user not in tags_db:
            tags_db[g.user] = {}

        d = tags_db[g.user]
        if tag in d:
            return "user has repeated tag"

        d[tag] = set()
        tags_db[g.user] = d

    logger.info(f"added empty tag {tag} for user {g.user}")
    return "ok"


@app.route("/add/<pid>/<tag>", methods=["GET", "POST"])
def add(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    pid = _normalize_name(pid)
    tag = _normalize_name(tag)
    if not pid:
        return "error, pid is required"
    err = _validate_tag_name(tag)
    if err:
        return err

    with get_tags_db(flag="c") as tags_db:
        # create the user if we don't know about them yet with an empty library
        if not g.user in tags_db:
            tags_db[g.user] = {}

        # fetch the user library object
        d = tags_db[g.user]

        # add the paper to the tag
        if tag not in d:
            d[tag] = set()
        d[tag].add(pid)

        # write back to database
        tags_db[g.user] = d

    logger.info(f"added paper {pid} to tag {tag} for user {g.user}")
    return "ok"


@app.route("/sub/<pid>/<tag>", methods=["GET", "POST"])
def sub(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    pid = _normalize_name(pid)
    tag = _normalize_name(tag)
    if not pid or not tag:
        return "error, pid and tag are required"

    with get_tags_db(flag="c") as tags_db:
        # if the user doesn't have any tags, there is nothing to do
        if not g.user in tags_db:
            return r"user has no library of tags ¯\_(ツ)_/¯"

        # fetch the user library object
        d = tags_db[g.user]

        # add the paper to the tag
        if tag not in d:
            return f"user doesn't have the tag {tag}"
        else:
            if pid in d[tag]:
                # remove this pid from the tag
                d[tag].remove(pid)

                # if this was the last paper in this tag, also delete the tag
                if len(d[tag]) == 0:
                    del d[tag]

                # write back the resulting dict to database
                tags_db[g.user] = d
                return f"ok removed pid {pid} from tag {tag}"
            else:
                return f"user doesn't have paper {pid} in tag {tag}"


@app.route("/del/<tag>", methods=["GET", "POST"])
def delete_tag(tag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    tag = _normalize_name(tag)
    if not tag:
        return "error, tag is required"

    with get_tags_db(flag="c") as tags_db:
        with get_combined_tags_db(flag="c") as ctags_db:
            if g.user not in tags_db:
                return "user does not have a library"

            d = tags_db[g.user]

            if tag not in d:
                return "user does not have this tag"

            # delete the tag
            del d[tag]

            # write back to database
            tags_db[g.user] = d
            # remove ctag
            if g.user in ctags_db:
                ctags = ctags_db[g.user]
                for ctag in list(ctags):
                    if tag in ctag.split(","):
                        ctags.remove(ctag)
                ctags_db[g.user] = ctags

    logger.info(f"deleted tag {tag} for user {g.user}")
    return "ok"


@app.route("/rename/<otag>/<ntag>", methods=["GET", "POST"])
def rename_tag(otag=None, ntag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    otag = _normalize_name(otag)
    ntag = _normalize_name(ntag)
    if not otag or not ntag:
        return "error, tag is required"
    err = _validate_tag_name(ntag)
    if err:
        return err

    with get_tags_db(flag="c") as tags_db:
        with get_combined_tags_db(flag="c") as ctags_db:
            if g.user not in tags_db:
                return "user does not have a library"

            d = tags_db[g.user]

            if otag not in d:
                return "user does not have this tag"

            # logger.trace(d)
            o_pids = d[otag]
            del d[otag]
            if ntag not in d:
                d[ntag] = o_pids
            else:
                d[ntag] = d[ntag].union(o_pids)
            # logger.trace(d)

            # write back to database
            tags_db[g.user] = d

            # rename ctag
            if g.user in ctags_db:
                ctags = ctags_db[g.user]
                for ctag in list(ctags):
                    if otag in (ctag_split := ctag.split(",")):
                        ctag_split = [ct_s if ct_s != otag else ntag for ct_s in ctag_split]
                        n_ctag = ",".join(ctag_split)
                        ctags.remove(ctag)
                        ctags.add(n_ctag)
                ctags_db[g.user] = ctags

    logger.info(f"renamed tag {otag} to {ntag} for user {g.user}")
    return "ok"


@app.route("/add_ctag/<ctag>", methods=["GET", "POST"])
def add_ctag(ctag=None):
    if g.user is None:
        return "error, not logged in"
    ctag = _normalize_name(ctag)
    if not ctag:
        return "error, ctag is required"
    if ctag == "null":
        return "error, cannot add the ctag 'null'"
    _csrf_protect()

    tags = get_tags()
    for tag in map(str.strip, ctag.split(",")):
        if tag not in tags:
            return "invalid ctag"

    with get_combined_tags_db(flag="c") as ctags_db:
        # logger.trace(f"{ctags_db}")
        # create the user if we don't know about them yet with an empty library
        if g.user not in ctags_db:
            ctags_db[g.user] = set()

        # fetch the user library object
        d = ctags_db[g.user]

        # if isinstance(d, dict):
        #     d = set(d.keys())

        # add the paper to the key
        if ctag in d:
            return "user has repeated ctag"

        d.add(ctag)

        # write back to database
        ctags_db[g.user] = d

    logger.info(f"added ctag {ctag} for user {g.user}")
    return "ok"


@app.route("/del_ctag/<ctag>", methods=["GET", "POST"])
def delete_ctag(ctag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    ctag = _normalize_name(ctag)
    if not ctag:
        return "error, ctag is required"

    with get_combined_tags_db(flag="c") as ctags_db:
        if g.user not in ctags_db:
            return "user does not have a library"

        d = ctags_db[g.user]

        if ctag not in d:
            return "user does not have this ctag"

        # delete the tag
        d.remove(ctag)

        # write back to database
        ctags_db[g.user] = d

    logger.info(f"deleted ctag {ctag} for user {g.user}")
    return "ok"


@app.route("/rename_ctag/<otag>/<ntag>", methods=["POST"])
def rename_ctag(otag=None, ntag=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()

    otag = (otag or "").strip()
    ntag = (ntag or "").strip()
    if not otag or not ntag:
        return "error, tag is required"
    if ntag == "null":
        return "error, cannot add the ctag 'null'"
    if otag == ntag:
        return "ok"

    tags = get_tags()
    for tag in map(str.strip, ntag.split(",")):
        if tag not in tags:
            return "invalid ctag"

    with get_combined_tags_db(flag="c") as ctags_db:
        if g.user not in ctags_db:
            return "user does not have a library"

        d = ctags_db[g.user]
        if otag not in d:
            return "user does not have this ctag"
        if ntag in d:
            return "user has repeated ctag"

        d.remove(otag)
        d.add(ntag)
        ctags_db[g.user] = d

    logger.info(f"renamed ctag {otag} to {ntag} for user {g.user}")
    return "ok"


@app.route("/add_key/<keyword>", methods=["GET", "POST"])
def add_key(keyword=None):
    if g.user is None:
        return "error, not logged in"
    keyword = _normalize_name(keyword)
    err = _validate_keyword_name(keyword)
    if err:
        return err
    _csrf_protect()

    with get_keywords_db(flag="c") as keys_db:
        # create the user if we don't know about them yet with an empty library
        if g.user not in keys_db:
            keys_db[g.user] = {}

        # fetch the user library object
        d = keys_db[g.user]

        # add the paper to the key
        if keyword in d:
            return "user has repeated keywords"

        d[keyword] = set()

        # write back to database
        keys_db[g.user] = d

    logger.info(f"added keyword {keyword} for user {g.user}")
    return "ok"


@app.route("/del_key/<keyword>", methods=["GET", "POST"])
def delete_key(keyword=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()
    keyword = _normalize_name(keyword)
    if not keyword:
        return "error, keyword is required"

    with get_keywords_db(flag="c") as keys_db:
        if g.user not in keys_db:
            return "user does not have a library"

        d = keys_db[g.user]

        if keyword not in d:
            return "user does not have this keyword"

        # delete the keyword
        del d[keyword]

        # write back to database
        keys_db[g.user] = d

    logger.info(f"deleted keyword {keyword} for user {g.user}")
    return "ok"


@app.route("/rename_key/<okey>/<nkey>", methods=["POST"])
def rename_key(okey=None, nkey=None):
    if g.user is None:
        return "error, not logged in"
    _csrf_protect()

    okey = _normalize_name(okey)
    nkey = _normalize_name(nkey)
    if not okey or not nkey:
        return "error, keyword is required"
    if nkey == "null":
        return "error, cannot add the protected keyword 'null'"
    if okey == nkey:
        return "ok"

    with get_keywords_db(flag="c") as keys_db:
        if g.user not in keys_db:
            return "user does not have a library"

        d = keys_db[g.user]
        if okey not in d:
            return "user does not have this keyword"

        if nkey in d:
            d[nkey] = d[nkey].union(d[okey])
        else:
            d[nkey] = d[okey]
        del d[okey]
        keys_db[g.user] = d

    logger.info(f"renamed keyword {okey} to {nkey} for user {g.user}")
    return "ok"


# -----------------------------------------------------------------------------
# endpoints to log in and out


@app.route("/login", methods=["POST"])
def login():
    _csrf_protect()
    # the user is logged out but wants to log in, ok
    username = (request.form.get("username") or "").strip()
    if g.user is None and username:
        if len(username) > 0:  # one more paranoid check
            session["user"] = username

    return redirect(url_for("profile"))


@app.route("/logout", methods=["GET", "POST"])
def logout():
    _csrf_protect()
    session.pop("user", None)
    return redirect(url_for("profile"))


# -----------------------------------------------------------------------------
# user settings and configurations


@app.route("/register_email", methods=["POST"])
def register_email():
    _csrf_protect()
    email = (request.form.get("email") or "").strip()

    if g.user:
        # do some basic input validation
        proper_email = re.match(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$", email, re.IGNORECASE)
        if email == "" or proper_email:  # allow empty email, meaning no email
            # everything checks out, write to the database
            with get_email_db(flag="c") as edb:
                edb[g.user] = email

    return redirect(url_for("profile"))


if __name__ == "__main__":
    from vars import SERVE_PORT

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    app.run(host="0.0.0.0", port=int(SERVE_PORT), threaded=True)
