"""Data access and caching services."""

from __future__ import annotations

import os
import pickle
import time
from threading import Lock
from typing import Any

from aslite.db import FEATURES_FILE, FEATURES_FILE_NEW, PAPERS_DB_FILE, load_features
from aslite.repositories import MetaRepository, PaperRepository
from config import settings

# Data cache (mtime-based invalidation, shared across warmup + request path)
_DATA_LOCK = Lock()
_PAPERS_CACHE: dict[str, Any] | None = None
_METAS_CACHE: dict[str, Any] | None = None
_PIDS_CACHE: list[str] | None = None
_PAPERS_DB_FILE_MTIME: float = 0.0
_PAPERS_DB_CACHE_TIME: float = 0.0

# Features cache (mtime-based)
_FEATURES_LOCK = Lock()
_FEATURES_CACHE: dict[str, Any] | None = None
_FEATURES_FILE_MTIME: float = 0.0
_FEATURES_CACHE_TIME: float = 0.0


def _cache_papers_in_memory() -> bool:
    return settings.web.cache_papers


def get_features_cached() -> dict[str, Any]:
    """Load features from disk with mtime-based caching."""
    global _FEATURES_CACHE, _FEATURES_FILE_MTIME, _FEATURES_CACHE_TIME

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    try:
        current_mtime = os.path.getmtime(FEATURES_FILE)
    except Exception:
        current_mtime = 0.0

    need_reload = _FEATURES_CACHE is None or current_mtime > _FEATURES_FILE_MTIME
    if need_reload:
        with _FEATURES_LOCK:
            try:
                current_mtime = os.path.getmtime(FEATURES_FILE)
            except Exception:
                current_mtime = 0.0
            need_reload = _FEATURES_CACHE is None or current_mtime > _FEATURES_FILE_MTIME

            if need_reload:
                try:
                    features = load_features()
                except (EOFError, pickle.UnpicklingError):
                    # During compute.py deployment, FEATURES_FILE is overwritten via shutil.copyfile,
                    # which is not atomic. If we catch a partial-read error, fall back to the
                    # atomic-written staging file.
                    if os.path.exists(FEATURES_FILE_NEW):
                        with open(FEATURES_FILE_NEW, "rb") as f:
                            features = pickle.load(f)
                    else:
                        raise

                # Build pid->index map for fast lookups (inspect, etc.)
                try:
                    if isinstance(features, dict) and "pids" in features and "pid_to_index" not in features:
                        features["pid_to_index"] = {pid: i for i, pid in enumerate(features["pids"])}
                except Exception:
                    pass

                _FEATURES_CACHE = features
                _FEATURES_FILE_MTIME = current_mtime
                _FEATURES_CACHE_TIME = time.time()

    return _FEATURES_CACHE or {}


def peek_features_cache() -> dict[str, Any] | None:
    """Return cached features dict if already loaded; does not trigger load."""
    return _FEATURES_CACHE


def get_features_file_mtime() -> float:
    return _FEATURES_FILE_MTIME


def get_features_cache_time() -> float:
    return _FEATURES_CACHE_TIME


def get_data_cached() -> dict[str, Any]:
    """Return cached {pids, papers, metas}.

    - Always caches metas/pids in memory (small and hot).
    - Optionally caches full papers table when ARXIV_SANITY_CACHE_PAPERS=1.
    - Cache invalidation is based on papers.db file modification time.
    """
    global _PAPERS_CACHE, _METAS_CACHE, _PIDS_CACHE
    global _PAPERS_DB_FILE_MTIME, _PAPERS_DB_CACHE_TIME

    if not os.path.exists(PAPERS_DB_FILE):
        raise FileNotFoundError(f"Papers database file not found: {PAPERS_DB_FILE}")

    try:
        current_mtime = os.path.getmtime(PAPERS_DB_FILE)
    except Exception:
        current_mtime = 0.0

    cache_papers = _cache_papers_in_memory()
    need_reload_metas = _METAS_CACHE is None or _PIDS_CACHE is None or current_mtime > _PAPERS_DB_FILE_MTIME
    need_reload_papers = cache_papers and (_PAPERS_CACHE is None or current_mtime > _PAPERS_DB_FILE_MTIME)
    need_reload = need_reload_metas or need_reload_papers

    if need_reload:
        with _DATA_LOCK:
            try:
                current_mtime = os.path.getmtime(PAPERS_DB_FILE)
            except Exception:
                current_mtime = 0.0

            cache_papers = _cache_papers_in_memory()
            need_reload_metas = _METAS_CACHE is None or _PIDS_CACHE is None or current_mtime > _PAPERS_DB_FILE_MTIME
            need_reload_papers = cache_papers and (_PAPERS_CACHE is None or current_mtime > _PAPERS_DB_FILE_MTIME)
            need_reload = need_reload_metas or need_reload_papers

            if need_reload:
                # Papers cache is optional
                if cache_papers:
                    _PAPERS_CACHE = {pid: paper for pid, paper in PaperRepository.iter_all_papers()}
                else:
                    _PAPERS_CACHE = None

                _METAS_CACHE = {pid: meta for pid, meta in MetaRepository.iter_all_metas()}
                _PIDS_CACHE = list(_METAS_CACHE.keys())

                _PAPERS_DB_FILE_MTIME = current_mtime
                _PAPERS_DB_CACHE_TIME = time.time()

    return {
        "pids": _PIDS_CACHE or [],
        "papers": _PAPERS_CACHE,
        "metas": _METAS_CACHE or {},
    }


def get_papers_db_file_mtime() -> float:
    return _PAPERS_DB_FILE_MTIME


def get_papers_db_cache_time() -> float:
    return _PAPERS_DB_CACHE_TIME


def get_pids() -> list[str]:
    """Get all paper IDs."""
    return get_data_cached()["pids"]


def get_papers() -> dict[str, Any]:
    """Get all papers."""
    papers = get_data_cached().get("papers")
    return papers or {}


def get_metas() -> dict[str, Any]:
    """Get all paper metadata."""
    return get_data_cached()["metas"]


def paper_exists(pid: str) -> bool:
    """Check if paper exists."""
    if not pid:
        return False
    return pid in get_metas()


def get_paper(pid: str) -> dict[str, Any] | None:
    """Get single paper by ID."""
    if not pid:
        return None
    papers = get_data_cached().get("papers")
    if isinstance(papers, dict):
        return papers.get(pid)
    return PaperRepository.get_by_id(pid)


def get_papers_bulk(pids: list[str]) -> dict[str, Any]:
    """Get multiple papers by IDs."""
    if not pids:
        return {}
    papers = get_data_cached().get("papers")
    if isinstance(papers, dict):
        return {pid: papers[pid] for pid in pids if pid in papers}
    return PaperRepository.get_by_ids(list(pids))


def invalidate_cache():
    """Invalidate all data caches."""
    global _PAPERS_CACHE, _METAS_CACHE, _PIDS_CACHE
    global _PAPERS_DB_FILE_MTIME, _PAPERS_DB_CACHE_TIME
    global _FEATURES_CACHE, _FEATURES_FILE_MTIME, _FEATURES_CACHE_TIME
    with _DATA_LOCK:
        _PAPERS_CACHE = None
        _METAS_CACHE = None
        _PIDS_CACHE = None
        _PAPERS_DB_FILE_MTIME = 0.0
        _PAPERS_DB_CACHE_TIME = 0.0
    with _FEATURES_LOCK:
        _FEATURES_CACHE = None
        _FEATURES_FILE_MTIME = 0.0
        _FEATURES_CACHE_TIME = 0.0


def warmup_data_cache():
    """Warm up data cache."""
    get_data_cached()


def warmup_ml_cache():
    """Warm up ML-related caches."""
    features = get_features_cached()
    return bool(features)
