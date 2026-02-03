"""Data access and caching services."""

from __future__ import annotations

import os
import pickle
import threading
import time
from threading import Lock
from typing import Any

from loguru import logger

from aslite.db import FEATURES_FILE, FEATURES_FILE_NEW, PAPERS_DB_FILE, load_features
from aslite.repositories import MetaRepository, PaperRepository
from config import settings

# Lock ordering notes (to prevent deadlocks):
# - Avoid nesting _DATA_LOCK, _FEATURES_LOCK, and semantic_service._EMBEDDINGS_LOCK.
# - semantic_service.get_paper_embeddings() is implemented to avoid holding _EMBEDDINGS_LOCK while
#   calling get_features_cached(); keep it that way to prevent FEATURES_LOCK <-> _EMBEDDINGS_LOCK cycles.

# Data cache (mtime-based invalidation, shared across warmup + request path)
_DATA_LOCK = Lock()
_PAPERS_CACHE: dict[str, Any] | None = None
_METAS_CACHE: dict[str, Any] | None = None
_PIDS_CACHE: list[str] | None = None
_PAPERS_DB_FILE_MTIME: float = 0.0
_PAPERS_DB_CACHE_TIME: float = 0.0

# Monotonic cache generation token (invalidate-safe against in-flight refresh threads).
_DATA_CACHE_GEN: int = 0

# Cold start loader state (avoid blocking gevent worker on heavy I/O under locks).
_DATA_COLD_LOAD_LOCK = Lock()
_DATA_COLD_LOAD_IN_PROGRESS: bool = False
_DATA_COLD_LOAD_LAST_START: float = 0.0
_DATA_COLD_LOAD_LAST_ERROR: str | None = None
_DATA_COLD_LOAD_SCHEDULED_GEN: int = 0

# Background refresh state (avoid blocking request threads on full reloads)
_DATA_REFRESH_LOCK = Lock()
_DATA_REFRESH_IN_PROGRESS: bool = False
_DATA_REFRESH_LAST_START: float = 0.0
_DATA_REFRESH_LAST_ERROR: str | None = None

# Features cache (mtime-based)
_FEATURES_LOCK = Lock()
_FEATURES_CACHE: dict[str, Any] | None = None
_FEATURES_FILE_MTIME: float = 0.0
_FEATURES_CACHE_TIME: float = 0.0

_FEATURES_REFRESH_LOCK = Lock()
_FEATURES_REFRESH_IN_PROGRESS: bool = False
_FEATURES_REFRESH_LAST_START: float = 0.0
_FEATURES_REFRESH_LAST_ERROR: str | None = None


def _get_native_thread_class():
    """Return a real OS Thread class even under gevent monkey-patching."""
    try:
        import gevent.monkey

        if gevent.monkey.is_module_patched("threading"):
            return gevent.monkey.get_original("threading", "Thread")
    except Exception:
        return threading.Thread
    return threading.Thread


def _start_daemon_thread(*, target, name: str) -> None:
    Thread = _get_native_thread_class()
    t = Thread(target=target, name=name, daemon=True)
    t.start()


def _cache_papers_in_memory() -> bool:
    return settings.web.cache_papers


def _sqlite_effective_mtime(db_path: str) -> float:
    """Return an mtime suitable for cache invalidation for SQLite (incl. WAL files).

    SQLite WAL mode writes to `*.db-wal` / `*.db-shm` and may not update the main
    database file mtime until checkpoint. Using the maximum mtime across these
    files avoids serving stale caches indefinitely.
    """
    if not db_path:
        return 0.0
    candidates = (db_path, f"{db_path}-wal", f"{db_path}-shm")
    mtimes: list[float] = []
    for p in candidates:
        try:
            if os.path.exists(p):
                mtimes.append(float(os.path.getmtime(p)))
        except Exception:
            continue
    return max(mtimes) if mtimes else 0.0


def _should_throttle(last_start: float, min_interval_s: float) -> bool:
    try:
        min_interval_s = float(min_interval_s or 0.0)
    except Exception:
        min_interval_s = 0.0
    if min_interval_s <= 0:
        return False
    return (time.time() - float(last_start or 0.0)) < min_interval_s


def _load_all_papers_and_metas(*, cache_papers: bool) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    """Load full metas (and optionally full papers) from papers.db.

    NOTE: This can be very expensive for large corpora; do not call it on the
    request thread unless it's a cold start.
    """
    t0 = time.time()

    papers_cache: dict[str, Any] | None
    if cache_papers:
        logger.trace("[BLOCKING] data_cache: loading all papers from papers.db...")
        papers_cache = {pid: paper for pid, paper in PaperRepository.iter_all_papers()}
        logger.trace(f"[BLOCKING] data_cache: loaded {len(papers_cache)} papers in {time.time() - t0:.2f}s")
    else:
        papers_cache = None

    t1 = time.time()
    logger.trace("[BLOCKING] data_cache: loading all metas from papers.db...")
    metas_cache = {pid: meta for pid, meta in MetaRepository.iter_all_metas()}
    pids_cache = list(metas_cache.keys())
    logger.trace(
        f"[BLOCKING] data_cache: loaded {len(metas_cache)} metas in {time.time() - t1:.2f}s, total {time.time() - t0:.2f}s"
    )

    return papers_cache, metas_cache, pids_cache


def _data_cache_ready(*, cache_papers: bool) -> bool:
    if _METAS_CACHE is None or _PIDS_CACHE is None:
        return False
    if cache_papers and _PAPERS_CACHE is None:
        return False
    return True


def _ensure_cold_load_started(*, current_mtime: float, cache_papers: bool) -> None:
    """Start cold load in a background OS thread (best-effort)."""
    global _DATA_COLD_LOAD_IN_PROGRESS, _DATA_COLD_LOAD_LAST_START, _DATA_COLD_LOAD_LAST_ERROR
    global _DATA_COLD_LOAD_SCHEDULED_GEN, _DATA_CACHE_GEN

    with _DATA_COLD_LOAD_LOCK:
        if _DATA_COLD_LOAD_IN_PROGRESS:
            return
        _DATA_COLD_LOAD_IN_PROGRESS = True
        _DATA_COLD_LOAD_LAST_START = time.time()
        _DATA_COLD_LOAD_LAST_ERROR = None
        _DATA_COLD_LOAD_SCHEDULED_GEN = int(_DATA_CACHE_GEN)

    scheduled_gen = int(_DATA_COLD_LOAD_SCHEDULED_GEN)

    def _run() -> None:
        global _DATA_COLD_LOAD_IN_PROGRESS, _DATA_COLD_LOAD_LAST_ERROR
        global _PAPERS_CACHE, _METAS_CACHE, _PIDS_CACHE
        global _PAPERS_DB_FILE_MTIME, _PAPERS_DB_CACHE_TIME
        global _DATA_CACHE_GEN

        try:
            papers_cache, metas_cache, pids_cache = _load_all_papers_and_metas(cache_papers=cache_papers)
            refreshed_mtime = _sqlite_effective_mtime(PAPERS_DB_FILE)
            with _DATA_LOCK:
                if int(_DATA_CACHE_GEN) != int(scheduled_gen):
                    return
                _PAPERS_CACHE = papers_cache
                _METAS_CACHE = metas_cache
                _PIDS_CACHE = pids_cache
                _PAPERS_DB_FILE_MTIME = float(refreshed_mtime or current_mtime or 0.0)
                _PAPERS_DB_CACHE_TIME = time.time()
        except Exception as exc:
            _DATA_COLD_LOAD_LAST_ERROR = str(exc)
            logger.warning(f"Cold-start data cache load failed: {exc}")
        finally:
            with _DATA_COLD_LOAD_LOCK:
                _DATA_COLD_LOAD_IN_PROGRESS = False

    try:
        _start_daemon_thread(target=_run, name="data-cache-cold-load")
    except Exception as exc:
        # Avoid wedging cold-start waiters when thread start fails.
        with _DATA_COLD_LOAD_LOCK:
            _DATA_COLD_LOAD_IN_PROGRESS = False
            _DATA_COLD_LOAD_LAST_ERROR = str(exc)
        logger.warning(f"Cold-start data cache thread failed to start: {exc}")


def _schedule_data_refresh(current_mtime: float) -> None:
    """Trigger a best-effort background refresh for papers/metas cache."""
    global _DATA_REFRESH_IN_PROGRESS, _DATA_REFRESH_LAST_START, _DATA_REFRESH_LAST_ERROR
    global _DATA_CACHE_GEN

    min_interval_s = float(getattr(settings.web, "data_cache_refresh_min_interval", 60) or 0)
    # If a previous refresh got stuck (e.g., slow filesystem), allow a new one eventually.
    max_stuck_s = float(getattr(settings.web, "data_cache_refresh_min_interval", 60) or 0) * 10
    if max_stuck_s < 600:
        max_stuck_s = 600.0

    with _DATA_REFRESH_LOCK:
        if _DATA_REFRESH_IN_PROGRESS:
            if max_stuck_s > 0 and (time.time() - float(_DATA_REFRESH_LAST_START or 0.0)) > max_stuck_s:
                logger.warning(
                    f"Data cache refresh appears stuck (> {max_stuck_s:.0f}s); allowing a new refresh attempt"
                )
                _DATA_REFRESH_IN_PROGRESS = False
            else:
                return
        if _should_throttle(_DATA_REFRESH_LAST_START, min_interval_s):
            return
        _DATA_REFRESH_IN_PROGRESS = True
        _DATA_REFRESH_LAST_START = time.time()
        _DATA_REFRESH_LAST_ERROR = None
        scheduled_gen = int(_DATA_CACHE_GEN)

    def _run() -> None:
        global _PAPERS_CACHE, _METAS_CACHE, _PIDS_CACHE
        global _PAPERS_DB_FILE_MTIME, _PAPERS_DB_CACHE_TIME
        global _DATA_REFRESH_IN_PROGRESS, _DATA_REFRESH_LAST_ERROR
        global _DATA_CACHE_GEN

        try:
            cache_papers = _cache_papers_in_memory()
            papers_cache, metas_cache, pids_cache = _load_all_papers_and_metas(cache_papers=cache_papers)
            # Recompute mtime after read (daemon may have advanced the WAL while we were loading).
            refreshed_mtime = _sqlite_effective_mtime(PAPERS_DB_FILE)
            with _DATA_LOCK:
                # If cache was invalidated while we were refreshing, do not write back stale results.
                if int(_DATA_CACHE_GEN) != int(scheduled_gen):
                    return
                _PAPERS_CACHE = papers_cache
                _METAS_CACHE = metas_cache
                _PIDS_CACHE = pids_cache
                _PAPERS_DB_FILE_MTIME = float(refreshed_mtime or current_mtime or 0.0)
                _PAPERS_DB_CACHE_TIME = time.time()
        except Exception as exc:
            _DATA_REFRESH_LAST_ERROR = str(exc)
            logger.warning(f"Background data cache refresh failed: {exc}")
        finally:
            with _DATA_REFRESH_LOCK:
                _DATA_REFRESH_IN_PROGRESS = False

    try:
        _start_daemon_thread(target=_run, name="data-cache-refresh")
    except Exception as exc:
        with _DATA_REFRESH_LOCK:
            _DATA_REFRESH_IN_PROGRESS = False
            _DATA_REFRESH_LAST_ERROR = str(exc)
        logger.warning(f"Background data cache refresh thread failed to start: {exc}")


def _schedule_features_refresh(effective_mtime: float) -> None:
    """Trigger a best-effort background refresh for features cache."""
    global _FEATURES_REFRESH_IN_PROGRESS, _FEATURES_REFRESH_LAST_START, _FEATURES_REFRESH_LAST_ERROR

    min_interval_s = float(getattr(settings.web, "features_cache_refresh_min_interval", 300) or 0)

    with _FEATURES_REFRESH_LOCK:
        if _FEATURES_REFRESH_IN_PROGRESS:
            return
        if _should_throttle(_FEATURES_REFRESH_LAST_START, min_interval_s):
            return
        _FEATURES_REFRESH_IN_PROGRESS = True
        _FEATURES_REFRESH_LAST_START = time.time()
        _FEATURES_REFRESH_LAST_ERROR = None

    def _run() -> None:
        global _FEATURES_CACHE, _FEATURES_FILE_MTIME, _FEATURES_CACHE_TIME
        global _FEATURES_REFRESH_IN_PROGRESS, _FEATURES_REFRESH_LAST_ERROR

        def _mtime(path: str) -> float:
            try:
                if path and os.path.exists(path):
                    return float(os.path.getmtime(path))
            except Exception:
                return 0.0
            return 0.0

        try:
            t0 = time.time()
            mtime_main = _mtime(FEATURES_FILE)
            mtime_new = _mtime(FEATURES_FILE_NEW)
            effective = max(mtime_main, mtime_new)
            used_new = bool(mtime_new > 0 and mtime_new >= mtime_main and os.path.exists(FEATURES_FILE_NEW))
            try:
                if used_new:
                    with open(FEATURES_FILE_NEW, "rb") as f:
                        features = pickle.load(f)
                else:
                    features = load_features()
            except (EOFError, pickle.UnpicklingError, OSError) as exc:
                # Fallback to the other file when the preferred one is corrupted/partial.
                try:
                    if used_new:
                        features = load_features()
                    elif os.path.exists(FEATURES_FILE_NEW):
                        with open(FEATURES_FILE_NEW, "rb") as f:
                            features = pickle.load(f)
                    else:
                        raise exc
                except Exception:
                    raise exc

            try:
                if isinstance(features, dict) and "pids" in features and "pid_to_index" not in features:
                    features["pid_to_index"] = {pid: i for i, pid in enumerate(features["pids"])}
            except Exception:
                pass

            with _FEATURES_LOCK:
                _FEATURES_CACHE = features
                _FEATURES_FILE_MTIME = float(effective or effective_mtime or 0.0)
                _FEATURES_CACHE_TIME = time.time()
            logger.trace(f"[BLOCKING] features_cache: refreshed in {time.time() - t0:.2f}s")
        except Exception as exc:
            _FEATURES_REFRESH_LAST_ERROR = str(exc)
            logger.warning(f"Background features cache refresh failed: {exc}")
        finally:
            with _FEATURES_REFRESH_LOCK:
                _FEATURES_REFRESH_IN_PROGRESS = False

    try:
        _start_daemon_thread(target=_run, name="features-cache-refresh")
    except Exception as exc:
        with _FEATURES_REFRESH_LOCK:
            _FEATURES_REFRESH_IN_PROGRESS = False
            _FEATURES_REFRESH_LAST_ERROR = str(exc)
        logger.warning(f"Background features cache refresh thread failed to start: {exc}")


def get_features_cached() -> dict[str, Any]:
    """Load features from disk with mtime-based caching."""
    global _FEATURES_CACHE, _FEATURES_FILE_MTIME, _FEATURES_CACHE_TIME

    if not os.path.exists(FEATURES_FILE) and not os.path.exists(FEATURES_FILE_NEW):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    def _mtime(path: str) -> float:
        try:
            if path and os.path.exists(path):
                return float(os.path.getmtime(path))
        except Exception:
            return 0.0
        return 0.0

    mtime_main = _mtime(FEATURES_FILE)
    mtime_new = _mtime(FEATURES_FILE_NEW)
    # Reload whenever either file changes (including rollbacks where mtime decreases).
    effective_mtime = max(mtime_main, mtime_new)

    need_reload = _FEATURES_CACHE is None or effective_mtime != _FEATURES_FILE_MTIME
    if need_reload:
        # Cold start: block and load in-process.
        if _FEATURES_CACHE is None:
            with _FEATURES_LOCK:
                mtime_main = _mtime(FEATURES_FILE)
                mtime_new = _mtime(FEATURES_FILE_NEW)
                effective_mtime = max(mtime_main, mtime_new)
                if _FEATURES_CACHE is None or effective_mtime != _FEATURES_FILE_MTIME:
                    t0 = time.time()
                    logger.trace("[BLOCKING] get_features_cached: starting to load features.p file...")
                    used_new = bool(mtime_new > 0 and mtime_new >= mtime_main and os.path.exists(FEATURES_FILE_NEW))
                    try:
                        if used_new:
                            with open(FEATURES_FILE_NEW, "rb") as f:
                                features = pickle.load(f)
                        else:
                            features = load_features()
                    except (EOFError, pickle.UnpicklingError, OSError) as exc:
                        # Fallback to the other file when the preferred one is corrupted/partial.
                        try:
                            if used_new:
                                features = load_features()
                            elif os.path.exists(FEATURES_FILE_NEW):
                                with open(FEATURES_FILE_NEW, "rb") as f:
                                    features = pickle.load(f)
                            else:
                                raise exc
                        except Exception:
                            raise exc

                    try:
                        if isinstance(features, dict) and "pids" in features and "pid_to_index" not in features:
                            features["pid_to_index"] = {pid: i for i, pid in enumerate(features["pids"])}
                    except Exception:
                        pass

                    _FEATURES_CACHE = features
                    _FEATURES_FILE_MTIME = float(effective_mtime or 0.0)
                    _FEATURES_CACHE_TIME = time.time()
                    logger.trace(f"[BLOCKING] get_features_cached: loaded features.p in {time.time() - t0:.2f}s")
        else:
            # Warm cache but underlying file changed: refresh asynchronously and serve stale cache now.
            _schedule_features_refresh(effective_mtime)

    return _FEATURES_CACHE or {}


def peek_features_cache() -> dict[str, Any] | None:
    """Return cached features dict if already loaded; does not trigger load."""
    return _FEATURES_CACHE


def get_features_file_mtime() -> float:
    return _FEATURES_FILE_MTIME


def get_features_cache_time() -> float:
    return _FEATURES_CACHE_TIME


def get_data_cached(*, wait: bool = True, max_wait_s: float | None = None) -> dict[str, Any]:
    """Return cached {pids, papers, metas}.

    - Always caches metas/pids in memory (small and hot).
    - Optionally caches full papers table when ARXIV_SANITY_CACHE_PAPERS=1.
    - Cache invalidation is based on papers.db file modification time.
    - On cold start, a background loader is started. By default (`wait=True`),
      this call cooperatively waits until the cache becomes ready (legacy behavior).
      When `wait=False`, it returns immediately with the current snapshot.
      When `max_wait_s` is provided, waiting is bounded to that duration.
    """
    global _PAPERS_CACHE, _METAS_CACHE, _PIDS_CACHE
    global _PAPERS_DB_FILE_MTIME, _PAPERS_DB_CACHE_TIME

    if not os.path.exists(PAPERS_DB_FILE):
        # Graceful fallback: allow web/UI to load even when papers.db is absent.
        # Keep caches empty so callers get safe defaults instead of 500s.
        with _DATA_LOCK:
            _PAPERS_CACHE = None
            _METAS_CACHE = {}
            _PIDS_CACHE = []
            _PAPERS_DB_FILE_MTIME = 0.0
            _PAPERS_DB_CACHE_TIME = 0.0
        logger.warning(f"Papers database file not found: {PAPERS_DB_FILE}")
        return {"pids": [], "papers": None, "metas": {}}

    current_mtime = _sqlite_effective_mtime(PAPERS_DB_FILE)

    cache_papers = _cache_papers_in_memory()
    cold_start = not _data_cache_ready(cache_papers=cache_papers)
    # Reload whenever effective mtime changes (including rollbacks where mtime decreases).
    stale = current_mtime != _PAPERS_DB_FILE_MTIME

    if cold_start:
        # Cold start: trigger background load and wait cooperatively until it becomes available.
        # This avoids holding _DATA_LOCK during heavy I/O, and under gevent it prevents
        # blocking the worker event loop on lock acquisition.
        _ensure_cold_load_started(current_mtime=current_mtime, cache_papers=cache_papers)

        if wait:
            start_wait = time.time()
            deadline = None
            if max_wait_s is not None:
                try:
                    deadline = start_wait + float(max_wait_s)
                except Exception:
                    deadline = start_wait

            last_log = 0.0
            while True:
                with _DATA_LOCK:
                    ready = _data_cache_ready(cache_papers=cache_papers)
                    last_err = _DATA_COLD_LOAD_LAST_ERROR
                    in_prog = _DATA_COLD_LOAD_IN_PROGRESS
                    started_at = float(_DATA_COLD_LOAD_LAST_START or 0.0)
                if ready:
                    break
                # If background loader finished but cache still isn't ready, surface error.
                if (not in_prog) and last_err:
                    raise RuntimeError(last_err)
                now = time.time()
                if deadline is not None and now >= deadline:
                    break
                if started_at and (now - started_at) > 60.0 and (now - last_log) > 60.0:
                    last_log = now
                    logger.warning(
                        f"[BLOCKING] get_data_cached: cold start still loading after {now - started_at:.0f}s"
                    )
                # Yield to other greenlets/threads.
                time.sleep(0.05)
        else:
            # Non-blocking peek: still surface a previous cold-start error if one already happened.
            with _DATA_LOCK:
                last_err = _DATA_COLD_LOAD_LAST_ERROR
                in_prog = _DATA_COLD_LOAD_IN_PROGRESS
            if (not in_prog) and last_err:
                raise RuntimeError(last_err)
    else:
        # Warm cache: never block request threads on a full reload.
        if stale:
            _schedule_data_refresh(current_mtime)

    # Return a consistent snapshot (avoid mixing pids/metas/papers across refresh boundaries).
    with _DATA_LOCK:
        pids = _PIDS_CACHE or []
        papers = _PAPERS_CACHE
        metas = _METAS_CACHE or {}

    return {"pids": pids, "papers": papers, "metas": metas}


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
    global _DATA_CACHE_GEN
    with _DATA_LOCK:
        _DATA_CACHE_GEN += 1
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
