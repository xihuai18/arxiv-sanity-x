"""Summary generation and caching services."""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from loguru import logger

from aslite.repositories import SummaryStatusRepository
from config import settings
from tools.paper_summarizer import (
    acquire_summary_lock,
    atomic_write_json,
    atomic_write_text,
)
from tools.paper_summarizer import (
    generate_paper_summary as generate_paper_summary_from_module,
)
from tools.paper_summarizer import (
    normalize_summary_result,
    normalize_summary_source,
    read_summary_meta,
    release_summary_lock,
    resolve_cache_pid,
    summary_cache_paths,
    summary_source_matches,
)

DATA_DIR = str(settings.data_dir)
SUMMARY_DIR = str(settings.summary_dir)
LLM_NAME = settings.llm.name
SUMMARY_MARKDOWN_SOURCE = settings.summary.markdown_source

# -----------------------------------------------------------------------------
# Summary coverage / cache stats (fast path + incremental updates + periodic full rebuild)
#
# We keep:
# - a small snapshot for UI under `_SUMMARY_CACHE_STATS_KEY`
# - per-model counts under `_SUMMARY_CACHE_MODEL_PREFIX + model`
# - per-pid counts under `_SUMMARY_CACHE_PID_PREFIX + pid`
# - aggregate totals under `_SUMMARY_CACHE_TOTALS_KEY`
#
# And we periodically rebuild from disk to ensure correctness.
# -----------------------------------------------------------------------------

_SUMMARY_CACHE_STATS_KEY = "stats::summary_cache"  # UI snapshot
_SUMMARY_CACHE_TOTALS_KEY = "stats::summary_cache::totals"
_SUMMARY_CACHE_PID_PREFIX = "stats::summary_cache::pid::"
_SUMMARY_CACHE_MODEL_PREFIX = "stats::summary_cache::model::"


def _get_stats_db(flag: str = "r"):
    # Late import to avoid import-time side effects.
    from aslite.db import get_summary_status_db

    return get_summary_status_db(flag=flag)


def _stats_lock_path() -> Path:
    return Path(SUMMARY_DIR) / ".summary_cache_stats.lock"


def _with_stats_lock(timeout_s: int = 10):
    """Context manager for cross-process stats updates."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        lock_path = _stats_lock_path()
        fd = acquire_summary_lock(lock_path, timeout_s=timeout_s)
        try:
            yield fd
        finally:
            if fd is not None:
                try:
                    release_summary_lock(fd, lock_path)
                except Exception:
                    pass

    return _cm()


def _read_snapshot() -> dict | None:
    try:
        with _get_stats_db("r") as sdb:
            payload = sdb.get(_SUMMARY_CACHE_STATS_KEY)
        return payload if isinstance(payload, dict) else None
    except Exception as e:
        logger.warning(f"Failed to read persisted summary cache snapshot: {e}")
        return None


def _write_snapshot(data: dict, duration: float = 0.0) -> None:
    payload = {
        "data": data,
        "updated_time": time.time(),
        "duration": float(duration or 0.0),
    }
    try:
        with _get_stats_db("c") as sdb:
            sdb[_SUMMARY_CACHE_STATS_KEY] = payload
    except Exception as e:
        logger.warning(f"Failed to persist summary cache snapshot: {e}")


def _read_totals(sdb) -> dict:
    totals = sdb.get(_SUMMARY_CACHE_TOTALS_KEY)
    if not isinstance(totals, dict):
        totals = {}
    return {
        "total": int(totals.get("total") or 0),
        "paper_count": int(totals.get("paper_count") or 0),
        "updated_time": float(totals.get("updated_time") or 0.0),
        "last_full_scan_time": float(totals.get("last_full_scan_time") or 0.0),
    }


def _write_totals(sdb, totals: dict) -> None:
    sdb[_SUMMARY_CACHE_TOTALS_KEY] = {
        "total": int(totals.get("total") or 0),
        "paper_count": int(totals.get("paper_count") or 0),
        "updated_time": float(totals.get("updated_time") or time.time()),
        "last_full_scan_time": float(totals.get("last_full_scan_time") or 0.0),
    }


def _iter_model_counts(sdb) -> dict:
    counts: dict[str, int] = {}
    try:
        for key, val in sdb.items_with_prefix(_SUMMARY_CACHE_MODEL_PREFIX):
            model = key[len(_SUMMARY_CACHE_MODEL_PREFIX) :]
            try:
                counts[model] = int(val or 0)
            except Exception:
                continue
    except Exception:
        pass
    return counts


def _build_snapshot_data(total: int, paper_count: int, model_counts: dict) -> dict:
    return {
        "summary_cache_total": int(total),
        "summary_cache_paper_count": int(paper_count),
        "summary_cache_model_counts": [
            {"model": m, "count": int(c)}
            for m, c in sorted(model_counts.items(), key=lambda x: (-x[1], x[0]))
            if int(c) > 0
        ],
    }


def invalidate_summary_cache_stats() -> None:
    """Mark persisted snapshot/totals as stale; next request triggers a refresh."""
    try:
        with _get_stats_db("c") as sdb:
            snap = sdb.get(_SUMMARY_CACHE_STATS_KEY)
            if isinstance(snap, dict):
                sdb[_SUMMARY_CACHE_STATS_KEY] = {**snap, "updated_time": 0.0}
            totals = sdb.get(_SUMMARY_CACHE_TOTALS_KEY)
            if isinstance(totals, dict):
                sdb[_SUMMARY_CACHE_TOTALS_KEY] = {**totals, "updated_time": 0.0}
    except Exception:
        pass


def _normalize_model_for_stats(model: str | None, meta: dict | None = None) -> str:
    if isinstance(meta, dict):
        m = (meta.get("model") or meta.get("llm_model") or "").strip()
        if m:
            return m
    return (model or "").strip()


def summary_cache_stats_increment(pid: str, model: str) -> None:
    pid = (pid or "").strip()
    model = (model or "").strip()
    if not pid or not model:
        return

    with _with_stats_lock(timeout_s=10):
        with _get_stats_db("c") as sdb:
            totals = _read_totals(sdb)

            pid_key = f"{_SUMMARY_CACHE_PID_PREFIX}{pid}"
            old_pid_n = int(sdb.get(pid_key) or 0)
            new_pid_n = old_pid_n + 1
            sdb[pid_key] = new_pid_n
            if old_pid_n == 0:
                totals["paper_count"] += 1

            model_key = f"{_SUMMARY_CACHE_MODEL_PREFIX}{model}"
            old_model_n = int(sdb.get(model_key) or 0)
            sdb[model_key] = old_model_n + 1

            totals["total"] += 1
            totals["updated_time"] = time.time()
            _write_totals(sdb, totals)

            model_counts = _iter_model_counts(sdb)
            snapshot = _build_snapshot_data(totals["total"], totals["paper_count"], model_counts)
            _write_snapshot(snapshot, duration=0.0)


def summary_cache_stats_decrement(pid: str, model: str) -> None:
    pid = (pid or "").strip()
    model = (model or "").strip()
    if not pid or not model:
        return

    with _with_stats_lock(timeout_s=10):
        with _get_stats_db("c") as sdb:
            totals = _read_totals(sdb)

            pid_key = f"{_SUMMARY_CACHE_PID_PREFIX}{pid}"
            old_pid_n = int(sdb.get(pid_key) or 0)

            model_key = f"{_SUMMARY_CACHE_MODEL_PREFIX}{model}"
            old_model_n = int(sdb.get(model_key) or 0)

            # If counters are missing, don't mutate them; just mark stale so full refresh can fix.
            if old_pid_n <= 0 or old_model_n <= 0:
                try:
                    snap = sdb.get(_SUMMARY_CACHE_STATS_KEY)
                    if isinstance(snap, dict):
                        sdb[_SUMMARY_CACHE_STATS_KEY] = {**snap, "updated_time": 0.0}
                except Exception:
                    pass
                totals["updated_time"] = 0.0
                _write_totals(sdb, totals)
                return

            if old_pid_n > 0:
                new_pid_n = old_pid_n - 1
                if new_pid_n <= 0:
                    try:
                        del sdb[pid_key]
                    except Exception:
                        sdb[pid_key] = 0
                    if totals["paper_count"] > 0:
                        totals["paper_count"] -= 1
                else:
                    sdb[pid_key] = new_pid_n

            if old_model_n > 0:
                new_model_n = old_model_n - 1
                if new_model_n <= 0:
                    try:
                        del sdb[model_key]
                    except Exception:
                        sdb[model_key] = 0
                else:
                    sdb[model_key] = new_model_n

            if totals["total"] > 0:
                totals["total"] = max(0, totals["total"] - 1)
            totals["updated_time"] = time.time()
            _write_totals(sdb, totals)

            model_counts = _iter_model_counts(sdb)
            snapshot = _build_snapshot_data(totals["total"], totals["paper_count"], model_counts)
            _write_snapshot(snapshot, duration=0.0)


def refresh_summary_cache_stats_full() -> dict:
    """Full rebuild from disk, then rewrite pid/model/totals + snapshot."""
    start = time.time()
    data = compute_summary_cache_stats()
    duration = time.time() - start

    # Extract computed counts.
    model_counts = data.get("_model_counts") if isinstance(data, dict) else None
    pid_counts = data.get("_pid_counts") if isinstance(data, dict) else None
    if not isinstance(model_counts, dict) or not isinstance(pid_counts, dict):
        # Fall back to legacy output.
        model_counts = {}
        pid_counts = {}
        for row in data.get("summary_cache_model_counts") or []:
            try:
                model_counts[str(row.get("model"))] = int(row.get("count") or 0)
            except Exception:
                continue

    total = int(data.get("summary_cache_total") or 0)
    paper_count = int(data.get("summary_cache_paper_count") or 0)

    with _with_stats_lock(timeout_s=30):
        with _get_stats_db("c") as sdb:
            # Clear existing pid/model keys.
            try:
                for key, _ in list(sdb.items_with_prefix(_SUMMARY_CACHE_PID_PREFIX)):
                    try:
                        del sdb[key]
                    except Exception:
                        pass
                for key, _ in list(sdb.items_with_prefix(_SUMMARY_CACHE_MODEL_PREFIX)):
                    try:
                        del sdb[key]
                    except Exception:
                        pass
            except Exception:
                pass

            for pid, n in pid_counts.items():
                try:
                    n = int(n or 0)
                except Exception:
                    continue
                if n <= 0:
                    continue
                sdb[f"{_SUMMARY_CACHE_PID_PREFIX}{pid}"] = n

            for model, n in model_counts.items():
                try:
                    n = int(n or 0)
                except Exception:
                    continue
                if n <= 0:
                    continue
                sdb[f"{_SUMMARY_CACHE_MODEL_PREFIX}{model}"] = n

            totals = {
                "total": total,
                "paper_count": paper_count,
                "updated_time": time.time(),
                "last_full_scan_time": time.time(),
            }
            _write_totals(sdb, totals)
            snapshot = _build_snapshot_data(total, paper_count, model_counts)
            _write_snapshot(snapshot, duration=duration)

    return {"data": data, "updated_time": time.time(), "duration": duration}


# TL;DR cache


class _TLDRCache:
    def __init__(self, maxsize: int = 2000, ttl_s: float = 600.0):
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._data = {}
        self._lock = Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self._ttl_s:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str):
        with self._lock:
            if len(self._data) >= self._maxsize:
                oldest = min(self._data.items(), key=lambda x: x[1][0])
                del self._data[oldest[0]]
            self._data[key] = (time.time(), value)


TLDR_CACHE = _TLDRCache()


def write_summary_meta(meta_path: Path, data: dict) -> None:
    """Write summary metadata to file."""
    try:
        atomic_write_json(meta_path, data)
    except Exception as e:
        logger.warning(f"Failed to write summary meta: {e}")


def public_summary_meta(meta: dict) -> dict:
    """Filter summary metadata for client responses."""
    if not isinstance(meta, dict):
        return {}
    allowed = ("generated_at", "source", "llm", "llm_model", "llm_fallback_attempts")
    return {key: meta.get(key) for key in allowed if meta.get(key) is not None}


def sanitize_summary_meta(meta: dict) -> dict:
    """Remove internal fields from summary metadata."""
    if not isinstance(meta, dict):
        return {}
    clean = dict(meta)
    clean.pop("prompt", None)
    clean.pop("updated_at", None)
    clean.pop("quality", None)
    clean.pop("chinese_ratio", None)
    clean.pop("model", None)
    return clean


# Summary cache stats
_SUMMARY_CACHE_STATS_LOCK = Lock()
_SUMMARY_CACHE_STATS = {
    "updated_time": 0.0,
    "data": None,
    "in_progress": False,
    "duration": 0.0,
}


def get_summary_status(pid: str, model: str | None = None) -> tuple[str, str | None]:
    """Return (status, last_error) for summary generation."""
    model = (model or LLM_NAME or "").strip()
    if not model:
        return "", None

    summary_source = normalize_summary_source(SUMMARY_MARKDOWN_SOURCE)
    cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(pid, model)

    if cache_file.exists() or legacy_cache.exists():
        meta = read_summary_meta(meta_file) if meta_file.exists() else read_summary_meta(legacy_meta)
        if summary_source_matches(meta, summary_source):
            return "ok", None

    if lock_file.exists() or legacy_lock.exists():
        return "running", None

    try:
        info = SummaryStatusRepository.get_status(pid, model)
        if isinstance(info, dict):
            status = info.get("status") or ""
            last_error = info.get("last_error")
            return status, last_error
    except Exception as e:
        logger.warning(f"Failed to read summary status for {pid}: {e}")

    return "", None


def extract_tldr_from_summary(pid: str) -> str:
    """Extract TL;DR from cached summary file."""
    from backend.utils.summary_utils import read_tldr_from_summary_file

    cached = TLDR_CACHE.get(pid)
    if cached is not None:
        return cached

    tldr = read_tldr_from_summary_file(pid)
    if tldr:
        TLDR_CACHE.set(pid, tldr)
    return tldr


def compute_summary_cache_stats() -> dict:
    """Compute summary cache statistics."""
    cache_models = defaultdict(int)
    pid_counts = defaultdict(int)
    cache_total = 0
    summary_dir = Path(SUMMARY_DIR)

    if summary_dir.exists():
        for entry in summary_dir.iterdir():
            if entry.is_dir():
                pid = entry.name
                for meta_path in entry.glob("*.meta.json"):
                    meta = read_summary_meta(meta_path)
                    model = (meta.get("model") or meta.get("llm_model") or "").strip()
                    if not model:
                        name = meta_path.name
                        model = name[: -len(".meta.json")] if name.endswith(".meta.json") else meta_path.stem
                    cache_models[model] += 1
                    cache_total += 1
                    pid_counts[pid] += 1
            elif entry.is_file() and entry.name.endswith(".meta.json") and not entry.name.startswith("."):
                pid = entry.name[: -len(".meta.json")]
                meta = read_summary_meta(entry)
                model = (meta.get("model") or meta.get("llm_model") or "").strip() or "legacy"
                cache_models[model] += 1
                cache_total += 1
                pid_counts[pid] += 1

    model_counts = dict(cache_models)
    pid_counts_dict = dict(pid_counts)

    return {
        "summary_cache_total": int(cache_total),
        "summary_cache_paper_count": int(len(pid_counts_dict)),
        "summary_cache_model_counts": [
            {"model": model, "count": int(count)}
            for model, count in sorted(model_counts.items(), key=lambda x: (-x[1], x[0]))
        ],
        # For full rebuild
        "_model_counts": model_counts,
        "_pid_counts": pid_counts_dict,
    }


def get_summary_cache_stats(ttl: int = 300) -> dict:
    """Get summary cache stats with caching."""
    import threading

    # Fast path: persisted snapshot (shared across processes)
    snap = _read_snapshot()
    if isinstance(snap, dict):
        pdata = snap.get("data")
        pupdated = float(snap.get("updated_time") or 0.0)
        pduration = float(snap.get("duration") or 0.0)
        now = time.time()
        if pdata and (now - pupdated) < ttl:
            return {"data": pdata, "updated_time": pupdated, "in_progress": False, "duration": pduration, "ttl": ttl}

    now = time.time()
    with _SUMMARY_CACHE_STATS_LOCK:
        data = _SUMMARY_CACHE_STATS.get("data")
        updated_time = _SUMMARY_CACHE_STATS.get("updated_time", 0.0)
        duration = _SUMMARY_CACHE_STATS.get("duration", 0.0)

    if data and (now - updated_time) < ttl:
        return {"data": data, "updated_time": updated_time, "in_progress": False, "duration": duration, "ttl": ttl}

    if data:
        # Refresh async
        def _run():
            try:
                result = refresh_summary_cache_stats_full()
                new_data = result.get("data") if isinstance(result, dict) else None
                dur = float(result.get("duration") or 0.0) if isinstance(result, dict) else 0.0
                if isinstance(new_data, dict):
                    with _SUMMARY_CACHE_STATS_LOCK:
                        _SUMMARY_CACHE_STATS["data"] = new_data
                        _SUMMARY_CACHE_STATS["updated_time"] = time.time()
                        _SUMMARY_CACHE_STATS["duration"] = dur
            except Exception as e:
                logger.warning(f"Failed to refresh summary cache stats: {e}")
            finally:
                with _SUMMARY_CACHE_STATS_LOCK:
                    _SUMMARY_CACHE_STATS["in_progress"] = False

        with _SUMMARY_CACHE_STATS_LOCK:
            if not _SUMMARY_CACHE_STATS["in_progress"]:
                _SUMMARY_CACHE_STATS["in_progress"] = True
                threading.Thread(target=_run, daemon=True).start()

        return {"data": data, "updated_time": updated_time, "in_progress": True, "duration": duration, "ttl": ttl}

    # Try quick assemble from totals+model counts without disk scan.
    try:
        with _get_stats_db("r") as sdb:
            totals = _read_totals(sdb)
            model_counts = _iter_model_counts(sdb)
        if totals.get("total") or totals.get("paper_count") or model_counts:
            assembled = _build_snapshot_data(totals["total"], totals["paper_count"], model_counts)
            with _SUMMARY_CACHE_STATS_LOCK:
                _SUMMARY_CACHE_STATS["data"] = assembled
                _SUMMARY_CACHE_STATS["updated_time"] = time.time()
                _SUMMARY_CACHE_STATS["duration"] = 0.0
                updated_time = _SUMMARY_CACHE_STATS["updated_time"]
            return {"data": assembled, "updated_time": updated_time, "in_progress": False, "duration": 0.0, "ttl": ttl}
    except Exception:
        pass

    # First time: do a synchronous full rebuild
    result = refresh_summary_cache_stats_full()
    new_data = result.get("data") if isinstance(result, dict) else None
    dur = float(result.get("duration") or 0.0) if isinstance(result, dict) else 0.0
    if not isinstance(new_data, dict):
        new_data = compute_summary_cache_stats()
    with _SUMMARY_CACHE_STATS_LOCK:
        _SUMMARY_CACHE_STATS["data"] = new_data
        _SUMMARY_CACHE_STATS["updated_time"] = time.time()
        _SUMMARY_CACHE_STATS["duration"] = dur
        _SUMMARY_CACHE_STATS["in_progress"] = False
        updated_time = _SUMMARY_CACHE_STATS["updated_time"]

    return {"data": new_data, "updated_time": updated_time, "in_progress": False, "duration": dur, "ttl": ttl}


def safe_unlink(path: Path) -> bool:
    """Safely unlink a file."""
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")
    return False


def safe_rmtree(path: Path) -> bool:
    """Safely remove a directory tree."""
    if path.exists():
        try:
            shutil.rmtree(path)
            return True
        except Exception as e:
            logger.warning(f"Failed to remove directory {path}: {e}")
    return False


def clear_model_summary(pid: str, model: str, metas_getter=None, user: str | None = None):
    """Clear summary cache for a specific model."""
    meta = metas_getter().get(pid.split("v")[0] if "v" in pid else pid) if metas_getter and pid else None
    cache_pid, raw_pid, _ = resolve_cache_pid(pid, meta)
    ids_to_clear = {cache_pid, raw_pid}
    model = (model or "").strip()
    cleared = False
    cancel_info = {"canceled_task_ids": [], "epoch": 0}

    # Best-effort: cancel any in-flight summary tasks for this pid+model.
    try:
        from tasks import cancel_summary_tasks

        cancel_info = cancel_summary_tasks(raw_pid, model, user=user, reason="Canceled by user (clear current summary)")
    except Exception:
        cancel_info = {"canceled_task_ids": [], "epoch": 0}

    # Best-effort: check whether a summary for (pid, model) exists BEFORE deletion.
    # Track per pid so we can decrement accurately when both cache_pid/raw_pid have entries.
    had_model_summary_by_pid: dict[str, bool] = {}
    for paper_id in ids_to_clear:
        try:
            cache_file, meta_file, _, legacy_cache, legacy_meta, _ = summary_cache_paths(paper_id, model)
            if cache_file.exists() or meta_file.exists():
                had_model_summary_by_pid[paper_id] = True
                continue
            if legacy_meta.exists():
                legacy_meta_data = read_summary_meta(legacy_meta)
                legacy_model = _normalize_model_for_stats(None, legacy_meta_data)
                if legacy_model and legacy_model == model:
                    # compute_summary_cache_stats counts legacy meta.json.
                    had_model_summary_by_pid[paper_id] = True
        except Exception:
            continue

    for paper_id in ids_to_clear:
        cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(paper_id, model)
        for file_path in (cache_file, meta_file, lock_file):
            if safe_unlink(file_path):
                cleared = True

        if legacy_meta.exists():
            try:
                legacy_meta_data = read_summary_meta(legacy_meta)
            except Exception:
                legacy_meta_data = {}
            legacy_model = (legacy_meta_data.get("model") or legacy_meta_data.get("llm_model") or "").strip()
            if legacy_model and legacy_model == model:
                for file_path in (legacy_cache, legacy_meta, legacy_lock):
                    safe_unlink(file_path)
                cleared = True

        try:
            if cancel_info.get("canceled_task_ids"):
                SummaryStatusRepository.set_status(
                    paper_id,
                    model,
                    "canceled",
                    "Canceled by user (clear current summary)",
                    task_id=None,
                    task_user=None,
                )
            else:
                SummaryStatusRepository.delete_status(paper_id, model)
            cleared = True
        except Exception as e:
            logger.warning(f"Failed to clear summary status for {paper_id}::{model}: {e}")

    if cleared:
        logger.debug(f"Cleared summary for model '{model}' for paper {pid}")
        try:
            did_decrement = False
            for paper_id in ids_to_clear:
                if had_model_summary_by_pid.get(paper_id):
                    summary_cache_stats_decrement(paper_id, model)
                    did_decrement = True
            if not did_decrement:
                invalidate_summary_cache_stats()
        except Exception:
            invalidate_summary_cache_stats()


def clear_paper_cache(pid: str, metas_getter=None, user: str | None = None):
    """Clear all caches for a paper."""
    meta = metas_getter().get(pid.split("v")[0] if "v" in pid else pid) if metas_getter and pid else None
    cache_pid, raw_pid, _ = resolve_cache_pid(pid, meta)
    ids_to_clear = {cache_pid, raw_pid}

    # Best-effort: cancel any in-flight summary tasks for this paper (all models).
    try:
        from tasks import cancel_paper_summary_tasks

        cancel_paper_summary_tasks(raw_pid, user=user, reason="Canceled by user (clear all caches)")
    except Exception:
        pass

    # Collect all per-model summaries under each cache dir for decrement.
    # Use meta['model'/'llm_model'] when present; fallback to filename for older caches.
    per_pid_models: dict[str, set[str]] = {}
    try:
        for paper_id in ids_to_clear:
            cache_dir = Path(SUMMARY_DIR) / paper_id
            if cache_dir.exists() and cache_dir.is_dir():
                models = set()
                for meta_path in cache_dir.glob("*.meta.json"):
                    meta_d = read_summary_meta(meta_path)
                    m = _normalize_model_for_stats(None, meta_d)
                    if not m:
                        name = meta_path.name
                        m = name[: -len(".meta.json")] if name.endswith(".meta.json") else meta_path.stem
                    if m:
                        models.add(m)
                if models:
                    per_pid_models[paper_id] = models
            # Legacy root meta.json
            legacy_meta = Path(SUMMARY_DIR) / f"{paper_id}.meta.json"
            if legacy_meta.exists():
                meta_d = read_summary_meta(legacy_meta)
                m = _normalize_model_for_stats(None, meta_d) or "legacy"
                if m:
                    per_pid_models.setdefault(paper_id, set()).add(m)
    except Exception:
        per_pid_models = {}

    for paper_id in ids_to_clear:
        safe_rmtree(Path(SUMMARY_DIR) / paper_id)
        for path in (
            Path(SUMMARY_DIR) / f"{paper_id}.md",
            Path(SUMMARY_DIR) / f"{paper_id}.meta.json",
            Path(SUMMARY_DIR) / f".{paper_id}.lock",
        ):
            safe_unlink(path)

        # Clear all persisted per-model status records for this paper to avoid stale UI states.
        try:
            from aslite.db import get_summary_status_db

            with get_summary_status_db(flag="c") as sdb:
                keys = [k for k, _v in sdb.items_with_prefix(f"{paper_id}::")]
                for k in keys:
                    try:
                        del sdb[k]
                    except Exception:
                        pass
        except Exception:
            pass
        if safe_rmtree(Path(DATA_DIR) / "html_md" / paper_id):
            logger.debug(f"Cleared HTML cache for {paper_id}")
        if safe_rmtree(Path(DATA_DIR) / "mineru" / paper_id):
            logger.debug(f"Cleared MinerU cache for {paper_id}")
            # For uploaded papers, reset parse_status since MinerU cache is required
            if paper_id.startswith("up_"):
                try:
                    from aslite.repositories import UploadedPaperRepository

                    UploadedPaperRepository.update(
                        paper_id,
                        {"parse_status": "pending", "parse_error": "Cache cleared, re-parsing required"},
                    )
                    logger.debug(f"Reset parse_status for uploaded paper {paper_id}")
                except Exception as e:
                    logger.warning(f"Failed to reset parse_status for {paper_id}: {e}")

    # Best-effort decrement based on collected meta files.
    try:
        for paper_id in ids_to_clear:
            for m in per_pid_models.get(paper_id, set()):
                if m:
                    summary_cache_stats_decrement(paper_id, m)
    except Exception:
        invalidate_summary_cache_stats()

    invalidate_summary_cache_stats()


# -----------------------------------------------------------------------------
# Summary generation
# -----------------------------------------------------------------------------


class SummaryCacheMiss(Exception):
    """Raised when summary cache is not found and cache_only=True."""


def generate_paper_summary(
    pid: str,
    model: str | None = None,
    force_refresh: bool = False,
    cache_only: bool = False,
    *,
    metas_getter: Callable[[], dict[str, Any]] | None = None,
    paper_exists_fn: Callable[[str], bool] | None = None,
) -> tuple[str, dict]:
    """
    Generate paper summary with intelligent caching mechanism.

    1. Check if SUMMARY_DIR/{pid}.md exists
    2. If exists, return cached summary (unless force_refresh=True)
    3. If not exists, call paper_summarizer to generate and cache the summary

    Args:
        pid: Paper ID
        model: LLM model name
        force_refresh: Force regeneration even if cached
        cache_only: Only return cached summary, raise SummaryCacheMiss if not found
        metas_getter: Function to get paper metas (for dependency injection)
        paper_exists_fn: Function to check if paper exists (for dependency injection)

    Returns:
        Tuple of (summary_content, summary_meta)
    """
    try:
        # Use shared resolve_cache_pid with local meta lookup
        meta = metas_getter().get(pid.split("v")[0] if "v" in pid else pid) if metas_getter and pid else None
        cache_pid, raw_pid, has_explicit_version = resolve_cache_pid(pid, meta)

        if paper_exists_fn and not paper_exists_fn(raw_pid):
            return "# Error\n\nPaper not found.", {}

        summary_source = normalize_summary_source(SUMMARY_MARKDOWN_SOURCE)

        # Cooperative cancellation epoch snapshot (bumped by clear actions).
        start_epoch = 0
        try:
            start_epoch = SummaryStatusRepository.get_generation_epoch(cache_pid, model or "")
        except Exception:
            start_epoch = 0

        # Define cache file path
        cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(cache_pid, model)
        if legacy_cache.exists() and not cache_file.exists():
            lock_file = legacy_lock

        # Ensure cache directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        def _read_from_paths(body_path: Path, meta_path: Path):
            if not body_path.exists():
                return None, {}
            try:
                with open(body_path, encoding="utf-8") as f:
                    cached = f.read()
                cached = cached if cached.strip() else None
                meta = read_summary_meta(meta_path)
                # Backfill generated_at for old caches without mutating meaning.
                if "generated_at" not in meta:
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

            legacy_cached, legacy_meta_data = _read_from_paths(legacy_cache, legacy_meta)
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
                logger.debug(f"Using cached paper summary: {pid}")
                return cached_summary, sanitize_summary_meta(cached_meta)

        if cache_only:
            # If another request is generating the same cache (lock held), tell client.
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
                return cached_summary, sanitize_summary_meta(cached_meta)
            return "# Error\n\nSummary is being generated, please retry shortly.", {}

        try:
            # If canceled while waiting for lock, abort without writing caches.
            try:
                if SummaryStatusRepository.get_generation_epoch(cache_pid, model or "") != start_epoch:
                    return "# Error\n\nSummary canceled.", {}
            except Exception:
                pass

            cached_summary, cached_meta = _read_cached_summary()
            if cached_summary and not force_refresh:
                if not summary_source_matches(cached_meta, summary_source):
                    cached_summary = None
                else:
                    logger.debug(f"Using cached paper summary after lock: {pid}")
                    return cached_summary, sanitize_summary_meta(cached_meta)

            if cache_only:
                raise SummaryCacheMiss("Summary cache not found")

            # Generate new summary using paper_summarizer module
            logger.debug(f"Generating new paper summary: {pid}")
            if summary_source == "html":
                pid_for_summary = pid if has_explicit_version else cache_pid
            else:
                pid_for_summary = pid if has_explicit_version else raw_pid

            summary_result = generate_paper_summary_from_module(pid_for_summary, source=summary_source, model=model)
            summary_content, summary_meta = normalize_summary_result(summary_result)
            summary_meta = summary_meta if isinstance(summary_meta, dict) else {}
            response_meta = sanitize_summary_meta(summary_meta)

            # If paper_summarizer fell back to a different model, cache under the
            # actual model id so future requests hit the right file.
            actual_model = (summary_meta.get("llm_model") or "").strip()
            if actual_model and model and actual_model != model:
                cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(
                    cache_pid, actual_model
                )
                if legacy_cache.exists() and not cache_file.exists():
                    lock_file = legacy_lock
                cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Decide the stats model id and whether this pid+model already existed.
            stat_model = _normalize_model_for_stats(actual_model or model, summary_meta)
            existed_before = False
            if stat_model:
                try:
                    _cf, _mf, _lf, _lc, _lm, _ll = summary_cache_paths(cache_pid, stat_model)
                    existed_before = _cf.exists() or _lc.exists()
                except Exception:
                    existed_before = cache_file.exists() or legacy_cache.exists()
            else:
                existed_before = cache_file.exists() or legacy_cache.exists()

            # Only cache successful summaries (not error messages)
            is_error = summary_content.startswith("# Error") or summary_content.startswith(
                "# PDF Parsing Service Unavailable"
            )

            # If canceled after generation, do not write cache and do not return a "ready" signal.
            # Note: cancellation is keyed by the requested model epoch snapshot.
            try:
                if SummaryStatusRepository.get_generation_epoch(cache_pid, model or "") != start_epoch:
                    return "# Error\n\nSummary canceled.", {}
            except Exception:
                pass

            if not is_error:
                try:
                    atomic_write_text(cache_file, summary_content)
                    meta = {}
                    meta.update(summary_meta)
                    meta.setdefault("source", summary_source)
                    meta.setdefault("generated_at", time.time())
                    if stat_model:
                        meta.setdefault("model", stat_model)
                    write_summary_meta(meta_file, meta)
                    logger.debug(f"Paper summary cached to: {cache_file}")
                    response_meta = sanitize_summary_meta(meta)
                    # Incremental coverage update (best-effort).
                    try:
                        if not existed_before and stat_model:
                            summary_cache_stats_increment(cache_pid, stat_model)
                        else:
                            # Conservative: mark stale; periodic full scan will correct.
                            invalidate_summary_cache_stats()
                    except Exception:
                        invalidate_summary_cache_stats()
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
