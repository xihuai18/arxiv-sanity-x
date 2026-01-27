from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huey import SqliteHuey
from loguru import logger

from aslite.repositories import (
    MetaRepository,
    ReadingListRepository,
    SummaryStatusRepository,
)
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
    split_pid_version,
    summary_cache_paths,
    summary_source_matches,
)

DATA_DIR = str(settings.data_dir)
LLM_NAME = settings.llm.name
SUMMARY_MARKDOWN_SOURCE = settings.summary.markdown_source


def _is_huey_consumer_process() -> bool:
    """Best-effort detection of a Huey consumer process.

    Huey runs tasks in a separate process, so in-process SSE emits are ineffective.
    """
    try:
        return any("huey" in (arg or "").lower() for arg in sys.argv)
    except Exception:
        return False


# SSE can only reach clients in the web worker process. Disable SSE in Huey consumer.
_TASKS_SSE_ENABLED = settings.huey.tasks_sse_enabled and not _is_huey_consumer_process()


# Import SSE event emitters lazily to avoid circular imports.
def _emit_user_event(user, payload):
    if not _TASKS_SSE_ENABLED:
        return
    try:
        from backend.utils.sse import emit_user_event as _emit

        _emit(user, payload)
    except Exception:
        pass


def _emit_all_event(payload):
    if not _TASKS_SSE_ENABLED:
        return
    try:
        from backend.utils.sse import emit_all_event as _emit

        _emit(payload)
    except Exception:
        pass


def _allow_task_id(task_user: str | None, request_user: str | None) -> bool:
    """Whether it's safe to return a task_id to the requester."""
    if not task_user:
        return True
    return bool(request_user) and task_user == request_user


def _find_active_task(pid: str, model: str) -> tuple[str | None, str | None]:
    """Return (task_id, task_user) for an active queued/running task, if any."""
    try:
        for key, info in SummaryStatusRepository.get_items_with_prefix("task::"):
            if not isinstance(info, dict):
                continue
            task_status = info.get("status")
            if task_status not in ("queued", "running"):
                continue
            if info.get("pid") != pid or info.get("model") != model:
                continue
            return key.replace("task::", ""), info.get("user")
    except Exception as e:
        logger.warning(f"Failed to scan active tasks for {pid}::{model}: {e}")
    return None, None


def _enqueue_lock_path(pid: str, model: str) -> Path:
    # Reuse model_key sanitization from summary_cache_paths()
    _cache_file, _meta_file, lock_file, _legacy_cache, _legacy_meta, _legacy_lock = summary_cache_paths(pid, model)
    name = lock_file.name
    model_key = name[1 : -len(".lock")] if (name.startswith(".") and name.endswith(".lock")) else "default"
    return lock_file.parent / f".{model_key}.enqueue.lock"


HUEY_DB_PATH = settings.huey.db_path or os.path.join(DATA_DIR, "huey.db")

SUMMARY_PRIORITY_HIGH = settings.huey.summary_priority_high
SUMMARY_PRIORITY_LOW = settings.huey.summary_priority_low

HUEY_SQLITE_TIMEOUT = (
    float(settings.huey.sqlite_timeout_worker)
    if _is_huey_consumer_process()
    else float(settings.huey.sqlite_timeout_web)
)
huey = SqliteHuey("arxiv-sanity", filename=HUEY_DB_PATH, timeout=HUEY_SQLITE_TIMEOUT)


def _is_upload_pid(pid: str) -> bool:
    """Check if pid is an uploaded paper ID."""
    return bool(pid and pid.startswith("up_"))


def _paper_exists(pid: str) -> bool:
    # Check for upload pid first
    if _is_upload_pid(pid):
        from aslite.repositories import UploadedPaperRepository

        record = UploadedPaperRepository.get(pid)
        return record is not None and record.get("parse_status") == "ok"

    raw_pid, _ = split_pid_version(pid)
    if not raw_pid:
        return False
    try:
        return MetaRepository.get_by_id(raw_pid) is not None
    except Exception as e:
        logger.warning(f"Failed to check paper existence for {pid}: {e}")
        return False


def _update_summary_status_db(
    pid: str,
    model: str | None,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
    task_user: str | None = None,
) -> None:
    model = (model or LLM_NAME or "").strip()
    if not model:
        return
    try:
        extra = {}
        if task_id is not None:
            extra["task_id"] = str(task_id)
        if task_user is not None:
            extra["task_user"] = task_user
        if status not in ("queued", "running"):
            extra["task_id"] = None
            extra["task_user"] = None
        SummaryStatusRepository.set_status(pid, model, status, error, **extra)
        _emit_all_event({"type": "summary_status", "pid": pid, "status": status, "error": error})
    except Exception as e:
        logger.warning(f"Failed to update summary status db for {pid}: {e}")


def _update_task_status(task_id: str | None, status: str, error: str | None = None, **extra) -> None:
    if not task_id:
        return
    try:
        SummaryStatusRepository.set_task_status(task_id, status, error, **extra)
    except Exception as e:
        logger.warning(f"Failed to update task status {task_id}: {e}")


def _update_readinglist_summary_status(
    user: str,
    pid: str,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
) -> None:
    if not user:
        return
    try:
        if ReadingListRepository.get_reading_list_item(user, pid) is None:
            return
        updates = {
            "summary_status": status,
            "summary_last_error": error,
            "summary_updated_time": time.time(),
        }
        if status not in ("queued", "running"):
            updates["summary_task_id"] = None
        elif task_id is not None:
            updates["summary_task_id"] = str(task_id)
        ReadingListRepository.update_reading_list_item(user, pid, updates)
        payload = {"type": "summary_status", "pid": pid, "status": status, "error": error}
        if task_id is not None:
            payload["task_id"] = str(task_id)
        _emit_user_event(user, payload)
    except Exception as e:
        logger.warning(f"Failed to update summary status for {user}:{pid}: {e}")


def _read_cached_summary(cache_file: Path, meta_file: Path, legacy_cache: Path, legacy_meta: Path, model: str):
    def _read_from_paths(body_path: Path, meta_path: Path, inject_model: bool = True):
        if not body_path.exists():
            return None, {}
        try:
            with open(body_path, encoding="utf-8") as f:
                cached = f.read()
            cached = cached if cached.strip() else None
            meta = read_summary_meta(meta_path)
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
            if inject_model and "model" not in meta:
                meta["model"] = model
            if not cached:
                return None, {}
            return cached, meta
        except Exception as e:
            logger.warning(f"Failed to read cached summary: {e}")
            return None, {}

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


def _is_error_summary(summary_content: str) -> bool:
    if not summary_content:
        return True
    return summary_content.startswith("# Error") or summary_content.startswith("# PDF Parsing Service Unavailable")


def _generate_and_cache_summary(pid: str, model: str) -> tuple[str, dict]:
    cache_pid, raw_pid, has_explicit_version = resolve_cache_pid(pid, None)
    if not _paper_exists(raw_pid):
        return "# Error\n\nPaper not found.", {}

    summary_source = normalize_summary_source(SUMMARY_MARKDOWN_SOURCE)

    cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = summary_cache_paths(cache_pid, model)
    if legacy_cache.exists() and not cache_file.exists():
        lock_file = legacy_lock

    existed_before = False
    try:
        _cf, _mf, _lf, _lc, _lm, _ll = summary_cache_paths(cache_pid, model)
        existed_before = _cf.exists() or _lc.exists()
    except Exception:
        existed_before = cache_file.exists() or legacy_cache.exists()

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached_summary, cached_meta = _read_cached_summary(cache_file, meta_file, legacy_cache, legacy_meta, model)
    if cached_summary:
        if summary_source_matches(cached_meta, summary_source):
            return cached_summary, cached_meta

    lock_fd = acquire_summary_lock(lock_file, timeout_s=300)
    if lock_fd is None:
        if cached_summary:
            return cached_summary, cached_meta
        return "# Error\n\nSummary is being generated, please retry shortly.", {}

    # Keep track of the original lock file for proper release
    acquired_lock_file = lock_file

    try:
        cached_summary, cached_meta = _read_cached_summary(cache_file, meta_file, legacy_cache, legacy_meta, model)
        if cached_summary:
            if summary_source_matches(cached_meta, summary_source):
                return cached_summary, cached_meta

        if summary_source == "html":
            pid_for_summary = pid if has_explicit_version else cache_pid
        else:
            pid_for_summary = pid if has_explicit_version else raw_pid

        summary_result = generate_paper_summary_from_module(pid_for_summary, source=summary_source, model=model)
        summary_content, summary_meta = normalize_summary_result(summary_result)
        summary_meta = summary_meta if isinstance(summary_meta, dict) else {}

        actual_model = (summary_meta.get("llm_model") or "").strip()
        # Use actual_model for caching/stats if LLM fell back to a different model
        stat_model = actual_model if actual_model else model
        if actual_model and model and actual_model != model:
            # Model changed, update cache paths but keep original lock for release
            cache_file, meta_file, _, legacy_cache, legacy_meta, _ = summary_cache_paths(cache_pid, actual_model)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Re-check if this actual_model cache already existed
            try:
                existed_before = cache_file.exists() or legacy_cache.exists()
            except Exception:
                pass

        if not _is_error_summary(summary_content):
            meta = {}
            meta.update(summary_meta)
            meta.setdefault("source", summary_source)
            meta.setdefault("generated_at", time.time())
            meta.setdefault("model", stat_model)  # Use actual model for meta
            atomic_write_text(cache_file, summary_content)
            atomic_write_json(meta_file, meta)
            # Incremental coverage update (best-effort).
            try:
                from backend.services.summary_service import (
                    invalidate_summary_cache_stats,
                    summary_cache_stats_increment,
                )

                if not existed_before:
                    summary_cache_stats_increment(cache_pid, stat_model)  # Use actual model for stats
                else:
                    invalidate_summary_cache_stats()
            except Exception:
                pass
            return summary_content, meta

        return summary_content, summary_meta
    finally:
        release_summary_lock(lock_fd, acquired_lock_file)


@huey.task(retries=3, retry_delay=60, context=True)
def generate_summary_task(pid: str, model: str | None = None, user: str | None = None, task=None) -> None:
    model = (model or LLM_NAME or "").strip()
    task_id = getattr(task, "id", None)

    max_attempts = 1
    try:
        max_attempts = int(getattr(task, "retries", 0) or 0) + 1
    except Exception:
        max_attempts = 1

    attempt = 1
    try:
        existing = SummaryStatusRepository.get_task_status(str(task_id)) if task_id else None
        if isinstance(existing, dict):
            attempt = int(existing.get("attempt") or 0) + 1
    except Exception:
        attempt = 1

    _update_task_status(
        task_id,
        "running",
        pid=pid,
        model=model,
        user=user,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    _update_summary_status_db(pid, model, "running", None, task_id=task_id, task_user=user)
    if user:
        _update_readinglist_summary_status(user, pid, "running", None, task_id=task_id)

    try:
        # Uploaded papers are private; ensure the task user owns the paper.
        if _is_upload_pid(pid):
            from aslite.repositories import UploadedPaperRepository

            record = UploadedPaperRepository.get(pid)
            if not user or not record or record.get("owner") != user:
                raise RuntimeError("Uploaded paper not found")
            if record.get("parse_status") != "ok":
                raise RuntimeError(f"Uploaded paper not parsed (status: {record.get('parse_status')})")

        summary_content, _meta = _generate_and_cache_summary(pid, model)

        # Check if this is a lock contention error (not a real failure)
        if _is_error_summary(summary_content):
            if "Summary is being generated" in summary_content:
                # This is lock contention, not a real failure - retry
                logger.info(f"Lock contention for {pid}, will retry")
                # Reflect that we're waiting on another generator rather than doing work.
                _update_task_status(
                    task_id,
                    "queued",
                    pid=pid,
                    model=model,
                    user=user,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                _update_summary_status_db(pid, model, "queued", None, task_id=task_id, task_user=user)
                if user:
                    _update_readinglist_summary_status(user, pid, "queued", None, task_id=task_id)

                # If we've exhausted our retries, mark as failed to avoid stuck states.
                if attempt >= max_attempts:
                    raise RuntimeError("Lock contention exhausted")
                raise RuntimeError("Lock contention, retrying")
            else:
                # Real generation failure
                raise RuntimeError("Summary generation failed")

        _update_task_status(task_id, "ok", pid=pid, model=model, user=user)
        _update_summary_status_db(pid, model, "ok", None, task_id=task_id, task_user=user)
        if user:
            _update_readinglist_summary_status(user, pid, "ok", None, task_id=task_id)
    except Exception as e:
        logger.warning(f"Failed to generate summary for {pid}: {e}")
        err = str(e)

        # Don't mark as failed if it's just lock contention
        if "Lock contention" in err:
            if attempt >= max_attempts:
                _update_task_status(task_id, "failed", error=err, pid=pid, model=model, user=user)
                _update_summary_status_db(pid, model, "failed", err, task_id=task_id, task_user=user)
                if user:
                    _update_readinglist_summary_status(user, pid, "failed", err, task_id=task_id)
        else:
            _update_task_status(task_id, "failed", error=err, pid=pid, model=model, user=user)
            _update_summary_status_db(pid, model, "failed", err, task_id=task_id, task_user=user)
            if user:
                _update_readinglist_summary_status(user, pid, "failed", err, task_id=task_id)

        raise


def enqueue_summary_task(
    pid: str,
    model: str | None = None,
    user: str | None = None,
    priority: int | None = None,
) -> str:
    model = (model or LLM_NAME or "").strip()
    if not model:
        raise ValueError("Model is required")

    # Normalize to raw pid to avoid duplicates between versioned/unversioned inputs.
    raw_pid, _ = split_pid_version(pid)
    pid = raw_pid or pid

    # Fast path: if status record already points to an active task, return it without scanning.
    try:
        info = SummaryStatusRepository.get_status(pid, model)
        if isinstance(info, dict) and (info.get("status") in ("queued", "running")):
            existing_task_id = info.get("task_id")
            existing_task_user = info.get("task_user")
            if existing_task_id and _allow_task_id(existing_task_user, user):
                return str(existing_task_id)
            if existing_task_id and not _allow_task_id(existing_task_user, user):
                return ""
    except Exception:
        pass

    # Serialize enqueue across processes to reduce race-induced duplicate tasks.
    enqueue_lock = _enqueue_lock_path(pid, model)
    lock_fd = acquire_summary_lock(enqueue_lock, timeout_s=2)
    if lock_fd is None:
        # Another process is enqueueing this pid/model. Avoid enqueueing duplicates.
        existing_task_id, existing_task_user = _find_active_task(pid, model)
        if existing_task_id:
            if _allow_task_id(existing_task_user, user):
                return existing_task_id
            return ""
        # No active task found; best-effort: let caller fall back to summary_status polling.
        logger.warning(f"Enqueue lock timeout for {pid}::{model}; skipping enqueue to avoid duplicates")
        return ""

    try:
        # Check if there's already a queued/running task for this pid+model.
        existing_task_id, existing_task_user = _find_active_task(pid, model)
        if existing_task_id:
            if _allow_task_id(existing_task_user, user):
                logger.debug(f"Task already exists for {pid}::{model}: {existing_task_id}")
                return existing_task_id
            # Don't leak task_id across users.
            logger.debug(f"Active task exists for {pid}::{model} but belongs to a different user")
            return ""

        task_priority = SUMMARY_PRIORITY_HIGH if priority is None else priority
        task = generate_summary_task.s(pid, model=model, user=user)
        task.priority = task_priority
        result = huey.enqueue(task)
        task_id = task.id or getattr(result, "id", None)
        if task_id is None:
            logger.warning(f"Failed to obtain Huey task id for {pid}::{model}")
            return ""

        _update_task_status(task_id, "queued", pid=pid, model=model, user=user, priority=task_priority)
        _update_summary_status_db(pid, model, "queued", None, task_id=task_id, task_user=user)
        if user:
            _update_readinglist_summary_status(user, pid, "queued", None, task_id=task_id)

        return str(task_id)
    finally:
        try:
            release_summary_lock(lock_fd, enqueue_lock)
        except Exception:
            pass


def _collect_task_priority_map() -> dict:
    """Return latest priority per (pid, model) from task status records."""
    priority_map = {}
    try:
        for _key, info in SummaryStatusRepository.get_items_with_prefix("task::"):
            if not isinstance(info, dict):
                continue
            pid = info.get("pid")
            model = info.get("model")
            priority = info.get("priority")
            updated_time = info.get("updated_time") or 0
            if not pid or not model or priority is None:
                continue
            map_key = (pid, model)
            prev = priority_map.get(map_key)
            if not prev or updated_time >= prev.get("updated_time", 0):
                priority_map[map_key] = {"priority": priority, "updated_time": updated_time}
    except Exception as e:
        logger.warning(f"Failed to collect task priorities: {e}")
    return priority_map


def _collect_task_user_map() -> dict:
    """Return latest user per (pid, model) from task status records."""
    user_map = {}
    try:
        for _key, info in SummaryStatusRepository.get_items_with_prefix("task::"):
            if not isinstance(info, dict):
                continue
            pid = info.get("pid")
            model = info.get("model")
            user = info.get("user")
            updated_time = info.get("updated_time") or 0
            if not pid or not model or not user:
                continue
            map_key = (pid, model)
            prev = user_map.get(map_key)
            if not prev or updated_time >= prev.get("updated_time", 0):
                user_map[map_key] = {"user": user, "updated_time": updated_time}
    except Exception as e:
        logger.warning(f"Failed to collect task users: {e}")
    return user_map


def repair_stale_summary_tasks(max_age_s: int | None = None, requeue: bool = False) -> int:
    """Repair stale running summary statuses and cleanup expired lock files.

    Args:
        max_age_s: Age threshold in seconds; running entries older than this are repaired.
        requeue: If True, re-enqueue stale running summaries.

    Returns:
        Number of repaired entries.
    """
    max_age_s = int(max_age_s or settings.huey.summary_repair_ttl)
    now = time.time()
    repaired = 0
    priority_map = _collect_task_priority_map() if requeue else {}
    user_map = _collect_task_user_map()

    try:
        for key, info in SummaryStatusRepository.get_all_items():
            if key.startswith("task::"):
                continue
            if not isinstance(info, dict):
                continue
            status = info.get("status")
            updated_time = info.get("updated_time") or 0
            if status != "running":
                continue
            if now - float(updated_time) < max_age_s:
                continue

            if "::" not in key:
                continue
            pid, model = key.split("::", 1)
            if not pid or not model:
                continue

            _cache_file, _meta_file, lock_file, _legacy_cache, _legacy_meta, legacy_lock = summary_cache_paths(
                pid, model
            )
            for lock_path in (lock_file, legacy_lock):
                try:
                    if lock_path.exists():
                        lock_mtime = lock_path.stat().st_mtime
                        if now - lock_mtime >= max_age_s:
                            lock_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove stale lock {lock_path}: {e}")

            user_info = user_map.get((pid, model))
            user = user_info.get("user") if user_info else None
            task_id = info.get("task_id") if isinstance(info, dict) else None

            if requeue:
                if task_id:
                    try:
                        SummaryStatusRepository.set_task_status(
                            str(task_id),
                            "failed",
                            error="stale_running_repaired",
                            pid=pid,
                            model=model,
                            user=user,
                        )
                    except Exception:
                        pass
                priority_info = priority_map.get((pid, model))
                priority = priority_info.get("priority") if priority_info else None
                enqueue_summary_task(pid, model=model, user=user, priority=priority)
            else:
                SummaryStatusRepository.update_status(
                    pid,
                    model,
                    {
                        "status": "failed",
                        "last_error": "stale_running_repaired",
                        "updated_time": now,
                    },
                )
                if task_id:
                    try:
                        SummaryStatusRepository.set_task_status(
                            str(task_id),
                            "failed",
                            error="stale_running_repaired",
                            pid=pid,
                            model=model,
                            user=user,
                        )
                    except Exception:
                        pass
                if user:
                    _update_readinglist_summary_status(user, pid, "failed", "stale_running_repaired")
            repaired += 1
    except Exception as e:
        logger.warning(f"Failed to repair stale summary tasks: {e}")

    return repaired


if settings.huey.summary_repair_on_start:
    # Only run repair if we're in a Huey consumer process (not during import in web server)
    # Check if we're being imported by huey consumer
    is_huey_consumer = any("huey" in arg.lower() for arg in sys.argv)

    if is_huey_consumer or settings.huey.force_repair:
        try:
            repair_requeue = settings.huey.summary_repair_requeue
            repaired_count = repair_stale_summary_tasks(requeue=repair_requeue)
            if repaired_count > 0:
                logger.info(f"Repaired {repaired_count} stale summary tasks on startup")
        except Exception as e:
            logger.warning(f"Stale summary repair failed on start: {e}")


def cleanup_tasks(
    status_filter: str | list[str] | None = None,
    pid_filter: str | None = None,
    model_filter: str | None = None,
    max_age_s: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Clean up task entries from the database.

    Args:
        status_filter: Only clean tasks with this status (e.g., 'running', 'queued', 'ok', 'failed')
                      Can be a single status or list of statuses. None means all.
        pid_filter: Only clean tasks for this pid. None means all.
        model_filter: Only clean tasks for this model. None means all.
        max_age_s: Only clean tasks older than this many seconds. None means all.
        dry_run: If True, only report what would be cleaned without actually deleting.

    Returns:
        dict with 'cleaned' count and 'details' list
    """
    if isinstance(status_filter, str):
        status_filter = [status_filter]

    now = time.time()
    cleaned = 0
    details = []

    try:
        for key, info in list(SummaryStatusRepository.get_items_with_prefix("task::")):
            if not isinstance(info, dict):
                continue

            task_id = key.replace("task::", "")
            status = info.get("status")
            pid = info.get("pid")
            model = info.get("model")
            updated_time = info.get("updated_time") or 0

            # Apply filters
            if status_filter and status not in status_filter:
                continue
            if pid_filter and pid != pid_filter:
                continue
            if model_filter and model != model_filter:
                continue
            if max_age_s is not None and (now - updated_time) < max_age_s:
                continue

            details.append(
                {
                    "task_id": task_id,
                    "pid": pid,
                    "model": model,
                    "status": status,
                    "age_s": int(now - updated_time) if updated_time else None,
                }
            )

            if not dry_run:
                # Delete task entry directly from db
                task_key = f"task::{task_id}"
                try:
                    from aslite.db import get_summary_status_db

                    with get_summary_status_db(flag="c") as sdb:
                        if task_key in sdb:
                            del sdb[task_key]
                except Exception as del_err:
                    logger.warning(f"Failed to delete task {task_id}: {del_err}")
                # Also reset the summary status if it was queued/running
                if pid and model and status in ("queued", "running"):
                    try:
                        current = SummaryStatusRepository.get_status(pid, model)
                        if isinstance(current, dict) and current.get("status") in ("queued", "running"):
                            SummaryStatusRepository.delete_status(pid, model)
                    except Exception:
                        pass
                # Best-effort: clear readinglist task linkage to avoid UI stuck in queued/running.
                task_user = info.get("user")
                if pid and task_user and status in ("queued", "running"):
                    try:
                        _update_readinglist_summary_status(task_user, pid, "failed", "task_cleaned", task_id=None)
                    except Exception:
                        pass

            cleaned += 1

    except Exception as e:
        logger.warning(f"Failed to cleanup tasks: {e}")

    return {"cleaned": cleaned, "details": details, "dry_run": dry_run}


# -----------------------------------------------------------------------------
# Uploaded PDF Processing Task
# -----------------------------------------------------------------------------


@huey.task(retries=2, retry_delay=60)
def process_uploaded_pdf_task(pid: str, user: str, model: str | None = None) -> None:
    """
    Process an uploaded PDF: MinerU parsing + LLM metadata extraction.
    After parsing completes, automatically triggers summary generation.

    Args:
        pid: Upload PID (e.g., up_V1StGXR8_Z5j)
        user: Username (owner)
        model: LLM model for summary (optional, uses default)
    """
    from backend.services.upload_service import process_uploaded_pdf

    try:
        process_uploaded_pdf(pid, user, model)
    except Exception as e:
        logger.error(f"Failed to process uploaded PDF {pid}: {e}")
        raise


@huey.task(retries=2, retry_delay=60)
def parse_uploaded_pdf_task(pid: str, user: str) -> None:
    """
    Parse an uploaded PDF with MinerU only (no metadata extraction).

    Args:
        pid: Upload PID
        user: Username (owner)
    """
    from backend.services.upload_service import do_parse_only

    try:
        do_parse_only(pid, user)
    except Exception as e:
        logger.error(f"Failed to parse uploaded PDF {pid}: {e}")
        raise


@huey.task(retries=2, retry_delay=30)
def extract_info_task(pid: str, user: str) -> None:
    """
    Extract metadata from an already-parsed uploaded PDF.

    Args:
        pid: Upload PID
        user: Username (owner)
    """
    from backend.services.upload_service import do_extract_metadata

    try:
        do_extract_metadata(pid, user)
    except Exception as e:
        logger.error(f"Failed to extract info for {pid}: {e}")
        raise
