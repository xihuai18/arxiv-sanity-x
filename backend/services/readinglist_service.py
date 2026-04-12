"""Reading list service.

This module handles all reading list operations including:
- Getting user's reading list
- Adding/removing papers from reading list
- Summary status management
- Async summary triggering
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from aslite.repositories import (
    ReadingListRepository,
    SummaryStatusRepository,
    UploadedPaperRepository,
)
from config import settings

from ..utils.sse import emit_all_event, emit_user_event
from .summary_state_machine import (
    build_readinglist_summary_transition,
    build_summary_status_transition,
)
from .user_context import resolve_user

if TYPE_CHECKING:
    pass


def _default_summary_model() -> str:
    return str(settings.llm.name or "")


def _is_upload_pid(pid: str) -> bool:
    return bool(pid and pid.startswith("up_"))


# Optional task queue integration (Huey)
try:
    from tasks import SUMMARY_PRIORITY_HIGH, enqueue_summary_task

    _TASK_QUEUE_AVAILABLE = True
except Exception:
    SUMMARY_PRIORITY_HIGH = None
    enqueue_summary_task = None
    _TASK_QUEUE_AVAILABLE = False


def get_user_readinglist(user: str | None = None) -> dict:
    """Get reading list for a user.

    Args:
        user: Username. If None, uses g.user from Flask context.

    Returns:
        Dict mapping pid to reading list item info
    """
    user = resolve_user(user)
    if user is None:
        return {}

    return ReadingListRepository.get_user_reading_list(user)


def _overlay_summary_status_for_user(
    user: str,
    pid: str,
    item: dict,
    *,
    prefetched_status_info: dict | None = None,
    prefetched_summary_snapshot: dict | None = None,
    prefetched_upload_record: dict | None = None,
) -> dict:
    model = (_default_summary_model() or "").strip()
    if not model:
        return item

    if _is_upload_pid(pid):
        try:
            from backend.services.upload_service import _normalize_upload_parse_status

            record = prefetched_upload_record
            if not isinstance(record, dict) or record.get("owner") != user:
                item["summary_status"] = ""
                item["summary_last_error"] = None
                item["summary_task_id"] = None
                return item
            parse_status, _parse_error = _normalize_upload_parse_status(pid, record)
            if parse_status != "ok":
                item["summary_status"] = ""
                item["summary_last_error"] = None
                item["summary_task_id"] = None
                return item
        except Exception:
            item["summary_status"] = ""
            item["summary_last_error"] = None
            item["summary_task_id"] = None
            return item

    normalized_status = ""
    normalized_last_error = None
    if isinstance(prefetched_summary_snapshot, dict):
        normalized_status = str(prefetched_summary_snapshot.get("status") or "")
        normalized_last_error = prefetched_summary_snapshot.get("last_error")
    else:
        try:
            from backend.services.summary_service import (
                get_summary_status,
                get_summary_status_info,
            )

            normalized_status, normalized_last_error = get_summary_status(pid, model)
        except Exception:
            # Transient failures while probing summary reality should not erase the
            # last persisted reading-list state.
            return item

    if isinstance(prefetched_status_info, dict):
        info = prefetched_status_info
    else:
        try:
            from backend.services.summary_service import get_summary_status_info

            info = get_summary_status_info(pid, model)
        except Exception:
            info = None

    if not isinstance(info, dict):
        if normalized_status:
            item["summary_status"] = normalized_status
            item["summary_last_error"] = normalized_last_error
            item["summary_task_id"] = None
        return item

    raw_status = str(info.get("status") or "").strip()
    status = normalized_status or raw_status
    if not status:
        return item

    if not normalized_status and raw_status:
        item["summary_status"] = ""
        item["summary_last_error"] = None
        item["summary_task_id"] = None
        return item

    task_user = info.get("task_user")
    allow_sensitive = task_user is None or task_user == user
    last_error = info.get("last_error") if allow_sensitive else None
    if last_error is None and allow_sensitive:
        last_error = normalized_last_error
    item["summary_status"] = status
    item["summary_updated_time"] = info.get("updated_time") or item.get("summary_updated_time")
    item["summary_last_error"] = last_error
    if status in ("queued", "running") and allow_sensitive and info.get("task_id"):
        item["summary_task_id"] = str(info.get("task_id"))
    else:
        item["summary_task_id"] = None
    return item


def overlay_summary_status_for_user(
    user: str,
    pid: str,
    item: dict,
    *,
    prefetched_status_info: dict | None = None,
    prefetched_summary_snapshot: dict | None = None,
) -> dict:
    """Public wrapper for aligning reading-list summary fields with summary status DB."""
    prefetched_upload_record = None
    if _is_upload_pid(pid):
        try:
            prefetched_upload_record = UploadedPaperRepository.get(pid)
        except Exception:
            prefetched_upload_record = None

    return _overlay_summary_status_for_user(
        user,
        pid,
        item,
        prefetched_status_info=prefetched_status_info,
        prefetched_summary_snapshot=prefetched_summary_snapshot,
        prefetched_upload_record=prefetched_upload_record,
    )


def overlay_summary_statuses_for_user(
    user: str,
    items_by_pid: dict[str, dict],
    *,
    prefetched_status_rows: dict[str, dict] | None = None,
    prefetched_summary_snapshots: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Bulk wrapper for aligning reading-list summary fields with summary status DB."""
    if not items_by_pid:
        return {}

    model = (_default_summary_model() or "").strip()
    status_rows: dict[str, dict] = dict(prefetched_status_rows or {})
    if model and not status_rows:
        batch_pids = [pid for pid in items_by_pid.keys() if pid and not _is_upload_pid(pid)]
        if batch_pids:
            try:
                from backend.services.summary_service import (
                    get_summary_status_info_many,
                )

                status_rows = get_summary_status_info_many(batch_pids, model)
            except Exception:
                status_rows = {}

    upload_records: dict[str, dict] = {}
    upload_pids = [pid for pid in items_by_pid.keys() if pid and _is_upload_pid(pid)]
    if upload_pids:
        try:
            upload_records = UploadedPaperRepository.get_by_owner_for_pids(user, upload_pids)
        except Exception:
            upload_records = {}
        missing_upload_pids = [pid for pid in upload_pids if pid not in upload_records]
        for pid in missing_upload_pids:
            try:
                record = UploadedPaperRepository.get(pid)
            except Exception:
                record = None
            if isinstance(record, dict) and record.get("owner") == user:
                upload_records[pid] = record

    result: dict[str, dict] = {}
    for pid, item in items_by_pid.items():
        result[pid] = _overlay_summary_status_for_user(
            user,
            pid,
            dict(item or {}),
            prefetched_status_info=status_rows.get(pid),
            prefetched_summary_snapshot=(prefetched_summary_snapshots or {}).get(pid),
            prefetched_upload_record=upload_records.get(pid),
        )
    return result


def update_summary_status(
    user: str,
    pid: str,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
    model: str | None = None,
) -> None:
    """Update summary generation status in user's reading list.

    Args:
        user: Username
        pid: Paper ID
        status: Status string (queued, running, ok, failed)
        error: Error message if failed
        task_id: Task ID if queued
        model: Model for this status update; reading list persistence tracks the default model only
    """
    try:
        # Check if item exists first
        if ReadingListRepository.get_reading_list_item(user, pid) is None:
            return

        transition = build_readinglist_summary_transition(
            user,
            pid,
            status,
            error,
            task_id,
            model,
            default_model=_default_summary_model(),
            updated_time=time.time(),
        )
        if transition is None:
            return

        updates = transition.persisted_updates()
        if updates is not None:
            ReadingListRepository.update_reading_list_item(user, pid, updates)

        emit_user_event(user, transition.event_payload())
    except Exception as e:
        logger.warning(f"Failed to update summary status for {user}:{pid}: {e}")


def update_summary_status_db(
    pid: str,
    model: str | None,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
    task_user: str | None = None,
    resolved_model: str | None = None,
    default_model: str | None = None,
) -> None:
    """Persist summary status for main list usage.

    Args:
        pid: Paper ID
        model: Model name
        status: Status string
        error: Error message if failed
        task_id: Task ID if queued
        task_user: User who triggered the task
        resolved_model: Actual model used for successful generation
        default_model: Default model name to use if model is None
    """
    transition = build_summary_status_transition(
        pid,
        model,
        status,
        error,
        task_id,
        task_user,
        resolved_model,
        default_model=default_model,
    )
    if transition is None:
        return

    try:
        SummaryStatusRepository.set_status(
            transition.pid,
            transition.model,
            transition.status,
            cast(Any, transition.error),
            **transition.repository_extra(),
        )
        if _is_upload_pid(transition.pid) and transition.task_user:
            emit_user_event(transition.task_user, transition.private_event_payload())
        else:
            emit_all_event(transition.public_event_payload())
    except Exception as e:
        logger.warning(f"Failed to update summary status db for {pid}: {e}")


def trigger_summary_async(
    user: str | None,
    pid: str,
    model: str | None = None,
    priority: int | None = None,
    force_refresh: bool = False,
    generate_summary_fn: Callable | None = None,
    update_readinglist_fn: Callable | None = None,
    update_db_fn: Callable | None = None,
    default_model: str | None = None,
) -> str | None:
    """Trigger summary generation in a background worker or thread.

    Args:
        user: Username (optional)
        pid: Paper ID
        model: Model name
        priority: Task priority
        generate_summary_fn: Function to generate summary
        update_readinglist_fn: Function to update reading list status
        update_db_fn: Function to update summary status db
        default_model: Default model name

    Returns:
        Task ID when using the task queue, otherwise None
    """
    model = (model or default_model or "").strip() or None

    if _TASK_QUEUE_AVAILABLE and enqueue_summary_task:
        try:
            return enqueue_summary_task(
                pid,
                model=model,
                user=user,
                priority=(priority if priority is not None else SUMMARY_PRIORITY_HIGH),
                force_refresh=force_refresh,
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue summary task for {pid}: {e}")
            if not settings.huey.allow_thread_fallback:
                err = f"Failed to enqueue summary task: {e}"
                try:
                    if user and update_readinglist_fn:
                        update_readinglist_fn(user, pid, "failed", err)
                except Exception:
                    pass
                try:
                    if update_db_fn:
                        update_db_fn(pid, model, "failed", err, task_user=user)
                except Exception:
                    pass
                return None

    # Fallback to thread-based execution
    if not settings.huey.allow_thread_fallback:
        err = "Summary queue unavailable (Huey disabled); enable ARXIV_SANITY_HUEY_ALLOW_THREAD_FALLBACK=true to use in-process fallback."
        try:
            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "failed", err)
        except Exception:
            pass
        try:
            if update_db_fn:
                update_db_fn(pid, model, "failed", err, task_user=user)
        except Exception:
            pass
        return None

    # Thread fallback has no task id, but it should still preserve the same
    # owner-scoped status semantics as queued worker execution.
    try:
        if user and update_readinglist_fn:
            update_readinglist_fn(user, pid, "queued", None, task_id=None, model=model)
    except Exception:
        pass
    try:
        if update_db_fn:
            update_db_fn(pid, model, "queued", None, task_id=None, task_user=user)
    except Exception:
        pass

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

    def _run():
        start_epoch = 0
        try:
            # Cooperative cancellation: clear actions bump this epoch to stop in-flight work.
            if model:
                start_epoch = SummaryStatusRepository.get_generation_epoch(pid, model)
        except Exception:
            start_epoch = 0

        try:
            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "running", None, task_id=None, model=model)
            if update_db_fn:
                update_db_fn(pid, model, "running", None, task_id=None, task_user=user)

            result = None
            if generate_summary_fn:
                result = generate_summary_fn(
                    pid,
                    model=model,
                    force_refresh=bool(force_refresh),
                    cache_only=False,
                )
                summary_content = ""
                resolved_model = (model or "").strip()
                if isinstance(result, tuple):
                    summary_content = str(result[0] or "")
                    if len(result) > 1 and isinstance(result[1], dict):
                        resolved_model = str(
                            result[1].get("resolved_model") or result[1].get("llm_model") or resolved_model or ""
                        ).strip()
                elif result is not None:
                    summary_content = str(result or "")

                from tools.paper_summarizer import (
                    looks_like_valid_cached_summary_markdown,
                )

                if not looks_like_valid_cached_summary_markdown(summary_content):
                    if "Summary canceled." in summary_content or "Canceled by user" in summary_content:
                        if user and update_readinglist_fn:
                            update_readinglist_fn(
                                user,
                                pid,
                                "canceled",
                                "Canceled by user",
                                task_id=None,
                                model=model,
                            )
                        if update_db_fn:
                            update_db_fn(
                                pid,
                                model,
                                "canceled",
                                "Canceled by user",
                                task_id=None,
                                task_user=user,
                            )
                        return
                    if "Summary is being generated" in summary_content:
                        if user and update_readinglist_fn:
                            update_readinglist_fn(user, pid, "queued", None, task_id=None, model=model)
                        if update_db_fn:
                            update_db_fn(pid, model, "queued", None, task_id=None, task_user=user)
                        return
                    raise RuntimeError("Summary generation failed: invalid summary content")
            else:
                resolved_model = (model or "").strip()

            # If canceled mid-flight, do not mark ok.
            try:
                if model and SummaryStatusRepository.get_generation_epoch(pid, model) != start_epoch:
                    if user and update_readinglist_fn:
                        update_readinglist_fn(user, pid, "canceled", "Canceled by user", task_id=None)
                    if update_db_fn:
                        update_db_fn(
                            pid,
                            model,
                            "canceled",
                            "Canceled by user",
                            task_id=None,
                            task_user=user,
                        )
                    return
            except Exception:
                pass

            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "ok", None, task_id=None, model=model)
            if update_db_fn:
                update_db_fn(
                    pid,
                    model,
                    "ok",
                    None,
                    task_id=None,
                    task_user=user,
                    resolved_model=resolved_model,
                )
        except Exception as e:
            logger.warning(f"Failed to generate summary for {pid}: {e}")
            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "failed", str(e), task_id=None, model=model)
            if update_db_fn:
                update_db_fn(pid, model, "failed", str(e), task_id=None, task_user=user)

    _start_daemon_thread(target=_run, name="summary-thread-fallback")
    return None


def add_to_readinglist(
    pid: str,
    user: str | None = None,
    compute_top_tags_fn: Callable | None = None,
    get_tags_fn: Callable | None = None,
    trigger_summary_fn: Callable | None = None,
) -> dict:
    """Add a paper to the reading list.

    Args:
        pid: Paper ID (raw, without version)
        user: Username. If None, uses g.user
        compute_top_tags_fn: Function to compute top tags
        get_tags_fn: Function to get user tags
        trigger_summary_fn: Function to trigger summary generation

    Returns:
        Dict with result info (message, top_tags, task_id, already_exists)
    """
    user = resolve_user(user)
    if user is None:
        return {"error": "Not logged in"}

    current_status = ""
    model = (_default_summary_model() or "").strip()
    if model:
        try:
            from backend.services.summary_service import get_summary_status

            current_status, _current_error = get_summary_status(pid, model)
        except Exception:
            current_status = ""

    # Check if already in reading list
    existing = ReadingListRepository.get_reading_list_item(user, pid)
    if existing is not None:
        # If summary is already ready, do not re-trigger it.
        current_status = current_status or (existing.get("summary_status") or "").strip()
        task_id = str(existing.get("summary_task_id") or "") or None
        if current_status not in ("ok", "queued", "running"):
            ReadingListRepository.update_reading_list_item(
                user,
                pid,
                {
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": time.time(),
                },
            )
            if trigger_summary_fn:
                task_id = trigger_summary_fn(user, pid)

        return {
            "message": "Already in reading list",
            "top_tags": existing.get("top_tags", []),
            "task_id": task_id,
            "already_exists": True,
            "summary_status": current_status,
        }

    # Compute top tags
    top_tags = []
    if compute_top_tags_fn and get_tags_fn:
        user_tags = get_tags_fn()
        top_tags = compute_top_tags_fn(pid, user_tags)

    initial_status = current_status if current_status in ("ok", "queued", "running") else "queued"

    # Add to reading list
    ReadingListRepository.add_to_reading_list(
        user,
        pid,
        {
            "added_time": time.time(),
            "top_tags": top_tags,
            "summary_triggered": initial_status != "ok",
            "summary_status": initial_status,
            "summary_last_error": None,
            "summary_updated_time": time.time(),
        },
    )

    task_id = None
    if trigger_summary_fn and initial_status != "ok":
        task_id = trigger_summary_fn(user, pid)

    logger.debug(f"Added paper {pid} to reading list for user {user}, top_tags={top_tags}")
    emit_user_event(user, {"type": "readinglist_changed", "action": "add", "pid": pid})

    return {
        "pid": pid,
        "top_tags": top_tags,
        "message": "Added to reading list",
        "task_id": task_id,
        "already_exists": False,
        "summary_status": initial_status,
    }


def remove_from_readinglist(pid: str, user: str | None = None) -> dict:
    """Remove a paper from the reading list.

    Args:
        pid: Paper ID (raw, without version)
        user: Username. If None, uses g.user

    Returns:
        Dict with result info (success, message, error)
    """
    user = resolve_user(user)
    if user is None:
        return {"error": "Not logged in"}

    removed = ReadingListRepository.remove_from_reading_list(user, pid)
    if not removed:
        return {"error": "Paper not in reading list"}

    logger.debug(f"Removed paper {pid} from reading list for user {user}")
    emit_user_event(user, {"type": "readinglist_changed", "action": "remove", "pid": pid})

    return {"pid": pid, "message": "Removed from reading list"}


def list_readinglist(user: str | None = None) -> list:
    """Get reading list data for a user.

    Args:
        user: Username. If None, uses g.user

    Returns:
        List of reading list items sorted by added_time descending
    """
    user = resolve_user(user)
    if user is None:
        return []

    readinglist = get_user_readinglist(user)

    # Sort by added_time descending
    sorted_items = sorted(readinglist.items(), key=lambda x: x[1].get("added_time", 0), reverse=True)

    from backend.services.data_service import paper_exists

    base_items: list[tuple[str, dict]] = []
    for pid, info in sorted_items:
        if not _is_upload_pid(pid) and not paper_exists(pid):
            continue
        item = {
            "pid": pid,
            "added_time": info.get("added_time", 0),
            "top_tags": info.get("top_tags", []),
            "summary_status": info.get("summary_status"),
            "summary_last_error": info.get("summary_last_error"),
            "summary_updated_time": info.get("summary_updated_time"),
            "summary_task_id": info.get("summary_task_id"),
        }
        base_items.append((pid, item))

    from backend.services.summary_service import get_summary_render_snapshots

    model = (_default_summary_model() or "").strip()
    batch_pids = [pid for pid, _item in base_items if pid and not _is_upload_pid(pid)]
    prefetched_status_rows = {}
    if model and batch_pids:
        try:
            from backend.services.summary_service import get_summary_status_info_many

            prefetched_status_rows = get_summary_status_info_many(batch_pids, model)
        except Exception:
            prefetched_status_rows = {}
    summary_snapshots = get_summary_render_snapshots(
        [pid for pid, _item in base_items],
        include_tldr=False,
        prefetched_status_rows=prefetched_status_rows,
    )
    overlaid = overlay_summary_statuses_for_user(
        user,
        {pid: item for pid, item in base_items},
        prefetched_status_rows=prefetched_status_rows,
        prefetched_summary_snapshots=summary_snapshots,
    )
    return [overlaid.get(pid, item) for pid, item in base_items]
