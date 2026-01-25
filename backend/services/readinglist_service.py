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
from typing import TYPE_CHECKING, Callable

from flask import g
from loguru import logger

from aslite.repositories import ReadingListRepository, SummaryStatusRepository

from ..utils.sse import emit_all_event, emit_user_event

if TYPE_CHECKING:
    pass


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
    user = user or getattr(g, "user", None)
    if user is None:
        return {}

    return ReadingListRepository.get_user_reading_list(user)


def update_summary_status(
    user: str,
    pid: str,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
) -> None:
    """Update summary generation status in user's reading list.

    Args:
        user: Username
        pid: Paper ID
        status: Status string (queued, running, ok, failed)
        error: Error message if failed
        task_id: Task ID if queued
    """
    try:
        # Check if item exists first
        if ReadingListRepository.get_reading_list_item(user, pid) is None:
            return

        # Update the item
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
        emit_user_event(user, payload)
    except Exception as e:
        logger.warning(f"Failed to update summary status for {user}:{pid}: {e}")


def update_summary_status_db(
    pid: str,
    model: str | None,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
    task_user: str | None = None,
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
        default_model: Default model name to use if model is None
    """
    model = (model or default_model or "").strip()
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
        emit_all_event({"type": "summary_status", "pid": pid, "status": status, "error": error})
    except Exception as e:
        logger.warning(f"Failed to update summary status db for {pid}: {e}")


def trigger_summary_async(
    user: str | None,
    pid: str,
    model: str | None = None,
    priority: int | None = None,
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
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue summary task for {pid}: {e}")

    # Fallback to thread-based execution
    def _run():
        try:
            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "running", None)
            if update_db_fn:
                update_db_fn(pid, model, "running", None)

            if generate_summary_fn:
                generate_summary_fn(pid, model=model, force_refresh=False, cache_only=False)

            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "ok", None)
            if update_db_fn:
                update_db_fn(pid, model, "ok", None)
        except Exception as e:
            logger.warning(f"Failed to generate summary for {pid}: {e}")
            if user and update_readinglist_fn:
                update_readinglist_fn(user, pid, "failed", str(e))
            if update_db_fn:
                update_db_fn(pid, model, "failed", str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
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
    user = user or getattr(g, "user", None)
    if user is None:
        return {"error": "Not logged in"}

    # Check if already in reading list
    existing = ReadingListRepository.get_reading_list_item(user, pid)
    if existing is not None:
        # Update existing item
        ReadingListRepository.update_reading_list_item(
            user,
            pid,
            {
                "summary_status": "queued",
                "summary_last_error": None,
                "summary_updated_time": time.time(),
            },
        )

        task_id = None
        if trigger_summary_fn:
            task_id = trigger_summary_fn(user, pid)

        return {
            "message": "Already in reading list",
            "top_tags": existing.get("top_tags", []),
            "task_id": task_id,
            "already_exists": True,
        }

    # Compute top tags
    top_tags = []
    if compute_top_tags_fn and get_tags_fn:
        user_tags = get_tags_fn()
        top_tags = compute_top_tags_fn(pid, user_tags)

    # Add to reading list
    ReadingListRepository.add_to_reading_list(
        user,
        pid,
        {
            "added_time": time.time(),
            "top_tags": top_tags,
            "summary_triggered": True,
            "summary_status": "queued",
            "summary_last_error": None,
            "summary_updated_time": time.time(),
        },
    )

    task_id = None
    if trigger_summary_fn:
        task_id = trigger_summary_fn(user, pid)

    logger.debug(f"Added paper {pid} to reading list for user {user}, top_tags={top_tags}")
    emit_user_event(user, {"type": "readinglist_changed", "action": "add", "pid": pid})

    return {
        "pid": pid,
        "top_tags": top_tags,
        "message": "Added to reading list",
        "task_id": task_id,
        "already_exists": False,
    }


def remove_from_readinglist(pid: str, user: str | None = None) -> dict:
    """Remove a paper from the reading list.

    Args:
        pid: Paper ID (raw, without version)
        user: Username. If None, uses g.user

    Returns:
        Dict with result info (success, message, error)
    """
    user = user or getattr(g, "user", None)
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
    user = user or getattr(g, "user", None)
    if user is None:
        return []

    readinglist = get_user_readinglist(user)

    # Sort by added_time descending
    sorted_items = sorted(readinglist.items(), key=lambda x: x[1].get("added_time", 0), reverse=True)

    result = []
    for pid, info in sorted_items:
        result.append(
            {
                "pid": pid,
                "added_time": info.get("added_time", 0),
                "top_tags": info.get("top_tags", []),
                "summary_status": info.get("summary_status"),
                "summary_last_error": info.get("summary_last_error"),
                "summary_updated_time": info.get("summary_updated_time"),
                "summary_task_id": info.get("summary_task_id"),
            }
        )

    return result
