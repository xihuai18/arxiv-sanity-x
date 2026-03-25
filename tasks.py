from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from huey import SqliteHuey
from loguru import logger

from aslite.repositories import (
    MetaRepository,
    ReadingListRepository,
    SummaryStatusRepository,
    UploadedPaperRepository,
    safe_closing,
)
from backend.services.summary_state_machine import (
    build_readinglist_summary_transition,
    build_summary_status_transition,
)
from config import settings
from config.sentry import initialize_sentry
from tools.paper_summarizer import (
    acquire_summary_lock,
    atomic_write_json,
    atomic_write_text,
)
from tools.paper_summarizer import (
    generate_paper_summary as generate_paper_summary_from_module,
)
from tools.paper_summarizer import (
    is_error_summary_content,
    looks_like_valid_cached_summary_markdown,
    normalize_summary_result,
    normalize_summary_source,
    read_summary_meta,
    release_summary_lock,
    resolve_cache_pid,
    split_pid_version,
    summary_cache_paths,
    summary_source_matches,
)

# Optional Sentry error reporting (no-op unless configured).
initialize_sentry()


class SummaryCanceled(Exception):
    """Raised when a running/queued summary task is canceled."""


def _is_huey_consumer_process() -> bool:
    """Best-effort detection of a Huey consumer process.

    Huey runs tasks in a separate process, so in-process SSE emits are ineffective.
    """
    # Prefer explicit environment markers over argv sniffing.
    # - bin/huey_consumer.py sets ARXIV_SANITY_HUEY_CONSUMER=1.
    # - Users may run huey consumer directly; keep argv fallback for compatibility.
    try:
        marker = (os.environ.get("ARXIV_SANITY_HUEY_CONSUMER") or "").strip().lower()
        if marker in {"1", "true", "yes", "on"}:
            return True
    except Exception:
        pass
    try:
        return any("huey" in (arg or "").lower() for arg in sys.argv)
    except Exception:
        return False


def _data_dir() -> str:
    return str(settings.data_dir)


def _default_llm_name() -> str:
    return str(settings.llm.name or "")


def _summary_markdown_source() -> str:
    return str(settings.summary.markdown_source or "")


def _tasks_sse_enabled() -> bool:
    return bool(settings.huey.tasks_sse_enabled)


# Import SSE event emitters lazily to avoid circular imports.
def _emit_user_event(user, payload):
    if not _tasks_sse_enabled():
        return
    try:
        from backend.utils.sse import emit_user_event as _emit

        _emit(user, payload)
    except Exception:
        pass


def _emit_all_event(payload):
    if not _tasks_sse_enabled():
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
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix("task::")
        ) as items:
            for key, info in items:
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
    _cache_file, _meta_file, lock_file, _legacy_cache, _legacy_meta, _legacy_lock = (
        summary_cache_paths(pid, model)
    )
    name = lock_file.name
    model_key = (
        name[1 : -len(".lock")]
        if (name.startswith(".") and name.endswith(".lock"))
        else "default"
    )
    return lock_file.parent / f".{model_key}.enqueue.lock"


HUEY_DB_PATH = settings.huey.db_path or os.path.join(_data_dir(), "huey.db")

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


def _update_upload_task_reference(
    pid: str,
    *,
    record_field: str,
    task_id: str | None = None,
    clear: bool = False,
) -> None:
    """Best-effort sync of upload task pointers stored on upload records."""
    if not _is_upload_pid(pid):
        return
    try:
        record = UploadedPaperRepository.get(pid)
        if not isinstance(record, dict):
            return
        current = str(record.get(record_field) or "").strip()
        if clear:
            expected = str(task_id or "").strip()
            if expected and current and current != expected:
                return
            if not current:
                return
            UploadedPaperRepository.update(pid, {record_field: None})
            return

        desired = str(task_id or "").strip()
        if not desired or current == desired:
            return
        UploadedPaperRepository.update(pid, {record_field: desired})
    except Exception as e:
        logger.debug(
            f"Failed to sync upload task reference for {pid}:{record_field}: {e}"
        )


def _is_current_upload_task(
    pid: str,
    *,
    record_field: str,
    task_id: str | None,
    allowed_statuses: set[str] | None = None,
) -> bool:
    """Whether a worker task is still the active upload task for a record."""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id or not _is_upload_pid(pid):
        return False

    try:
        record = UploadedPaperRepository.get(pid)
    except Exception:
        return False
    if not isinstance(record, dict):
        return False
    if record.get("deleting") is True:
        return False

    current_task_id = str(record.get(record_field) or "").strip()
    if current_task_id != normalized_task_id:
        return False

    if not allowed_statuses:
        return True

    current_status = str(record.get("parse_status") or "").strip().lower()
    return current_status in {status.lower() for status in allowed_statuses}


def _maybe_mark_upload_task_canceled(
    pid: str,
    *,
    record_field: str,
    task_id: str | None,
    model: str,
    user: str,
) -> None:
    """Best-effort mark a replaced upload task as canceled without recreating deleted state."""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return

    try:
        info = SummaryStatusRepository.get_task_status(normalized_task_id)
    except Exception:
        info = None
    if isinstance(info, dict):
        current_task_status = str(info.get("status") or "").strip().lower()
        if current_task_status and current_task_status not in {"queued", "running"}:
            return

    try:
        record = UploadedPaperRepository.get(pid)
    except Exception:
        record = None
    if not isinstance(record, dict):
        return
    if record.get("deleting") is True:
        return

    current_task_id = str(record.get(record_field) or "").strip()
    if current_task_id != normalized_task_id:
        _update_task_status(
            normalized_task_id,
            "canceled",
            error="superseded_upload_task",
            pid=pid,
            model=model,
            user=user,
        )


def _emit_summary_status_event(transition) -> None:
    if _is_upload_pid(transition.pid):
        if not transition.task_user:
            return
        _emit_user_event(transition.task_user, transition.private_event_payload())
        return

    _emit_all_event(transition.public_event_payload())


def _paper_exists(pid: str) -> bool:
    # Check for upload pid first
    if _is_upload_pid(pid):
        from aslite.repositories import UploadedPaperRepository

        record = UploadedPaperRepository.get(pid)
        if record is None:
            return False
        # Treat deleting uploads as non-existent to prevent background tasks
        # from working on records being removed.
        if record.get("deleting") is True:
            return False
        return record.get("parse_status") == "ok"

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
    resolved_model: str | None = None,
) -> None:
    transition = build_summary_status_transition(
        pid,
        model,
        status,
        error,
        task_id,
        task_user,
        resolved_model,
        default_model=_default_llm_name(),
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
        if _is_upload_pid(transition.pid):
            if transition.is_active and task_id is not None:
                _update_upload_task_reference(
                    transition.pid,
                    record_field="summary_task_id",
                    task_id=task_id,
                )
            elif not transition.is_active and task_id is not None:
                _update_upload_task_reference(
                    transition.pid,
                    record_field="summary_task_id",
                    task_id=task_id,
                    clear=True,
                )
        _emit_summary_status_event(transition)
    except Exception as e:
        logger.warning(f"Failed to update summary status db for {pid}: {e}")


def _update_task_status(
    task_id: str | None, status: str, error: str | None = None, **extra
) -> None:
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
    model: str | None = None,
) -> None:
    if not user:
        return
    try:
        if ReadingListRepository.get_reading_list_item(user, pid) is None:
            return
        transition = build_readinglist_summary_transition(
            user,
            pid,
            status,
            error,
            task_id,
            model,
            default_model=_default_llm_name(),
            updated_time=time.time(),
        )
        if transition is None:
            return
        updates = transition.persisted_updates()
        if updates is not None:
            ReadingListRepository.update_reading_list_item(user, pid, updates)
        _emit_user_event(user, transition.event_payload())
    except Exception as e:
        logger.warning(f"Failed to update summary status for {user}:{pid}: {e}")


def _read_cached_summary(
    cache_file: Path,
    meta_file: Path,
    legacy_cache: Path,
    legacy_meta: Path,
    model: str,
    *,
    status_pid: str | None = None,
):
    def _read_from_paths(body_path: Path, meta_path: Path, inject_model: bool = True):
        if not body_path.exists():
            return None, {}
        try:
            with open(body_path, encoding="utf-8") as f:
                cached = f.read()
            cached = cached if cached.strip() else None
            if cached and not looks_like_valid_cached_summary_markdown(cached):
                cached = None
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

    legacy_cached, legacy_meta_data = _read_from_paths(
        legacy_cache, legacy_meta, inject_model=False
    )
    if legacy_cached:
        legacy_model = (
            legacy_meta_data.get("model") or legacy_meta_data.get("llm_model") or ""
        ).strip()
        if not model or (legacy_model and legacy_model == model):
            return legacy_cached, legacy_meta_data

    if not status_pid:
        return None, {}
    try:
        info = SummaryStatusRepository.get_status(status_pid, model)
    except Exception:
        info = None
    if not isinstance(info, dict):
        return None, {}

    resolved_model = (info.get("resolved_model") or info.get("llm_model") or "").strip()
    if not resolved_model or resolved_model == model:
        return None, {}

    (
        resolved_cache,
        resolved_meta,
        _resolved_lock,
        resolved_legacy_cache,
        resolved_legacy_meta,
        _resolved_legacy_lock,
    ) = summary_cache_paths(status_pid, resolved_model)
    resolved_cached, resolved_meta_data = _read_from_paths(
        resolved_cache, resolved_meta
    )
    if resolved_cached:
        return resolved_cached, resolved_meta_data

    resolved_legacy_cached, resolved_legacy_meta_data = _read_from_paths(
        resolved_legacy_cache, resolved_legacy_meta, inject_model=False
    )
    if not resolved_legacy_cached:
        return None, {}

    resolved_legacy_model = (
        resolved_legacy_meta_data.get("model")
        or resolved_legacy_meta_data.get("llm_model")
        or ""
    ).strip()
    if resolved_legacy_model and resolved_legacy_model == resolved_model:
        return resolved_legacy_cached, resolved_legacy_meta_data
    return None, {}


def _is_error_summary(summary_content: str) -> bool:
    if not summary_content:
        return True
    text = str(summary_content or "")
    if is_error_summary_content(text):
        return True
    # Keep write/read criteria consistent to avoid "task ok but cache unusable".
    return not looks_like_valid_cached_summary_markdown(text)


def _extract_error_reason_from_summary(
    summary_content: str, *, max_len: int = 500
) -> str:
    """Extract a concise human-readable error reason from a '# Error' summary body."""
    if not summary_content:
        return "Summary generation failed: empty summary content"

    text = str(summary_content).strip()
    if not text:
        return "Summary generation failed: empty summary content"

    reason = ""
    if is_error_summary_content(text):
        # Typically formatted as:
        #   # Error
        #
        #   <reason>
        # Keep the first non-empty line after the heading.
        rest = text.split("\n", 1)[1] if "\n" in text else ""
        for line in rest.splitlines():
            candidate = (line or "").strip()
            if candidate:
                reason = candidate
                break
    else:
        reason = text.splitlines()[0].strip()

    reason = reason or "Summary generation failed"
    if len(reason) > max_len:
        reason = reason[: max_len - 1] + "…"
    return reason


def _is_task_canceled(task_id: str | None) -> bool:
    if not task_id:
        return False
    try:
        info = SummaryStatusRepository.get_task_status(str(task_id))
        return bool(isinstance(info, dict) and info.get("status") == "canceled")
    except Exception:
        return False


def _revoke_task_by_id(task_id: str) -> None:
    """Best-effort revoke (prevents queued tasks from executing)."""
    if not task_id:
        return
    try:
        # Huey expects a Task instance (uses task.revoke_id). We can construct a dummy
        # task and override its revoke_id to match the target task id.
        dummy = generate_summary_task.s(
            "_", model=_default_llm_name() or "default", user=None
        )
        dummy.revoke_id = f"r:{task_id}"
        huey.revoke(dummy, revoke_once=True)
    except Exception:
        # Revoke is best-effort; cooperative cancellation will still prevent writes.
        pass


def cancel_summary_tasks(
    pid: str,
    model: str | None = None,
    *,
    user: str | None = None,
    reason: str | None = None,
) -> dict:
    """Cancel queued/running summary tasks for a given (pid, model).

    Behavior:
    - Marks matching tasks as canceled in the task status store.
    - Revokes tasks in Huey best-effort (prevents queued execution).
    - Bumps per-(pid, model) generation epoch for cooperative cancellation.
    """
    model = (model or _default_llm_name() or "").strip()
    if not pid or not model:
        return {"canceled_task_ids": [], "epoch": 0}

    # Normalize to raw pid (unversioned) to match enqueue de-duplication.
    raw_pid, _ = split_pid_version(pid)
    pid = raw_pid or pid

    reason = (reason or "Canceled by user").strip()

    # Cooperative cancellation: bump epoch first, so any in-flight task can see it quickly.
    try:
        epoch = SummaryStatusRepository.bump_generation_epoch(pid, model)
    except Exception:
        epoch = 0

    def _task_epoch(info: dict | None) -> int:
        if not isinstance(info, dict):
            return 0
        try:
            return int(info.get("epoch") or 0)
        except Exception:
            return 0

    def _should_cancel_task(task_id: str, info: dict | None = None) -> bool:
        if epoch <= 0:
            return True
        if info is None:
            try:
                info = SummaryStatusRepository.get_task_status(task_id)
            except Exception:
                info = None
        if not isinstance(info, dict):
            # Avoid canceling unknown tasks when epoch filtering is enabled.
            return False
        return _task_epoch(info) < epoch

    canceled_task_ids: list[str] = []

    # 1) Fast path: summary status record points to the active task.
    try:
        info = SummaryStatusRepository.get_status(pid, model)
        if isinstance(info, dict) and info.get("status") in ("queued", "running"):
            tid = info.get("task_id")
            if tid and _should_cancel_task(str(tid)):
                canceled_task_ids.append(str(tid))
    except Exception:
        pass

    # 2) Best-effort scan for any duplicate queued/running tasks for this pid/model.
    try:
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix("task::")
        ) as items:
            for key, tinfo in items:
                if not isinstance(tinfo, dict):
                    continue
                if tinfo.get("status") not in ("queued", "running"):
                    continue
                if tinfo.get("pid") != pid or tinfo.get("model") != model:
                    continue
                task_id = str(key).replace("task::", "")
                if _should_cancel_task(task_id, tinfo):
                    canceled_task_ids.append(task_id)
    except Exception:
        pass

    # De-dup while preserving order
    seen = set()
    canceled_task_ids = [
        x for x in canceled_task_ids if x and (x not in seen and not seen.add(x))
    ]

    for tid in canceled_task_ids:
        existing_user = None
        try:
            existing = SummaryStatusRepository.get_task_status(tid)
            if isinstance(existing, dict):
                existing_user = existing.get("user")
        except Exception:
            existing_user = None
        try:
            # Mark task canceled (do not rely on Huey revoke alone).
            SummaryStatusRepository.set_task_status(
                tid,
                "canceled",
                error=reason,
                pid=pid,
                model=model,
                user=(user if user is not None else existing_user),
                canceled_time=time.time(),
            )
        except Exception:
            pass
        _revoke_task_by_id(tid)
        if _is_upload_pid(pid):
            _update_upload_task_reference(
                pid,
                record_field="summary_task_id",
                task_id=tid,
                clear=True,
            )

    # Clear summary status so UI doesn't stay in queued/running state,
    # but avoid overwriting a newly enqueued task (epoch >= current).
    try:
        current = SummaryStatusRepository.get_status(pid, model)
        if isinstance(current, dict) and current.get("status") in ("queued", "running"):
            cur_tid = current.get("task_id")
            if cur_tid and _should_cancel_task(str(cur_tid)):
                SummaryStatusRepository.set_status(
                    pid, model, "canceled", reason, task_id=None, task_user=None
                )
                if _is_upload_pid(pid):
                    _update_upload_task_reference(
                        pid,
                        record_field="summary_task_id",
                        task_id=str(cur_tid),
                        clear=True,
                    )
    except Exception:
        pass

    return {"canceled_task_ids": canceled_task_ids, "epoch": epoch}


def cancel_paper_summary_tasks(
    pid: str,
    *,
    user: str | None = None,
    reason: str | None = None,
) -> dict:
    """Cancel queued/running summary tasks for a paper (all models)."""
    if not pid:
        return {"canceled": {}, "total": 0}

    raw_pid, _ = split_pid_version(pid)
    pid = raw_pid or pid

    reason = (reason or "Canceled by user").strip()

    # Collect active tasks grouped by model.
    model_to_task_ids: dict[str, list[str]] = {}
    try:
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix("task::")
        ) as items:
            for key, info in items:
                if not isinstance(info, dict):
                    continue
                if info.get("status") not in ("queued", "running"):
                    continue
                if info.get("pid") != pid:
                    continue
                m = (info.get("model") or "").strip()
                if not m:
                    continue
                model_to_task_ids.setdefault(m, []).append(
                    str(key).replace("task::", "")
                )
    except Exception:
        model_to_task_ids = {}
    # Include models from status entries (even if no task:: record is present).
    try:
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix(f"{pid}::")
        ) as items:
            for key, info in items:
                if not isinstance(info, dict):
                    continue
                if info.get("status") not in ("queued", "running"):
                    continue
                if "::" not in key:
                    continue
                _pid, m = key.split("::", 1)
                if not m:
                    continue
                model_to_task_ids.setdefault(m, [])
    except Exception:
        pass

    results: dict[str, dict] = {}
    total = 0
    for m, tids in sorted(model_to_task_ids.items(), key=lambda kv: kv[0]):
        # cancel_summary_tasks scans too, but we already have tids; still OK for simplicity
        res = cancel_summary_tasks(pid, m, user=user, reason=reason)
        results[m] = res
        total += len(res.get("canceled_task_ids") or [])

    return {"canceled": results, "total": total}


def _generate_and_cache_summary(
    pid: str,
    model: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
    force_refresh: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[str, dict]:
    cache_pid, raw_pid, has_explicit_version = resolve_cache_pid(pid, None)
    if not _paper_exists(raw_pid):
        return "# Error\n\nPaper not found.", {}

    summary_source = normalize_summary_source(_summary_markdown_source())

    cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock = (
        summary_cache_paths(cache_pid, model)
    )
    # Use a paper-level lock (legacy_lock) to avoid cross-model races (e.g. model fallback).
    # This trades some parallelism for correctness and stability under concurrency.
    lock_file = legacy_lock

    existed_before = False
    try:
        _cf, _mf, _lf, _lc, _lm, _ll = summary_cache_paths(cache_pid, model)
        existed_before = _cf.exists() or _lc.exists()
    except Exception:
        existed_before = cache_file.exists() or legacy_cache.exists()

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached_summary, cached_meta = _read_cached_summary(
        cache_file, meta_file, legacy_cache, legacy_meta, model, status_pid=cache_pid
    )
    if cached_summary and not force_refresh:
        if summary_source_matches(cached_meta, summary_source):
            return cached_summary, cached_meta

    if cancel_check and cancel_check():
        raise SummaryCanceled("Canceled before lock acquisition")

    if progress_cb:
        progress_cb("acquiring_lock")

    lock_fd = acquire_summary_lock(lock_file, timeout_s=300)
    if lock_fd is None:
        if (
            cached_summary
            and not force_refresh
            and summary_source_matches(cached_meta, summary_source)
        ):
            return cached_summary, cached_meta
        return "# Error\n\nSummary is being generated, please retry shortly.", {}

    # Keep track of the original lock file for proper release
    acquired_lock_file = lock_file

    try:
        if progress_cb:
            progress_cb("lock_acquired")

        if cancel_check and cancel_check():
            raise SummaryCanceled("Canceled after lock acquisition")

        cached_summary, cached_meta = _read_cached_summary(
            cache_file,
            meta_file,
            legacy_cache,
            legacy_meta,
            model,
            status_pid=cache_pid,
        )
        if cached_summary and not force_refresh:
            if summary_source_matches(cached_meta, summary_source):
                return cached_summary, cached_meta

        if summary_source == "html":
            pid_for_summary = pid if has_explicit_version else cache_pid
        else:
            pid_for_summary = pid if has_explicit_version else raw_pid

        if cancel_check and cancel_check():
            raise SummaryCanceled("Canceled before LLM request")

        if progress_cb:
            progress_cb("llm_request")

        summary_result = generate_paper_summary_from_module(
            pid_for_summary, source=summary_source, model=model
        )
        summary_content, summary_meta = normalize_summary_result(summary_result)
        summary_meta = summary_meta if isinstance(summary_meta, dict) else {}

        actual_model = (summary_meta.get("llm_model") or "").strip()
        # Use actual_model for caching/stats if LLM fell back to a different model
        stat_model = actual_model if actual_model else model
        if actual_model and model and actual_model != model:
            # Model changed, update cache paths but keep original lock for release
            cache_file, meta_file, _, legacy_cache, legacy_meta, _ = (
                summary_cache_paths(cache_pid, actual_model)
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Re-check if this actual_model cache already existed
            try:
                existed_before = cache_file.exists() or legacy_cache.exists()
            except Exception:
                pass

        if cancel_check and cancel_check():
            raise SummaryCanceled("Canceled before cache write")

        if progress_cb:
            progress_cb("writing_cache")

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
                    summary_cache_stats_increment(
                        cache_pid, stat_model
                    )  # Use actual model for stats
                else:
                    invalidate_summary_cache_stats()
            except Exception:
                pass
            if progress_cb:
                progress_cb("done")
            return summary_content, meta

        return summary_content, summary_meta
    finally:
        release_summary_lock(lock_fd, acquired_lock_file)


@huey.task(retries=3, retry_delay=60, context=True)
def generate_summary_task(
    pid: str,
    model: str | None = None,
    user: str | None = None,
    epoch: int | None = None,
    force_refresh: bool = False,
    task=None,
    **_kwargs,
) -> None:
    model = (model or _default_llm_name() or "").strip()
    task_id = getattr(task, "id", None)

    max_attempts = 1
    try:
        max_attempts = int(getattr(task, "retries", 0) or 0) + 1
    except Exception:
        max_attempts = 1

    attempt = 1
    status_force_refresh = False
    try:
        existing = (
            SummaryStatusRepository.get_task_status(str(task_id)) if task_id else None
        )
        if isinstance(existing, dict):
            status_force_refresh = bool(existing.get("force_refresh"))
            # Respect pre-cancellation (e.g. user cleared summary while task was queued).
            if existing.get("status") == "canceled":
                reason = (existing.get("error") or "Canceled by user").strip()
                _update_task_status(
                    task_id, "canceled", error=reason, pid=pid, model=model, user=user
                )
                _update_summary_status_db(
                    pid, model, "canceled", reason, task_id=None, task_user=user
                )
                if user:
                    _update_readinglist_summary_status(
                        user, pid, "canceled", reason, task_id=None, model=model
                    )
                return
            attempt = int(existing.get("attempt") or 0) + 1
    except Exception:
        attempt = 1

    # NOTE: For backward compatibility, some enqueue paths intentionally do not
    # pass force_refresh to Huey payload (old workers would crash on new kwargs).
    # We still persist force_refresh in task status, so the worker can recover it.
    force_refresh = bool(force_refresh) or bool(status_force_refresh)

    start_epoch = 0
    if epoch is not None:
        try:
            start_epoch = int(epoch or 0)
        except Exception:
            start_epoch = 0
    else:
        try:
            start_epoch = SummaryStatusRepository.get_generation_epoch(pid, model)
        except Exception:
            start_epoch = 0

    # If the epoch has moved since enqueue, treat as canceled (cooperative cancellation).
    try:
        cur_epoch = SummaryStatusRepository.get_generation_epoch(pid, model)
    except Exception:
        cur_epoch = start_epoch
    if cur_epoch != start_epoch:
        reason = "Canceled by user"
        _update_task_status(
            task_id,
            "canceled",
            error=reason,
            pid=pid,
            model=model,
            user=user,
            epoch=start_epoch,
        )
        _update_summary_status_db(
            pid, model, "canceled", reason, task_id=None, task_user=user
        )
        if user:
            _update_readinglist_summary_status(
                user, pid, "canceled", reason, task_id=None, model=model
            )
        return

    # Uploaded papers are private; ensure the task user owns the paper before marking running.
    if _is_upload_pid(pid):
        from aslite.repositories import UploadedPaperRepository

        record = UploadedPaperRepository.get(pid)
        if not user or not record or record.get("owner") != user:
            raise RuntimeError("Uploaded paper not found")
        if record.get("deleting") is True:
            raise SummaryCanceled("Uploaded paper deleted")
        if record.get("parse_status") != "ok":
            raise RuntimeError(
                f"Uploaded paper not parsed (status: {record.get('parse_status')})"
            )

    _update_task_status(
        task_id,
        "running",
        pid=pid,
        model=model,
        user=user,
        attempt=attempt,
        max_attempts=max_attempts,
        epoch=start_epoch,
        force_refresh=bool(force_refresh),
    )
    _update_summary_status_db(
        pid, model, "running", None, task_id=task_id, task_user=user
    )
    if user:
        _update_readinglist_summary_status(
            user, pid, "running", None, task_id=task_id, model=model
        )

    try:
        # If cancellation happened between the pre-check and the "running" update, abort quickly.
        try:
            if SummaryStatusRepository.get_generation_epoch(pid, model) != start_epoch:
                raise SummaryCanceled("Canceled by user")
        except SummaryCanceled:
            raise
        except Exception:
            pass

        def _cancel_check() -> bool:
            if _is_task_canceled(str(task_id) if task_id else None):
                return True
            try:
                return (
                    SummaryStatusRepository.get_generation_epoch(pid, model)
                    != start_epoch
                )
            except Exception:
                return False

        def _progress(stage: str) -> None:
            # Persist a coarse-grained stage for UI polling (best effort).
            _update_task_status(task_id, "running", stage=stage)

        summary_content, summary_meta = _generate_and_cache_summary(
            pid,
            model,
            cancel_check=_cancel_check,
            force_refresh=bool(force_refresh),
            progress_cb=_progress,
        )

        if _cancel_check():
            raise SummaryCanceled("Canceled after generation")

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
                _update_summary_status_db(
                    pid, model, "queued", None, task_id=task_id, task_user=user
                )
                if user:
                    _update_readinglist_summary_status(
                        user, pid, "queued", None, task_id=task_id, model=model
                    )

                # If we've exhausted our retries, mark as failed to avoid stuck states.
                if attempt >= max_attempts:
                    raise RuntimeError("Lock contention exhausted")
                raise RuntimeError("Lock contention, retrying")
            else:
                reason = _extract_error_reason_from_summary(summary_content)
                # Best-effort: include meta hint (e.g. last LLM error) if available.
                try:
                    if isinstance(summary_meta, dict):
                        attempts = summary_meta.get("llm_fallback_attempts")
                        if isinstance(attempts, list) and attempts:
                            last = attempts[-1]
                            if isinstance(last, dict):
                                last_err = (last.get("error") or "").strip()
                                if last_err and last_err not in reason:
                                    reason = f"{reason} (last_error={last_err})"
                except Exception:
                    pass
                raise RuntimeError(reason)

        _update_task_status(task_id, "ok", pid=pid, model=model, user=user)
        resolved_model = (
            (summary_meta.get("llm_model") or model or "").strip()
            if isinstance(summary_meta, dict)
            else model
        )
        _update_summary_status_db(
            pid,
            model,
            "ok",
            None,
            task_id=task_id,
            task_user=user,
            resolved_model=resolved_model,
        )
        if user:
            _update_readinglist_summary_status(
                user, pid, "ok", None, task_id=task_id, model=model
            )
    except SummaryCanceled as e:
        reason = str(e) if str(e) else "Canceled by user"
        _update_task_status(
            task_id,
            "canceled",
            error=reason,
            pid=pid,
            model=model,
            user=user,
            epoch=start_epoch,
        )
        _update_summary_status_db(
            pid, model, "canceled", reason, task_id=None, task_user=user
        )
        if user:
            _update_readinglist_summary_status(
                user, pid, "canceled", reason, task_id=None, model=model
            )
    except Exception as e:
        logger.warning(
            f"Failed to generate summary for {pid} model={model} attempt={attempt}/{max_attempts}: {e}"
        )
        err = str(e)

        # Don't mark as failed if it's just lock contention
        if "Lock contention" in err:
            if attempt >= max_attempts:
                _update_task_status(
                    task_id, "failed", error=err, pid=pid, model=model, user=user
                )
                _update_summary_status_db(
                    pid, model, "failed", err, task_id=task_id, task_user=user
                )
                if user:
                    _update_readinglist_summary_status(
                        user, pid, "failed", err, task_id=task_id, model=model
                    )
        else:
            _update_task_status(
                task_id, "failed", error=err, pid=pid, model=model, user=user
            )
            _update_summary_status_db(
                pid, model, "failed", err, task_id=task_id, task_user=user
            )
            if user:
                _update_readinglist_summary_status(
                    user, pid, "failed", err, task_id=task_id, model=model
                )

        raise


def enqueue_summary_task(
    pid: str,
    model: str | None = None,
    user: str | None = None,
    priority: int | None = None,
    force_refresh: bool = False,
) -> str:
    model = (model or _default_llm_name() or "").strip()
    if not model:
        raise ValueError("Model is required")

    # Normalize to raw pid to avoid duplicates between versioned/unversioned inputs.
    raw_pid, _ = split_pid_version(pid)
    pid = raw_pid or pid

    if not force_refresh:
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
        # No active task found; this is an enqueue failure. Returning "" would cause the caller
        # to treat it as "queued but task_id hidden", which can leave a queued status stuck.
        logger.warning(f"Enqueue lock timeout for {pid}::{model}; enqueue failed")
        raise RuntimeError("enqueue_lock_timeout")

    try:
        existing_info = None
        try:
            existing_info = SummaryStatusRepository.get_status(pid, model)
        except Exception:
            existing_info = None

        if force_refresh:
            # Cancel any existing queued/running tasks and bump epoch before enqueueing.
            # This enables "force regenerate" semantics via cancel+enqueue.
            cancel_summary_tasks(pid, model, user=user, reason="Force regenerate")
            # NOTE: Avoid passing new kwargs to Huey payload to stay compatible with
            # older workers that don't accept force_refresh.
            _purge_summary_cache(pid, model)
            if isinstance(existing_info, dict):
                resolved_model = str(
                    existing_info.get("resolved_model")
                    or existing_info.get("llm_model")
                    or ""
                ).strip()
                if resolved_model and resolved_model != model:
                    _purge_summary_cache(pid, resolved_model)

        # Check if there's already a queued/running task for this pid+model.
        if not force_refresh:
            existing_task_id, existing_task_user = _find_active_task(pid, model)
            if existing_task_id:
                if _allow_task_id(existing_task_user, user):
                    logger.debug(
                        f"Task already exists for {pid}::{model}: {existing_task_id}"
                    )
                    return existing_task_id
                # Don't leak task_id across users.
                logger.debug(
                    f"Active task exists for {pid}::{model} but belongs to a different user"
                )
                return ""

        task_priority = SUMMARY_PRIORITY_HIGH if priority is None else priority
        epoch = 0
        try:
            epoch = SummaryStatusRepository.get_generation_epoch(pid, model)
        except Exception:
            epoch = 0

        # Keep Huey payload minimal for forward/backward compatibility.
        task = generate_summary_task.s(pid, model=model, user=user, epoch=epoch)
        task.priority = task_priority
        result = huey.enqueue(task)
        task_id = task.id or getattr(result, "id", None)
        if task_id is None:
            logger.warning(f"Failed to obtain Huey task id for {pid}::{model}")
            raise RuntimeError("enqueue_task_id_missing")

        _update_task_status(
            task_id,
            "queued",
            pid=pid,
            model=model,
            user=user,
            priority=task_priority,
            epoch=epoch,
            force_refresh=bool(force_refresh),
        )
        _update_summary_status_db(
            pid, model, "queued", None, task_id=task_id, task_user=user
        )
        if user:
            _update_readinglist_summary_status(
                user, pid, "queued", None, task_id=task_id, model=model
            )

        return str(task_id)
    finally:
        try:
            release_summary_lock(lock_fd, enqueue_lock)
        except Exception:
            pass


def _purge_summary_cache(pid: str, model: str) -> None:
    """Best-effort delete cached summary files for (pid, model).

    This is used to implement force regeneration semantics without relying on
    Huey task kwargs, which can break older workers when new kwargs are added.
    """
    try:
        cache_file, meta_file, _lock_file, legacy_cache, legacy_meta, _legacy_lock = (
            summary_cache_paths(pid, model)
        )
    except Exception:
        return

    for path in (cache_file, meta_file, legacy_cache, legacy_meta):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    # Remove empty dir best-effort (don't fail force refresh on cleanup).
    try:
        base_dir = cache_file.parent
        if base_dir.exists() and not any(base_dir.iterdir()):
            base_dir.rmdir()
    except Exception:
        pass


def _collect_task_priority_map() -> dict:
    """Return latest priority per (pid, model) from task status records."""
    priority_map = {}
    try:
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix("task::")
        ) as items:
            for _key, info in items:
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
                    priority_map[map_key] = {
                        "priority": priority,
                        "updated_time": updated_time,
                    }
    except Exception as e:
        logger.warning(f"Failed to collect task priorities: {e}")
    return priority_map


def _collect_task_user_map() -> dict:
    """Return latest user per (pid, model) from task status records."""
    user_map = {}
    try:
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix("task::")
        ) as items:
            for _key, info in items:
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


def repair_stale_summary_tasks(
    max_age_s: int | None = None, requeue: bool = False
) -> int:
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

    stale_items: list[tuple[str, str, dict]] = []
    try:
        with safe_closing(SummaryStatusRepository.get_all_items()) as items:
            for key, info in items:
                if key.startswith("task::"):
                    continue
                if not isinstance(info, dict):
                    continue
                status = info.get("status")
                updated_time = info.get("updated_time") or 0
                if status not in {"queued", "running"}:
                    continue
                if now - float(updated_time) < max_age_s:
                    continue

                if "::" not in key:
                    continue
                pid, model = key.split("::", 1)
                if not pid or not model:
                    continue
                # Do not mutate the summary-status DB while iterating its items: snapshot stale keys
                # and repair them after closing the iterator to avoid SQLite cursor/locking issues.
                stale_items.append((pid, model, dict(info)))
    except Exception as e:
        logger.warning(f"Failed to repair stale summary tasks: {e}")

    for pid, model, info in stale_items:
        try:
            from backend.services.summary_service import has_active_summary_lock

            if has_active_summary_lock(pid, model):
                continue
        except Exception:
            pass

        task_id = info.get("task_id") if isinstance(info, dict) else None
        if task_id:
            try:
                task_info = SummaryStatusRepository.get_task_status(str(task_id))
            except Exception:
                task_info = None
            if isinstance(task_info, dict):
                task_status = str(task_info.get("status") or "").strip().lower()
                try:
                    task_updated_time = float(task_info.get("updated_time") or 0.0)
                except Exception:
                    task_updated_time = 0.0
                if (
                    task_status in {"queued", "running"}
                    and task_updated_time > 0
                    and now - task_updated_time < max_age_s
                ):
                    continue

        try:
            (
                _cache_file,
                _meta_file,
                lock_file,
                _legacy_cache,
                _legacy_meta,
                legacy_lock,
            ) = summary_cache_paths(pid, model)
            for lock_path in (lock_file, legacy_lock):
                try:
                    if lock_path.exists():
                        lock_mtime = lock_path.stat().st_mtime
                        if now - lock_mtime >= max_age_s:
                            lock_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove stale lock {lock_path}: {e}")
        except Exception:
            pass

        user_info = user_map.get((pid, model))
        user = user_info.get("user") if user_info else None
        stale_status = str(info.get("status") or "").strip().lower()
        stale_error = f"stale_{stale_status or 'running'}_repaired"

        if requeue:
            if task_id:
                try:
                    SummaryStatusRepository.set_task_status(
                        str(task_id),
                        "failed",
                        error=stale_error,
                        pid=pid,
                        model=model,
                        user=user,
                    )
                except Exception:
                    pass
            if _is_upload_pid(pid):
                _update_upload_task_reference(
                    pid,
                    record_field="summary_task_id",
                    task_id=str(task_id or ""),
                    clear=True,
                )
            priority_info = priority_map.get((pid, model))
            priority = priority_info.get("priority") if priority_info else None
            enqueue_summary_task(pid, model=model, user=user, priority=priority)
        else:
            try:
                SummaryStatusRepository.update_status(
                    pid,
                    model,
                    {
                        "status": "failed",
                        "last_error": stale_error,
                        "task_id": None,
                        "task_user": None,
                        "resolved_model": None,
                        "updated_time": now,
                    },
                )
            except Exception:
                pass
            if task_id:
                try:
                    SummaryStatusRepository.set_task_status(
                        str(task_id),
                        "failed",
                        error=stale_error,
                        pid=pid,
                        model=model,
                        user=user,
                    )
                except Exception:
                    pass
            if _is_upload_pid(pid):
                _update_upload_task_reference(
                    pid,
                    record_field="summary_task_id",
                    task_id=str(task_id or ""),
                    clear=True,
                )
            if user:
                try:
                    _update_readinglist_summary_status(
                        user,
                        pid,
                        "failed",
                        stale_error,
                        model=model,
                    )
                except Exception:
                    pass
        repaired += 1

    return repaired


if settings.huey.summary_repair_on_start:
    # Only run repair if we're in a Huey consumer process (not during import in web server)
    if _is_huey_consumer_process() or settings.huey.force_repair:
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
        with safe_closing(
            SummaryStatusRepository.get_items_with_prefix("task::")
        ) as items:
            for key, info in list(items):
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
                            if isinstance(current, dict) and current.get("status") in (
                                "queued",
                                "running",
                            ):
                                SummaryStatusRepository.delete_status(pid, model)
                        except Exception:
                            pass
                    # Best-effort: clear readinglist task linkage to avoid UI stuck in queued/running.
                    task_user = info.get("user")
                    if pid and task_user and status in ("queued", "running"):
                        try:
                            _update_readinglist_summary_status(
                                task_user,
                                pid,
                                "failed",
                                "task_cleaned",
                                task_id=None,
                                model=model,
                            )
                        except Exception:
                            pass

                cleaned += 1

    except Exception as e:
        logger.warning(f"Failed to cleanup tasks: {e}")

    return {"cleaned": cleaned, "details": details, "dry_run": dry_run}


# -----------------------------------------------------------------------------
# Uploaded PDF Processing Task
# -----------------------------------------------------------------------------


@huey.task(retries=2, retry_delay=60, context=True)
def process_uploaded_pdf_task(
    pid: str, user: str, model: str | None = None, task=None, **_kwargs
) -> None:
    """
    Process an uploaded PDF: MinerU parsing + LLM metadata extraction.
    After parsing completes, automatically triggers summary generation.

    Args:
        pid: Upload PID (e.g., up_V1StGXR8_Z5j)
        user: Username (owner)
        model: LLM model for summary (optional, uses default)
    """
    from backend.services.upload_service import (
        UPLOAD_TASK_MODEL_PROCESS,
        UploadServiceError,
        process_uploaded_pdf,
    )

    task_id = str(getattr(task, "id", None) or "")
    if not _is_current_upload_task(
        pid,
        record_field="parse_task_id",
        task_id=task_id,
        allowed_statuses={"queued", "running"},
    ):
        _maybe_mark_upload_task_canceled(
            pid,
            record_field="parse_task_id",
            task_id=task_id,
            model=UPLOAD_TASK_MODEL_PROCESS,
            user=user,
        )
        return
    _update_task_status(
        task_id, "running", pid=pid, model=UPLOAD_TASK_MODEL_PROCESS, user=user
    )

    try:
        process_uploaded_pdf(pid, user, model, current_task_id=task_id)
        if not _is_current_upload_task(
            pid,
            record_field="parse_task_id",
            task_id=task_id,
            allowed_statuses={"queued", "running", "ok"},
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_PROCESS,
                user=user,
            )
            return
        _update_task_status(
            task_id, "ok", pid=pid, model=UPLOAD_TASK_MODEL_PROCESS, user=user
        )
        _update_upload_task_reference(
            pid, record_field="parse_task_id", task_id=task_id, clear=True
        )
    except UploadServiceError as e:
        if not _is_current_upload_task(
            pid, record_field="parse_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_PROCESS,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="parse_task_id", task_id=task_id, clear=True
            )
            return
        logger.error(f"Failed to process uploaded PDF {pid}: {e}")
        _update_task_status(
            task_id,
            "failed",
            error=e.code,
            pid=pid,
            model=UPLOAD_TASK_MODEL_PROCESS,
            user=user,
        )
        _update_upload_task_reference(
            pid, record_field="parse_task_id", task_id=task_id, clear=True
        )
        raise
    except Exception as e:
        if not _is_current_upload_task(
            pid, record_field="parse_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_PROCESS,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="parse_task_id", task_id=task_id, clear=True
            )
            return
        logger.error(f"Failed to process uploaded PDF {pid}: {e}")
        _update_task_status(
            task_id,
            "failed",
            error=str(e),
            pid=pid,
            model=UPLOAD_TASK_MODEL_PROCESS,
            user=user,
        )
        _update_upload_task_reference(
            pid, record_field="parse_task_id", task_id=task_id, clear=True
        )
        raise


@huey.task(retries=2, retry_delay=60, context=True)
def parse_uploaded_pdf_task(pid: str, user: str, task=None, **_kwargs) -> None:
    """
    Parse an uploaded PDF with MinerU only (no metadata extraction).

    Args:
        pid: Upload PID
        user: Username (owner)
    """
    from backend.services.upload_service import (
        UPLOAD_TASK_MODEL_PARSE,
        UploadServiceError,
        do_parse_only,
    )

    task_id = str(getattr(task, "id", None) or "")
    if not _is_current_upload_task(
        pid,
        record_field="parse_task_id",
        task_id=task_id,
        allowed_statuses={"queued", "running"},
    ):
        _maybe_mark_upload_task_canceled(
            pid,
            record_field="parse_task_id",
            task_id=task_id,
            model=UPLOAD_TASK_MODEL_PARSE,
            user=user,
        )
        return
    _update_task_status(
        task_id, "running", pid=pid, model=UPLOAD_TASK_MODEL_PARSE, user=user
    )

    try:
        ok = do_parse_only(pid, user, current_task_id=task_id)
        if ok:
            if not _is_current_upload_task(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                allowed_statuses={"queued", "running", "ok"},
            ):
                _maybe_mark_upload_task_canceled(
                    pid,
                    record_field="parse_task_id",
                    task_id=task_id,
                    model=UPLOAD_TASK_MODEL_PARSE,
                    user=user,
                )
                return
            _update_task_status(
                task_id, "ok", pid=pid, model=UPLOAD_TASK_MODEL_PARSE, user=user
            )
            _update_upload_task_reference(
                pid, record_field="parse_task_id", task_id=task_id, clear=True
            )
            return
        if not _is_current_upload_task(
            pid, record_field="parse_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_PARSE,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="parse_task_id", task_id=task_id, clear=True
            )
            return
        _update_task_status(
            task_id,
            "failed",
            error="parse_failed",
            pid=pid,
            model=UPLOAD_TASK_MODEL_PARSE,
            user=user,
        )
        _update_upload_task_reference(
            pid, record_field="parse_task_id", task_id=task_id, clear=True
        )
    except UploadServiceError:
        if not _is_current_upload_task(
            pid, record_field="parse_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_PARSE,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="parse_task_id", task_id=task_id, clear=True
            )
            return
        raise
    except Exception as e:
        if not _is_current_upload_task(
            pid, record_field="parse_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="parse_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_PARSE,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="parse_task_id", task_id=task_id, clear=True
            )
            return
        logger.error(f"Failed to parse uploaded PDF {pid}: {e}")
        _update_task_status(
            task_id,
            "failed",
            error=str(e),
            pid=pid,
            model=UPLOAD_TASK_MODEL_PARSE,
            user=user,
        )
        _update_upload_task_reference(
            pid, record_field="parse_task_id", task_id=task_id, clear=True
        )
        raise


@huey.task(retries=2, retry_delay=30, context=True)
def extract_info_task(pid: str, user: str, task=None, **_kwargs) -> None:
    """
    Extract metadata from an already-parsed uploaded PDF.

    Args:
        pid: Upload PID
        user: Username (owner)
    """
    from backend.services.upload_service import (
        UPLOAD_TASK_MODEL_EXTRACT,
        UploadServiceError,
        do_extract_metadata,
    )

    task_id = str(getattr(task, "id", None) or "")
    if not _is_current_upload_task(
        pid,
        record_field="extract_task_id",
        task_id=task_id,
    ):
        _maybe_mark_upload_task_canceled(
            pid,
            record_field="extract_task_id",
            task_id=task_id,
            model=UPLOAD_TASK_MODEL_EXTRACT,
            user=user,
        )
        return
    _update_task_status(
        task_id, "running", pid=pid, model=UPLOAD_TASK_MODEL_EXTRACT, user=user
    )

    try:
        ok = do_extract_metadata(pid, user, current_task_id=task_id)
        if ok:
            if not _is_current_upload_task(
                pid, record_field="extract_task_id", task_id=task_id
            ):
                _maybe_mark_upload_task_canceled(
                    pid,
                    record_field="extract_task_id",
                    task_id=task_id,
                    model=UPLOAD_TASK_MODEL_EXTRACT,
                    user=user,
                )
                return
            _update_task_status(
                task_id, "ok", pid=pid, model=UPLOAD_TASK_MODEL_EXTRACT, user=user
            )
            _update_upload_task_reference(
                pid, record_field="extract_task_id", task_id=task_id, clear=True
            )
            return
        if not _is_current_upload_task(
            pid, record_field="extract_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="extract_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_EXTRACT,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="extract_task_id", task_id=task_id, clear=True
            )
            return
        _update_task_status(
            task_id,
            "failed",
            error="extract_failed",
            pid=pid,
            model=UPLOAD_TASK_MODEL_EXTRACT,
            user=user,
        )
        _update_upload_task_reference(
            pid, record_field="extract_task_id", task_id=task_id, clear=True
        )
    except UploadServiceError:
        if not _is_current_upload_task(
            pid, record_field="extract_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="extract_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_EXTRACT,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="extract_task_id", task_id=task_id, clear=True
            )
            return
        raise
    except Exception as e:
        if not _is_current_upload_task(
            pid, record_field="extract_task_id", task_id=task_id
        ):
            _maybe_mark_upload_task_canceled(
                pid,
                record_field="extract_task_id",
                task_id=task_id,
                model=UPLOAD_TASK_MODEL_EXTRACT,
                user=user,
            )
            _update_upload_task_reference(
                pid, record_field="extract_task_id", task_id=task_id, clear=True
            )
            return
        logger.error(f"Failed to extract info for {pid}: {e}")
        _update_task_status(
            task_id,
            "failed",
            error=str(e),
            pid=pid,
            model=UPLOAD_TASK_MODEL_EXTRACT,
            user=user,
        )
        _update_upload_task_reference(
            pid, record_field="extract_task_id", task_id=task_id, clear=True
        )
        raise
