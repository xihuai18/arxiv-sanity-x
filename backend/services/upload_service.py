"""
Upload service for handling uploaded PDF papers.

This module provides business logic for:
- Uploading and storing PDF files
- Triggering MinerU parsing
- Extracting metadata via LLM
- Managing uploaded paper lifecycle
"""

import importlib
import re
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

import tools.paper_summarizer as paper_summarizer
from aslite.repositories import (
    NegativeTagRepository,
    SummaryStatusRepository,
    TagRepository,
    UploadedPaperRepository,
    safe_closing,
)
from backend.services.opencode_service import generate_structured_json
from backend.utils.upload_utils import (
    compute_bytes_sha256,
    generate_upload_pid,
    get_upload_dir,
    get_upload_pdf_path,
    sanitize_filename,
    validate_upload_pid,
)
from config import settings
from config.model_aliases import build_model_candidate_chain

from .user_service import build_pid_tag_reverse_index as _build_pid_tag_reverse_index


def _data_dir() -> str:
    return str(settings.data_dir)


def _summary_dir() -> str:
    return str(settings.summary_dir)


def _get_uploaded_papers_db(**kwargs):
    from aslite.db import get_uploaded_papers_db

    return get_uploaded_papers_db(**kwargs)


# SSE enabled flag - check if we're in a web context
_SSE_ENABLED = True


def _get_tasks_module():
    """Lazy import tasks module to avoid circular imports."""
    return importlib.import_module("tasks")


def _emit_upload_event(user: str, payload: dict) -> None:
    """Emit SSE event for upload status changes.

    Args:
        user: Username to send event to
        payload: Event payload dict
    """
    if not _SSE_ENABLED:
        return
    try:
        from backend.utils.sse import emit_user_event

        emit_user_event(user, payload)
    except Exception as e:
        logger.debug(f"Failed to emit upload event: {e}")


def _llm_name() -> str:
    return str(settings.llm.name or "")


def _extract_model_name() -> str:
    return str(settings.extract_info.model_name or "")


# Reasonable limits to prevent abuse / DB bloat.
# Keep them generous to avoid surprising users, but bounded.
MAX_META_TITLE_CHARS = 512
MAX_META_ABSTRACT_CHARS = 20000
MAX_META_AUTHOR_COUNT = 200
MAX_META_AUTHOR_CHARS = 200

UPLOAD_TASK_MODEL_PROCESS = "upload_process"
UPLOAD_TASK_MODEL_PARSE = "upload_parse"
UPLOAD_TASK_MODEL_EXTRACT = "upload_extract"

_UPLOAD_TASK_SNAPSHOT: ContextVar[dict[str, Any] | None] = ContextVar(
    "upload_task_snapshot",
    default=None,
)


def _extract_huey_task_id(task: Any = None, enqueue_result: Any = None) -> str:
    """Best-effort extract Huey task id from task/result objects."""
    for obj in (task, enqueue_result):
        if obj is None:
            continue
        task_id = getattr(obj, "id", None)
        if task_id:
            return str(task_id)
    return ""


def _get_upload_task_meta(task_type: str) -> tuple[str, str]:
    """Return (model, record_field) for upload task bookkeeping."""
    if task_type == "extract":
        return UPLOAD_TASK_MODEL_EXTRACT, "extract_task_id"
    if task_type == "parse":
        return UPLOAD_TASK_MODEL_PARSE, "parse_task_id"
    return UPLOAD_TASK_MODEL_PROCESS, "parse_task_id"


def register_upload_task_enqueue(
    *,
    task_type: str,
    pid: str,
    user: str,
    task: Any = None,
    enqueue_result: Any = None,
) -> str:
    """Persist upload task queued status and return task id if available."""
    task_id = _extract_huey_task_id(task, enqueue_result)
    if not task_id:
        return ""

    model, record_field = _get_upload_task_meta(task_type)
    try:
        UploadedPaperRepository.update(pid, {record_field: task_id})
    except Exception as e:
        logger.warning(f"Failed to update {record_field} for {pid}: {e}")

    try:
        SummaryStatusRepository.set_task_status(task_id, "queued", None, pid=pid, model=model, user=user)
    except Exception as e:
        logger.warning(f"Failed to write upload task queued status for {pid}: {e}")

    return task_id


def _get_active_upload_task_id(record: dict[str, Any], record_field: str) -> str:
    """Return active queued/running task id from upload record, if any."""
    task_state, task_id, _info = _classify_or_recover_upload_task(record, record_field)
    if task_state == "active":
        return task_id
    return ""


def _upload_task_repair_ttl() -> int:
    try:
        return max(0, int(settings.huey.upload_repair_ttl or 0))
    except Exception:
        return 0


def _upload_task_pointer_grace_seconds() -> int:
    """Grace window for recently queued upload records without stable task metadata.

    This avoids falsely repairing records during the short interval between:
    - setting `parse_status=queued`
    - persisting the concrete Huey task id / task status record
    """
    return 15


def _get_upload_record_task_models(record_field: str) -> set[str]:
    if record_field == "extract_task_id":
        return {UPLOAD_TASK_MODEL_EXTRACT}
    if record_field == "parse_task_id":
        return {UPLOAD_TASK_MODEL_PARSE, UPLOAD_TASK_MODEL_PROCESS}
    return set()


def _get_upload_record_field_for_task_model(task_model: str) -> str | None:
    normalized_task_model = str(task_model or "").strip()
    if normalized_task_model in {UPLOAD_TASK_MODEL_PROCESS, UPLOAD_TASK_MODEL_PARSE}:
        return "parse_task_id"
    if normalized_task_model == UPLOAD_TASK_MODEL_EXTRACT:
        return "extract_task_id"
    return None


def _build_upload_task_snapshot(user: str, records_by_pid: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    normalized_user = str(user or "").strip()
    candidate_pids = {str(pid or "").strip() for pid in (records_by_pid or {}).keys() if str(pid or "").strip()}
    snapshot: dict[str, Any] = {
        "complete": False,
        "task_by_id": {},
        "active_by_record_field": {},
    }
    if not normalized_user or not candidate_pids:
        snapshot["complete"] = True
        return snapshot

    task_by_id: dict[str, dict[str, Any]] = {}
    active_candidates: dict[tuple[str, str, str], tuple[float, str, dict[str, Any]]] = {}

    try:
        with safe_closing(SummaryStatusRepository.get_items_with_prefix("task::")) as items:
            for key, info in items:
                if not isinstance(info, dict):
                    continue
                pid = str(info.get("pid") or "").strip()
                task_user = str(info.get("user") or "").strip()
                if pid not in candidate_pids or task_user != normalized_user:
                    continue

                task_id = str(key).replace("task::", "")
                if not task_id:
                    continue

                task_by_id[task_id] = info

                record_field = _get_upload_record_field_for_task_model(info.get("model") or "")
                task_status = str(info.get("status") or "").strip().lower()
                if not record_field or task_status not in {"queued", "running"}:
                    continue

                try:
                    updated_time = float(info.get("updated_time") or 0.0)
                except Exception:
                    updated_time = 0.0

                active_key = (pid, task_user, record_field)
                candidate = (updated_time, task_id, info)
                existing = active_candidates.get(active_key)
                if existing is None or (updated_time, task_id) > (
                    existing[0],
                    existing[1],
                ):
                    active_candidates[active_key] = candidate
    except Exception as e:
        logger.debug(f"Failed to build upload task snapshot for {normalized_user}: {e}")
        return snapshot

    snapshot["complete"] = True
    snapshot["task_by_id"] = task_by_id
    snapshot["active_by_record_field"] = {
        key: (task_id, info) for key, (_updated_time, task_id, info) in active_candidates.items()
    }
    return snapshot


def _get_upload_task_snapshot() -> dict[str, Any] | None:
    snapshot = _UPLOAD_TASK_SNAPSHOT.get()
    return snapshot if isinstance(snapshot, dict) else None


@contextmanager
def _use_upload_task_snapshot(snapshot: dict[str, Any] | None):
    token = _UPLOAD_TASK_SNAPSHOT.set(snapshot if isinstance(snapshot, dict) else None)
    try:
        yield
    finally:
        _UPLOAD_TASK_SNAPSHOT.reset(token)


def _upload_task_pointer_is_within_grace(record: dict[str, Any]) -> bool:
    grace_seconds = _upload_task_pointer_grace_seconds()
    if grace_seconds <= 0:
        return False

    reference_ts = 0.0
    for field in ("updated_time", "created_time"):
        try:
            reference_ts = max(reference_ts, float(record.get(field) or 0.0))
        except Exception:
            continue

    if reference_ts <= 0:
        return True
    return (time.time() - reference_ts) < grace_seconds


def _find_active_upload_task_for_record(
    pid: str,
    user: str,
    record_field: str,
) -> tuple[str, dict[str, Any] | None]:
    """Best-effort recover the active task id for an upload record field."""
    normalized_pid = str(pid or "").strip()
    normalized_user = str(user or "").strip()
    task_models = _get_upload_record_task_models(record_field)
    if not normalized_pid or not normalized_user or not task_models:
        return "", None

    snapshot = _get_upload_task_snapshot()
    if snapshot is not None:
        active_by_record_field = snapshot.get("active_by_record_field") or {}
        candidate = active_by_record_field.get((normalized_pid, normalized_user, record_field))
        if isinstance(candidate, tuple) and len(candidate) == 2:
            task_id, info = candidate
            return str(task_id or ""), info if isinstance(info, dict) else None
        if snapshot.get("complete"):
            return "", None

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    try:
        with safe_closing(SummaryStatusRepository.get_items_with_prefix("task::")) as items:
            for key, info in items:
                if not isinstance(info, dict):
                    continue
                if info.get("pid") != normalized_pid or info.get("user") != normalized_user:
                    continue
                task_model = str(info.get("model") or "").strip()
                if task_model not in task_models:
                    continue
                task_status = str(info.get("status") or "").strip().lower()
                if task_status not in {"queued", "running"}:
                    continue
                task_id = str(key).replace("task::", "")
                if not task_id:
                    continue
                try:
                    updated_time = float(info.get("updated_time") or 0.0)
                except Exception:
                    updated_time = 0.0
                candidates.append((updated_time, task_id, info))
    except Exception as e:
        logger.debug(f"Failed to scan active upload tasks for {normalized_pid}:{record_field}: {e}")
        return "", None

    if not candidates:
        return "", None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _updated_time, task_id, info = candidates[0]
    return task_id, info


def _classify_or_recover_upload_task(
    record: dict[str, Any], record_field: str
) -> tuple[str, str, dict[str, Any] | None]:
    """Classify task pointer state, with upload-specific recovery for races/orphans."""
    current_task_id = str(record.get(record_field) or "").strip()
    task_state, task_id, task_info = _classify_upload_task(current_task_id)
    if task_state == "active":
        return task_state, task_id, task_info

    pid = str(record.get("pid") or "").strip()
    owner = str(record.get("owner") or "").strip()
    recovered_task_id, recovered_task_info = _find_active_upload_task_for_record(pid, owner, record_field)
    if recovered_task_id:
        if recovered_task_id != current_task_id and pid:
            try:
                UploadedPaperRepository.update(pid, {record_field: recovered_task_id})
            except Exception as e:
                logger.debug(f"Failed to recover upload task pointer for {pid}:{record_field}: {e}")
            record[record_field] = recovered_task_id
        return "active", recovered_task_id, recovered_task_info

    if task_state == "missing" and _upload_task_pointer_is_within_grace(record):
        return "pending_registration", current_task_id, task_info

    return task_state, task_id, task_info


def _ensure_upload_record_active(
    pid: str,
    user: str,
    *,
    record_field: str | None = None,
    task_id: str | None = None,
    allowed_parse_statuses: set[str] | None = None,
) -> dict[str, Any]:
    """Ensure upload record still exists, is owned, and task pointer is still current."""
    record = _get_owned_upload_record(pid, user)

    if record_field:
        expected_task_id = str(task_id or "").strip()
        current_task_id = str(record.get(record_field) or "").strip()
        if expected_task_id and current_task_id != expected_task_id:
            raise UploadServiceError("superseded_task", "Task is no longer current")

    if allowed_parse_statuses:
        current_status = str(record.get("parse_status") or "").strip().lower()
        normalized_allowed = {status.lower() for status in allowed_parse_statuses}
        if current_status not in normalized_allowed:
            raise UploadServiceError("superseded_task", "Task is no longer runnable")

    return record


def _is_upload_record_current(
    pid: str,
    user: str,
    *,
    record_field: str | None = None,
    task_id: str | None = None,
    allowed_parse_statuses: set[str] | None = None,
) -> bool:
    """Best-effort check whether the upload record still matches the running task."""
    try:
        _ensure_upload_record_active(
            pid,
            user,
            record_field=record_field,
            task_id=task_id,
            allowed_parse_statuses=allowed_parse_statuses,
        )
        return True
    except UploadServiceError:
        return False


def _update_upload_record_if_current(
    pid: str,
    user: str,
    *,
    updates: dict[str, Any],
    record_field: str | None = None,
    task_id: str | None = None,
    allowed_parse_statuses: set[str] | None = None,
) -> bool:
    """Atomically update an upload record only if the running task is still current."""
    if not isinstance(updates, dict) or not updates:
        return _is_upload_record_current(
            pid,
            user,
            record_field=record_field,
            task_id=task_id,
            allowed_parse_statuses=allowed_parse_statuses,
        )

    expected_task_id = str(task_id or "").strip()
    allowed_statuses = {status.lower() for status in (allowed_parse_statuses or set())}

    if not expected_task_id:
        try:
            _ensure_upload_record_active(
                pid,
                user,
                allowed_parse_statuses=allowed_parse_statuses,
            )
        except UploadServiceError:
            return False
        try:
            return bool(UploadedPaperRepository.update(pid, updates))
        except Exception as e:
            logger.warning(f"Failed fallback upload record update for {pid}: {e}")
            return False

    try:
        with _get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                record = updb.get(pid)
                if not isinstance(record, dict):
                    return False
                if record.get("owner") != user:
                    return False
                if record.get("deleting") is True:
                    return False
                if record_field:
                    current_task_id = str(record.get(record_field) or "").strip()
                    if expected_task_id and current_task_id != expected_task_id:
                        return False
                if allowed_statuses:
                    current_status = str(record.get("parse_status") or "").strip().lower()
                    if current_status not in allowed_statuses:
                        return False

                next_record = dict(record)
                next_record.update(updates)
                next_record["updated_time"] = time.time()
                updb[pid] = next_record
                return True
    except Exception as e:
        logger.warning(f"Failed guarded upload record update for {pid}: {e}")
        return False


def _emit_upload_event_if_current(
    pid: str,
    user: str,
    payload: dict[str, Any],
    *,
    record_field: str | None = None,
    task_id: str | None = None,
    allowed_parse_statuses: set[str] | None = None,
) -> bool:
    """Emit an upload SSE event only while the task still owns the record."""
    if not _is_upload_record_current(
        pid,
        user,
        record_field=record_field,
        task_id=task_id,
        allowed_parse_statuses=allowed_parse_statuses,
    ):
        return False
    _emit_upload_event(user, payload)
    return True


def _classify_upload_task(task_id: Any) -> tuple[str, str, dict[str, Any] | None]:
    """Classify an upload task pointer as active/stale/terminal/missing."""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return "missing", "", None

    snapshot = _get_upload_task_snapshot()
    if snapshot is not None:
        info = (snapshot.get("task_by_id") or {}).get(normalized_task_id)
    else:
        info = None

    try:
        if info is None:
            info = SummaryStatusRepository.get_task_status(normalized_task_id)
    except Exception:
        return "missing", normalized_task_id, None

    if not isinstance(info, dict):
        return "missing", normalized_task_id, None

    status = str(info.get("status") or "").strip().lower()
    if status not in {"queued", "running"}:
        return "terminal", normalized_task_id, info

    ttl = _upload_task_repair_ttl()
    if ttl > 0:
        try:
            updated_time = float(info.get("updated_time") or 0)
        except Exception:
            updated_time = 0.0
        if updated_time > 0 and (time.time() - updated_time) >= ttl:
            return "stale", normalized_task_id, info

    return "active", normalized_task_id, info


def _repair_upload_task_status(
    task_id: str,
    *,
    pid: str,
    user: str,
    info: dict[str, Any] | None = None,
    sync_record: bool = True,
) -> None:
    """Best-effort mark a stale upload task as failed so it does not stay stuck."""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return

    task_info = info if isinstance(info, dict) else SummaryStatusRepository.get_task_status(normalized_task_id) or {}
    status = str(task_info.get("status") or "").strip().lower()
    if status not in {"queued", "running"}:
        return

    task_model = str(task_info.get("model") or "").strip() or UPLOAD_TASK_MODEL_PARSE
    try:
        SummaryStatusRepository.set_task_status(
            normalized_task_id,
            "failed",
            "stale_running_repaired",
            pid=pid,
            model=task_model,
            user=user,
        )
    except Exception as e:
        logger.warning(f"Failed to repair stale upload task {normalized_task_id} for {pid}: {e}")

    record_field = None
    record_updates: dict[str, Any] = {}
    if task_model in {UPLOAD_TASK_MODEL_PROCESS, UPLOAD_TASK_MODEL_PARSE}:
        record_field = "parse_task_id"
        record_updates.update(
            {
                "parse_status": "failed",
                "parse_error": "stale_running_repaired",
            }
        )
    elif task_model == UPLOAD_TASK_MODEL_EXTRACT:
        record_field = "extract_task_id"

    if sync_record and record_field:
        record_updates[record_field] = None
        try:
            UploadedPaperRepository.update(pid, record_updates)
        except Exception as e:
            logger.warning(f"Failed to update upload record while repairing stale task {normalized_task_id}: {e}")


def _get_owned_upload_record(pid: str, user: str, *, allow_deleting: bool = False) -> dict[str, Any]:
    """Return an upload record owned by user or raise UploadServiceError."""
    if not validate_upload_pid(pid):
        logger.error(f"Invalid upload PID format: {pid}")
        raise UploadServiceError("invalid_pid", "Invalid paper ID")

    record = UploadedPaperRepository.get(pid)
    if not isinstance(record, dict):
        raise UploadServiceError("not_found", "Paper not found")

    if record.get("owner") != user:
        raise UploadServiceError("not_owner", "Paper not found")

    if not allow_deleting and record.get("deleting") is True:
        raise UploadServiceError("deleting", "Paper is being deleted")

    return record


def _cancel_active_upload_tasks(pid: str, user: str, *, reason: str) -> dict[str, Any]:
    """Best-effort cancel queued/running upload parse/process/extract tasks."""
    task_models = {
        UPLOAD_TASK_MODEL_PROCESS,
        UPLOAD_TASK_MODEL_PARSE,
        UPLOAD_TASK_MODEL_EXTRACT,
    }
    tasks_by_id: dict[str, dict[str, Any]] = {}

    try:
        record = UploadedPaperRepository.get(pid) or {}
    except Exception:
        record = {}

    for field in ("parse_task_id", "extract_task_id"):
        task_id = str(record.get(field) or "").strip()
        if not task_id:
            continue
        try:
            info = SummaryStatusRepository.get_task_status(task_id) or {}
        except Exception:
            info = {}
        if isinstance(info, dict):
            tasks_by_id[task_id] = info

    try:
        with safe_closing(SummaryStatusRepository.get_items_with_prefix("task::")) as items:
            for key, info in items:
                if not isinstance(info, dict):
                    continue
                if info.get("pid") != pid:
                    continue
                task_model = str(info.get("model") or "").strip()
                if task_model not in task_models:
                    continue
                task_id = str(key).replace("task::", "")
                if task_id:
                    tasks_by_id[task_id] = info
    except Exception as e:
        logger.warning(f"Failed to scan upload tasks for {pid}: {e}")

    canceled_task_ids: list[str] = []
    revoked_task_ids: list[str] = []
    cleared_parse = False
    cleared_extract = False

    for task_id, info in tasks_by_id.items():
        status = str(info.get("status") or "").strip().lower()
        task_model = str(info.get("model") or "").strip()
        if status not in {"queued", "running"} or task_model not in task_models:
            continue

        try:
            SummaryStatusRepository.set_task_status(
                task_id,
                "canceled",
                reason,
                pid=pid,
                model=task_model,
                user=user,
                canceled_time=time.time(),
            )
            canceled_task_ids.append(task_id)
        except Exception as e:
            logger.warning(f"Failed to cancel upload task {task_id} for {pid}: {e}")

        try:
            _get_tasks_module()._revoke_task_by_id(task_id)
            revoked_task_ids.append(task_id)
        except Exception as e:
            logger.debug(f"Failed to revoke upload task {task_id} for {pid}: {e}")

        if task_model in {UPLOAD_TASK_MODEL_PROCESS, UPLOAD_TASK_MODEL_PARSE}:
            cleared_parse = True
        if task_model == UPLOAD_TASK_MODEL_EXTRACT:
            cleared_extract = True

    updates: dict[str, Any] = {}
    if cleared_parse:
        updates.update(
            {
                "parse_status": "failed",
                "parse_error": "paper_deleted",
                "parse_task_id": None,
            }
        )
    if cleared_extract:
        updates["extract_task_id"] = None
    if updates:
        try:
            UploadedPaperRepository.update(pid, updates)
        except Exception as e:
            logger.warning(f"Failed to clear upload task pointers for {pid}: {e}")

    return {
        "canceled_task_ids": canceled_task_ids,
        "parse_canceled": cleared_parse,
        "extract_canceled": cleared_extract,
        "revoked_task_ids": revoked_task_ids,
    }


def _infer_meta_extracted_ok(record: dict[str, Any], title: str, abstract: str, authors: list) -> bool:
    """Infer meta_extracted_ok status from available metadata.

    For backward compatibility with older records that don't have the meta_extracted_ok field,
    we infer it from the presence of title, abstract, or authors.

    Args:
        record: The uploaded paper record
        title: Paper title (may be empty)
        abstract: Paper abstract (may be empty)
        authors: List of authors (may be empty)

    Returns:
        bool: True if metadata was extracted (or can be inferred), False otherwise
    """
    if "meta_extracted_ok" in record:
        return bool(record.get("meta_extracted_ok"))

    # Infer from available metadata
    inferred_title = str(title or "").strip()
    # Do not treat original filename as a "real" extracted title.
    try:
        orig = str(record.get("original_filename") or "").strip()
        if orig and inferred_title == orig:
            inferred_title = ""
    except Exception:
        pass
    inferred_abs = str(abstract or "").strip()
    inferred_authors = authors if isinstance(authors, list) else []
    inferred_authors = [str(a).strip() for a in inferred_authors if str(a).strip()]

    return bool(inferred_title or inferred_abs or inferred_authors)


def _infer_upload_mineru_parsed_ok(pid: str) -> bool:
    """Best-effort infer whether MinerU parsing is complete for an uploaded paper.

    This is used for backward compatibility with older upload records that may
    miss `parse_status` but already have MinerU cache on disk.
    """
    # Keep aligned with tools.paper_summarizer.PaperSummarizer._MINERU_MD_MIN_SIZE
    min_size = 4096
    base_dir = Path(_data_dir()) / "mineru" / pid
    candidates = [
        base_dir / "auto" / f"{pid}.md",
        base_dir / "vlm" / f"{pid}.md",
        base_dir / "api" / f"{pid}.md",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file() and p.stat().st_size >= min_size:
                return True
        except Exception:
            continue

    try:
        if base_dir.exists() and base_dir.is_dir():
            for p in base_dir.glob(f"*/{pid}.md"):
                try:
                    if p.exists() and p.is_file() and p.stat().st_size >= min_size:
                        return True
                except Exception:
                    continue
    except Exception:
        return False

    return False


def _normalize_upload_parse_status(pid: str, record: dict[str, Any]) -> tuple[str, str]:
    """Return (parse_status, parse_error) with backward-compatible inference.

    Older upload records may miss `parse_status` even if MinerU cache exists on disk.
    """
    parse_status_raw = record.get("parse_status")
    parse_error_raw = record.get("parse_error") or ""

    parse_status = str(parse_status_raw).strip() if parse_status_raw is not None else ""
    parse_error = str(parse_error_raw).strip()

    if parse_status in {"queued", "running"}:
        task_state, task_id, task_info = _classify_or_recover_upload_task(record, "parse_task_id")
        if task_state == "active":
            task_status = str((task_info or {}).get("status") or parse_status).strip().lower()
            if task_status in {"queued", "running"}:
                return task_status, parse_error
            return parse_status, parse_error
        if task_state == "pending_registration":
            return parse_status, parse_error

        owner = str(record.get("owner") or "").strip()
        repaired_error = parse_error or "stale_running_repaired"
        if task_state == "stale" and owner:
            _repair_upload_task_status(task_id, pid=pid, user=owner, info=task_info)
            try:
                refreshed = UploadedPaperRepository.get(pid)
            except Exception:
                refreshed = None
            if isinstance(refreshed, dict):
                return (
                    str(refreshed.get("parse_status") or "failed").strip() or "failed",
                    str(refreshed.get("parse_error") or repaired_error).strip(),
                )
        else:
            updates = {
                "parse_status": "failed",
                "parse_error": repaired_error,
                "parse_task_id": None,
            }
            if owner:
                try:
                    UploadedPaperRepository.update(pid, updates)
                except Exception as e:
                    logger.warning(f"Failed to normalize stale parse status for {pid}: {e}")
            return "failed", repaired_error

        return "failed", repaired_error

    if parse_status:
        return parse_status, parse_error

    if _infer_upload_mineru_parsed_ok(pid):
        return "ok", ""

    # Keep UI consistent: treat unknown as pending (not ready).
    return "pending", parse_error


def _validate_meta_override_inputs(
    *,
    title: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    abstract: str | None = None,
) -> tuple[str | None, list[str] | None, int | None, str | None]:
    """Validate and normalize user-supplied meta override inputs.

    Raises:
        ValueError: for invalid types/values.
    """
    norm_title: str | None = None
    if title is not None:
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        norm_title = title.strip()
        if len(norm_title) > MAX_META_TITLE_CHARS:
            raise ValueError(f"title too long (max {MAX_META_TITLE_CHARS} chars)")

    norm_authors: list[str] | None = None
    if authors is not None:
        if not isinstance(authors, list):
            raise ValueError("authors must be a list")
        if len(authors) > MAX_META_AUTHOR_COUNT:
            raise ValueError(f"authors too many (max {MAX_META_AUTHOR_COUNT})")
        out: list[str] = []
        for a in authors:
            if a is None:
                continue
            if not isinstance(a, str):
                raise ValueError("author must be a string")
            s = a.strip()
            if not s:
                continue
            if len(s) > MAX_META_AUTHOR_CHARS:
                raise ValueError(f"author name too long (max {MAX_META_AUTHOR_CHARS} chars)")
            out.append(s)
        norm_authors = out

    norm_year: int | None = None
    if year is not None:
        if not isinstance(year, int):
            raise ValueError("year must be an integer")
        # Keep wide bounds; year is not displayed for uploads, but can be stored.
        if year < 1900 or year > 2100:
            raise ValueError("year out of range")
        norm_year = int(year)

    norm_abs: str | None = None
    if abstract is not None:
        if not isinstance(abstract, str):
            raise ValueError("abstract must be a string")
        norm_abs = abstract.strip()
        if len(norm_abs) > MAX_META_ABSTRACT_CHARS:
            raise ValueError(f"abstract too long (max {MAX_META_ABSTRACT_CHARS} chars)")

    return norm_title, norm_authors, norm_year, norm_abs


def _invalidate_upload_features(pid: str) -> None:
    """Invalidate cached upload feature file (best effort)."""
    try:
        from backend.services.upload_similarity_service import get_upload_features_path

        feat_path = get_upload_features_path(pid)
        if feat_path.exists():
            feat_path.unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass


def _extract_model_candidates() -> list[str]:
    return build_model_candidate_chain(_extract_model_name(), _llm_name())


def _normalize_extracted_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM metadata output to expected types."""
    if not isinstance(meta, dict):
        return {"title": "", "authors": [], "year": None, "abstract": None}

    title = meta.get("title")
    title = title.strip() if isinstance(title, str) else ""

    abstract = meta.get("abstract")
    if abstract is not None and not isinstance(abstract, str):
        abstract = str(abstract)
    abstract = abstract.strip() if isinstance(abstract, str) else None

    authors_raw = meta.get("authors")
    authors_list: list[str] = []
    if isinstance(authors_raw, (list, tuple)):
        for a in authors_raw:
            if not isinstance(a, str):
                a = str(a)
            s = a.strip()
            if s:
                authors_list.append(s)
    elif isinstance(authors_raw, str):
        s = authors_raw.strip()
        if s:
            if "," in s or ";" in s:
                parts = re.split(r"[;,]", s)
            elif " and " in s:
                parts = s.split(" and ")
            else:
                parts = [s]
            authors_list = [p.strip() for p in parts if p.strip()]

    return {
        "title": title,
        "authors": authors_list,
        "year": None,
        "abstract": abstract,
    }


# Patterns to detect Introduction section (stop extracting front matter here)
INTRO_PATTERNS = [
    r"^#{1,3}\s*(?:\d+\.?\s*)?Introduction\b",
    r"^(?:\d+\.?\s*)?Introduction\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?引言\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Background\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Related\s+Work\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Preliminaries\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Problem\s+Statement\b",
]

METADATA_EXTRACTION_PROMPT = """Extract metadata from this academic paper's front matter.

Input:
---
{content}
---

Return JSON only:
{{"title": "...", "authors": ["..."], "abstract": "..."}}

Rules:
- Use null for missing fields, [] for missing authors
- No affiliations/emails in author names
- JSON only, no explanation"""

METADATA_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "authors": {"type": "array", "items": {"type": "string"}},
        "year": {"type": ["integer", "null"]},
        "abstract": {"type": ["string", "null"]},
    },
    "required": ["title", "authors", "year", "abstract"],
    "additionalProperties": False,
}


def _redact_error_message(msg: str, max_len: int = 300) -> str:
    """Redact likely filesystem paths from user-visible error messages."""
    if not msg:
        return ""
    # Collapse whitespace/newlines to keep the UI compact.
    msg = " ".join(str(msg).split())
    try:
        msg = msg.replace(_data_dir(), "<DATA_DIR>")
    except Exception:
        pass

    # Redact deep absolute paths (avoid replacing short URL paths like /api).
    msg = re.sub(r"/(?:[^\s/]+/){2,}[^\s/]+", "<PATH>", msg)
    msg = re.sub(r"[A-Za-z]:\\\\(?:[^\s\\\\]+\\\\){2,}[^\s\\\\]+", "<PATH>", msg)
    return msg[:max_len]


def extract_front_matter(md_content: str, max_chars: int = 12000) -> str:
    """Extract front matter from markdown content (before Introduction).

    Args:
        md_content: Full markdown content
        max_chars: Maximum characters to return

    Returns:
        Front matter text
    """
    lines = md_content.split("\n")
    for i, line in enumerate(lines):
        for pattern in INTRO_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                front = "\n".join(lines[:i])
                return front[:max_chars] if len(front) > max_chars else front
    return md_content[:max_chars]


def extract_metadata_with_llm(front_matter: str) -> dict[str, Any]:
    """Extract metadata from front matter using LLM.

    Uses OpenCode structured output with the configured extract/default/fallback models.

    Args:
        front_matter: Text content before Introduction

    Returns:
        Dictionary with title, authors, year (always None), abstract
    """
    if not front_matter.strip():
        return {"title": "", "authors": [], "year": None, "abstract": None}

    prompt = METADATA_EXTRACTION_PROMPT.format(content=front_matter[:8000])

    for model_name in _extract_model_candidates():
        try:
            result = generate_structured_json(
                model=model_name,
                prompt=prompt,
                system=(
                    "Return structured JSON only. Do not call tools other than the "
                    "structured output required by the schema."
                ),
                schema=METADATA_EXTRACTION_SCHEMA,
                timeout=settings.extract_info.timeout,
            )
            meta = result.get("json")
            if isinstance(meta, dict) and (meta.get("title") or meta.get("authors")):
                resolved_model = str(result.get("resolved_model") or model_name).strip()
                logger.info(f"Successfully extracted metadata using {resolved_model}")
                return _normalize_extracted_metadata(meta)
            logger.warning(f"Structured metadata extraction returned no usable fields for model={model_name}")
        except Exception as e:
            logger.warning(f"Failed to extract metadata with {model_name}: {e}")

    return {"title": "", "authors": [], "year": None, "abstract": None}


def create_uploaded_paper(
    user: str,
    file_content: bytes,
    original_filename: str,
    max_uploads_per_user: int = 100,
) -> tuple[str, dict[str, Any], bool]:
    """Create a new uploaded paper record.

    Args:
        user: Username (owner)
        file_content: PDF file content
        original_filename: Original filename
        max_uploads_per_user: Maximum uploads allowed per user (for quota enforcement)

    Returns:
        Tuple of (pid, paper_data, is_new)

    Raises:
        QuotaExceededError: If user has reached upload limit
    """
    sha256 = compute_bytes_sha256(file_content)

    # Fast path: return existing record if present.
    existing = UploadedPaperRepository.get_by_sha256(user, sha256)
    if existing:
        pid, data = existing
        logger.info(f"Duplicate upload detected for user {user}, returning existing pid {pid}")
        return pid, data, False

    now = time.time()
    safe_name = sanitize_filename(original_filename)

    def _build_record(upload_pid: str) -> dict[str, Any]:
        return {
            "pid": upload_pid,
            "owner": user,
            "created_time": now,
            "updated_time": now,
            "original_filename": safe_name,
            "size_bytes": len(file_content),
            "sha256": sha256,
            "parse_status": "queued",
            "parse_error": None,
            "meta_extracted": {
                "title": "",
                "authors": [],
                "year": None,
                "abstract": None,
            },
            "meta_extracted_ok": False,
            "meta_override": {},
            "parse_task_id": None,
            "extract_task_id": None,
            "summary_task_id": None,
        }

    def _prepare_pdf_paths(upload_pid: str):
        upload_dir = get_upload_dir(upload_pid, _data_dir())
        upload_dir.mkdir(parents=True, exist_ok=True)
        out_path = get_upload_pdf_path(upload_pid, _data_dir())
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        return out_path, tmp_path

    # Create record + sha mapping atomically to avoid duplicate PIDs under concurrent uploads.
    # Quota check is also done inside the transaction to prevent race conditions.
    # PDF is written before DB commit; on failure we best-effort clean up both.
    from aslite.db import get_uploaded_papers_db

    sha_key = UploadedPaperRepository.sha256_mapping_key(user, sha256)
    pid: str | None = None
    paper_data: dict[str, Any] | None = None
    pdf_path = None
    tmp_pdf_path = None

    try:
        with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                from aslite import repositories as _repos

                conn = updb.conn
                decode = updb._decode
                encode = updb._encode
                _repos._ensure_kv_table(conn, "uploaded_papers_index")

                mapped_pid = updb.get(sha_key)
                if isinstance(mapped_pid, str) and mapped_pid:
                    record = UploadedPaperRepository.get(mapped_pid)
                    if isinstance(record, dict) and record.get("owner") == user and record.get("sha256") == sha256:
                        return mapped_pid, record, False
                    # Stale mapping - overwrite below.

                # Atomic quota check inside transaction
                # Count user's existing uploads by scanning the index
                indexed_pids = _repos._kv_get(conn, "uploaded_papers_index", user, decode) or []
                # Verify each indexed PID actually exists and belongs to user
                valid_count = 0
                for indexed_pid in indexed_pids:
                    rec = updb.get(indexed_pid)
                    if isinstance(rec, dict) and rec.get("owner") == user:
                        valid_count += 1

                if valid_count >= max_uploads_per_user:
                    raise QuotaExceededError(f"Upload limit reached (max {max_uploads_per_user} papers)")

                # Generate a PID that doesn't collide with an existing record.
                for _ in range(5):
                    candidate = generate_upload_pid()
                    if not updb.get(candidate):
                        pid = candidate
                        break
                if not pid:
                    raise RuntimeError("Failed to generate unique upload pid")

                # Write PDF to disk before committing DB records so we don't leave orphan DB rows on IO failure.
                pdf_path, tmp_pdf_path = _prepare_pdf_paths(pid)
                with open(tmp_pdf_path, "wb") as f:
                    f.write(file_content)
                tmp_pdf_path.replace(pdf_path)
                tmp_pdf_path = None
                paper_data = _build_record(pid)

                updb[pid] = paper_data
                updb[sha_key] = pid
                if pid not in indexed_pids:
                    indexed_pids.append(pid)
                _repos._kv_set(conn, "uploaded_papers_index", user, indexed_pids, encode)

        logger.info(f"Created uploaded paper {pid} for user {user}")
        return pid, paper_data or {}, True

    except QuotaExceededError:
        raise
    except Exception:
        # Best-effort cleanup to avoid stale mapping/record.
        try:
            if pid:
                UploadedPaperRepository.delete(pid)
                UploadedPaperRepository.remove_from_index(user, pid)
            UploadedPaperRepository.remove_sha256_mapping(user, sha256, pid=pid)
        except Exception:
            logger.opt(exception=True).warning(
                f"Best-effort cleanup failed (db rows/index/sha mapping): pid={pid}, user={user}"
            )
        try:
            if tmp_pdf_path and tmp_pdf_path.exists():
                tmp_pdf_path.unlink()
        except Exception:
            logger.opt(exception=True).warning(f"Best-effort cleanup failed (tmp pdf): pid={pid}, path={tmp_pdf_path}")
        try:
            if pdf_path and pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            logger.opt(exception=True).warning(f"Best-effort cleanup failed (pdf): pid={pid}, path={pdf_path}")
        raise


class UploadServiceError(Exception):
    """Raised for expected upload service errors.

    `code` is a stable machine-readable identifier for API mapping.
    `detail` is a short user-safe message.
    """

    def __init__(self, code: str, detail: str = ""):
        self.code = str(code or "").strip() or "upload_error"
        self.detail = str(detail or "").strip() or self.code
        super().__init__(self.detail)


class QuotaExceededError(UploadServiceError):
    """Raised when user has exceeded their upload quota."""

    def __init__(self, detail: str = ""):
        super().__init__("quota_exceeded", detail or "Upload limit reached")


@dataclass(frozen=True)
class UploadEnqueueResult:
    status: Literal["queued", "already_in_progress"]
    task_id: str


def process_uploaded_pdf(
    pid: str,
    user: str,
    model: str | None = None,
    *,
    current_task_id: str | None = None,
):
    """Process an uploaded PDF: parse with MinerU and extract metadata.

    This is called by the Huey task.

    Args:
        pid: Upload PID
        user: Username
        model: LLM model for summary (optional)
    """
    t_start = time.time()
    logger.trace(f"[BLOCKING] process_uploaded_pdf: starting pid={pid}, user={user}")
    try:
        tasks = _get_tasks_module()
        task_id = str(current_task_id or "").strip()
        _ensure_upload_record_active(
            pid,
            user,
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running"},
        )

        # Update status to running
        UploadedPaperRepository.update(pid, {"parse_status": "running", "parse_error": None})
        _emit_upload_event(
            user,
            {
                "type": "upload_parse_status",
                "pid": pid,
                "status": "running",
                "error": "",
            },
        )

        # Get PDF path
        pdf_path = get_upload_pdf_path(pid, _data_dir())
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Parse with MinerU (use pid as cache_pid, keep_pdf=True to preserve uploaded file)
        summarizer = paper_summarizer.PaperSummarizer()
        t_parse = time.time()
        logger.trace(f"[BLOCKING] process_uploaded_pdf: starting MinerU parse pid={pid}, pdf={pdf_path}")
        md_path = summarizer.parse_pdf_with_mineru(pdf_path, cache_pid=pid, keep_pdf=True)
        _ensure_upload_record_active(
            pid,
            user,
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running"},
        )
        logger.trace(
            f"[BLOCKING] process_uploaded_pdf: MinerU parse completed in {time.time() - t_parse:.2f}s pid={pid}"
        )

        if not md_path or not md_path.exists():
            raise RuntimeError("MinerU parsing returned empty content")

        # Read the markdown content
        t_read = time.time()
        md_content = md_path.read_text(encoding="utf-8")
        logger.trace(
            f"[BLOCKING] process_uploaded_pdf: markdown read completed in {time.time() - t_read:.2f}s "
            f"(len={len(md_content)}) pid={pid}"
        )

        # Update parse status first (parsing succeeded)
        if not _update_upload_record_if_current(
            pid,
            user,
            updates={
                "parse_status": "ok",
                "parse_error": None,
            },
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running"},
        ):
            return
        _emit_upload_event(
            user,
            {"type": "upload_parse_status", "pid": pid, "status": "ok", "error": ""},
        )

        # Extract metadata from front matter (separate step, can fail independently)
        meta_extracted_ok = False
        _emit_upload_event(user, {"type": "upload_extract_status", "pid": pid, "status": "running"})
        try:
            front_matter = extract_front_matter(md_content)
            t_meta = time.time()
            logger.trace(f"[BLOCKING] process_uploaded_pdf: starting metadata extraction via LLM pid={pid}")
            meta_extracted = extract_metadata_with_llm(front_matter)
            logger.trace(
                f"[BLOCKING] process_uploaded_pdf: metadata extraction completed in {time.time() - t_meta:.2f}s pid={pid}"
            )
            _ensure_upload_record_active(
                pid,
                user,
                record_field="parse_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok", "running"},
            )
            # Check if we got meaningful data
            if meta_extracted.get("title") or meta_extracted.get("authors"):
                meta_extracted_ok = True
                if not _update_upload_record_if_current(
                    pid,
                    user,
                    updates={
                        "meta_extracted": meta_extracted,
                        "meta_extracted_ok": True,
                    },
                    record_field="parse_task_id",
                    task_id=task_id,
                    allowed_parse_statuses={"ok", "running"},
                ):
                    return
                authors_list = meta_extracted.get("authors") or []
                _emit_upload_event(
                    user,
                    {
                        "type": "upload_extract_status",
                        "pid": pid,
                        "status": "ok",
                        "meta_extracted_ok": True,
                        "title": meta_extracted.get("title") or "",
                        "authors": ", ".join(authors_list) if authors_list else "",
                        "abstract": meta_extracted.get("abstract") or "",
                    },
                )
            else:
                logger.warning(f"Metadata extraction returned empty for {pid}")
                _emit_upload_event_if_current(
                    pid,
                    user,
                    {"type": "upload_extract_status", "pid": pid, "status": "failed"},
                    record_field="parse_task_id",
                    task_id=task_id,
                    allowed_parse_statuses={"ok", "running"},
                )
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {pid}: {e}")
            _emit_upload_event_if_current(
                pid,
                user,
                {"type": "upload_extract_status", "pid": pid, "status": "failed"},
                record_field="parse_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok", "running"},
            )

        logger.info(f"Successfully processed uploaded paper {pid} (meta_ok={meta_extracted_ok})")

        # Trigger summary generation
        if not _is_upload_record_current(
            pid,
            user,
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"ok", "running"},
        ):
            return
        try:
            t_enqueue = time.time()
            summary_task_id = tasks.enqueue_summary_task(pid, model=model, user=user)
            logger.trace(
                f"[BLOCKING] process_uploaded_pdf: enqueue_summary_task completed in {time.time() - t_enqueue:.2f}s pid={pid}"
            )
            if summary_task_id and not _is_upload_record_current(
                pid,
                user,
                record_field="parse_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok", "running"},
            ):
                try:
                    tasks.cancel_paper_summary_tasks(pid, user=user, reason="Upload superseded during processing")
                except Exception as cancel_exc:
                    logger.debug(f"Failed to cancel late summary task for {pid}: {cancel_exc}")
                return
            if summary_task_id:
                _update_upload_record_if_current(
                    pid,
                    user,
                    updates={"summary_task_id": summary_task_id},
                    record_field="parse_task_id",
                    task_id=task_id,
                    allowed_parse_statuses={"ok", "running"},
                )
        except Exception as e:
            logger.warning(f"Failed to enqueue summary for {pid}: {e}")
            err_msg = f"Failed to enqueue summary task: {e}"
            if not _is_upload_record_current(
                pid,
                user,
                record_field="parse_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok", "running"},
            ):
                return
            try:
                _update_upload_record_if_current(
                    pid,
                    user,
                    updates={"summary_task_id": None},
                    record_field="parse_task_id",
                    task_id=task_id,
                    allowed_parse_statuses={"ok", "running"},
                )
            except Exception:
                pass
            try:
                tasks._update_summary_status_db(pid, model, "failed", err_msg, task_user=user)
            except Exception as emit_exc:
                logger.debug(f"Failed to persist upload summary enqueue failure for {pid}: {emit_exc}")
            try:
                tasks._update_readinglist_summary_status(user, pid, "failed", err_msg, model=model)
            except Exception:
                pass

        logger.trace(f"[BLOCKING] process_uploaded_pdf: completed in {time.time() - t_start:.2f}s pid={pid}")
    except UploadServiceError:
        logger.trace(f"[BLOCKING] process_uploaded_pdf: rejected after {time.time() - t_start:.2f}s pid={pid}")
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded paper {pid}: {e}")
        parse_error = _redact_error_message(f"{type(e).__name__}: {e}") or type(e).__name__
        _update_upload_record_if_current(
            pid,
            user,
            updates={
                "parse_status": "failed",
                "parse_error": parse_error,
            },
            record_field="parse_task_id",
            task_id=str(current_task_id or "").strip(),
            allowed_parse_statuses={"queued", "running", "ok"},
        )
        _emit_upload_event_if_current(
            pid,
            user,
            {
                "type": "upload_parse_status",
                "pid": pid,
                "status": "failed",
                "error": parse_error,
            },
            record_field="parse_task_id",
            task_id=str(current_task_id or "").strip(),
            allowed_parse_statuses={"queued", "running", "ok", "failed"},
        )
        logger.trace(f"[BLOCKING] process_uploaded_pdf: failed after {time.time() - t_start:.2f}s pid={pid}")
        raise


def get_uploaded_papers_list(user: str) -> list[dict[str, Any]]:
    """Get list of uploaded papers for a user in display format.

    Args:
        user: Username

    Returns:
        List of paper items for frontend display
    """
    t_start = time.time()
    from backend.services.summary_service import get_summary_render_snapshots

    papers = UploadedPaperRepository.get_by_owner(user)
    logger.trace(
        f"[BLOCKING] get_uploaded_papers_list: loaded {len(papers)} record(s) in {time.time() - t_start:.2f}s user={user}"
    )

    # Get user tags for these papers
    t_tags = time.time()
    user_tags = TagRepository.get_user_tags(user)
    user_neg_tags = NegativeTagRepository.get_user_neg_tags(user)
    logger.trace(f"[BLOCKING] get_uploaded_papers_list: loaded tag db in {time.time() - t_tags:.2f}s user={user}")

    pid_set = {str(pid or "").strip() for pid in papers.keys() if str(pid or "").strip()}
    pid_to_utags = _build_pid_tag_reverse_index(user_tags, candidate_pids=pid_set)
    pid_to_ntags = _build_pid_tag_reverse_index(user_neg_tags, candidate_pids=pid_set)
    task_snapshot = _build_upload_task_snapshot(user, papers)

    upload_rows = []
    parse_ok_pids: list[str] = []
    with _use_upload_task_snapshot(task_snapshot):
        for pid, data in papers.items():
            meta = data.get("meta_extracted", {})
            override = data.get("meta_override", {})

            # Merge meta with overrides
            title = override.get("title") or meta.get("title") or data.get("original_filename", pid)
            authors_list = override.get("authors") or meta.get("authors") or []
            # Do not expose year/time for uploaded papers.
            abstract = override.get("abstract") or meta.get("abstract")

            # Backward compatibility: older records may not have `meta_extracted_ok`.
            # Infer it from available meta fields so frontend gating stays consistent with API behavior.
            meta_extracted_ok = _infer_meta_extracted_ok(data, title, abstract, authors_list)

            utags = pid_to_utags.get(pid, [])
            ntags = pid_to_ntags.get(pid, [])

            parse_status, parse_error = _normalize_upload_parse_status(pid, data)
            parse_task_id = _get_active_upload_task_id(data, "parse_task_id")

            extract_task_state, extract_task_id, extract_task_info = _classify_or_recover_upload_task(
                data, "extract_task_id"
            )
            if extract_task_state == "stale":
                _repair_upload_task_status(extract_task_id, pid=pid, user=user, info=extract_task_info)
                extract_task_id = ""
            elif extract_task_state != "active":
                extract_task_id = ""

            summary_task_id = _get_active_upload_task_id(data, "summary_task_id") if parse_status == "ok" else ""
            if parse_status == "ok":
                parse_ok_pids.append(pid)

            upload_rows.append(
                {
                    "id": pid,
                    "kind": "upload",
                    "title": title,
                    "authors": ", ".join(authors_list) if authors_list else "",
                    "time": "",
                    "summary": abstract,
                    "utags": utags,
                    "ntags": ntags,
                    "parse_status": parse_status,
                    "parse_error": parse_error,
                    "meta_extracted_ok": meta_extracted_ok,
                    "parse_task_id": parse_task_id,
                    "extract_task_id": extract_task_id,
                    "summary_task_id": summary_task_id,
                    "created_time": data.get("created_time", 0),
                    "original_filename": data.get("original_filename", ""),
                }
            )

    summary_snapshots = get_summary_render_snapshots(parse_ok_pids) if parse_ok_pids else {}

    result = []
    for row in upload_rows:
        snapshot = summary_snapshots.get(row["id"]) or {}
        summary_status = str(snapshot.get("status") or "")
        summary_last_error = snapshot.get("last_error") or ""
        result.append(
            {
                **row,
                "tldr": str(snapshot.get("tldr") or "") if summary_status == "ok" else "",
                "summary_status": summary_status,
                "summary_last_error": summary_last_error,
            }
        )

    # Sort by created_time descending
    result.sort(key=lambda x: x.get("created_time", 0), reverse=True)
    logger.trace(
        f"[BLOCKING] get_uploaded_papers_list: completed in {time.time() - t_start:.2f}s user={user}, items={len(result)}"
    )
    return result


def update_uploaded_paper_meta(
    pid: str,
    user: str,
    title: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    abstract: str | None = None,
) -> None:
    """Update metadata override for an uploaded paper.

    Args:
        pid: Upload PID
        user: Username (must be owner)
        title: New title (optional)
        authors: New authors list (optional)
        year: New year (optional)
        abstract: New abstract (optional)

    Raises:
        UploadServiceError: if not found / not owner / invalid inputs.
    """
    record = _get_owned_upload_record(pid, user)

    # Validate inputs early.
    try:
        title, authors, year, abstract = _validate_meta_override_inputs(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
        )
    except ValueError as e:
        raise UploadServiceError("invalid_meta", str(e)) from e

    override = record.get("meta_override", {})

    if title is not None:
        override["title"] = title
    if authors is not None:
        override["authors"] = authors
    if year is not None:
        override["year"] = year
    if abstract is not None:
        override["abstract"] = abstract

    UploadedPaperRepository.update(pid, {"meta_override": override})
    _invalidate_upload_features(pid)
    return None


def delete_uploaded_paper(pid: str, user: str) -> None:
    """Delete an uploaded paper and all associated data.

    Uses two-phase delete: first delete files, then delete DB records.
    This ensures we don't leave orphaned files on disk if DB delete succeeds
    but file delete fails.

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Raises:
        UploadServiceError: for expected failures (not found, not owner, delete failed).
    """
    record = _get_owned_upload_record(pid, user, allow_deleting=True)
    original_record = dict(record)

    sha256 = record.get("sha256")

    # Mark as deleting early to prevent concurrent tasks from treating it as valid.
    try:
        UploadedPaperRepository.update(pid, {"deleting": True, "deleting_started_at": time.time()})
    except Exception:
        pass

    # Best-effort: cancel any in-flight summary tasks before deleting files.
    try:
        tasks = _get_tasks_module()
        tasks.cancel_paper_summary_tasks(pid, user=user, reason="Paper deleted")
    except Exception as e:
        logger.warning(f"Failed to cancel summary tasks for {pid}: {e}")

    cancel_result: dict[str, Any] = {
        "parse_canceled": False,
        "extract_canceled": False,
    }
    try:
        cancel_result = _cancel_active_upload_tasks(pid, user, reason="Paper deleted") or cancel_result
    except Exception as e:
        logger.warning(f"Failed to cancel upload tasks for {pid}: {e}")

    # Invalidate upload feature caches early (best effort).
    try:
        _invalidate_upload_features(pid)
    except Exception:
        pass

    # Phase 1: Delete files first (critical paths)
    # If this fails, we abort and keep DB intact so user can retry
    critical_errors = []

    upload_dir = get_upload_dir(pid, _data_dir())
    if upload_dir.exists():
        try:
            shutil.rmtree(upload_dir)
        except Exception as e:
            critical_errors.append(f"upload_dir: {e}")
            logger.error(f"Failed to delete upload directory for {pid}: {e}")

    # If critical file deletion failed, abort
    if critical_errors:
        # Revert deleting marker so user can retry operations.
        try:
            rollback_updates: dict[str, Any] = {"deleting": False}
            if "deleting_started_at" in original_record:
                rollback_updates["deleting_started_at"] = original_record.get("deleting_started_at")
            else:
                rollback_updates["deleting_started_at"] = None

            if cancel_result.get("parse_canceled"):
                rollback_updates.update(
                    {
                        "parse_status": "failed",
                        "parse_error": "delete_failed_after_cancel",
                        "parse_task_id": None,
                    }
                )
            else:
                for key in ("parse_status", "parse_error", "parse_task_id"):
                    rollback_updates[key] = original_record.get(key)

            if cancel_result.get("extract_canceled"):
                rollback_updates["extract_task_id"] = None
            else:
                rollback_updates["extract_task_id"] = original_record.get("extract_task_id")

            UploadedPaperRepository.update(pid, rollback_updates)
        except Exception:
            pass
        raise UploadServiceError("file_delete_failed", "; ".join(critical_errors))

    # Phase 2: Delete DB records (now safe since files are gone)
    try:
        UploadedPaperRepository.delete(pid)
        UploadedPaperRepository.remove_from_index(user, pid)

        # Remove sha256 mapping so future uploads can re-create a clean mapping.
        if sha256:
            try:
                UploadedPaperRepository.remove_sha256_mapping(user, str(sha256), pid=pid)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to delete DB records for {pid}: {e}")
        # Files are already deleted, so we should still try to clean up
        # but report partial failure
        raise UploadServiceError("db_delete_failed", str(e))

    # Phase 3: Clean up non-critical caches (best effort, don't fail on errors)
    # Delete MinerU cache
    mineru_dir = Path(_data_dir()) / "mineru" / pid
    if mineru_dir.exists():
        try:
            shutil.rmtree(mineru_dir)
        except Exception as e:
            logger.warning(f"Failed to delete MinerU cache for {pid}: {e}")

    # Delete HTML->Markdown cache
    html_md_dir = Path(_data_dir()) / "html_md" / pid
    if html_md_dir.exists():
        try:
            shutil.rmtree(html_md_dir)
        except Exception as e:
            logger.warning(f"Failed to delete HTML cache for {pid}: {e}")

    # Delete summary cache
    summary_root = Path(_summary_dir())
    summary_dir = summary_root / pid
    if summary_dir.exists():
        try:
            shutil.rmtree(summary_dir)
        except Exception as e:
            logger.warning(f"Failed to delete summary cache for {pid}: {e}")
    for legacy_path in (
        summary_root / f"{pid}.md",
        summary_root / f"{pid}.meta.json",
        summary_root / f".{pid}.lock",
    ):
        try:
            legacy_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Best-effort: clean up summary task/status records after delete.
    # Important: do NOT delete epoch markers here; they are used for cooperative cancellation
    # and may still be needed by in-flight tasks to stop promptly.
    try:
        from aslite.db import get_summary_status_db

        with get_summary_status_db(flag="c") as sdb:
            status_keys = [k for k, _v in sdb.items_with_prefix(f"{pid}::")]
            for k in status_keys:
                try:
                    del sdb[k]
                except Exception:
                    pass

            task_keys = []
            for k, v in sdb.items_with_prefix("task::"):
                if isinstance(v, dict) and v.get("pid") == pid:
                    task_keys.append(k)
            for k in task_keys:
                try:
                    del sdb[k]
                except Exception:
                    pass

        if status_keys or task_keys:
            logger.debug(f"Cleaned up {len(status_keys)} status and {len(task_keys)} task records for {pid}")
    except Exception as e:
        logger.warning(f"Failed to clean up summary task/status records for {pid}: {e}")

    # Clean up tags (best effort)
    try:
        from aslite.db import get_neg_tags_db, get_tags_db

        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {}) or {}
            if not isinstance(tags, dict):
                tags = {}
            changed = False
            for tag in list(tags.keys()):
                tag_pids = tags.get(tag)
                if not isinstance(tag_pids, set):
                    continue
                if pid in tag_pids:
                    tag_pids.discard(pid)
                    if not tag_pids:
                        del tags[tag]
                    changed = True
            if changed:
                tdb[user] = tags

        with get_neg_tags_db(flag="c") as ntdb:
            neg_tags = ntdb.get(user, {}) or {}
            if not isinstance(neg_tags, dict):
                neg_tags = {}
            changed = False
            for tag in list(neg_tags.keys()):
                tag_pids = neg_tags.get(tag)
                if not isinstance(tag_pids, set):
                    continue
                if pid in tag_pids:
                    tag_pids.discard(pid)
                    if not tag_pids:
                        del neg_tags[tag]
                    changed = True
            if changed:
                ntdb[user] = neg_tags
    except Exception as e:
        logger.warning(f"Failed to clean up tags for {pid}: {e}")

    # Clean up reading list (best effort)
    try:
        from aslite.repositories import ReadingListRepository

        ReadingListRepository.remove_from_reading_list(user, pid)
    except Exception as e:
        logger.warning(f"Failed to clean up reading list for {pid}: {e}")

    logger.info(f"Deleted uploaded paper {pid} for user {user}")
    try:
        _emit_upload_event(user, {"type": "upload_deleted", "pid": pid})
    except Exception:
        pass
    return None


def retry_parse_uploaded_paper(pid: str, user: str) -> UploadEnqueueResult:
    """Retry parsing for a failed uploaded paper.

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        UploadEnqueueResult with status/task_id.

    Raises:
        UploadServiceError: for expected failures.
    """
    status, existing_task_id = _prepare_upload_parse_enqueue(
        pid=pid,
        user=user,
        require_failed=True,
        reject_if_ok=False,
        set_updated_time=False,
    )
    if status == "already_in_progress":
        return UploadEnqueueResult(status="already_in_progress", task_id=existing_task_id)

    tasks = _get_tasks_module()

    try:
        task_id = _enqueue_upload_task(
            task_type="process",
            pid=pid,
            user=user,
            task_builder=lambda: tasks.process_uploaded_pdf_task.s(pid, user),
        )
    except UploadServiceError:
        try:
            UploadedPaperRepository.update(pid, {"parse_status": "failed", "parse_error": "enqueue_failed"})
        except Exception:
            pass
        raise

    return UploadEnqueueResult(status="queued", task_id=task_id)


def _prepare_upload_parse_enqueue(
    *,
    pid: str,
    user: str,
    require_failed: bool,
    reject_if_ok: bool,
    set_updated_time: bool = False,
) -> tuple[Literal["ready", "already_in_progress"], str]:
    """Validate ownership and atomically transition parse_status to queued.

    Returns (status, existing_task_id).
    - status="already_in_progress": idempotent, already queued/running
    - status="ready": transitioned to queued, caller should enqueue a task

    Raises:
        UploadServiceError: for expected failures (not found, not owner, etc.)
    """
    record = _get_owned_upload_record(pid, user)

    current_status = (record.get("parse_status") or "").strip()
    if current_status in ("queued", "running"):
        task_state, task_id, task_info = _classify_or_recover_upload_task(record, "parse_task_id")
        if task_state == "active":
            return "already_in_progress", task_id
        if task_state == "pending_registration":
            return "already_in_progress", task_id
        if task_state == "stale":
            _repair_upload_task_status(task_id, pid=pid, user=user, info=task_info)
        current_status = "failed"

    if require_failed and current_status != "failed":
        raise UploadServiceError("not_failed", "Paper is not in failed state")

    if reject_if_ok and current_status == "ok":
        raise UploadServiceError("already_parsed", "Paper already parsed")

    from aslite.db import get_uploaded_papers_db

    try:
        with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                current_record = updb.get(pid)
                if not isinstance(current_record, dict):
                    raise UploadServiceError("not_found", "Paper not found")
                if current_record.get("owner") != user:
                    raise UploadServiceError("not_owner", "Paper not found")
                if current_record.get("deleting") is True:
                    raise UploadServiceError("deleting", "Paper is being deleted")

                actual_status = (current_record.get("parse_status") or "").strip()
                if actual_status in ("queued", "running"):
                    task_state, task_id, task_info = _classify_or_recover_upload_task(current_record, "parse_task_id")
                    if task_state == "active":
                        return "already_in_progress", task_id
                    if task_state == "pending_registration":
                        return "already_in_progress", task_id
                    if task_state == "stale":
                        _repair_upload_task_status(
                            task_id,
                            pid=pid,
                            user=user,
                            info=task_info,
                            sync_record=False,
                        )
                    if task_state != "active":
                        current_record["parse_status"] = "failed"
                        current_record["parse_error"] = "stale_running_repaired"
                        current_record["parse_task_id"] = None
                        actual_status = "failed"

                if require_failed and actual_status != "failed":
                    raise UploadServiceError("not_failed", "Paper is not in failed state")

                if reject_if_ok and actual_status == "ok":
                    raise UploadServiceError("already_parsed", "Paper already parsed")

                current_record["parse_status"] = "queued"
                current_record["parse_error"] = None
                current_record["parse_task_id"] = None
                current_record["updated_time"] = time.time()
                updb[pid] = current_record
    except UploadServiceError:
        raise
    except Exception as e:
        logger.error(f"Failed to update parse status for {pid}: {e}")
        raise UploadServiceError("db_error", "Failed to update parse status") from e

    return "ready", ""


def _enqueue_upload_task(
    *,
    task_type: str,
    pid: str,
    user: str,
    task_builder,
) -> str:
    """Enqueue a Huey task and register task status.

    Returns:
        task_id (may be empty string if task id not available).

    Raises:
        UploadServiceError: when enqueue fails.
    """
    try:
        tasks = _get_tasks_module()

        task = task_builder()
        enqueue_result = tasks.huey.enqueue(task)
        task_id = register_upload_task_enqueue(
            task_type=task_type,
            pid=pid,
            user=user,
            task=task,
            enqueue_result=enqueue_result,
        )
        return task_id
    except Exception as e:
        logger.error(f"Failed to enqueue upload task (type={task_type}) for {pid}: {e}")
        raise UploadServiceError("enqueue_failed", "Failed to enqueue task") from e


def trigger_parse_only(pid: str, user: str) -> UploadEnqueueResult:
    """Trigger MinerU parsing only (without metadata extraction).

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        UploadEnqueueResult with status/task_id.

    Raises:
        UploadServiceError: for expected failures.
    """
    status, existing_task_id = _prepare_upload_parse_enqueue(
        pid=pid,
        user=user,
        require_failed=False,
        reject_if_ok=True,
        set_updated_time=False,
    )
    if status == "already_in_progress":
        return UploadEnqueueResult(status="already_in_progress", task_id=existing_task_id)

    tasks = _get_tasks_module()

    try:
        task_id = _enqueue_upload_task(
            task_type="parse",
            pid=pid,
            user=user,
            task_builder=lambda: tasks.parse_uploaded_pdf_task.s(pid, user),
        )
    except UploadServiceError:
        # Rollback status on enqueue failure
        try:
            UploadedPaperRepository.update(pid, {"parse_status": "failed", "parse_error": "enqueue_failed"})
        except Exception:
            pass
        raise

    return UploadEnqueueResult(status="queued", task_id=task_id)


def trigger_process_uploaded_paper(pid: str, user: str) -> UploadEnqueueResult:
    """Trigger the full upload processing pipeline: parse + extract + summary.

    This is intended for the UI "one-click" flow. Individual steps are still
    available via separate endpoints (parse-only / extract-only / trigger summary).

    Returns:
        UploadEnqueueResult with status/task_id.

    Raises:
        UploadServiceError: for expected failures.
    """
    status, existing_task_id = _prepare_upload_parse_enqueue(
        pid=pid,
        user=user,
        require_failed=False,
        reject_if_ok=True,
        set_updated_time=True,
    )
    if status == "already_in_progress":
        return UploadEnqueueResult(status="already_in_progress", task_id=existing_task_id)

    tasks = _get_tasks_module()

    try:
        task_id = _enqueue_upload_task(
            task_type="process",
            pid=pid,
            user=user,
            task_builder=lambda: tasks.process_uploaded_pdf_task.s(pid, user),
        )
    except UploadServiceError:
        try:
            UploadedPaperRepository.update(pid, {"parse_status": "failed", "parse_error": "enqueue_failed"})
        except Exception:
            pass
        raise

    return UploadEnqueueResult(status="queued", task_id=task_id)


def trigger_extract_info(pid: str, user: str) -> str:
    """Trigger metadata extraction only (requires parsing to be done).

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        task_id (may be empty string if task id not available).
        If an extract task is already queued/running, returns the existing task id.

    Raises:
        UploadServiceError: for expected failures.
    """
    record = _get_owned_upload_record(pid, user)

    parse_status, _parse_error = _normalize_upload_parse_status(pid, record)
    if parse_status != "ok":
        logger.warning(f"Paper {pid} not parsed yet (status: {parse_status})")
        raise UploadServiceError("not_parsed", "Paper not parsed yet")

    if record.get("meta_extracted_ok") is True:
        logger.warning(f"Paper {pid} already has extracted metadata")
        raise UploadServiceError("already_extracted", "Metadata already extracted")

    task_state, existing_task_id, task_info = _classify_upload_task(record.get("extract_task_id"))
    if task_state == "active":
        logger.info(f"Extract metadata task already active for {pid}: {existing_task_id}")
        return existing_task_id
    if task_state == "stale":
        _repair_upload_task_status(existing_task_id, pid=pid, user=user, info=task_info)

    try:
        tasks = _get_tasks_module()

        with _get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                record = updb.get(pid)
                if not isinstance(record, dict):
                    logger.warning(f"Record not found for {pid}")
                    raise UploadServiceError("not_found", "Paper not found")

                if record.get("owner") != user:
                    logger.warning(f"User {user} does not own {pid}")
                    raise UploadServiceError("not_owner", "Paper not found")
                if record.get("deleting") is True:
                    raise UploadServiceError("deleting", "Paper is being deleted")

                parse_status, _parse_error = _normalize_upload_parse_status(pid, record)
                if parse_status != "ok":
                    logger.warning(f"Paper {pid} not parsed yet (status: {parse_status})")
                    raise UploadServiceError("not_parsed", "Paper not parsed yet")

                if record.get("meta_extracted_ok") is True:
                    logger.warning(f"Paper {pid} already has extracted metadata")
                    raise UploadServiceError("already_extracted", "Metadata already extracted")

                task_state, existing_task_id, task_info = _classify_upload_task(record.get("extract_task_id"))
                if task_state == "active":
                    logger.info(f"Extract metadata task already active for {pid}: {existing_task_id}")
                    return existing_task_id
                if task_state == "stale":
                    _repair_upload_task_status(
                        existing_task_id,
                        pid=pid,
                        user=user,
                        info=task_info,
                        sync_record=False,
                    )
                    record["extract_task_id"] = None

                task = tasks.extract_info_task.s(pid, user)
                enqueue_result = tasks.huey.enqueue(task)
                task_id = _extract_huey_task_id(task, enqueue_result)
                if not task_id:
                    raise UploadServiceError("enqueue_failed", "Failed to enqueue task")

                record["extract_task_id"] = task_id
                record["updated_time"] = time.time()
                updb[pid] = record
                SummaryStatusRepository.set_task_status(
                    task_id,
                    "queued",
                    None,
                    pid=pid,
                    model=UPLOAD_TASK_MODEL_EXTRACT,
                    user=user,
                )
    except Exception as e:
        if isinstance(e, UploadServiceError):
            raise
        logger.error(f"Failed to enqueue extract info for {pid}: {e}")
        raise UploadServiceError("enqueue_failed", "Failed to enqueue task") from e

    return task_id


def do_extract_metadata(pid: str, user: str, *, current_task_id: str | None = None) -> bool:
    """Actually perform metadata extraction (called by task).

    Args:
        pid: Upload PID
        user: Username

    Returns:
        True if extraction succeeded
    """
    if not validate_upload_pid(pid):
        return False

    record = UploadedPaperRepository.get(pid)
    if not record or record.get("owner") != user:
        return False

    if record.get("deleting") is True:
        return False

    parse_status, _parse_error = _normalize_upload_parse_status(pid, record)
    if parse_status != "ok":
        return False

    task_id = str(current_task_id or "").strip()
    try:
        _ensure_upload_record_active(
            pid,
            user,
            record_field="extract_task_id",
            task_id=task_id,
            allowed_parse_statuses={"ok"},
        )
    except UploadServiceError:
        return False

    # Emit running status
    _emit_upload_event(user, {"type": "upload_extract_status", "pid": pid, "status": "running"})

    try:
        summarizer = paper_summarizer.PaperSummarizer()
        backend = summarizer._normalize_mineru_backend()
        md_path = summarizer._find_mineru_markdown(pid, backend=backend)

        if not md_path or not md_path.exists():
            logger.error(f"MinerU markdown not found for {pid}")
            _emit_upload_event_if_current(
                pid,
                user,
                {"type": "upload_extract_status", "pid": pid, "status": "failed"},
                record_field="extract_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok"},
            )
            return False

        md_content = md_path.read_text(encoding="utf-8")
        front_matter = extract_front_matter(md_content)
        meta_extracted = extract_metadata_with_llm(front_matter)
        try:
            _ensure_upload_record_active(
                pid,
                user,
                record_field="extract_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok"},
            )
        except UploadServiceError:
            return False

        if meta_extracted.get("title") or meta_extracted.get("authors"):
            if not _update_upload_record_if_current(
                pid,
                user,
                updates={
                    "meta_extracted": meta_extracted,
                    "meta_extracted_ok": True,
                },
                record_field="extract_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok"},
            ):
                return False
            logger.info(f"Successfully extracted metadata for {pid}")
            # Emit success status with extracted metadata
            authors_list = meta_extracted.get("authors") or []
            _emit_upload_event(
                user,
                {
                    "type": "upload_extract_status",
                    "pid": pid,
                    "status": "ok",
                    "meta_extracted_ok": True,
                    "title": meta_extracted.get("title") or "",
                    "authors": ", ".join(authors_list) if authors_list else "",
                    "abstract": meta_extracted.get("abstract") or "",
                },
            )
            return True
        else:
            logger.warning(f"Metadata extraction returned empty for {pid}")
            _emit_upload_event_if_current(
                pid,
                user,
                {"type": "upload_extract_status", "pid": pid, "status": "failed"},
                record_field="extract_task_id",
                task_id=task_id,
                allowed_parse_statuses={"ok"},
            )
            return False

    except Exception as e:
        logger.error(f"Failed to extract metadata for {pid}: {e}")
        _emit_upload_event_if_current(
            pid,
            user,
            {"type": "upload_extract_status", "pid": pid, "status": "failed"},
            record_field="extract_task_id",
            task_id=task_id,
            allowed_parse_statuses={"ok"},
        )
        return False


def do_parse_only(pid: str, user: str, *, current_task_id: str | None = None) -> bool:
    """Actually perform MinerU parsing only (called by task).

    Args:
        pid: Upload PID
        user: Username

    Returns:
        True if parsing succeeded
    """
    if not validate_upload_pid(pid):
        return False

    record = UploadedPaperRepository.get(pid)
    if not record or record.get("owner") != user:
        return False

    if record.get("deleting") is True:
        return False

    task_id = str(current_task_id or "").strip()
    try:
        _ensure_upload_record_active(
            pid,
            user,
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running"},
        )
    except UploadServiceError:
        return False

    UploadedPaperRepository.update(pid, {"parse_status": "running", "parse_error": None})
    # Emit running status
    _emit_upload_event(
        user,
        {"type": "upload_parse_status", "pid": pid, "status": "running", "error": ""},
    )

    try:
        pdf_path = get_upload_pdf_path(pid, _data_dir())
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        summarizer = paper_summarizer.PaperSummarizer()
        md_path = summarizer.parse_pdf_with_mineru(pdf_path, cache_pid=pid, keep_pdf=True)
        try:
            _ensure_upload_record_active(
                pid,
                user,
                record_field="parse_task_id",
                task_id=task_id,
                allowed_parse_statuses={"queued", "running"},
            )
        except UploadServiceError:
            return False

        if not md_path or not md_path.exists():
            raise RuntimeError("MinerU parsing returned empty content")

        if not _update_upload_record_if_current(
            pid,
            user,
            updates={
                "parse_status": "ok",
                "parse_error": None,
            },
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running"},
        ):
            return False
        logger.info(f"Successfully parsed uploaded paper {pid}")
        # Emit success status
        _emit_upload_event(
            user,
            {"type": "upload_parse_status", "pid": pid, "status": "ok", "error": ""},
        )
        return True

    except Exception as e:
        logger.error(f"Failed to parse uploaded paper {pid}: {e}")
        parse_error = _redact_error_message(f"{type(e).__name__}: {e}") or type(e).__name__
        _update_upload_record_if_current(
            pid,
            user,
            updates={
                "parse_status": "failed",
                "parse_error": parse_error,
            },
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running", "ok"},
        )
        # Emit failure status
        _emit_upload_event_if_current(
            pid,
            user,
            {
                "type": "upload_parse_status",
                "pid": pid,
                "status": "failed",
                "error": parse_error,
            },
            record_field="parse_task_id",
            task_id=task_id,
            allowed_parse_statuses={"queued", "running", "ok", "failed"},
        )
        return False


def get_upload_summary_context(pid: str, user: str) -> dict[str, Any] | None:
    """Get context for rendering summary page for an uploaded paper.

    Args:
        pid: Upload PID
        user: Username

    Returns:
        Context dictionary for template, or None if not found/unauthorized
    """
    record = UploadedPaperRepository.get(pid)
    if not record:
        return None

    if record.get("owner") != user:
        return None

    meta = record.get("meta_extracted", {})
    override = record.get("meta_override", {})

    title = override.get("title") or meta.get("title") or record.get("original_filename", pid)
    authors_list = override.get("authors") or meta.get("authors") or []
    abstract = override.get("abstract") or meta.get("abstract")
    meta_extracted_ok = _infer_meta_extracted_ok(record, title, abstract, authors_list)

    parse_status, parse_error = _normalize_upload_parse_status(pid, record)

    # Format upload time for display
    created_time = record.get("created_time", 0)
    if created_time:
        from datetime import datetime

        dt = datetime.fromtimestamp(created_time)
        time_str = f"Uploaded: {dt.strftime('%Y-%m-%d %H:%M')}"
    else:
        time_str = ""

    # Get user tags for this paper
    utags = []
    ntags = []
    try:
        user_tags = TagRepository.get_user_tags(user) or {}
        user_neg_tags = NegativeTagRepository.get_user_neg_tags(user) or {}
        for tag, tag_pids in user_tags.items():
            if pid in tag_pids:
                utags.append(tag)
        for tag, tag_pids in user_neg_tags.items():
            if pid in tag_pids:
                ntags.append(tag)
    except Exception as e:
        logger.warning(f"Failed to get tags for uploaded paper {pid}: {e}")

    # Build paper-like structure for template
    paper = {
        "id": pid,
        "kind": "upload",
        "title": title,
        "authors": ", ".join(authors_list) if authors_list else "",
        "time": time_str,
        "summary": abstract or "",
        "tags": "",
        "utags": utags,
        "ntags": ntags,
        "parse_status": parse_status,
        "parse_error": parse_error,
        "meta_extracted_ok": meta_extracted_ok,
        "created_time": created_time,
    }

    return {
        "paper": paper,
        "pid": pid,
    }
