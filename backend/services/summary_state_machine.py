"""Shared summary state transition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTIVE_SUMMARY_STATUSES = frozenset({"queued", "running"})


@dataclass(frozen=True)
class SummaryStatusTransition:
    """Normalized summary status write contract."""

    pid: str
    model: str
    status: str
    error: str | None = None
    task_id: str | None = None
    task_user: str | None = None
    resolved_model: str | None = None

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_SUMMARY_STATUSES

    def repository_extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.task_id is not None:
            extra["task_id"] = self.task_id
        if self.task_user is not None:
            extra["task_user"] = self.task_user
        if self.status == "ok":
            if self.resolved_model:
                extra["resolved_model"] = self.resolved_model
        else:
            extra["resolved_model"] = None
        if not self.is_active:
            extra["task_id"] = None
            extra["task_user"] = None
        return extra

    def public_event_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "summary_status",
            "pid": self.pid,
            "model": self.model,
            "status": self.status,
            "error": "failed" if self.error else None,
        }
        if self.resolved_model:
            payload["resolved_model"] = self.resolved_model
        return payload

    def private_event_payload(self) -> dict[str, Any]:
        payload = self.public_event_payload()
        payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ReadingListSummaryTransition:
    """Normalized reading-list summary write contract."""

    user: str
    pid: str
    model: str | None
    status: str
    error: str | None = None
    task_id: str | None = None
    updated_time: float | None = None
    should_persist: bool = True

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_SUMMARY_STATUSES

    def persisted_updates(self) -> dict[str, Any] | None:
        if not self.should_persist:
            return None
        updates: dict[str, Any] = {
            "summary_status": self.status,
            "summary_last_error": self.error,
            "summary_updated_time": self.updated_time,
        }
        if not self.is_active:
            updates["summary_task_id"] = None
        elif self.task_id is not None:
            updates["summary_task_id"] = self.task_id
        return updates

    def event_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "summary_status",
            "pid": self.pid,
            "model": self.model or None,
            "status": self.status,
            "error": self.error,
        }
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        return payload


def build_summary_status_transition(
    pid: str,
    model: str | None,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
    task_user: str | None = None,
    resolved_model: str | None = None,
    *,
    default_model: str | None = None,
) -> SummaryStatusTransition | None:
    """Build a normalized summary status transition."""

    normalized_model = str(model or default_model or "").strip()
    if not normalized_model:
        return None

    normalized_resolved_model = None
    if status == "ok":
        normalized_resolved_model = str(resolved_model or "").strip() or None

    normalized_task_id = None if task_id is None else str(task_id)
    normalized_task_user = None if task_user is None else str(task_user)

    return SummaryStatusTransition(
        pid=str(pid or "").strip(),
        model=normalized_model,
        status=str(status or "").strip(),
        error=error,
        task_id=normalized_task_id,
        task_user=normalized_task_user,
        resolved_model=normalized_resolved_model,
    )


def build_readinglist_summary_transition(
    user: str,
    pid: str,
    status: str,
    error: str | None = None,
    task_id: str | None = None,
    model: str | None = None,
    *,
    default_model: str | None = None,
    updated_time: float | None = None,
) -> ReadingListSummaryTransition | None:
    """Build a normalized reading-list summary transition."""

    normalized_user = str(user or "").strip()
    if not normalized_user:
        return None

    normalized_default_model = str(default_model or "").strip()
    normalized_model = str(model or normalized_default_model or "").strip() or None
    should_persist = (
        not normalized_model
        or not normalized_default_model
        or normalized_model == normalized_default_model
    )

    normalized_task_id = None if task_id is None else str(task_id)

    return ReadingListSummaryTransition(
        user=normalized_user,
        pid=str(pid or "").strip(),
        model=normalized_model,
        status=str(status or "").strip(),
        error=error,
        task_id=normalized_task_id,
        updated_time=updated_time,
        should_persist=should_persist,
    )
