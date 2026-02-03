"""Pydantic schemas for summary-related APIs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SummaryBaseModel(BaseModel):
    """Base model that ignores extra fields (like csrf_token) and strips whitespace."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    # csrf_token is handled separately by _csrf_protect(), not validated here


class SummaryPidRequest(SummaryBaseModel):
    pid: str


class SummaryGetRequest(SummaryPidRequest):
    model: str | None = None
    force: bool = False
    force_regenerate: bool = False
    cache_only: bool = False


class SummaryTriggerRequest(SummaryPidRequest):
    model: str | None = None
    priority: int | str | None = None
    # Backward compatibility: some legacy clients use `force` instead of `force_regenerate`.
    force: bool = False
    force_regenerate: bool = False


class SummaryStatusRequest(SummaryBaseModel):
    pid: str | None = None
    pids: list[str] | str | None = None
    model: str | None = None


class SummaryClearModelRequest(SummaryPidRequest):
    model: str  # Required for this endpoint
