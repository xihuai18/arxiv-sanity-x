"""Pydantic schemas for tag and keyword APIs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class TagBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class TagFeedbackRequest(TagBaseModel):
    pid: str
    tag: str
    label: int  # 1 for positive, -1 for negative, 0 for remove

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError("label must be -1, 0, or 1")
        return v


class PaperTitlesRequest(TagBaseModel):
    pids: list[str]
