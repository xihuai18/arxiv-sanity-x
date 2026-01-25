"""Pydantic schemas for reading list APIs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReadingListBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class ReadingListPidRequest(ReadingListBaseModel):
    pid: str
