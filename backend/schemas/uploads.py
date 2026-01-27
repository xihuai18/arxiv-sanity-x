"""
Pydantic schemas for uploaded papers API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class UploadedPaperMeta(BaseModel):
    """Metadata extracted from uploaded paper."""

    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None


class UploadedPaperResponse(BaseModel):
    """Response schema for uploaded paper."""

    pid: str
    owner: str
    created_time: float
    updated_time: float
    original_filename: str
    size_bytes: int
    sha256: str
    parse_status: str
    parse_error: Optional[str] = None
    meta_extracted: UploadedPaperMeta = Field(default_factory=UploadedPaperMeta)
    meta_override: dict = Field(default_factory=dict)
    summary_task_id: Optional[str] = None


class UploadPdfRequest(BaseModel):
    """Request schema for PDF upload (multipart form, not JSON)."""


class UpdateMetaRequest(BaseModel):
    """Request schema for updating paper metadata."""

    pid: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    abstract: Optional[str] = None


class DeleteUploadRequest(BaseModel):
    """Request schema for deleting uploaded paper."""

    pid: str


class RetryParseRequest(BaseModel):
    """Request schema for retrying parse."""

    pid: str


class UploadedPaperListItem(BaseModel):
    """List item for uploaded papers."""

    id: str  # pid
    kind: str = "upload"
    title: str
    authors: str  # Comma-separated for display
    time: str  # Year as string
    summary: Optional[str] = None  # Abstract
    utags: List[str] = Field(default_factory=list)
    ntags: List[str] = Field(default_factory=list)
    parse_status: str
    summary_status: Optional[str] = None
    created_time: float
    original_filename: str
