"""Paper rendering services."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

from flask import abort, send_file

from tools.paper_summarizer import split_pid_version

from .summary_service import extract_tldr_from_summary, get_summary_status

# Image MIME types for serving paper images
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}


class _ThumbCache:
    """Simple cache for thumbnail URLs."""

    def __init__(self, maxsize: int = 4096, ttl_s: float = 600.0):
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._data = {}
        self._lock = Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self._ttl_s:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str):
        with self._lock:
            if len(self._data) >= self._maxsize:
                oldest = min(self._data.items(), key=lambda x: x[1][0])
                del self._data[oldest[0]]
            self._data[key] = (time.time(), value)


THUMB_CACHE = _ThumbCache()


def get_thumb_url(pid: str) -> str:
    """Get thumbnail URL for a paper."""
    cached = THUMB_CACHE.get(pid)
    if cached is not None:
        return cached
    thumb_path = Path("static/thumb") / f"{pid}.jpg"
    thumb_url = f"static/thumb/{pid}.jpg" if thumb_path.is_file() else ""
    THUMB_CACHE.set(pid, thumb_url)
    return thumb_url


def render_pid(
    pid: str,
    pid_to_utags: dict[str, list[str]] | None = None,
    pid_to_ntags: dict[str, list[str]] | None = None,
    paper: dict[str, Any] | None = None,
    get_paper_fn=None,
    get_tags_fn=None,
    get_neg_tags_fn=None,
    *,
    include_tldr: bool = True,
    include_summary_status: bool = True,
) -> dict[str, Any]:
    """Render a single paper for the UI."""
    thumb_url = get_thumb_url(pid)
    tldr = extract_tldr_from_summary(pid) if include_tldr else ""

    d = paper if paper is not None else (get_paper_fn(pid) if get_paper_fn else None)
    if d is None:
        return dict(
            weight=0.0,
            id=pid,
            title="(missing paper)",
            time="",
            authors="",
            tags="",
            utags=pid_to_utags.get(pid, []) if pid_to_utags else [],
            ntags=pid_to_ntags.get(pid, []) if pid_to_ntags else [],
            summary="",
            tldr="",
            thumb_url=thumb_url,
            summary_status="",
            summary_last_error="",
        )

    if pid_to_utags is not None:
        utags = pid_to_utags.get(pid, [])
    elif get_tags_fn:
        tags = get_tags_fn()
        utags = [t for t, tpids in tags.items() if pid in tpids]
    else:
        utags = []

    if pid_to_ntags is not None:
        ntags = pid_to_ntags.get(pid, [])
    elif get_neg_tags_fn:
        neg_tags = get_neg_tags_fn()
        ntags = [t for t, tpids in neg_tags.items() if pid in tpids]
    else:
        ntags = []

    summary_status, summary_last_error = "", ""
    if include_summary_status:
        summary_status, summary_last_error = get_summary_status(pid)

    return dict(
        weight=0.0,
        id=d["_id"],
        title=d["title"],
        time=d["_time_str"],
        authors=", ".join(a["name"] for a in d["authors"]),
        tags=", ".join(t["term"] for t in d["tags"]),
        utags=utags,
        ntags=ntags,
        summary=d["summary"],
        tldr=tldr,
        thumb_url=thumb_url,
        summary_status=summary_status,
        summary_last_error=summary_last_error or "",
    )


def build_paper_text_fields(p: dict) -> dict:
    """Build normalized text fields for scoring."""
    from .search_service import normalize_text, normalize_text_loose

    title = p.get("title") or ""
    authors = p.get("authors") or []
    authors_str = " ".join(a.get("name", "") for a in authors if isinstance(a, dict))
    summary_text = p.get("summary") or ""
    tags = p.get("tags") or []
    tags_str = " ".join(t.get("term", "") for t in tags if isinstance(t, dict))

    return {
        "title": title,
        "title_lower": title.lower(),
        "title_norm": normalize_text(title),
        "title_norm_loose": normalize_text_loose(title),
        "authors": authors_str,
        "authors_lower": authors_str.lower(),
        "authors_norm": normalize_text(authors_str),
        "summary": summary_text,
        "summary_lower": summary_text.lower(),
        "summary_norm": normalize_text(summary_text),
        "summary_norm_loose": normalize_text_loose(summary_text),
        "tags": tags_str,
        "tags_lower": tags_str.lower(),
        "tags_norm": normalize_text(tags_str),
    }


def serve_paper_image(pid: str, filename: str, base_dir: Path, search_subdirs: list = None):
    """
    Common logic for serving paper images from cache directories.

    Args:
        pid: Paper ID (version will be stripped)
        filename: Image filename
        base_dir: Base directory (e.g., Path("data/html_md"))
        search_subdirs: Optional list of subdirectories to search (for MinerU)

    Returns:
        Flask response with image file
    """
    pid = pid.strip()
    filename = filename.strip()

    if not pid or not filename:
        abort(400, "Invalid paper ID or filename")

    # Prevent path traversal attacks
    if ".." in pid or ".." in filename or "/" in filename or "\\" in filename:
        abort(400, "Invalid path")

    raw_pid, _ = split_pid_version(pid)
    image_path = None

    if search_subdirs:
        # Search in multiple subdirectories (MinerU style)
        for subdir in search_subdirs:
            candidate = base_dir / raw_pid / subdir / "images" / filename
            if candidate.exists():
                image_path = candidate
                break
    else:
        # Direct path (HTML markdown style)
        image_path = base_dir / raw_pid / "images" / filename
        if not image_path.exists():
            image_path = None

    if not image_path:
        abort(404, f"Image not found: {filename}")

    mimetype = IMAGE_MIME_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
    return send_file(image_path, mimetype=mimetype)
