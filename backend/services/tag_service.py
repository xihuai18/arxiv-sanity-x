"""Tag management service.

This module handles all tag-related operations including:
- Regular tags (positive/negative)
- Combined tags
- Tag feedback
- Tag member listing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from flask import g
from loguru import logger

from aslite.repositories import CombinedTagRepository, TagRepository

from ..utils.sse import emit_user_event
from ..utils.validation import validate_tag_name
from .user_service import get_neg_tags, get_tags

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Regular Tag Operations
# -----------------------------------------------------------------------------


def create_empty_tag(tag: str) -> str:
    """Create an empty tag for the current user.

    Args:
        tag: Tag name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    err = validate_tag_name(tag)
    if err:
        return err

    result = TagRepository.create_tag(g.user, tag)
    if result != "ok":
        return result

    logger.debug(f"added empty tag {tag} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "add_tag", "tag": tag})
    return "ok"


def add_paper_to_tag(pid: str, tag: str) -> str:
    """Add a paper to a tag.

    Args:
        pid: Paper ID (already normalized)
        tag: Tag name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not pid:
        return "error, pid is required"

    err = validate_tag_name(tag)
    if err:
        return err

    try:
        TagRepository.add_paper_to_tag_and_remove_neg(g.user, pid, tag)
        logger.debug(f"added paper {pid} to tag {tag} for user {g.user}")
        emit_user_event(g.user, {"type": "user_state_changed", "reason": "add", "tag": tag, "pid": pid})
        return "ok"
    except Exception as e:
        logger.error(f"Failed to add paper {pid} to tag {tag}: {e}")
        return f"error: {e}"


def remove_paper_from_tag(pid: str, tag: str) -> str:
    """Remove a paper from a tag.

    Args:
        pid: Paper ID (already normalized)
        tag: Tag name (already normalized)

    Returns:
        Success message or error message
    """
    if g.user is None:
        return "error, not logged in"

    if not pid or not tag:
        return "error, pid and tag are required"

    try:
        result = TagRepository.remove_paper_from_tag_verbose(g.user, pid, tag)
        if result == "ok":
            emit_user_event(g.user, {"type": "user_state_changed", "reason": "sub", "tag": tag, "pid": pid})
            return f"ok removed pid {pid} from tag {tag}"
        return result
    except Exception as e:
        logger.error(f"Failed to remove paper {pid} from tag {tag}: {e}")
        return f"error: {e}"


def delete_tag(tag: str) -> str:
    """Delete a tag completely.

    Args:
        tag: Tag name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not tag:
        return "error, tag is required"

    # Capture affected combined tags before deletion (repository may delete them as cascade).
    deleted_ctags: list[str] = []
    try:
        combined = CombinedTagRepository.get_user_combined_tags(g.user) or set()
        deleted_ctags = sorted([ct for ct in combined if tag in map(str.strip, (ct or "").split(","))])
    except Exception:
        deleted_ctags = []

    result = TagRepository.delete_tag_full(g.user, tag)
    if result != "ok":
        return result

    logger.debug(f"deleted tag {tag} for user {g.user}")
    emit_user_event(
        g.user,
        {
            "type": "user_state_changed",
            "reason": "delete_tag",
            "tag": tag,
            "deleted_ctags": deleted_ctags,
        },
    )
    return "ok"


def rename_tag(old_tag: str, new_tag: str) -> str:
    """Rename a tag.

    Args:
        old_tag: Current tag name (already normalized)
        new_tag: New tag name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not old_tag or not new_tag:
        return "error, tag is required"

    err = validate_tag_name(new_tag)
    if err:
        return err

    # Capture affected combined tags so frontend can update caches.
    renamed_ctags: list[dict] = []
    try:
        combined = CombinedTagRepository.get_user_combined_tags(g.user) or set()
        for ct in sorted(combined):
            parts = [p.strip() for p in (ct or "").split(",")]
            if old_tag in parts:
                new_parts = [new_tag if p == old_tag else p for p in parts]
                renamed_ctags.append({"from": ct, "to": ",".join(new_parts)})
    except Exception:
        renamed_ctags = []

    result = TagRepository.rename_tag_full(g.user, old_tag, new_tag)
    if result != "ok":
        return result

    logger.debug(f"renamed tag {old_tag} to {new_tag} for user {g.user}")
    emit_user_event(
        g.user,
        {
            "type": "user_state_changed",
            "reason": "rename_tag",
            "from": old_tag,
            "to": new_tag,
            "renamed_ctags": renamed_ctags,
        },
    )
    return "ok"


# -----------------------------------------------------------------------------
# Combined Tag Operations
# -----------------------------------------------------------------------------


def create_combined_tag(ctag: str) -> str:
    """Create a combined tag.

    Args:
        ctag: Combined tag name (comma-separated tags, already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not ctag:
        return "error, ctag is required"

    if ctag == "null":
        return "error, cannot add the ctag 'null'"

    # Validate that all component tags exist (positive or negative tags are both allowed).
    pos_tags = get_tags() or {}
    neg_tags = get_neg_tags() or {}
    all_tags = set(pos_tags.keys()) | set(neg_tags.keys())
    for tag in map(str.strip, ctag.split(",")):
        if not tag:
            return "invalid ctag"
        err = validate_tag_name(tag)
        if err:
            return err
        if tag not in all_tags:
            return "invalid ctag"

    # Check if user already has this combined tag
    if CombinedTagRepository.has_combined_tag(g.user, ctag):
        return "user has repeated ctag"

    # Add the combined tag
    CombinedTagRepository.add_combined_tag(g.user, ctag)

    logger.debug(f"added ctag {ctag} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "add_ctag", "ctag": ctag})
    return "ok"


def delete_combined_tag(ctag: str) -> str:
    """Delete a combined tag.

    Args:
        ctag: Combined tag name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not ctag:
        return "error, ctag is required"

    # Check if user has this combined tag
    if not CombinedTagRepository.has_combined_tag(g.user, ctag):
        return "user does not have this ctag"

    # Delete the tag
    CombinedTagRepository.remove_combined_tag(g.user, ctag)

    logger.debug(f"deleted ctag {ctag} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "delete_ctag", "ctag": ctag})
    return "ok"


def rename_combined_tag(old_ctag: str, new_ctag: str) -> str:
    """Rename a combined tag.

    Args:
        old_ctag: Current combined tag name
        new_ctag: New combined tag name

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    old_ctag = (old_ctag or "").strip()
    new_ctag = (new_ctag or "").strip()

    if not old_ctag or not new_ctag:
        return "error, tag is required"

    if new_ctag == "null":
        return "error, cannot add the ctag 'null'"

    if old_ctag == new_ctag:
        return "ok"

    # Validate that all component tags exist (positive or negative tags are both allowed).
    pos_tags = get_tags() or {}
    neg_tags = get_neg_tags() or {}
    all_tags = set(pos_tags.keys()) | set(neg_tags.keys())
    for tag in map(str.strip, new_ctag.split(",")):
        if not tag:
            return "invalid ctag"
        err = validate_tag_name(tag)
        if err:
            return err
        if tag not in all_tags:
            return "invalid ctag"

    # Check if old tag exists and new tag doesn't
    if not CombinedTagRepository.has_combined_tag(g.user, old_ctag):
        return "user does not have this ctag"

    if CombinedTagRepository.has_combined_tag(g.user, new_ctag):
        return "user has repeated ctag"

    # Rename the tag
    CombinedTagRepository.rename_combined_tag(g.user, old_ctag, new_ctag)

    logger.debug(f"renamed ctag {old_ctag} to {new_ctag} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "rename_ctag", "from": old_ctag, "to": new_ctag})
    return "ok"


# -----------------------------------------------------------------------------
# Tag Feedback and Query Operations
# -----------------------------------------------------------------------------


def set_tag_feedback(pid: str, tag: str, label: int) -> None:
    """Set tag feedback (positive/negative/remove) for a paper.

    Args:
        pid: Paper ID
        tag: Tag name
        label: 1 for positive, -1 for negative, 0 to remove

    Raises:
        Exception: If operation fails
    """
    if g.user is None:
        raise ValueError("not logged in")

    err = validate_tag_name(tag)
    if err:
        raise ValueError(err)

    TagRepository.set_tag_label(g.user, pid, tag, label)
    emit_user_event(
        g.user,
        {"type": "user_state_changed", "reason": "tag_feedback", "pid": pid, "tag": tag, "label": label},
    )


def get_tag_members(
    tag: str,
    label: str = "all",
    search: str = "",
    page_number: int = 1,
    page_size: int = 20,
    get_metas_fn: Callable | None = None,
    get_papers_bulk_fn: Callable | None = None,
) -> dict:
    """Get papers under a tag with pagination and search.

    Args:
        tag: Tag name
        label: Filter by label ("all", "pos", "neg")
        search: Search query for title/authors/pid
        page_number: 1-based page index
        page_size: Items per page (max 200)
        get_metas_fn: Function to get paper metadata
        get_papers_bulk_fn: Function to get papers in bulk

    Returns:
        Dict with tag, label, pagination info, and items
    """
    pos_d = get_tags() or {}
    neg_d = get_neg_tags() or {}

    pos_set = set(pos_d.get(tag, set()))
    neg_set = set(neg_d.get(tag, set()))
    pos_total = len(pos_set)
    neg_total = len(neg_set)

    if label == "pos":
        pairs = [(pid, 1) for pid in pos_set]
    elif label == "neg":
        pairs = [(pid, -1) for pid in neg_set]
    else:
        pairs = [(pid, 1) for pid in pos_set] + [(pid, -1) for pid in (neg_set - pos_set)]

    # Pre-fetch all paper info for sorting and searching
    all_pids = [pid for pid, _ in pairs]
    mdb = get_metas_fn() if get_metas_fn else {}
    upload_records = {}
    upload_time = {}
    if all_pids:
        upload_pids = [pid for pid in all_pids if pid.startswith("up_")]
        if upload_pids:
            from flask import g

            from aslite.repositories import UploadedPaperRepository

            if g.user:
                all_uploads = UploadedPaperRepository.get_by_owner(g.user) or {}
                for pid in upload_pids:
                    record = all_uploads.get(pid)
                    if record:
                        upload_records[pid] = record
                        upload_time[pid] = float(record.get("created_time") or 0.0)

    def _format_upload_time(record: dict) -> str:
        created_time = record.get("created_time", 0)
        if created_time:
            from datetime import datetime

            dt = datetime.fromtimestamp(created_time)
            return f"Uploaded: {dt.strftime('%Y-%m-%d %H:%M')}"
        return ""

    def _build_upload_paper(record: dict, pid: str) -> dict:
        meta = record.get("meta_extracted", {})
        override = record.get("meta_override", {})
        title = override.get("title") or meta.get("title") or record.get("original_filename", pid)
        authors_list = override.get("authors") or meta.get("authors") or []
        return {
            "title": title,
            "authors": authors_list,
            "_time": float(record.get("created_time") or 0.0),
            "_time_str": _format_upload_time(record),
            "kind": "upload",
        }

    # Sort by time desc (fixed order)
    def _sort_time(pid: str) -> float:
        if pid in upload_time:
            return upload_time.get(pid, 0.0)
        return float((mdb.get(pid) or {}).get("_time", 0.0))

    pairs.sort(key=lambda x: _sort_time(x[0]), reverse=True)

    # If search query, filter and need paper details
    pid_to_paper = {}
    if search:
        pid_to_paper = get_papers_bulk_fn(all_pids) if get_papers_bulk_fn and all_pids else {}
        if upload_records:
            for pid, record in upload_records.items():
                pid_to_paper[pid] = _build_upload_paper(record, pid)
        filtered = []
        for pid, lab in pairs:
            p = pid_to_paper.get(pid)
            if not p:
                if search in pid.lower():
                    filtered.append((pid, lab))
                continue
            title = (p.get("title") or "").lower()
            authors_val = p.get("authors") or []
            if isinstance(authors_val, list):
                if authors_val and isinstance(authors_val[0], dict):
                    authors = " ".join(a.get("name", "") for a in authors_val if a).lower()
                else:
                    authors = " ".join(str(a) for a in authors_val if a).lower()
            else:
                authors = str(authors_val).lower()
            if search in pid.lower() or search in title or search in authors:
                filtered.append((pid, lab))
        pairs = filtered

    total_count = len(pairs)
    start = (page_number - 1) * page_size
    end = min(start + page_size, total_count)
    page_pairs = pairs[start:end]

    # Fetch paper details for current page only (if not already fetched)
    pids = [pid for pid, _lab in page_pairs]
    if not search:
        pid_to_paper = get_papers_bulk_fn(pids) if get_papers_bulk_fn and pids else {}
        if upload_records:
            for pid, record in upload_records.items():
                if pid in pids:
                    pid_to_paper[pid] = _build_upload_paper(record, pid)

    items = []
    for pid, lab in page_pairs:
        p = pid_to_paper.get(pid)
        if not p:
            items.append({"pid": pid, "title": "(missing paper)", "time": "", "authors": "", "label": lab})
            continue
        authors_val = p.get("authors") or []
        if isinstance(authors_val, list):
            if authors_val and isinstance(authors_val[0], dict):
                authors_text = ", ".join(a.get("name", "") for a in authors_val if a)
            else:
                authors_text = ", ".join(str(a) for a in authors_val if a)
        else:
            authors_text = str(authors_val)
        items.append(
            {
                "pid": pid,
                "title": p.get("title", ""),
                "time": p.get("_time_str", ""),
                "authors": authors_text,
                "label": lab,
            }
        )

    return {
        "tag": tag,
        "label": label,
        "page_number": page_number,
        "page_size": page_size,
        "total_count": total_count,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "items": items,
    }


def resolve_paper_titles(
    pids: list,
    get_papers_bulk_fn: Callable | None = None,
) -> list:
    """Resolve paper titles for a list of PIDs.

    Args:
        pids: List of paper IDs (already normalized and deduplicated)
        get_papers_bulk_fn: Function to get papers in bulk

    Returns:
        List of dicts with pid, title, exists
    """
    pid_to_paper = get_papers_bulk_fn(pids) if get_papers_bulk_fn and pids else {}
    items = []
    for pid in pids:
        p = pid_to_paper.get(pid)
        title = (p.get("title") if isinstance(p, dict) else "") or ""
        items.append({"pid": pid, "title": title, "exists": bool(title)})
    return items
