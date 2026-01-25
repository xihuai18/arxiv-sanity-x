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

    result = TagRepository.delete_tag_full(g.user, tag)
    if result != "ok":
        return result

    logger.debug(f"deleted tag {tag} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "delete_tag", "tag": tag})
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

    result = TagRepository.rename_tag_full(g.user, old_tag, new_tag)
    if result != "ok":
        return result

    logger.debug(f"renamed tag {old_tag} to {new_tag} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "rename_tag", "from": old_tag, "to": new_tag})
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

    # Validate that all component tags exist
    tags = get_tags()
    for tag in map(str.strip, ctag.split(",")):
        if tag not in tags:
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

    # Validate that all component tags exist
    tags = get_tags()
    for tag in map(str.strip, new_ctag.split(",")):
        if tag not in tags:
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

    # Sort by time desc (fixed order)
    pairs.sort(key=lambda x: (mdb.get(x[0]) or {}).get("_time", 0), reverse=True)

    # If search query, filter and need paper details
    pid_to_paper = {}
    if search:
        pid_to_paper = get_papers_bulk_fn(all_pids) if get_papers_bulk_fn and all_pids else {}
        filtered = []
        for pid, lab in pairs:
            p = pid_to_paper.get(pid)
            if not p:
                if search in pid.lower():
                    filtered.append((pid, lab))
                continue
            title = (p.get("title") or "").lower()
            authors = " ".join(a.get("name", "") for a in (p.get("authors") or []) if a).lower()
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

    items = []
    for pid, lab in page_pairs:
        p = pid_to_paper.get(pid)
        if not p:
            items.append({"pid": pid, "title": "(missing paper)", "time": "", "authors": "", "label": lab})
            continue
        items.append(
            {
                "pid": pid,
                "title": p.get("title", ""),
                "time": p.get("_time_str", ""),
                "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or []) if a),
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
