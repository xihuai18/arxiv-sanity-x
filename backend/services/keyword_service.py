"""Keyword management service.

This module handles all keyword-related operations including:
- Adding keywords
- Deleting keywords
- Renaming keywords
"""

from __future__ import annotations

from flask import g
from loguru import logger

from aslite.repositories import KeywordRepository

from ..utils.sse import emit_user_event
from ..utils.validation import validate_keyword_name


def add_keyword(keyword: str) -> str:
    """Add a keyword for the current user.

    Args:
        keyword: Keyword name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    err = validate_keyword_name(keyword)
    if err:
        return err

    # Check if keyword already exists
    keywords = KeywordRepository.get_user_keywords(g.user)
    if keyword in keywords:
        return "user has repeated keywords"

    # Add keyword using Repository
    KeywordRepository.add_keyword(g.user, keyword)

    logger.debug(f"added keyword {keyword} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "add_key", "keyword": keyword})
    return "ok"


def delete_keyword(keyword: str) -> str:
    """Delete a keyword for the current user.

    Args:
        keyword: Keyword name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not keyword:
        return "error, keyword is required"

    # Check if keyword exists
    keywords = KeywordRepository.get_user_keywords(g.user)
    if not keywords:
        return "user does not have a library"
    if keyword not in keywords:
        return "user does not have this keyword"

    # Delete keyword using Repository
    KeywordRepository.remove_keyword(g.user, keyword)

    logger.debug(f"deleted keyword {keyword} for user {g.user}")
    emit_user_event(g.user, {"type": "user_state_changed", "reason": "delete_key", "keyword": keyword})
    return "ok"


def rename_keyword(old_keyword: str, new_keyword: str) -> str:
    """Rename a keyword for the current user.

    Args:
        old_keyword: Current keyword name (already normalized)
        new_keyword: New keyword name (already normalized)

    Returns:
        "ok" on success, error message otherwise
    """
    if g.user is None:
        return "error, not logged in"

    if not old_keyword or not new_keyword:
        return "error, keyword is required"

    if new_keyword == "null":
        return "error, cannot add the protected keyword 'null'"

    if old_keyword == new_keyword:
        return "ok"

    # Rename keyword using Repository
    result = KeywordRepository.rename_keyword(g.user, old_keyword, new_keyword)
    if result != "ok":
        return result

    logger.debug(f"renamed keyword {old_keyword} to {new_keyword} for user {g.user}")
    emit_user_event(
        g.user, {"type": "user_state_changed", "reason": "rename_key", "from": old_keyword, "to": new_keyword}
    )
    return "ok"
