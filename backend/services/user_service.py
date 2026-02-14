"""User-related services: tags, keywords, authentication."""

from __future__ import annotations

from contextlib import contextmanager

from flask import g, session

from aslite.repositories import KeywordRepository, NegativeTagRepository, TagRepository


def get_tags():
    """Get user tags with request-level caching."""
    if g.user is None:
        return {}
    if not hasattr(g, "_tags"):
        g._tags = TagRepository.get_user_tags(g.user)
    return g._tags


def get_neg_tags():
    """Get user negative tags with request-level caching."""
    if g.user is None:
        return {}
    if not hasattr(g, "_neg_tags"):
        g._neg_tags = TagRepository.get_user_neg_tags(g.user)
    return g._neg_tags


def get_combined_tags():
    """Get user combined tags with request-level caching."""
    if g.user is None:
        return {}
    if not hasattr(g, "_combined_tags"):
        g._combined_tags = TagRepository.get_user_combined_tags(g.user)
    return g._combined_tags


def get_keys():
    """Get user keywords with request-level caching."""
    if g.user is None:
        return {}
    if not hasattr(g, "_keys"):
        g._keys = KeywordRepository.get_user_keywords(g.user)
    return g._keys


def build_user_tag_list():
    """Build tag list for frontend."""
    tags = get_tags()
    neg_tags = get_neg_tags()
    rtags = []
    for t in set(tags.keys()) | set(neg_tags.keys()):
        pos_n = len(tags.get(t, set()))
        neg_n = len(neg_tags.get(t, set()))
        rtags.append(
            {
                "name": t,
                "n": pos_n + neg_n,
                "pos_n": pos_n,
                "neg_n": neg_n,
                "neg_only": pos_n == 0 and neg_n > 0,
            }
        )
    if rtags:
        rtags.append({"name": "all", "n": 0, "pos_n": 0, "neg_n": 0, "neg_only": False})
    return rtags


def build_user_key_list():
    """Build keyword list for frontend."""
    keys = get_keys()
    return [{"name": k, "n": len(pids)} for k, pids in keys.items()]


def build_user_combined_tag_list():
    """Build combined tag list for frontend."""
    combined_tags = get_combined_tags()
    # combined_tags is a Set[str] when user is logged in, or {} when not
    # Handle both cases by iterating directly (sets are iterable, empty dict iterates over keys)
    return [{"name": ct} for ct in combined_tags]


# Import validation functions from utils.validation for consistency


def before_request():
    """Set up request context."""
    from .background import ensure_background_services_started

    ensure_background_services_started()
    g.user = session.get("user", None)


def close_connection(_error=None):
    """Clean up request context."""
    return None


@contextmanager
def temporary_user_context(user):
    """Context manager to temporarily set g.user and g._tags for API calls."""
    original_user = getattr(g, "user", None)
    original_tags = getattr(g, "_tags", None)
    original_neg_tags = getattr(g, "_neg_tags", None)

    try:
        # Get user tags
        user_tags = TagRepository.get_user_tags(user)
        user_neg_tags = NegativeTagRepository.get_user_neg_tags(user)

        # Set temporary context
        g.user = user
        g._tags = user_tags
        g._neg_tags = user_neg_tags

        yield user_tags

    finally:
        # Restore original context
        if original_user is not None:
            g.user = original_user
        else:
            if hasattr(g, "user"):
                delattr(g, "user")
        if original_tags is not None:
            g._tags = original_tags
        else:
            if hasattr(g, "_tags"):
                delattr(g, "_tags")
        if original_neg_tags is not None:
            g._neg_tags = original_neg_tags
        else:
            if hasattr(g, "_neg_tags"):
                delattr(g, "_neg_tags")
