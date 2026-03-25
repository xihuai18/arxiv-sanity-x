"""User-related services: tags, keywords, authentication."""

from __future__ import annotations

from contextlib import contextmanager

from flask import g, has_app_context, session

from aslite.repositories import KeywordRepository, NegativeTagRepository, TagRepository

from .user_context import resolve_user


def _request_user() -> str | None:
    if not has_app_context():
        return None
    return getattr(g, "user", None)


def _should_use_request_cache(user: str | None) -> bool:
    request_user = _request_user()
    return bool(user and request_user and user == request_user)


def invalidate_user_state_cache(user: str | None = None) -> None:
    """Clear request-local cached user state after successful mutations."""

    if not has_app_context():
        return

    request_user = _request_user()
    target_user = resolve_user(user)
    if request_user is None:
        return
    if target_user is not None and request_user != target_user:
        return

    for attr in ("_tags", "_neg_tags", "_keys", "_combined_tags"):
        if hasattr(g, attr):
            delattr(g, attr)


def get_tags(user: str | None = None):
    """Get user tags with request-level caching."""
    resolved_user = resolve_user(user)
    if resolved_user is None:
        return {}
    if _should_use_request_cache(resolved_user):
        if not hasattr(g, "_tags"):
            g._tags = TagRepository.get_user_tags(resolved_user)
        return g._tags
    return TagRepository.get_user_tags(resolved_user)


def get_neg_tags(user: str | None = None):
    """Get user negative tags with request-level caching."""
    resolved_user = resolve_user(user)
    if resolved_user is None:
        return {}
    if _should_use_request_cache(resolved_user):
        if not hasattr(g, "_neg_tags"):
            g._neg_tags = TagRepository.get_user_neg_tags(resolved_user)
        return g._neg_tags
    return TagRepository.get_user_neg_tags(resolved_user)


def get_combined_tags(user: str | None = None):
    """Get user combined tags with request-level caching."""
    resolved_user = resolve_user(user)
    if resolved_user is None:
        return {}
    if _should_use_request_cache(resolved_user):
        if not hasattr(g, "_combined_tags"):
            g._combined_tags = TagRepository.get_user_combined_tags(resolved_user)
        return g._combined_tags
    return TagRepository.get_user_combined_tags(resolved_user)


def get_keys(user: str | None = None):
    """Get user keywords with request-level caching."""
    resolved_user = resolve_user(user)
    if resolved_user is None:
        return {}
    if _should_use_request_cache(resolved_user):
        if not hasattr(g, "_keys"):
            g._keys = KeywordRepository.get_user_keywords(resolved_user)
        return g._keys
    return KeywordRepository.get_user_keywords(resolved_user)


def build_user_tag_list(user: str | None = None):
    """Build tag list for frontend."""
    tags = get_tags(user=user)
    neg_tags = get_neg_tags(user=user)
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


def build_pid_tag_reverse_index(tag_map, *, candidate_pids=None):
    """Build a pid -> [tags] reverse index for a target pid subset."""
    pid_to_tags = {}
    pid_filter = None
    if candidate_pids is not None:
        pid_filter = {
            str(pid or "").strip() for pid in candidate_pids if str(pid or "").strip()
        }

    for tag, tag_pids in (tag_map or {}).items():
        normalized_tag = str(tag or "").strip()
        if not normalized_tag:
            continue
        iterable = tag_pids or []
        for raw_pid in iterable:
            normalized_pid = str(raw_pid or "").strip()
            if not normalized_pid:
                continue
            if pid_filter is not None and normalized_pid not in pid_filter:
                continue
            pid_to_tags.setdefault(normalized_pid, []).append(normalized_tag)

    if pid_filter is not None:
        for pid in pid_filter:
            pid_to_tags.setdefault(pid, [])

    return pid_to_tags


def build_user_key_list(user: str | None = None):
    """Build keyword list for frontend."""
    keys = get_keys(user=user)
    return [{"name": k, "n": len(pids)} for k, pids in keys.items()]


def build_user_combined_tag_list(user: str | None = None):
    """Build combined tag list for frontend."""
    combined_tags = get_combined_tags(user=user)
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
    """Compatibility no-op teardown hook."""
    return None


@contextmanager
def temporary_user_context(user):
    """Context manager to temporarily set g.user and g._tags for API calls."""
    original_user = getattr(g, "user", None)
    original_tags = getattr(g, "_tags", None)
    original_neg_tags = getattr(g, "_neg_tags", None)
    original_keys = getattr(g, "_keys", None)
    original_combined_tags = getattr(g, "_combined_tags", None)

    try:
        # Get user tags
        user_tags = TagRepository.get_user_tags(user)
        user_neg_tags = NegativeTagRepository.get_user_neg_tags(user)
        user_keys = KeywordRepository.get_user_keywords(user)
        user_combined_tags = TagRepository.get_user_combined_tags(user)

        # Set temporary context
        g.user = user
        g._tags = user_tags
        g._neg_tags = user_neg_tags
        g._keys = user_keys
        g._combined_tags = user_combined_tags

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
        if original_keys is not None:
            g._keys = original_keys
        else:
            if hasattr(g, "_keys"):
                delattr(g, "_keys")
        if original_combined_tags is not None:
            g._combined_tags = original_combined_tags
        else:
            if hasattr(g, "_combined_tags"):
                delattr(g, "_combined_tags")
