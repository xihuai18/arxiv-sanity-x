"""Helpers for passing user context without hard-coding Flask globals."""

from __future__ import annotations

from dataclasses import dataclass

from flask import g, has_app_context, has_request_context, session


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class UserContext:
    """Minimal request-derived user context for service calls."""

    user: str | None = None
    csrf_token: str | None = None
    is_machine_auth: bool = False

    @classmethod
    def from_request(cls) -> "UserContext":
        if not has_app_context():
            return cls()
        try:
            user = _normalize_optional_string(getattr(g, "user", None))
        except Exception:
            user = None
        if has_request_context():
            try:
                csrf_token = _normalize_optional_string(session.get("_csrf_token"))
            except Exception:
                csrf_token = None
        else:
            csrf_token = None
        return cls(user=user, csrf_token=csrf_token)


def resolve_user(
    user: str | None = None, *, context: UserContext | None = None
) -> str | None:
    """Resolve an explicit user first, then fall back to request context."""

    normalized_user = _normalize_optional_string(user)
    if normalized_user is not None:
        return normalized_user
    if context is not None:
        return _normalize_optional_string(context.user)
    return UserContext.from_request().user
