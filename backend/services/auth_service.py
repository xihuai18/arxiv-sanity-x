"""Authentication service.

This module handles user authentication and session management:
- Login/logout
- Email registration
- Profile page rendering
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from flask import flash, g, session, url_for
from loguru import logger

from aslite.repositories import UserRepository

# Backward-compatible import for tests/patching (CSRF enforcement is done at API layer).
from ..utils.validation import csrf_protect  # noqa: F401

if TYPE_CHECKING:
    pass


_PROPER_EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}$", re.IGNORECASE)


def _parse_emails(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,\s;]+", text or "") if p.strip()]
    out: list[str] = []
    seen = set()
    for part in parts:
        e = part.lower()
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def validate_user_email_input(email: str | None) -> tuple[list[str], bool]:
    if email is None or not isinstance(email, str):
        return [], False
    raw = email.strip()
    emails = _parse_emails(raw)
    is_valid = raw == "" or (emails and all(_PROPER_EMAIL_RE.match(e) for e in emails))
    return emails, bool(is_valid)


def login_user(username: str) -> str:
    """Log in a user.

    Args:
        username: Username to log in

    Returns:
        Redirect URL
    """
    # the user is logged out but wants to log in, ok
    username = (username or "").strip()
    if g.user is None and username:
        if len(username) > 0:  # one more paranoid check
            if not re.match(r"^[a-zA-Z0-9_]{3,64}$", username):
                flash("Invalid username: use 3-64 chars of letters/numbers/_", "error")
            else:
                # Defensive: ensure no stale session fields leak across login boundary.
                # Keep CSRF token stable across login so the next request in the same client
                # session doesn't unexpectedly fail with 403.
                existing_csrf = session.get("_csrf_token")
                session.clear()
                if existing_csrf:
                    session["_csrf_token"] = existing_csrf
                session["user"] = username
                logger.debug(f"User {username} logged in")

    return url_for("web.profile")


def logout_user() -> str:
    """Log out the current user.

    Returns:
        Redirect URL
    """
    user = session.get("user")
    session.clear()
    if user:
        logger.debug(f"User {user} logged out")

    return url_for("web.profile")


def register_user_email(email: str) -> str:
    """Register or update user's email(s).

    Args:
        email: Email address(es) to register (comma/whitespace/newline separated)

    Returns:
        Redirect URL
    """
    raw = (email or "").strip()

    if g.user:
        emails, is_valid = validate_user_email_input(raw)
        if is_valid:
            UserRepository.set_emails(g.user, emails)
            if emails:
                logger.debug(f"User {g.user} registered emails: {emails}")
                flash("Email(s) updated.", "success")
            else:
                logger.debug(f"User {g.user} cleared emails")
                flash("Email(s) cleared.", "success")
        else:
            flash("Invalid email address(es).", "error")
    else:
        flash("Not logged in.", "error")

    return url_for("web.profile")


def get_user_email(user: str = None) -> str:
    """Get user's registered email.

    Args:
        user: Username. If None, uses g.user

    Returns:
        Email address or empty string
    """
    user = user or getattr(g, "user", None)
    if not user:
        return ""

    return UserRepository.get_email(user) or ""


def get_user_emails(user: str = None) -> list[str]:
    """Get user's registered emails."""
    user = user or getattr(g, "user", None)
    if not user:
        return []
    return UserRepository.get_emails(user)
