"""Authentication service.

This module handles user authentication and session management:
- Login/logout
- Email registration
- Profile page rendering
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from flask import g, session, url_for
from loguru import logger

from aslite.repositories import UserRepository

from ..utils.validation import csrf_protect

if TYPE_CHECKING:
    pass


def login_user(username: str) -> str:
    """Log in a user.

    Args:
        username: Username to log in

    Returns:
        Redirect URL
    """
    csrf_protect()

    # the user is logged out but wants to log in, ok
    username = (username or "").strip()
    if g.user is None and username:
        if len(username) > 0:  # one more paranoid check
            session["user"] = username
            logger.debug(f"User {username} logged in")

    return url_for("web.profile")


def logout_user() -> str:
    """Log out the current user.

    Returns:
        Redirect URL
    """
    csrf_protect()

    user = session.pop("user", None)
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
    csrf_protect()

    raw = (email or "").strip()

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

    if g.user:
        emails = _parse_emails(raw)

        # Do some basic input validation.
        # Keep validation lightweight but accept modern long TLDs (up to 63 chars).
        proper_email_re = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}$", re.IGNORECASE)
        if raw == "" or (emails and all(proper_email_re.match(e) for e in emails)):
            UserRepository.set_emails(g.user, emails)
            if emails:
                logger.debug(f"User {g.user} registered emails: {emails}")
            else:
                logger.debug(f"User {g.user} cleared emails")

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
