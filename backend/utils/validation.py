"""Request validation and CSRF helpers."""

from __future__ import annotations

import secrets
from urllib.parse import urlparse

from flask import abort, jsonify, request, session


def get_or_set_csrf_token() -> str:
    tok = session.get("_csrf_token")
    if not tok:
        tok = secrets.token_urlsafe(32)
        session["_csrf_token"] = tok
    return tok


def _is_same_origin_request() -> bool:
    """
    Best-effort same-origin check for legacy GET mutation endpoints.

    - For modern browsers, `Sec-Fetch-Site: cross-site` is a reliable CSRF signal.
    - Otherwise fall back to Origin/Referer checks.
    """
    sfs = (request.headers.get("Sec-Fetch-Site") or "").lower().strip()
    if sfs:
        return sfs in ("same-origin", "same-site", "none")

    origin = (request.headers.get("Origin") or "").strip()
    if origin:
        return origin.rstrip("/") == request.host_url.rstrip("/")

    referer = (request.headers.get("Referer") or "").strip()
    if referer:
        try:
            return urlparse(referer).netloc == request.host
        except Exception:
            return False

    return False


def _csrf_error(message: str):
    """Return a JSON error response for CSRF failures."""
    resp = jsonify({"success": False, "error": message})
    resp.status_code = 403
    abort(resp)


def csrf_protect() -> None:
    """
    CSRF protection for state-changing endpoints.

    - POST: require session token via header/form/json
    - GET (legacy): require same-origin *or* explicit query token
    """
    tok = get_or_set_csrf_token()

    if request.method == "GET":
        if request.args.get("csrf_token") == tok:
            return
        if _is_same_origin_request():
            return
        _csrf_error("CSRF blocked (cross-site GET)")

    token = (request.headers.get("X-CSRF-Token") or "").strip()
    if not token:
        token = (request.form.get("csrf_token") or "").strip()
    if not token and request.is_json:
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            token = (data.get("csrf_token") or "").strip()

    if not token or token != tok:
        _csrf_error("CSRF token missing/invalid")


# Import from api_helpers to avoid duplication
from ..services.api_helpers import parse_api_request as _shared_parse_api_request


def parse_api_request(
    require_login: bool = False,
    require_csrf: bool = True,
    require_pid: bool = False,
    user=None,
    paper_exists=None,
):
    """
    Common API request parsing and validation.

    Returns (data, error_response) tuple. If error_response is not None, return it immediately.
    """
    if require_pid and paper_exists is None:
        raise RuntimeError("parse_api_request requires paper_exists callable when require_pid=True")

    return _shared_parse_api_request(
        require_login=require_login,
        require_csrf=require_csrf,
        require_pid=require_pid,
        paper_exists_fn=paper_exists,
        csrf_protect_fn=csrf_protect,
    )


def validate_tag_name(tag: str) -> str | None:
    """Validate tag name. Returns error message or None if valid."""
    if not tag:
        return "error, tag is required"
    # Tag names are used in URL paths by some legacy endpoints; disallow slashes to avoid routing ambiguity.
    if "/" in tag or "\\" in tag:
        return "error, tag cannot contain slashes"
    if tag in ("all", "null"):
        return f"error, cannot use the protected tag '{tag}'"
    return None


def validate_keyword_name(keyword: str) -> str | None:
    """Validate keyword name. Returns error message or None if valid."""
    if not keyword:
        return "error, keyword is required"
    if "/" in keyword or "\\" in keyword:
        return "error, keyword cannot contain slashes"
    if keyword == "null":
        return "error, cannot use the protected keyword 'null'"
    return None
