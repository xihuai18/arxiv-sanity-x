"""API response helpers and request parsing utilities."""

from __future__ import annotations

from typing import Any

from flask import g, jsonify, request

from tools.paper_summarizer import split_pid_version


def api_error(error: str, status: int = 400, **extra) -> tuple[Any, int]:
    """Return a standardized JSON error response."""
    resp = {"success": False, "error": error}
    resp.update(extra)
    return jsonify(resp), status


def api_success(**data) -> Any:
    """Return a standardized JSON success response."""
    resp = {"success": True}
    resp.update(data)
    return jsonify(resp)


def parse_api_request(
    require_login: bool = False,
    require_csrf: bool = True,
    require_pid: bool = False,
    paper_exists_fn=None,
    csrf_protect_fn=None,
) -> tuple[dict | None, tuple[Any, int] | None]:
    """
    Common API request parsing and validation.

    Returns (data, error_response) tuple. If error_response is not None, return it immediately.
    """
    if require_login and g.user is None:
        return None, api_error("Not logged in", 401)

    if require_csrf and csrf_protect_fn:
        csrf_protect_fn()

    data = request.get_json()
    if not data:
        return None, api_error("No JSON data provided", 400)

    if require_pid:
        pid = (data.get("pid") or "").strip()
        if not pid:
            return None, api_error("Paper ID is required", 400)
        raw_pid, _ = split_pid_version(pid)
        if paper_exists_fn and not paper_exists_fn(raw_pid):
            return None, api_error("Paper not found", 404)
        data["_raw_pid"] = raw_pid

    return data, None


def normalize_name(value: str | None) -> str:
    """Normalize input name/value."""
    return (value or "").strip()
