"""Unit tests for :func:`backend.services.api_helpers.parse_api_request`."""

from __future__ import annotations

from flask import g

from backend.services.api_helpers import parse_api_request


def test_parse_api_request_requires_login(app):
    """Require login returns a 401 error when `g.user` is absent."""
    with app.test_request_context(json={"pid": "2301.00001"}):
        g.user = None
        data, error = parse_api_request(require_login=True)

    assert data is None
    assert error is not None
    assert error[1] == 401


def test_parse_api_request_calls_csrf_and_normalizes_pid(app):
    """CSRF hook runs and `_raw_pid` is set on success."""
    called = {"count": 0}

    def csrf_hook():
        called["count"] += 1

    with app.test_request_context(json={"pid": "2301.00001v2"}):
        g.user = "test_user"
        data, error = parse_api_request(
            require_pid=True,
            paper_exists_fn=lambda pid: True,
            csrf_protect_fn=csrf_hook,
        )

    assert error is None
    assert called["count"] == 1
    assert data["_raw_pid"] == "2301.00001"


def test_parse_api_request_rejects_missing_json(app):
    """Requests without JSON data should return a useful error."""
    with app.test_request_context(data="not-json", content_type="text/plain"):
        g.user = "test_user"
        data, error = parse_api_request()

    assert data is None
    assert error is not None
    assert error[1] == 400
    assert error[0].get_json()["error"] == "No JSON data provided"


def test_parse_api_request_requires_pid_field(app):
    """PID validation rejects requests that omit the PID."""
    with app.test_request_context(json={"foo": "bar"}):
        g.user = "test_user"
        data, error = parse_api_request(require_pid=True)

    assert data is None
    assert error is not None
    assert error[1] == 400
    assert error[0].get_json()["error"] == "Paper ID is required"


def test_parse_api_request_handles_missing_paper(app):
    """The optional existence checker propagates a 404 when a paper is missing."""
    with app.test_request_context(json={"pid": "2301.00001"}):
        g.user = "test_user"
        data, error = parse_api_request(require_pid=True, paper_exists_fn=lambda _: False)

    assert data is None
    assert error is not None
    assert error[1] == 404
    assert error[0].get_json()["error"] == "Paper not found"
