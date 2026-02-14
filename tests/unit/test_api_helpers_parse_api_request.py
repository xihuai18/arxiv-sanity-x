"""Unit tests for parse_api_request."""

from __future__ import annotations


class TestParseApiRequest:
    def test_require_login_blocks_anonymous(self, app):
        from flask import g

        from backend.services.api_helpers import parse_api_request

        with app.test_request_context("/api/test", method="POST", json={"x": 1}):
            g.user = None
            data, err = parse_api_request(require_login=True, require_csrf=False)

        assert data is None
        assert err is not None
        resp, status = err
        assert status == 401
        assert resp.get_json()["error"] == "Not logged in"

    def test_missing_json_returns_400(self, app):
        from flask import g

        from backend.services.api_helpers import parse_api_request

        with app.test_request_context("/api/test", method="POST"):
            g.user = "u"
            data, err = parse_api_request(require_login=False, require_csrf=False)

        assert data is None
        assert err is not None
        resp, status = err
        assert status == 400
        assert resp.get_json()["error"] == "No JSON data provided"

    def test_require_csrf_calls_hook(self, app):
        from flask import g

        from backend.services.api_helpers import parse_api_request

        called = {"n": 0}

        def csrf_protect():
            called["n"] += 1

        with app.test_request_context("/api/test", method="POST", json={"x": 1}):
            g.user = "u"
            data, err = parse_api_request(require_login=False, require_csrf=True, csrf_protect_fn=csrf_protect)

        assert err is None
        assert data is not None
        assert called["n"] == 1

    def test_require_pid_adds_raw_pid_and_checks_exists(self, app):
        from flask import g

        from backend.services.api_helpers import parse_api_request

        def exists(pid: str) -> bool:
            return pid == "2301.00001"

        with app.test_request_context("/api/test", method="POST", json={"pid": "2301.00001v2"}):
            g.user = "u"
            data, err = parse_api_request(
                require_login=False,
                require_csrf=False,
                require_pid=True,
                paper_exists_fn=exists,
            )

        assert err is None
        assert data is not None
        assert data["pid"] == "2301.00001v2"
        assert data["_raw_pid"] == "2301.00001"

    def test_require_pid_missing_returns_400(self, app):
        from flask import g

        from backend.services.api_helpers import parse_api_request

        with app.test_request_context("/api/test", method="POST", json={"pid": ""}):
            g.user = "u"
            data, err = parse_api_request(require_login=False, require_csrf=False, require_pid=True)

        assert data is None
        assert err is not None
        resp, status = err
        assert status == 400
        assert resp.get_json()["error"] == "Paper ID is required"

    def test_require_pid_not_found_returns_404(self, app):
        from flask import g

        from backend.services.api_helpers import parse_api_request

        with app.test_request_context("/api/test", method="POST", json={"pid": "2301.00001"}):
            g.user = "u"
            data, err = parse_api_request(
                require_login=False,
                require_csrf=False,
                require_pid=True,
                paper_exists_fn=lambda _pid: False,
            )

        assert data is None
        assert err is not None
        resp, status = err
        assert status == 404
        assert resp.get_json()["error"] == "Paper not found"
