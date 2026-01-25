"""Integration tests for authentication and CSRF protection."""

from __future__ import annotations


class TestCsrfProtection:
    """Tests for CSRF protection."""

    def test_summary_get_without_csrf_returns_403(self, client):
        """Test that POST to /api/get_paper_summary without CSRF returns 403."""
        resp = client.post("/api/get_paper_summary", json={"pid": "2301.00001"})
        assert resp.status_code == 403

    def test_summary_trigger_without_csrf_returns_403(self, client):
        """Test that POST to /api/trigger_paper_summary without CSRF returns 403."""
        resp = client.post("/api/trigger_paper_summary", json={"pid": "2301.00001"})
        assert resp.status_code == 403

    def test_summary_status_without_csrf_returns_403(self, client):
        """Test that POST to /api/summary_status without CSRF returns 403."""
        resp = client.post("/api/summary_status", json={"pids": ["2301.00001"]})
        assert resp.status_code == 403


class TestLoginRequired:
    """Tests for login-required endpoints."""

    def test_user_state_without_login_returns_401(self, client):
        """Test that GET /api/user_state without login returns 401."""
        resp = client.get("/api/user_state")
        assert resp.status_code == 401

        data = resp.get_json(silent=True) or {}
        assert data.get("success", True) is False

    def test_user_stream_without_login_returns_401(self, client):
        """Test that GET /api/user_stream without login returns 401."""
        resp = client.get("/api/user_stream")
        assert resp.status_code == 401

    def test_tag_feedback_without_login_returns_401(self, client):
        """Test that POST /api/tag_feedback without login returns 401."""
        resp = client.post("/api/tag_feedback", json={"pid": "x", "tag": "t", "label": 1})
        assert resp.status_code == 401

    def test_readinglist_add_without_login_returns_401(self, client):
        """Test that POST /api/readinglist/add without login returns 401."""
        resp = client.post("/api/readinglist/add", json={"pid": "2301.00001"})
        assert resp.status_code == 401

    def test_readinglist_remove_without_login_returns_401(self, client):
        """Test that POST /api/readinglist/remove without login returns 401."""
        resp = client.post("/api/readinglist/remove", json={"pid": "2301.00001"})
        assert resp.status_code == 401

    def test_readinglist_list_without_login_returns_401(self, client):
        """Test that GET /api/readinglist/list without login returns 401."""
        resp = client.get("/api/readinglist/list")
        assert resp.status_code == 401


class TestLoginRequiredWithCsrf:
    """Tests for login-required endpoints with CSRF but no login."""

    def test_tag_feedback_with_csrf_but_no_login_returns_401(self, client, csrf_token):
        """Test that tag_feedback with CSRF but no login returns 401."""
        resp = client.post(
            "/api/tag_feedback",
            json={"pid": "2301.00001", "tag": "test", "label": 1},
            headers={"X-CSRF-Token": csrf_token},
        )
        # Login check happens before CSRF check for these endpoints
        assert resp.status_code == 401

    def test_readinglist_add_with_csrf_but_no_login_returns_401(self, client, csrf_token):
        """Test that readinglist/add with CSRF but no login returns 401."""
        resp = client.post("/api/readinglist/add", json={"pid": "2301.00001"}, headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code == 401


class TestLoggedInWithoutCsrf:
    """Tests for logged-in users without CSRF token."""

    def test_tag_feedback_logged_in_without_csrf_returns_403(self, logged_in_client):
        """Test that tag_feedback without CSRF returns 403 even when logged in."""
        resp = logged_in_client.post("/api/tag_feedback", json={"pid": "2301.00001", "tag": "test", "label": 1})
        assert resp.status_code == 403

    def test_readinglist_add_logged_in_without_csrf_returns_403(self, logged_in_client):
        """Test that readinglist/add without CSRF returns 403 even when logged in."""
        resp = logged_in_client.post("/api/readinglist/add", json={"pid": "2301.00001"})
        assert resp.status_code == 403
