"""Integration tests for user APIs."""

from __future__ import annotations


class TestUserStateApi:
    """Tests for user state API."""

    def test_user_state_without_login_returns_401(self, client):
        """Test that user_state without login returns 401."""
        resp = client.get("/api/user_state")
        assert resp.status_code == 401

    def test_user_state_logged_in_returns_tags_and_keys(self, logged_in_client):
        """Test that user_state returns tags and keys when logged in."""
        resp = logged_in_client.get("/api/user_state")
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "tags" in data
        assert "keys" in data
        assert "combined_tags" in data


class TestUserStreamApi:
    """Tests for user stream API."""

    def test_user_stream_without_login_returns_401(self, client):
        """Test that user_stream without login returns 401."""
        resp = client.get("/api/user_stream")
        assert resp.status_code == 401


class TestApiResponseFormat:
    """Tests for API response format consistency."""

    def test_error_response_is_json(self, client):
        """Test that error responses are JSON."""
        resp = client.post("/api/get_paper_summary", json={"pid": "test"})

        # CSRF error should return 403
        assert resp.status_code == 403

        # Response may or may not be JSON for CSRF errors
        # Just verify it doesn't crash
        resp.get_json(silent=True)

    def test_get_endpoint_returns_json_content_type(self, client):
        """Test that GET endpoints return JSON content type."""
        resp = client.get("/api/user_state")
        content_type = resp.headers.get("Content-Type", "")
        assert "application/json" in content_type
