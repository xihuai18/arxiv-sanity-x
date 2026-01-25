"""Integration tests for tag APIs."""

from __future__ import annotations


class TestTagFeedbackApi:
    """Tests for tag feedback API."""

    def test_tag_feedback_without_login_returns_401(self, client):
        """Test that tag_feedback without login returns 401."""
        resp = client.post("/api/tag_feedback", json={"pid": "2301.00001", "tag": "test", "label": 1})
        assert resp.status_code == 401

    def test_tag_feedback_without_csrf_returns_403(self, logged_in_client):
        """Test that tag_feedback without CSRF returns 403."""
        resp = logged_in_client.post("/api/tag_feedback", json={"pid": "2301.00001", "tag": "test", "label": 1})
        assert resp.status_code == 403

    def test_tag_feedback_invalid_label_returns_400(self, logged_in_client, csrf_token):
        """Test that invalid label returns 400."""
        resp = logged_in_client.post(
            "/api/tag_feedback",
            json={"pid": "2301.00001", "tag": "test", "label": 999},
            headers={"X-CSRF-Token": csrf_token},
        )
        # Pydantic validation should reject label not in [-1, 0, 1]
        assert resp.status_code == 400

    def test_tag_feedback_missing_tag_returns_400(self, logged_in_client, csrf_token):
        """Test that missing tag returns 400."""
        resp = logged_in_client.post(
            "/api/tag_feedback", json={"pid": "2301.00001", "label": 1}, headers={"X-CSRF-Token": csrf_token}
        )
        assert resp.status_code == 400


class TestTagMembersApi:
    """Tests for tag members API."""

    def test_tag_members_without_login_returns_401(self, client):
        """Test that tag_members without login returns 401."""
        resp = client.get("/api/tag_members", query_string={"tag": "test"})
        # API requires login, should return 401
        assert resp.status_code == 401
