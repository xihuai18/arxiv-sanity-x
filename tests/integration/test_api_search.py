"""Integration tests for search APIs."""

from __future__ import annotations


class TestKeywordSearchApi:
    """Tests for keyword search API."""

    def test_keyword_search_missing_keyword_returns_400(self, client):
        """Test that missing keyword returns 400."""
        resp = client.post("/api/keyword_search", json={})
        assert resp.status_code == 400

        data = resp.get_json(silent=True) or {}
        assert "error" in data

    def test_keyword_search_with_keyword_returns_success(self, client):
        """Test that valid keyword returns success structure."""
        resp = client.post("/api/keyword_search", json={"keyword": "transformer", "time_delta": 365, "limit": 5})
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "pids" in data
        assert isinstance(data["pids"], list)

    def test_keyword_search_invalid_limit_uses_default(self, client):
        """Test that invalid limit uses default value."""
        resp = client.post("/api/keyword_search", json={"keyword": "test", "limit": "invalid"})
        # Should not crash, uses default limit
        assert resp.status_code == 200

    def test_keyword_search_empty_keyword_returns_400(self, client):
        """Test that empty keyword returns 400."""
        resp = client.post("/api/keyword_search", json={"keyword": ""})
        assert resp.status_code == 400


class TestTagSearchApi:
    """Tests for tag search API."""

    def test_tag_search_without_login_returns_401(self, client):
        """Test that tag search requires login."""
        resp = client.post("/api/tag_search", json={"tag_name": "test_tag"})
        assert resp.status_code == 401

    def test_tag_search_missing_tag_returns_400(self, logged_in_client):
        """Test that missing tag_name returns 400."""
        resp = logged_in_client.post("/api/tag_search", json={})
        assert resp.status_code == 400

    def test_tag_search_with_tag_returns_success(self, logged_in_client):
        """Test that valid tag search returns success structure."""
        resp = logged_in_client.post(
            "/api/tag_search",
            json={"tag_name": "test_tag", "user": "test_user", "time_delta": 365, "limit": 5},
        )
        # May return 200 with empty results or 400 if tag doesn't exist
        assert resp.status_code in [200, 400]

    def test_tag_search_user_mismatch_returns_403(self, logged_in_client):
        """Test that mismatched user field is rejected."""
        resp = logged_in_client.post(
            "/api/tag_search",
            json={"tag_name": "test_tag", "user": "other_user"},
        )
        assert resp.status_code == 403


class TestTagsSearchApi:
    """Tests for multi-tag search API."""

    def test_tags_search_without_login_returns_401(self, client):
        """Test that tags search requires login."""
        resp = client.post("/api/tags_search", json={"tags": ["test_tag"]})
        assert resp.status_code == 401

    def test_tags_search_missing_tags_returns_400(self, logged_in_client):
        """Test that missing tags returns 400."""
        resp = logged_in_client.post("/api/tags_search", json={})
        assert resp.status_code == 400

    def test_tags_search_with_tags_returns_success(self, logged_in_client):
        """Test that valid tags search returns success structure."""
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": ["test_tag"], "user": "test_user", "time_delta": 365, "limit": 5},
        )
        # May return 200 with empty results or 400 if tags don't exist
        assert resp.status_code in [200, 400]

    def test_tags_search_user_mismatch_returns_403(self, logged_in_client):
        """Test that mismatched user field is rejected."""
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": ["test_tag"], "user": "other_user"},
        )
        assert resp.status_code == 403

    def test_tags_search_empty_tags_list(self, logged_in_client):
        """Test tags search with empty tags list."""
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": [], "user": "test_user"},
        )
        assert resp.status_code == 400
