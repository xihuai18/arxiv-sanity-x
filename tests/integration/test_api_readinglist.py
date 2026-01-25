"""Integration tests for reading list APIs."""

from __future__ import annotations


class TestReadingListWithoutLogin:
    """Tests for reading list API without login."""

    def test_readinglist_list_without_login_returns_401(self, client):
        """Test that list without login returns 401."""
        resp = client.get("/api/readinglist/list")
        assert resp.status_code == 401

    def test_readinglist_add_without_login_returns_401(self, client):
        """Test that add without login returns 401."""
        resp = client.post("/api/readinglist/add", json={"pid": "2301.00001"})
        assert resp.status_code == 401

    def test_readinglist_remove_without_login_returns_401(self, client):
        """Test that remove without login returns 401."""
        resp = client.post("/api/readinglist/remove", json={"pid": "2301.00001"})
        assert resp.status_code == 401


class TestReadingListWithLogin:
    """Tests for reading list API with login."""

    def test_readinglist_list_logged_in_returns_success(self, logged_in_client):
        """Test that list returns success when logged in."""
        resp = logged_in_client.get("/api/readinglist/list")
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_readinglist_add_without_csrf_returns_403(self, logged_in_client):
        """Test that add without CSRF returns 403."""
        resp = logged_in_client.post("/api/readinglist/add", json={"pid": "2301.00001"})
        assert resp.status_code == 403

    def test_readinglist_remove_without_csrf_returns_403(self, logged_in_client):
        """Test that remove without CSRF returns 403."""
        resp = logged_in_client.post("/api/readinglist/remove", json={"pid": "2301.00001"})
        assert resp.status_code == 403
