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

    def test_readinglist_add_hidden_task_id_does_not_overwrite_status(self, logged_in_client, csrf_token, monkeypatch):
        """When enqueue returns hidden task marker, API should not rewrite summary status ownership."""
        from backend import legacy

        status_updates = []
        readinglist_updates = []

        monkeypatch.setattr(legacy, "_trigger_summary_async", lambda user, pid: "")
        monkeypatch.setattr(
            legacy, "_update_summary_status_db", lambda *args, **kwargs: status_updates.append((args, kwargs))
        )
        monkeypatch.setattr(
            legacy,
            "_update_readinglist_summary_status",
            lambda *args, **kwargs: readinglist_updates.append((args, kwargs)),
        )

        def _fake_add_to_readinglist(
            pid, user=None, compute_top_tags_fn=None, get_tags_fn=None, trigger_summary_fn=None
        ):
            if trigger_summary_fn:
                trigger_summary_fn(user or "alice", pid)
            return {"pid": pid}

        monkeypatch.setattr("backend.services.readinglist_service.add_to_readinglist", _fake_add_to_readinglist)

        resp = logged_in_client.post(
            "/api/readinglist/add",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        assert status_updates == []
        assert readinglist_updates == []
