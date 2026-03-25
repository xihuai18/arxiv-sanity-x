"""Integration tests for tag APIs."""

from __future__ import annotations

from aslite.repositories import (
    MetaRepository,
    PaperRepository,
    PaperTombstoneRepository,
    TagRepository,
)


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

    def test_tag_feedback_invalid_label_returns_400(self, logged_in_client, csrf_token, monkeypatch):
        """Test that invalid label returns 400."""
        monkeypatch.setattr("backend.legacy.paper_exists", lambda _pid: True)
        resp = logged_in_client.post(
            "/api/tag_feedback",
            json={"pid": "2301.00001", "tag": "test", "label": 999},
            headers={"X-CSRF-Token": csrf_token},
        )
        # Pydantic validation should reject label not in [-1, 0, 1]
        assert resp.status_code == 400

    def test_tag_feedback_missing_tag_returns_400(self, logged_in_client, csrf_token, monkeypatch):
        """Test that missing tag returns 400."""
        monkeypatch.setattr("backend.legacy.paper_exists", lambda _pid: True)
        resp = logged_in_client.post(
            "/api/tag_feedback",
            json={"pid": "2301.00001", "label": 1},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_tag_feedback_invalid_pid_returns_404(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/api/tag_feedback",
            json={"pid": "does-not-exist", "tag": "test", "label": 1},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404


class TestTagMembersApi:
    """Tests for tag members API."""

    def test_tag_members_without_login_returns_401(self, client):
        """Test that tag_members without login returns 401."""
        resp = client.get("/api/tag_members", query_string={"tag": "test"})
        # API requires login, should return 401
        assert resp.status_code == 401

    def test_tag_members_hides_only_tombstoned_public_pid(self, logged_in_client):
        live_pid = "2601.00002"
        dead_pid = "2601.00004"
        tag = "cleanup_tag"
        PaperRepository.save(live_pid, {"_id": live_pid, "title": "Live paper", "authors": []})
        MetaRepository.save_many({live_pid: {"_time": 1.0}})
        TagRepository.add_paper_to_tag("test_user", live_pid, tag)
        TagRepository.add_paper_to_tag("test_user", dead_pid, tag)
        PaperTombstoneRepository.save(dead_pid, {"pid": dead_pid, "reason": "withdrawn_only"})

        resp = logged_in_client.get("/api/tag_members", query_string={"tag": tag})

        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("total_count") == 1
        items = payload.get("items") or []
        assert len(items) == 1
        assert items[0].get("pid") == live_pid

    def test_tag_members_pagination_counts_only_visible_items(self, logged_in_client):
        live_pid = "2601.00005"
        dead_pid = "2601.00006"
        tag = "cleanup_tag_page"
        PaperRepository.save(live_pid, {"_id": live_pid, "title": "Live paper 2", "authors": []})
        MetaRepository.save_many({live_pid: {"_time": 5.0}})
        TagRepository.add_paper_to_tag("test_user", live_pid, tag)
        TagRepository.add_paper_to_tag("test_user", dead_pid, tag)
        PaperTombstoneRepository.save(dead_pid, {"pid": dead_pid, "reason": "withdrawn_only"})

        resp = logged_in_client.get(
            "/api/tag_members",
            query_string={"tag": tag, "page_size": 1, "page_number": 1},
        )

        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("total_count") == 1
        items = payload.get("items") or []
        assert len(items) == 1
        assert items[0].get("pid") == live_pid


class TestTagFeedbackBulkApi:
    def test_tag_feedback_bulk_invalid_pid_returns_item_error(self, logged_in_client, auth_headers):
        resp = logged_in_client.post(
            "/api/tag_feedback_bulk",
            headers=auth_headers,
            json={"items": [{"pid": "does-not-exist", "tag": "test", "label": 1}]},
        )
        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        results = payload.get("results") or []
        assert results[0]["success"] is False
        assert results[0]["error"] == "Paper not found"

    def test_tag_feedback_bulk_accepts_upload_pid_that_looks_versioned(
        self, logged_in_client, auth_headers, monkeypatch
    ):
        upload_pid = "up_abcdefghijv2"

        monkeypatch.setattr(
            "aslite.repositories.UploadedPaperRepository.get",
            lambda pid: {"owner": "test_user", "parse_status": "ok"} if pid == upload_pid else None,
        )

        resp = logged_in_client.post(
            "/api/tag_feedback_bulk",
            headers=auth_headers,
            json={"items": [{"pid": upload_pid, "tag": "test", "label": 1}]},
        )
        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        results = payload.get("results") or []
        assert results[0]["success"] is True
        assert results[0]["pid"] == upload_pid
