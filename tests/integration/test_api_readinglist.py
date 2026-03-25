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
        resp = logged_in_client.post(
            "/api/readinglist/remove", json={"pid": "2301.00001"}
        )
        assert resp.status_code == 403

    def test_readinglist_remove_empty_pid_returns_400(
        self, logged_in_client, csrf_token
    ):
        resp = logged_in_client.post(
            "/api/readinglist/remove",
            json={"pid": ""},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_readinglist_add_hidden_task_id_does_not_overwrite_status(
        self, logged_in_client, csrf_token, monkeypatch
    ):
        """When enqueue returns hidden task marker, API should not rewrite summary status ownership."""
        from backend import legacy

        status_updates = []
        readinglist_updates = []

        monkeypatch.setattr("backend.legacy.paper_exists", lambda _pid: True)
        monkeypatch.setattr(
            "backend.services.data_service.paper_exists", lambda _pid: True
        )

        monkeypatch.setattr(legacy, "_trigger_summary_async", lambda user, pid: "")
        monkeypatch.setattr(
            legacy,
            "_update_summary_status_db",
            lambda *args, **kwargs: status_updates.append((args, kwargs)),
        )
        monkeypatch.setattr(
            legacy,
            "_update_readinglist_summary_status",
            lambda *args, **kwargs: readinglist_updates.append((args, kwargs)),
        )

        def _fake_add_to_readinglist(
            pid,
            user=None,
            compute_top_tags_fn=None,
            get_tags_fn=None,
            trigger_summary_fn=None,
        ):
            if trigger_summary_fn:
                trigger_summary_fn(user or "alice", pid)
            return {"pid": pid}

        monkeypatch.setattr(
            "backend.services.readinglist_service.add_to_readinglist",
            _fake_add_to_readinglist,
        )

        resp = logged_in_client.post(
            "/api/readinglist/add",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        assert status_updates == []
        assert readinglist_updates == []

    def test_readinglist_add_upload_waits_for_parse_before_summary(
        self, logged_in_client, csrf_token, monkeypatch
    ):
        from backend import legacy

        trigger_calls = []

        monkeypatch.setattr(
            "aslite.repositories.UploadedPaperRepository.get",
            lambda _pid: {
                "owner": "test_user",
                "parse_status": "queued",
                "parse_error": "",
            },
        )
        monkeypatch.setattr(
            legacy,
            "_trigger_summary_async",
            lambda user, pid: trigger_calls.append((user, pid)) or "task123",
        )

        resp = logged_in_client.post(
            "/api/readinglist/add",
            json={"pid": "up_waitparse001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert trigger_calls == []

        listing = logged_in_client.get("/api/readinglist/list")
        assert listing.status_code == 200
        items = (listing.get_json(silent=True) or {}).get("items") or []
        item = next(
            (entry for entry in items if entry.get("pid") == "up_waitparse001"), None
        )
        assert item is not None
        assert item.get("summary_status") in ("", None)
        assert item.get("summary_task_id") in (None, "")

    def test_readinglist_add_reuses_ready_summary_without_enqueue(
        self, logged_in_client, csrf_token, monkeypatch
    ):
        from backend import legacy

        trigger_calls = []

        monkeypatch.setattr("backend.legacy.paper_exists", lambda _pid: True)
        monkeypatch.setattr(
            "backend.services.data_service.paper_exists", lambda _pid: True
        )

        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_status",
            lambda pid, model=None: ("ok", None),
        )
        monkeypatch.setattr(
            legacy,
            "_trigger_summary_async",
            lambda user, pid: trigger_calls.append((user, pid)) or "task123",
        )

        resp = logged_in_client.post(
            "/api/readinglist/add",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("summary_status") == "ok"
        assert payload.get("task_id") in (None, "")
        assert trigger_calls == []

        listing = logged_in_client.get("/api/readinglist/list")
        assert listing.status_code == 200
        items = (listing.get_json(silent=True) or {}).get("items") or []
        item = next(
            (entry for entry in items if entry.get("pid") == "2301.00001"), None
        )
        assert item is not None
        assert item.get("summary_status") == "ok"

    def test_readinglist_list_hides_tombstoned_public_pid(self, logged_in_client):
        from aslite.repositories import (
            MetaRepository,
            PaperRepository,
            PaperTombstoneRepository,
            ReadingListRepository,
        )
        from backend.services.data_service import invalidate_cache

        live_pid = "2601.00001"
        dead_pid = "2601.00003"
        PaperRepository.save(live_pid, {"_id": live_pid, "title": "Live paper"})
        MetaRepository.save_many({live_pid: {"_time": 1.0}})
        ReadingListRepository.add_to_reading_list(
            "test_user",
            live_pid,
            {"added_time": 2, "top_tags": []},
        )
        ReadingListRepository.add_to_reading_list(
            "test_user",
            dead_pid,
            {"added_time": 1, "top_tags": []},
        )
        PaperTombstoneRepository.save(
            dead_pid, {"pid": dead_pid, "reason": "withdrawn_only"}
        )
        invalidate_cache()

        resp = logged_in_client.get("/api/readinglist/list")

        assert resp.status_code == 200
        items = (resp.get_json(silent=True) or {}).get("items") or []
        item_pids = [item.get("pid") for item in items]
        assert live_pid in item_pids
        assert dead_pid not in item_pids

    def test_readinglist_paper_prefers_global_summary_status(
        self, logged_in_client, monkeypatch
    ):
        from aslite.repositories import ReadingListRepository

        ReadingListRepository.add_to_reading_list(
            "test_user",
            "2301.00001",
            {
                "added_time": 1,
                "top_tags": [],
                "summary_status": "queued",
                "summary_last_error": None,
                "summary_updated_time": 1,
                "summary_task_id": "stale-task",
            },
        )

        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status",
            lambda pid, model=None: {
                "status": "ok",
                "updated_time": 2,
                "task_user": "test_user",
                "task_id": "fresh-task",
            },
        )
        monkeypatch.setattr(
            "backend.legacy.get_papers_bulk",
            lambda _pids: {
                "2301.00001": {"_rawid": "2301.00001", "title": "Paper", "authors": []}
            },
        )
        monkeypatch.setattr(
            "backend.legacy.render_pid",
            lambda pid, **_kwargs: {"id": pid, "title": "Paper", "authors": []},
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_render_snapshots",
            lambda pids, include_tldr=True, prefetched_status_rows=None: {
                pid: {"status": "ok", "last_error": None, "tldr": ""} for pid in pids
            },
        )

        resp = logged_in_client.get(
            "/api/readinglist/paper", query_string={"pid": "2301.00001"}
        )
        assert resp.status_code == 200
        paper = (resp.get_json(silent=True) or {}).get("paper") or {}
        assert paper.get("summary_status") == "ok"
        assert paper.get("summary_task_id") in (None, "")

    def test_readinglist_paper_returns_uploaded_payload(
        self, logged_in_client, monkeypatch
    ):
        from aslite.repositories import ReadingListRepository

        ReadingListRepository.add_to_reading_list(
            "test_user",
            "up_uploaded001",
            {
                "added_time": 1,
                "top_tags": ["ml"],
                "summary_status": "queued",
                "summary_last_error": None,
                "summary_updated_time": 1,
                "summary_task_id": "task-upload",
            },
        )

        monkeypatch.setattr(
            "backend.services.upload_service.get_uploaded_papers_list",
            lambda user: [
                {
                    "id": "up_uploaded001",
                    "kind": "upload",
                    "title": "Uploaded paper",
                    "authors": "Alice",
                    "summary": "abstract",
                    "parse_status": "ok",
                    "parse_error": "",
                    "summary_status": "queued",
                    "summary_last_error": "",
                    "summary_task_id": "task-upload",
                    "created_time": 1,
                    "original_filename": "paper.pdf",
                    "utags": ["ml"],
                    "ntags": [],
                    "tldr": "",
                    "meta_extracted_ok": True,
                }
            ],
        )

        resp = logged_in_client.get(
            "/api/readinglist/paper", query_string={"pid": "up_uploaded001"}
        )

        assert resp.status_code == 200
        paper = (resp.get_json(silent=True) or {}).get("paper") or {}
        assert paper.get("id") == "up_uploaded001"
        assert paper.get("kind") == "upload"
        assert paper.get("top_tags") == ["ml"]
        assert paper.get("in_readinglist") is True
