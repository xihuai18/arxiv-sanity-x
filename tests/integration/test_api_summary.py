"""Integration tests for summary APIs."""

from __future__ import annotations


class TestSummaryStatusApi:
    """Tests for summary status API."""

    def test_summary_status_without_csrf_returns_403(self, client):
        """Test that summary_status without CSRF returns 403."""
        resp = client.post("/api/summary_status", json={"pids": ["2301.00001"]})
        assert resp.status_code == 403

    def test_summary_status_missing_pids_returns_400(self, client, csrf_token):
        """Test that missing pids returns 400."""
        resp = client.post("/api/summary_status", json={}, headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code == 400

    def test_summary_status_with_valid_pids_returns_success(self, client, csrf_token):
        """Test that valid pids returns success."""
        resp = client.post(
            "/api/summary_status",
            json={"pids": ["2301.00001"], "model": "test-model"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "statuses" in data

    def test_summary_status_hides_last_error_for_other_user(self, client, csrf_token, monkeypatch):
        """Do not leak last_error/task_id across users when task_user is set."""
        from backend import legacy

        # Force existence so the endpoint reaches the status DB path.
        monkeypatch.setattr(legacy, "paper_exists", lambda _pid: True)

        class _FakeDB:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_many(self, keys):
                return {
                    k: {
                        "status": "failed",
                        "last_error": "secret failure details",
                        "task_id": "task_abc",
                        "task_user": "alice",
                    }
                    for k in keys
                }

        monkeypatch.setattr(legacy, "get_summary_status_db", lambda *args, **kwargs: _FakeDB())

        pid = "9999.99999"
        resp = client.post(
            "/api/summary_status",
            json={"pids": [pid], "model": "test-model"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        info = (data.get("statuses") or {}).get(pid) or {}
        assert info.get("status") == "failed"
        assert info.get("last_error") is None
        assert "task_id" not in info


class TestSummaryGetApi:
    """Tests for get paper summary API."""

    def test_get_paper_summary_without_csrf_returns_403(self, client):
        """Test that get_paper_summary without CSRF returns 403."""
        resp = client.post("/api/get_paper_summary", json={"pid": "2301.00001"})
        assert resp.status_code == 403

    def test_get_paper_summary_forces_cache_only(self, client, csrf_token, monkeypatch):
        """P0: get_paper_summary must be cache-only (ignore cache_only=false and any force flags)."""
        from backend import legacy

        captured = {}

        def _fake_generate(pid, model=None, force_refresh=False, cache_only=False):
            captured["pid"] = pid
            captured["model"] = model
            captured["force_refresh"] = force_refresh
            captured["cache_only"] = cache_only
            return "cached-summary", {"model": model or "test-model"}

        monkeypatch.setattr(legacy, "generate_paper_summary", _fake_generate)

        resp = client.post(
            "/api/get_paper_summary",
            json={
                "pid": "2301.00001",
                "model": "test-model",
                "cache_only": False,
                "force_regenerate": True,
            },
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert data.get("summary_content") == "cached-summary"

        assert captured["cache_only"] is True
        assert captured["force_refresh"] is False


class TestSummaryTldrApi:
    """Tests for get paper TL;DR API."""

    def test_get_paper_tldr_without_csrf_returns_403(self, client):
        """Test that get_paper_tldr without CSRF returns 403."""
        resp = client.post("/api/get_paper_tldr", json={"pid": "2301.00001"})
        assert resp.status_code == 403

    def test_get_paper_tldr_returns_tldr_when_ok(self, client, csrf_token, monkeypatch):
        """Test that get_paper_tldr returns TL;DR when summary status is ok."""
        from backend import legacy, services

        # Avoid depending on a real papers.db in CI.
        monkeypatch.setattr(legacy, "paper_exists", lambda _pid: True)
        monkeypatch.setattr(services.summary_service, "get_summary_status", lambda _pid, model=None: ("ok", None))
        monkeypatch.setattr(services.summary_service, "extract_tldr_from_summary", lambda _pid: "Hello TL;DR")

        resp = client.post(
            "/api/get_paper_tldr",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert data.get("pid") == "2301.00001"
        assert data.get("summary_status") == "ok"
        assert data.get("tldr") == "Hello TL;DR"


class TestSummaryTriggerApi:
    """Tests for trigger paper summary API."""

    def test_trigger_paper_summary_without_csrf_returns_403(self, client):
        """Test that trigger_paper_summary without CSRF returns 403."""
        resp = client.post("/api/trigger_paper_summary", json={"pid": "2301.00001"})
        assert resp.status_code == 403

    def test_trigger_paper_summary_force_regenerate_passes_force_refresh(
        self, logged_in_client, csrf_token, monkeypatch
    ):
        """P0: trigger_paper_summary supports force_regenerate and forwards to enqueue(force_refresh)."""
        from backend.services import readinglist_service

        captured = {}

        def _fake_enqueue(pid, model=None, user=None, priority=None, force_refresh=False):
            captured["pid"] = pid
            captured["model"] = model
            captured["user"] = user
            captured["priority"] = priority
            captured["force_refresh"] = force_refresh
            return "task_test_123"

        monkeypatch.setattr(readinglist_service, "enqueue_summary_task", _fake_enqueue)

        resp = logged_in_client.post(
            "/api/trigger_paper_summary",
            json={"pid": "2301.00001", "model": "test-model", "force_regenerate": True},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert data.get("status") == "queued"
        assert data.get("task_id") == "task_test_123"
        assert captured["force_refresh"] is True

    def test_trigger_paper_summary_force_alias_passes_force_refresh(self, logged_in_client, csrf_token, monkeypatch):
        """Backward compatibility: trigger_paper_summary supports legacy `force` flag."""
        from backend.services import readinglist_service

        captured = {}

        def _fake_enqueue(pid, model=None, user=None, priority=None, force_refresh=False):
            captured["pid"] = pid
            captured["model"] = model
            captured["user"] = user
            captured["priority"] = priority
            captured["force_refresh"] = force_refresh
            return "task_test_124"

        monkeypatch.setattr(readinglist_service, "enqueue_summary_task", _fake_enqueue)

        resp = logged_in_client.post(
            "/api/trigger_paper_summary",
            json={"pid": "2301.00001", "model": "test-model", "force": True},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert data.get("status") == "queued"
        assert data.get("task_id") == "task_test_124"
        assert captured["force_refresh"] is True


class TestQueueStatsApi:
    """Tests for queue stats API."""

    def test_queue_stats_returns_success(self, client):
        """Test that queue_stats returns success."""
        resp = client.get("/api/queue_stats")
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "queued" in data
        assert "running" in data


class TestLlmModelsApi:
    """Tests for LLM models API."""

    def test_llm_models_returns_models_array(self, client):
        """Test that llm_models returns models array."""
        resp = client.get("/api/llm_models")
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert "models" in data
        assert isinstance(data["models"], list)


class TestTaskStatusApi:
    """Tests for task status API."""

    def test_task_status_invalid_task_id(self, client):
        """Test task status with invalid task ID."""
        resp = client.get("/api/task_status/invalid_task_id_12345")
        # Should return 200 with not found status or 404
        assert resp.status_code in [200, 404]

    def test_task_status_empty_task_id(self, client):
        """Test task status with empty task ID returns 404."""
        resp = client.get("/api/task_status/")
        assert resp.status_code == 404


class TestClearModelSummaryApi:
    """Tests for clear model summary API."""

    def test_clear_model_summary_without_login_returns_401(self, client):
        """Test that clear_model_summary requires login."""
        resp = client.post(
            "/api/clear_model_summary",
            json={"pid": "2301.00001", "model": "test-model"},
        )
        assert resp.status_code == 401

    def test_clear_model_summary_with_csrf_but_no_login_returns_401(self, client, csrf_token):
        """Test that clear_model_summary with CSRF but no login returns 401."""
        resp = client.post(
            "/api/clear_model_summary",
            json={"pid": "2301.00001", "model": "test-model"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    def test_clear_model_summary_without_csrf_returns_403(self, logged_in_client):
        """Test that clear_model_summary without CSRF returns 403 (when logged in)."""
        resp = logged_in_client.post(
            "/api/clear_model_summary",
            json={"pid": "2301.00001", "model": "test-model"},
        )
        assert resp.status_code == 403

    def test_clear_model_summary_missing_pid_returns_400(self, logged_in_client, csrf_token):
        """Test that missing pid returns 400."""
        resp = logged_in_client.post(
            "/api/clear_model_summary",
            json={"model": "test-model"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_clear_model_summary_missing_model_returns_400(self, logged_in_client, csrf_token):
        """Test that missing model returns 400."""
        resp = logged_in_client.post(
            "/api/clear_model_summary",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400


class TestClearPaperCacheApi:
    """Tests for clear paper cache API."""

    def test_clear_paper_cache_without_login_returns_401(self, client):
        """Test that clear_paper_cache requires login."""
        resp = client.post(
            "/api/clear_paper_cache",
            json={"pid": "2301.00001"},
        )
        assert resp.status_code == 401

    def test_clear_paper_cache_with_csrf_but_no_login_returns_401(self, client, csrf_token):
        """Test that clear_paper_cache with CSRF but no login returns 401."""
        resp = client.post(
            "/api/clear_paper_cache",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    def test_clear_paper_cache_without_csrf_returns_403(self, logged_in_client):
        """Test that clear_paper_cache without CSRF returns 403 (when logged in)."""
        resp = logged_in_client.post(
            "/api/clear_paper_cache",
            json={"pid": "2301.00001"},
        )
        assert resp.status_code == 403


class TestCheckPaperSummariesApi:
    """Tests for check paper summaries API."""

    def test_check_paper_summaries_returns_success(self, client):
        """Test that check_paper_summaries returns success structure."""
        resp = client.get("/api/check_paper_summaries", query_string={"pid": "2301.00001"})
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "available_models" in data

    def test_check_paper_summaries_missing_pid(self, client):
        """Test check_paper_summaries without pid."""
        resp = client.get("/api/check_paper_summaries")
        # Should return 400 or handle gracefully
        assert resp.status_code in [200, 400]
