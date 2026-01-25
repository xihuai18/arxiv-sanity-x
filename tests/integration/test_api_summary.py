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


class TestSummaryGetApi:
    """Tests for get paper summary API."""

    def test_get_paper_summary_without_csrf_returns_403(self, client):
        """Test that get_paper_summary without CSRF returns 403."""
        resp = client.post("/api/get_paper_summary", json={"pid": "2301.00001"})
        assert resp.status_code == 403


class TestSummaryTriggerApi:
    """Tests for trigger paper summary API."""

    def test_trigger_paper_summary_without_csrf_returns_403(self, client):
        """Test that trigger_paper_summary without CSRF returns 403."""
        resp = client.post("/api/trigger_paper_summary", json={"pid": "2301.00001"})
        assert resp.status_code == 403


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

    def test_clear_model_summary_without_csrf_returns_403(self, client):
        """Test that clear_model_summary without CSRF returns 403."""
        resp = client.post(
            "/api/clear_model_summary",
            json={"pid": "2301.00001", "model": "test-model"},
        )
        assert resp.status_code == 403

    def test_clear_model_summary_missing_pid_returns_400(self, client, csrf_token):
        """Test that missing pid returns 400."""
        resp = client.post(
            "/api/clear_model_summary",
            json={"model": "test-model"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_clear_model_summary_missing_model_returns_400(self, client, csrf_token):
        """Test that missing model returns 400."""
        resp = client.post(
            "/api/clear_model_summary",
            json={"pid": "2301.00001"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400


class TestClearPaperCacheApi:
    """Tests for clear paper cache API."""

    def test_clear_paper_cache_without_csrf_returns_403(self, client):
        """Test that clear_paper_cache without CSRF returns 403."""
        resp = client.post(
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
