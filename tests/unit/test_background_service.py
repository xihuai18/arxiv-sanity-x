"""Unit tests for background service functions.

Tests background service functions using mocks to avoid side effects.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestWarmupDataCache:
    """Tests for _warmup_data_cache function."""

    @patch("backend.services.background._is_data_cache_loaded", return_value=False)
    @patch("backend.services.data_service.warmup_data_cache")
    @patch("backend.services.background.logger")
    def test_warmup_data_cache_success(self, mock_logger, mock_warmup, mock_is_loaded):
        """Test successful data cache warmup."""
        from backend.services.background import _warmup_data_cache

        _warmup_data_cache()
        mock_warmup.assert_called_once()

    @patch("backend.services.background._is_data_cache_loaded", return_value=False)
    @patch("backend.services.data_service.warmup_data_cache")
    @patch("backend.services.background.logger")
    def test_warmup_data_cache_handles_exception(self, mock_logger, mock_warmup, mock_is_loaded):
        """Test that warmup handles exceptions gracefully."""
        from backend.services.background import _warmup_data_cache

        mock_warmup.side_effect = Exception("Test error")

        # Should not raise
        _warmup_data_cache()
        mock_logger.warning.assert_called()


class TestWarmupMlCache:
    """Tests for _warmup_ml_cache function."""

    @patch("backend.services.background._is_features_cache_loaded", return_value=False)
    @patch("backend.services.semantic_service.get_semantic_model")
    @patch("backend.services.semantic_service.get_paper_embeddings")
    @patch("backend.services.data_service.get_features_cached")
    @patch("backend.services.background.logger")
    def test_warmup_ml_cache_success(
        self,
        mock_logger,
        mock_get_features_cached,
        mock_get_paper_embeddings,
        mock_get_semantic_model,
        mock_is_loaded,
    ):
        """Test successful ML cache warmup."""
        from backend.services.background import _warmup_ml_cache

        _warmup_ml_cache()
        mock_get_features_cached.assert_called_once()
        mock_get_paper_embeddings.assert_called_once()
        mock_get_semantic_model.assert_called_once()


class TestIsSchedulerRunning:
    """Tests for is_scheduler_running function."""

    def test_is_scheduler_running_no_scheduler(self):
        """Test is_scheduler_running when no scheduler exists."""
        from backend.services import background

        # Save original value
        original = background._SCHEDULER

        try:
            background._SCHEDULER = None
            result = background.is_scheduler_running()
            assert result is False
        finally:
            background._SCHEDULER = original

    def test_is_scheduler_running_with_scheduler(self):
        """Test is_scheduler_running with a running scheduler."""
        from backend.services import background

        original = background._SCHEDULER

        try:
            mock_scheduler = MagicMock()
            mock_scheduler.running = True
            background._SCHEDULER = mock_scheduler

            result = background.is_scheduler_running()
            assert result is True
        finally:
            background._SCHEDULER = original

    def test_is_scheduler_running_stopped_scheduler(self):
        """Test is_scheduler_running with a stopped scheduler."""
        from backend.services import background

        original = background._SCHEDULER

        try:
            mock_scheduler = MagicMock()
            mock_scheduler.running = False
            background._SCHEDULER = mock_scheduler

            result = background.is_scheduler_running()
            assert result is False
        finally:
            background._SCHEDULER = original


class TestIsSummaryRepairEnabled:
    """Tests for is_summary_repair_enabled function."""

    def test_is_summary_repair_enabled_false(self):
        """Test is_summary_repair_enabled when disabled."""
        from backend.services import background

        original = background._SUMMARY_REPAIR_JOB

        try:
            background._SUMMARY_REPAIR_JOB = False
            result = background.is_summary_repair_enabled()
            assert result is False
        finally:
            background._SUMMARY_REPAIR_JOB = original

    def test_is_summary_repair_enabled_true(self):
        """Test is_summary_repair_enabled when enabled."""
        from backend.services import background

        original = background._SUMMARY_REPAIR_JOB

        try:
            background._SUMMARY_REPAIR_JOB = True
            result = background.is_summary_repair_enabled()
            assert result is True
        finally:
            background._SUMMARY_REPAIR_JOB = original


class TestEnsureBackgroundServicesStarted:
    """Tests for ensure_background_services_started function."""

    def test_ensure_background_services_idempotent(self):
        """Test that ensure_background_services_started is idempotent."""
        from backend.services import background

        # Save original state
        original_started = background._BACKGROUND_STARTED

        try:
            # If already started, should return immediately
            background._BACKGROUND_STARTED = True

            # This should not raise and should return quickly
            background.ensure_background_services_started()

            # State should remain True
            assert background._BACKGROUND_STARTED is True
        finally:
            background._BACKGROUND_STARTED = original_started

    @patch("backend.services.background.settings")
    @patch("backend.services.background.threading")
    def test_ensure_background_services_starts_warmup_threads(self, mock_threading, mock_settings):
        """Test that warmup threads are started when enabled."""
        from backend.services import background

        # Save original state
        original_started = background._BACKGROUND_STARTED
        original_scheduler = background._SCHEDULER

        try:
            background._BACKGROUND_STARTED = False
            background._SCHEDULER = None

            mock_settings.web.warmup_data = True
            mock_settings.web.warmup_ml = True
            mock_settings.web.enable_scheduler = False

            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread

            background.ensure_background_services_started()

            # Should have created threads for warmup
            assert mock_threading.Thread.call_count >= 1
            assert background._BACKGROUND_STARTED is True
        finally:
            background._BACKGROUND_STARTED = original_started
            background._SCHEDULER = original_scheduler


class TestSchedulerHelpers:
    """Tests for APScheduler log control helpers."""

    def test_scheduler_misfire_grace_time_is_bounded(self):
        """Small intervals should tolerate brief delays without noisy misfire logs."""
        from backend.services.background import _scheduler_misfire_grace_time

        assert _scheduler_misfire_grace_time(5) == 60
        assert _scheduler_misfire_grace_time(300) == 300
        assert _scheduler_misfire_grace_time(7200) == 1800

    @patch("backend.services.background.logger")
    def test_record_scheduler_job_event_logs_periodic_summary(self, mock_logger):
        """Periodic summary should replace per-run scheduler noise."""
        from backend.services import background

        original_labels = dict(background._SCHEDULER_JOB_LABELS)
        original_stats = dict(background._SCHEDULER_JOB_STATS)
        original_every_runs = background._SCHEDULER_SUMMARY_EVERY_RUNS
        original_every_seconds = background._SCHEDULER_SUMMARY_EVERY_SECONDS

        try:
            background._SCHEDULER_JOB_LABELS.clear()
            background._SCHEDULER_JOB_LABELS["repair"] = "repair_stale_summary_tasks"
            background._SCHEDULER_JOB_STATS.clear()
            background._SCHEDULER_SUMMARY_EVERY_RUNS = 2
            background._SCHEDULER_SUMMARY_EVERY_SECONDS = 10**9

            background._record_scheduler_job_event("repair", "runs", result=3)
            background._record_scheduler_job_event("repair", "runs", result=0)

            info_messages = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("repaired 3 stale item(s)" in msg for msg in info_messages)
            assert any("Scheduler job summary:" in msg for msg in info_messages)
            assert any("ran 2 time(s)" in msg for msg in info_messages)
            assert any("cumulative_result=3" in msg for msg in info_messages)
        finally:
            background._SCHEDULER_JOB_LABELS.clear()
            background._SCHEDULER_JOB_LABELS.update(original_labels)
            background._SCHEDULER_JOB_STATS.clear()
            background._SCHEDULER_JOB_STATS.update(original_stats)
            background._SCHEDULER_SUMMARY_EVERY_RUNS = original_every_runs
            background._SCHEDULER_SUMMARY_EVERY_SECONDS = original_every_seconds

    @patch("backend.services.background._record_scheduler_job_event")
    def test_on_scheduler_event_routes_execution_results(self, mock_record):
        """APScheduler events should be normalized into compact counters."""
        from backend.services import background

        class _Event:
            def __init__(self, code, job_id, retval=None):
                self.code = code
                self.job_id = job_id
                self.retval = retval

        background._on_scheduler_event(_Event(background.EVENT_JOB_EXECUTED, "repair", retval=4))
        background._on_scheduler_event(_Event(background.EVENT_JOB_MISSED, "repair"))
        background._on_scheduler_event(_Event(background.EVENT_JOB_ERROR, "repair"))

        assert mock_record.call_args_list[0].args == ("repair", "runs")
        assert mock_record.call_args_list[0].kwargs == {"result": 4}
        assert mock_record.call_args_list[1].args == ("repair", "missed")
        assert mock_record.call_args_list[2].args == ("repair", "errors")
