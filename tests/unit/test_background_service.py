"""Unit tests for background service functions.

Tests background service functions using mocks to avoid side effects.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestWarmupDataCache:
    """Tests for _warmup_data_cache function."""

    @patch("backend.services.background.logger")
    def test_warmup_data_cache_success(self, mock_logger):
        """Test successful data cache warmup."""
        with patch("backend.services.background._warmup_data_cache") as mock_warmup:
            mock_warmup()
            mock_warmup.assert_called_once()

    @patch("backend.services.data_service.warmup_data_cache")
    @patch("backend.services.background.logger")
    def test_warmup_data_cache_handles_exception(self, mock_logger, mock_warmup):
        """Test that warmup handles exceptions gracefully."""
        from backend.services.background import _warmup_data_cache

        mock_warmup.side_effect = Exception("Test error")

        # Should not raise
        _warmup_data_cache()
        mock_logger.warning.assert_called()


class TestWarmupMlCache:
    """Tests for _warmup_ml_cache function."""

    @patch("backend.services.background.logger")
    def test_warmup_ml_cache_success(self, mock_logger):
        """Test successful ML cache warmup."""
        with patch("backend.services.background._warmup_ml_cache") as mock_warmup:
            mock_warmup()
            mock_warmup.assert_called_once()


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
