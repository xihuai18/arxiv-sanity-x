"""Extended unit tests for tools.daemon module.

Tests daemon functions using mocks to avoid running actual subprocesses.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch


class TestGetEmailTimeDelta:
    """Tests for _get_email_time_delta function."""

    @patch("tools.daemon.datetime")
    @patch("tools.daemon.holidays")
    def test_get_email_time_delta_monday(self, mock_holidays, mock_datetime):
        """Test time delta on Monday (should be 4 days)."""
        import tools.daemon as d

        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta = datetime.timedelta

        mock_us_holidays = MagicMock()
        mock_us_holidays.__contains__ = MagicMock(return_value=False)
        mock_holidays.UnitedStates.return_value = mock_us_holidays

        result = d._get_email_time_delta()
        assert result == 4.0

    @patch("tools.daemon.datetime")
    @patch("tools.daemon.holidays")
    def test_get_email_time_delta_wednesday(self, mock_holidays, mock_datetime):
        """Test time delta on Wednesday (should be 2 days)."""
        import tools.daemon as d

        mock_now = MagicMock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta = datetime.timedelta

        mock_us_holidays = MagicMock()
        mock_us_holidays.__contains__ = MagicMock(return_value=False)
        mock_holidays.UnitedStates.return_value = mock_us_holidays

        result = d._get_email_time_delta()
        assert result == 2.0


class TestGenSummary:
    """Tests for gen_summary function."""

    def test_gen_summary_disabled(self, monkeypatch):
        """Test gen_summary returns True when disabled."""
        import tools.daemon as d

        monkeypatch.setattr(d, "ENABLE_SUMMARY", False)
        result = d.gen_summary()
        assert result is True

    def test_gen_summary_with_priority_queue(self, monkeypatch):
        """Test gen_summary includes priority queue args when enabled."""
        import tools.daemon as d

        calls = []

        def fake_run_cmd(cmd, name):
            calls.append((name, list(map(str, cmd))))
            return True

        monkeypatch.setattr(d, "_run_cmd", fake_run_cmd)
        monkeypatch.setattr(d, "ENABLE_SUMMARY", True)
        monkeypatch.setattr(d, "ENABLE_PRIORITY_QUEUE", True)
        monkeypatch.setattr(d, "ENABLE_SUMMARY_QUEUE", False)
        monkeypatch.setattr(d, "PRIORITY_DAYS", 3.0)
        monkeypatch.setattr(d, "PRIORITY_LIMIT", 50)

        d.gen_summary()

        assert len(calls) == 1
        cmd = calls[0][1]
        assert "--priority" in cmd
        assert "--priority-days" in cmd
        assert "--priority-limit" in cmd

    def test_gen_summary_with_queue(self, monkeypatch):
        """Test gen_summary includes --queue when enabled."""
        import tools.daemon as d

        calls = []

        def fake_run_cmd(cmd, name):
            calls.append((name, list(map(str, cmd))))
            return True

        monkeypatch.setattr(d, "_run_cmd", fake_run_cmd)
        monkeypatch.setattr(d, "ENABLE_SUMMARY", True)
        monkeypatch.setattr(d, "ENABLE_SUMMARY_QUEUE", True)
        monkeypatch.setattr(d, "ENABLE_PRIORITY_QUEUE", False)

        d.gen_summary()

        assert len(calls) == 1
        cmd = calls[0][1]
        assert "--queue" in cmd


class TestSendEmail:
    """Tests for send_email function."""

    @patch("tools.daemon.subprocess")
    @patch("tools.daemon.datetime")
    @patch("tools.daemon.holidays")
    def test_send_email_weekday(self, mock_holidays, mock_datetime, mock_subprocess):
        """Test send_email on a regular weekday."""
        import tools.daemon as d

        mock_now = MagicMock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta = datetime.timedelta

        mock_us_holidays = MagicMock()
        mock_us_holidays.__contains__ = MagicMock(return_value=False)
        mock_holidays.UnitedStates.return_value = mock_us_holidays

        d.send_email()

        mock_subprocess.call.assert_called_once()
        call_args = mock_subprocess.call.call_args[0][0]
        assert "-t" in call_args
        # Should be "2" for Wednesday
        t_index = call_args.index("-t")
        assert call_args[t_index + 1] == "2"

    @patch("tools.daemon.subprocess")
    @patch("tools.daemon.datetime")
    @patch("tools.daemon.holidays")
    def test_send_email_monday(self, mock_holidays, mock_datetime, mock_subprocess):
        """Test send_email on Monday (should use 4 days)."""
        import tools.daemon as d

        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_datetime.datetime.now.return_value = mock_now
        mock_datetime.timedelta = datetime.timedelta

        mock_us_holidays = MagicMock()
        mock_us_holidays.__contains__ = MagicMock(return_value=False)
        mock_holidays.UnitedStates.return_value = mock_us_holidays

        d.send_email()

        mock_subprocess.call.assert_called_once()
        call_args = mock_subprocess.call.call_args[0][0]
        t_index = call_args.index("-t")
        assert call_args[t_index + 1] == "4"


class TestBackupUserData:
    """Tests for backup_user_data function."""

    @patch("tools.daemon.settings")
    def test_backup_user_data_disabled(self, mock_settings):
        """Test backup_user_data does nothing when disabled."""
        import tools.daemon as d

        mock_settings.daemon.enable_git_backup = False

        # Should return without error
        d.backup_user_data()

    @patch("tools.daemon.subprocess")
    @patch("tools.daemon.os")
    @patch("tools.daemon.settings")
    def test_backup_user_data_no_file(self, mock_settings, mock_os, mock_subprocess):
        """Test backup_user_data handles missing file."""
        import tools.daemon as d

        mock_settings.daemon.enable_git_backup = True
        mock_os.path.isfile.return_value = False

        d.backup_user_data()

        mock_subprocess.run.assert_not_called()


class TestCreateScheduler:
    """Tests for create_scheduler function."""

    @patch("tools.daemon.settings")
    def test_create_scheduler_returns_scheduler(self, mock_settings):
        """Test create_scheduler returns a BlockingScheduler."""
        import tools.daemon as d

        mock_settings.daemon.timezone = "UTC"

        scheduler = d.create_scheduler()

        from apscheduler.schedulers.blocking import BlockingScheduler

        assert isinstance(scheduler, BlockingScheduler)

    @patch("tools.daemon.settings")
    def test_create_scheduler_has_jobs(self, mock_settings):
        """Test create_scheduler adds expected jobs."""
        import tools.daemon as d

        mock_settings.daemon.timezone = "UTC"

        scheduler = d.create_scheduler()
        jobs = scheduler.get_jobs()

        # Should have at least 3 types of jobs
        assert len(jobs) >= 3


class TestRunCmd:
    """Tests for _run_cmd function."""

    @patch("tools.daemon.subprocess")
    def test_run_cmd_success(self, mock_subprocess):
        """Test _run_cmd returns True on success."""
        import tools.daemon as d

        mock_subprocess.run.return_value = MagicMock()

        result = d._run_cmd(["echo", "test"], "test")
        assert result is True

    @patch("tools.daemon.subprocess")
    def test_run_cmd_failure(self, mock_subprocess):
        """Test _run_cmd returns False on CalledProcessError."""
        import subprocess

        import tools.daemon as d

        mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError

        result = d._run_cmd(["false"], "test")
        assert result is False

    @patch("tools.daemon.subprocess")
    def test_run_cmd_exception(self, mock_subprocess):
        """Test _run_cmd returns False on general exception."""
        import subprocess as real_subprocess

        import tools.daemon as d

        # Keep CalledProcessError as the real exception class
        mock_subprocess.CalledProcessError = real_subprocess.CalledProcessError
        mock_subprocess.run.side_effect = OSError("test error")

        result = d._run_cmd(["cmd"], "test")
        assert result is False


class TestLogStartupInfo:
    """Tests for _log_startup_info function."""

    @patch("tools.daemon.settings")
    @patch("builtins.print")
    def test_log_startup_info_runs(self, mock_print, mock_settings):
        """Test _log_startup_info runs without error."""
        import tools.daemon as d

        mock_settings.daemon.timezone = "UTC"
        mock_settings.log_level = "INFO"
        mock_settings.daemon.email_dry_run = False
        mock_settings.daemon.enable_git_backup = True

        # Patch module-level variables using contextlib.ExitStack
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(patch.object(d, "ENABLE_SUMMARY", True))
            stack.enter_context(patch.object(d, "ENABLE_PRIORITY_QUEUE", True))
            stack.enter_context(patch.object(d, "ENABLE_SUMMARY_QUEUE", False))
            stack.enter_context(patch.object(d, "ENABLE_EMBEDDINGS", True))
            stack.enter_context(patch.object(d, "PRIORITY_DAYS", 2.0))
            stack.enter_context(patch.object(d, "PRIORITY_LIMIT", 100))
            stack.enter_context(patch.object(d, "FETCH_NUM", 2000))
            stack.enter_context(patch.object(d, "FETCH_MAX", 1000))
            stack.enter_context(patch.object(d, "SUMMARY_NUM", 200))
            stack.enter_context(patch.object(d, "SUMMARY_WORKERS", 2))
            stack.enter_context(patch.object(d, "_get_email_time_delta", return_value=2.0))
            d._log_startup_info()

        mock_print.assert_called_once()
