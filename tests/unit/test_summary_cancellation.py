"""Unit tests for cooperative summary cancellation."""

from __future__ import annotations

from unittest.mock import patch


class TestGenerationEpoch:
    """Tests for per-(pid, model) generation epoch helpers."""

    def test_bump_and_get_generation_epoch(self):
        """Epoch starts at 0 and increments on bump."""

        class DummyDB(dict):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def items_with_prefix(self, prefix: str):
                return [(k, v) for k, v in self.items() if str(k).startswith(prefix)]

        db = DummyDB()

        def _get_db(flag="r", autocommit=True):
            return db

        with patch("aslite.repositories.get_summary_status_db", _get_db):
            from aslite.repositories import SummaryStatusRepository

            assert SummaryStatusRepository.get_generation_epoch("p", "m") == 0
            assert SummaryStatusRepository.bump_generation_epoch("p", "m") == 1
            assert SummaryStatusRepository.get_generation_epoch("p", "m") == 1
            assert SummaryStatusRepository.bump_generation_epoch("p", "m") == 2
            assert SummaryStatusRepository.get_generation_epoch("p", "m") == 2


class TestCancelSummaryTasks:
    """Tests for cancel_summary_tasks helper."""

    @patch("tasks._revoke_task_by_id")
    @patch("tasks.SummaryStatusRepository")
    def test_cancel_summary_tasks_cancels_task_ids(self, mock_repo, mock_revoke):
        """Cancel marks tasks canceled and bumps epoch."""
        mock_repo.get_status.return_value = {"status": "queued", "task_id": "t1", "task_user": "u"}
        mock_repo.get_items_with_prefix.return_value = [
            ("task::t2", {"status": "running", "pid": "p", "model": "m", "user": "u"}),
            ("task::t1", {"status": "queued", "pid": "p", "model": "m", "user": "u"}),
        ]
        mock_repo.bump_generation_epoch.return_value = 7
        mock_repo.get_task_status.return_value = {"user": "u"}

        import tasks

        res = tasks.cancel_summary_tasks("p", "m", user="u", reason="stop")

        assert res["epoch"] == 7
        assert res["canceled_task_ids"] == ["t1", "t2"]
        assert mock_repo.set_task_status.call_count == 2
        assert mock_revoke.call_count == 2
        mock_repo.set_status.assert_called()
