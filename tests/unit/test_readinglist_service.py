"""Unit tests for reading list service functions.

Tests reading list service functions using mocks to avoid database dependencies.
"""

from __future__ import annotations

from unittest.mock import patch


class TestGetUserReadinglist:
    """Tests for get_user_readinglist function."""

    def test_get_user_readinglist_no_user(self, app):
        """Test that get_user_readinglist returns empty dict when no user."""
        from backend.services.readinglist_service import get_user_readinglist

        with app.app_context():
            from flask import g

            g.user = None
            result = get_user_readinglist()
            assert result == {}

    @patch("backend.services.readinglist_service.ReadingListRepository")
    def test_get_user_readinglist_with_user(self, mock_repo, app):
        """Test get_user_readinglist with a logged in user."""
        from backend.services.readinglist_service import get_user_readinglist

        mock_repo.get_user_reading_list.return_value = {"2301.00001": {"added_time": 123}}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = get_user_readinglist()
            assert "2301.00001" in result
            mock_repo.get_user_reading_list.assert_called_once_with("test_user")

    @patch("backend.services.readinglist_service.ReadingListRepository")
    def test_get_user_readinglist_explicit_user(self, mock_repo, app):
        """Test get_user_readinglist with explicit user parameter."""
        from backend.services.readinglist_service import get_user_readinglist

        mock_repo.get_user_reading_list.return_value = {}

        with app.app_context():
            result = get_user_readinglist(user="explicit_user")
            mock_repo.get_user_reading_list.assert_called_once_with("explicit_user")


class TestUpdateSummaryStatus:
    """Tests for update_summary_status function."""

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_update_summary_status_item_not_found(self, mock_emit, mock_repo, app):
        """Test update_summary_status when item doesn't exist."""
        from backend.services.readinglist_service import update_summary_status

        mock_repo.get_reading_list_item.return_value = None

        with app.app_context():
            update_summary_status("test_user", "2301.00001", "running")
            mock_repo.update_reading_list_item.assert_not_called()
            mock_emit.assert_not_called()

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_update_summary_status_success(self, mock_emit, mock_repo, app):
        """Test successful summary status update."""
        from backend.services.readinglist_service import update_summary_status

        mock_repo.get_reading_list_item.return_value = {"pid": "2301.00001"}

        with app.app_context():
            update_summary_status("test_user", "2301.00001", "ok")
            mock_repo.update_reading_list_item.assert_called_once()
            mock_emit.assert_called_once()

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_update_summary_status_with_task_id(self, mock_emit, mock_repo, app):
        """Test summary status update with task_id."""
        from backend.services.readinglist_service import update_summary_status

        mock_repo.get_reading_list_item.return_value = {"pid": "2301.00001"}

        with app.app_context():
            update_summary_status("test_user", "2301.00001", "queued", task_id="task123")
            call_args = mock_repo.update_reading_list_item.call_args
            assert call_args is not None
            updates = call_args[0][2]
            assert updates.get("summary_task_id") == "task123"


class TestUpdateSummaryStatusDb:
    """Tests for update_summary_status_db function."""

    @patch("backend.services.readinglist_service.SummaryStatusRepository")
    @patch("backend.services.readinglist_service.emit_all_event")
    def test_update_summary_status_db_no_model(self, mock_emit, mock_repo, app):
        """Test update_summary_status_db with no model."""
        from backend.services.readinglist_service import update_summary_status_db

        with app.app_context():
            update_summary_status_db("2301.00001", None, "ok")
            mock_repo.set_status.assert_not_called()

    @patch("backend.services.readinglist_service.SummaryStatusRepository")
    @patch("backend.services.readinglist_service.emit_all_event")
    def test_update_summary_status_db_success(self, mock_emit, mock_repo, app):
        """Test successful summary status db update."""
        from backend.services.readinglist_service import update_summary_status_db

        with app.app_context():
            update_summary_status_db("2301.00001", "gpt-4", "ok")
            mock_repo.set_status.assert_called_once()
            mock_emit.assert_called_once()

    @patch("backend.services.readinglist_service.SummaryStatusRepository")
    @patch("backend.services.readinglist_service.emit_all_event")
    def test_update_summary_status_db_with_default_model(self, mock_emit, mock_repo, app):
        """Test summary status db update with default model."""
        from backend.services.readinglist_service import update_summary_status_db

        with app.app_context():
            update_summary_status_db("2301.00001", None, "ok", default_model="gpt-3.5")
            mock_repo.set_status.assert_called_once()


class TestTriggerSummaryAsync:
    """Tests for trigger_summary_async function."""

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", False)
    def test_trigger_summary_async_thread_fallback(self, app):
        """Test trigger_summary_async falls back to thread when queue unavailable."""
        from backend.services.readinglist_service import trigger_summary_async

        with app.app_context():
            result = trigger_summary_async(
                user="test_user",
                pid="2301.00001",
                model="gpt-4",
            )
            # Thread-based execution returns None
            assert result is None

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", True)
    @patch("backend.services.readinglist_service.enqueue_summary_task")
    def test_trigger_summary_async_with_queue(self, mock_enqueue, app):
        """Test trigger_summary_async uses task queue when available."""
        from backend.services.readinglist_service import trigger_summary_async

        mock_enqueue.return_value = "task123"

        with app.app_context():
            result = trigger_summary_async(
                user="test_user",
                pid="2301.00001",
                model="gpt-4",
            )
            assert result == "task123"
            mock_enqueue.assert_called_once()


class TestAddToReadingList:
    """Tests for add_to_readinglist function."""

    def test_add_to_readinglist_not_logged_in(self, app):
        """Test that add_to_readinglist returns error when not logged in."""
        from backend.services.readinglist_service import add_to_readinglist

        with app.app_context():
            from flask import g

            g.user = None
            result = add_to_readinglist("2301.00001")
            assert "error" in result or "Not logged in" in result.get("error", "")

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_add_to_readinglist_success(self, mock_emit, mock_repo, app):
        """Test successful addition to reading list."""
        from backend.services.readinglist_service import add_to_readinglist

        mock_repo.get_reading_list_item.return_value = None  # Not already in list
        mock_repo.add_to_reading_list.return_value = None

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_to_readinglist("2301.00001")
            mock_repo.add_to_reading_list.assert_called_once()
            assert result.get("pid") == "2301.00001"

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_add_to_readinglist_already_exists(self, mock_emit, mock_repo, app):
        """Test adding paper that already exists in reading list."""
        from backend.services.readinglist_service import add_to_readinglist

        mock_repo.get_reading_list_item.return_value = {"pid": "2301.00001", "top_tags": ["ml"]}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_to_readinglist("2301.00001")
            assert result.get("already_exists") is True


class TestRemoveFromReadingList:
    """Tests for remove_from_readinglist function."""

    def test_remove_from_readinglist_not_logged_in(self, app):
        """Test that remove_from_readinglist returns error when not logged in."""
        from backend.services.readinglist_service import remove_from_readinglist

        with app.app_context():
            from flask import g

            g.user = None
            result = remove_from_readinglist("2301.00001")
            assert "error" in result or "Not logged in" in result.get("error", "")

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_remove_from_readinglist_success(self, mock_emit, mock_repo, app):
        """Test successful removal from reading list."""
        from backend.services.readinglist_service import remove_from_readinglist

        mock_repo.remove_from_reading_list.return_value = True

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = remove_from_readinglist("2301.00001")
            mock_repo.remove_from_reading_list.assert_called_once()
            assert result.get("pid") == "2301.00001"

    @patch("backend.services.readinglist_service.ReadingListRepository")
    def test_remove_from_readinglist_not_found(self, mock_repo, app):
        """Test removing paper not in reading list."""
        from backend.services.readinglist_service import remove_from_readinglist

        mock_repo.remove_from_reading_list.return_value = False

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = remove_from_readinglist("2301.00001")
            assert "error" in result


class TestListReadinglist:
    """Tests for list_readinglist function."""

    def test_list_readinglist_not_logged_in(self, app):
        """Test that list_readinglist returns empty list when not logged in."""
        from backend.services.readinglist_service import list_readinglist

        with app.app_context():
            from flask import g

            g.user = None
            result = list_readinglist()
            assert result == []

    @patch("backend.services.readinglist_service.get_user_readinglist")
    def test_list_readinglist_success(self, mock_get, app):
        """Test successful reading list retrieval."""
        from backend.services.readinglist_service import list_readinglist

        mock_get.return_value = {
            "2301.00001": {"added_time": 200, "top_tags": ["ml"]},
            "2301.00002": {"added_time": 100, "top_tags": []},
        }

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = list_readinglist()
            assert len(result) == 2
            # Should be sorted by added_time descending
            assert result[0]["pid"] == "2301.00001"
            assert result[1]["pid"] == "2301.00002"
