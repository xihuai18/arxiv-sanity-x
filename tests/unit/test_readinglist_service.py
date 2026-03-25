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

    @patch("backend.services.readinglist_service.SummaryStatusRepository")
    @patch("backend.services.readinglist_service.emit_all_event")
    def test_update_summary_status_db_ok_with_resolved_model(self, mock_emit, mock_repo, app):
        """Successful status should forward resolved_model."""
        from backend.services.readinglist_service import update_summary_status_db

        with app.app_context():
            update_summary_status_db("2301.00001", "gpt-4", "ok", resolved_model="fallback-model")
            call = mock_repo.set_status.call_args
            assert call is not None
            assert call.kwargs.get("resolved_model") == "fallback-model"
            payload = mock_emit.call_args[0][0]
            assert payload.get("resolved_model") == "fallback-model"

    @patch("backend.services.readinglist_service.SummaryStatusRepository")
    @patch("backend.services.readinglist_service.emit_all_event")
    def test_update_summary_status_db_redacts_global_error_payload(self, mock_emit, mock_repo, app):
        """Global SSE payload should not include raw internal error text."""
        from backend.services.readinglist_service import update_summary_status_db

        with app.app_context():
            update_summary_status_db("2301.00001", "gpt-4", "failed", "internal stack trace")
            payload = mock_emit.call_args[0][0]
            assert payload.get("status") == "failed"
            assert payload.get("error") == "failed"

    @patch("backend.services.readinglist_service.SummaryStatusRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    @patch("backend.services.readinglist_service.emit_all_event")
    def test_update_summary_status_db_upload_events_are_owner_scoped(
        self, mock_emit_all, mock_emit_user, mock_repo, app
    ):
        """Uploaded paper summary events should only go to the owner."""
        from backend.services.readinglist_service import update_summary_status_db

        with app.app_context():
            update_summary_status_db("up_secret002", "gpt-4", "failed", "private detail", task_user="alice")

        mock_repo.set_status.assert_called_once()
        mock_emit_all.assert_not_called()
        mock_emit_user.assert_called_once()
        payload = mock_emit_user.call_args[0][1]
        assert payload.get("error") == "private detail"


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

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", False)
    def test_trigger_summary_async_thread_fallback_invalid_summary_marks_failed(self, app, monkeypatch):
        """Thread fallback should not mark ok for invalid summary output."""
        import backend.services.readinglist_service as rs

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        monkeypatch.setattr(rs.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(rs.settings.huey, "allow_thread_fallback", True)
        monkeypatch.setattr(rs.SummaryStatusRepository, "get_generation_epoch", lambda *_a, **_k: 0)

        db_calls = []

        def _update_db(pid, model, status, error, **extra):
            db_calls.append((pid, model, status, error, extra))

        with app.app_context():
            task_id = rs.trigger_summary_async(
                user="test_user",
                pid="2301.00001",
                model="gpt-4",
                generate_summary_fn=lambda *_a, **_k: ("# Error\n\nboom", {}),
                update_db_fn=_update_db,
                default_model="gpt-4",
            )

        assert task_id is None
        assert db_calls
        assert db_calls[-1][2] == "failed"

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", False)
    def test_trigger_summary_async_thread_fallback_records_resolved_model(self, app, monkeypatch):
        """Thread fallback should persist resolved_model for successful fallback output."""
        import backend.services.readinglist_service as rs

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        monkeypatch.setattr(rs.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(rs.settings.huey, "allow_thread_fallback", True)
        monkeypatch.setattr(rs.SummaryStatusRepository, "get_generation_epoch", lambda *_a, **_k: 0)

        db_calls = []

        def _update_db(pid, model, status, error, **extra):
            db_calls.append((pid, model, status, error, extra))

        long_body = " ".join(["detail"] * 80)
        valid_summary = f"# Title\n\n## TL;DR\n\nhello\n\n## Body\n\n{long_body}"
        with app.app_context():
            task_id = rs.trigger_summary_async(
                user="test_user",
                pid="2301.00001",
                model="requested-model",
                generate_summary_fn=lambda *_a, **_k: (
                    valid_summary,
                    {"llm_model": "fallback-model"},
                ),
                update_db_fn=_update_db,
                default_model="requested-model",
            )

        assert task_id is None
        assert db_calls
        assert db_calls[-1][2] == "ok"
        assert db_calls[-1][4].get("resolved_model") == "fallback-model"

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", False)
    def test_trigger_summary_async_thread_fallback_preserves_task_user_and_queued_status(self, app, monkeypatch):
        """Thread fallback should write queued/running/ok with owner context."""
        import backend.services.readinglist_service as rs

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        monkeypatch.setattr(rs.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(rs.settings.huey, "allow_thread_fallback", True)
        monkeypatch.setattr(rs.SummaryStatusRepository, "get_generation_epoch", lambda *_a, **_k: 0)

        db_calls = []

        def _update_db(pid, model, status, error, **extra):
            db_calls.append((pid, model, status, error, dict(extra)))

        long_body = " ".join(["detail"] * 80)
        valid_summary = f"# Title\n\n## TL;DR\n\nhello\n\n## Body\n\n{long_body}"
        with app.app_context():
            task_id = rs.trigger_summary_async(
                user="alice",
                pid="up_secret_thread_success",
                model="requested-model",
                generate_summary_fn=lambda *_a, **_k: (
                    valid_summary,
                    {"llm_model": "fallback-model"},
                ),
                update_db_fn=_update_db,
                default_model="requested-model",
            )

        assert task_id is None
        assert [call[2] for call in db_calls] == ["queued", "running", "ok"]
        assert all(call[4].get("task_user") == "alice" for call in db_calls)
        assert db_calls[-1][4].get("resolved_model") == "fallback-model"

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", False)
    def test_trigger_summary_async_thread_fallback_failure_preserves_task_user(self, app, monkeypatch):
        """Thread fallback failures should remain owner-scoped for upload summaries."""
        import backend.services.readinglist_service as rs

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        monkeypatch.setattr(rs.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(rs.settings.huey, "allow_thread_fallback", True)
        monkeypatch.setattr(rs.SummaryStatusRepository, "get_generation_epoch", lambda *_a, **_k: 0)

        db_calls = []

        def _update_db(pid, model, status, error, **extra):
            db_calls.append((pid, model, status, error, dict(extra)))

        with app.app_context():
            task_id = rs.trigger_summary_async(
                user="alice",
                pid="up_secret_thread_fail",
                model="requested-model",
                generate_summary_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
                update_db_fn=_update_db,
                default_model="requested-model",
            )

        assert task_id is None
        assert [call[2] for call in db_calls] == ["queued", "running", "failed"]
        assert all(call[4].get("task_user") == "alice" for call in db_calls)

    @patch("backend.services.readinglist_service._TASK_QUEUE_AVAILABLE", False)
    def test_trigger_summary_async_thread_fallback_cancel_marks_canceled(self, app, monkeypatch):
        """Thread fallback should preserve canceled semantics from summary service."""
        import backend.services.readinglist_service as rs

        class _ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

        monkeypatch.setattr(rs.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(rs.settings.huey, "allow_thread_fallback", True)
        monkeypatch.setattr(rs.SummaryStatusRepository, "get_generation_epoch", lambda *_a, **_k: 0)

        db_calls = []

        def _update_db(pid, model, status, error, **extra):
            db_calls.append((pid, model, status, error, dict(extra)))

        with app.app_context():
            task_id = rs.trigger_summary_async(
                user="alice",
                pid="up_secret_thread_cancel",
                model="requested-model",
                generate_summary_fn=lambda *_a, **_k: (
                    "# Error\n\nSummary canceled.",
                    {},
                ),
                update_db_fn=_update_db,
                default_model="requested-model",
            )

        assert task_id is None
        assert [call[2] for call in db_calls] == ["queued", "running", "canceled"]
        assert db_calls[-1][3] == "Canceled by user"


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
    def test_add_to_readinglist_accepts_explicit_user(self, mock_emit, mock_repo, app, monkeypatch):
        from backend.services.readinglist_service import add_to_readinglist

        mock_repo.get_reading_list_item.return_value = None
        mock_repo.add_to_reading_list.return_value = None
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "",
        )

        with app.app_context():
            result = add_to_readinglist("2301.00001", user="explicit_user")

        mock_repo.add_to_reading_list.assert_called_once()
        assert mock_repo.add_to_reading_list.call_args[0][0] == "explicit_user"
        assert result.get("pid") == "2301.00001"

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_add_to_readinglist_already_exists(self, mock_emit, mock_repo, app):
        """Test adding paper that already exists in reading list."""
        from backend.services.readinglist_service import add_to_readinglist

        mock_repo.get_reading_list_item.return_value = {
            "pid": "2301.00001",
            "top_tags": ["ml"],
        }

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_to_readinglist("2301.00001")
            assert result.get("already_exists") is True

    @patch("backend.services.readinglist_service.ReadingListRepository")
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_add_to_readinglist_does_not_retrigger_when_already_queued(self, mock_emit, mock_repo, app):
        from backend.services.readinglist_service import add_to_readinglist

        calls = []
        mock_repo.get_reading_list_item.return_value = {
            "pid": "2301.00001",
            "top_tags": ["ml"],
            "summary_status": "queued",
            "summary_task_id": "task123",
        }

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_to_readinglist(
                "2301.00001",
                trigger_summary_fn=lambda *_a, **_k: calls.append("triggered"),
            )

        assert result.get("already_exists") is True
        assert result.get("task_id") == "task123"
        assert calls == []
        mock_repo.update_reading_list_item.assert_not_called()


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


class TestListReadingList:
    def test_list_readinglist_prefers_global_summary_status(self, app, monkeypatch):
        from backend.services.readinglist_service import list_readinglist

        monkeypatch.setattr("backend.services.data_service.paper_exists", lambda pid: True)
        monkeypatch.setattr(
            "backend.services.readinglist_service.get_user_readinglist",
            lambda user=None: {
                "2301.00001": {
                    "added_time": 1,
                    "top_tags": [],
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "stale-task",
                }
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status_many",
            lambda pids, model=None: {
                pid: {
                    "status": "ok",
                    "updated_time": 2,
                    "task_user": "alice",
                    "task_id": "new-task",
                }
                for pid in pids
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )
        monkeypatch.setattr(
            "backend.services.data_service.paper_exists",
            lambda _pid: True,
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_render_snapshots",
            lambda pids, include_tldr=False, prefetched_status_rows=None: {
                pid: {"status": "ok", "last_error": None} for pid in pids
            },
        )

        with app.app_context():
            from flask import g

            g.user = "alice"
            items = list_readinglist()

        assert items[0]["summary_status"] == "ok"
        assert items[0]["summary_task_id"] is None

    def test_list_readinglist_uses_prefetched_status_rows(self, app, monkeypatch):
        from backend.services.readinglist_service import list_readinglist

        monkeypatch.setattr(
            "backend.services.readinglist_service.get_user_readinglist",
            lambda user=None: {
                "2301.00001": {
                    "added_time": 2,
                    "top_tags": [],
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "stale-task",
                },
                "2301.00002": {
                    "added_time": 1,
                    "top_tags": [],
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "stale-task-2",
                },
            },
        )
        calls = {"count": 0}

        def _fake_get_status_many(pids, model=None):
            calls["count"] += 1
            return {
                pid: {
                    "status": "ok",
                    "updated_time": 2,
                    "task_user": "alice",
                    "task_id": f"task-{pid}",
                }
                for pid in pids
            }

        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status_many",
            _fake_get_status_many,
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("single get_status should not be used")),
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )
        monkeypatch.setattr(
            "backend.services.data_service.paper_exists",
            lambda _pid: True,
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_render_snapshots",
            lambda pids, include_tldr=False, prefetched_status_rows=None: {
                pid: {"status": "ok", "last_error": None} for pid in pids
            },
        )

        with app.app_context():
            from flask import g

            g.user = "alice"
            items = list_readinglist()

        assert calls["count"] == 1
        assert [item["pid"] for item in items] == ["2301.00001", "2301.00002"]

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
    @patch("backend.services.readinglist_service.emit_user_event")
    def test_remove_from_readinglist_accepts_explicit_user(self, mock_emit, mock_repo, app):
        from backend.services.readinglist_service import remove_from_readinglist

        mock_repo.remove_from_reading_list.return_value = True

        with app.app_context():
            result = remove_from_readinglist("2301.00001", user="explicit_user")

        mock_repo.remove_from_reading_list.assert_called_once_with("explicit_user", "2301.00001")
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
    @patch("backend.services.data_service.paper_exists")
    def test_list_readinglist_success(self, mock_exists, mock_get, app):
        """Test successful reading list retrieval."""
        from backend.services.readinglist_service import list_readinglist

        mock_exists.return_value = True
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

    @patch("backend.services.readinglist_service.get_user_readinglist")
    @patch("backend.services.data_service.paper_exists")
    def test_list_readinglist_accepts_explicit_user(self, mock_exists, mock_get, app):
        from backend.services.readinglist_service import list_readinglist

        mock_exists.return_value = True
        mock_get.return_value = {"2301.00001": {"added_time": 200, "top_tags": []}}

        with app.app_context():
            result = list_readinglist(user="explicit_user")

        mock_get.assert_called_once_with("explicit_user")
        assert [item["pid"] for item in result] == ["2301.00001"]

    @patch("backend.services.readinglist_service.get_user_readinglist")
    @patch("backend.services.data_service.paper_exists")
    def test_list_readinglist_skips_missing_public_pids(self, mock_exists, mock_get, app):
        """Missing/tombstoned public papers should be hidden from API results."""
        from backend.services.readinglist_service import list_readinglist

        mock_exists.side_effect = lambda pid: pid == "2301.00001"
        mock_get.return_value = {
            "2301.00001": {"added_time": 200, "top_tags": ["ml"]},
            "2301.00002": {"added_time": 100, "top_tags": []},
        }

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = list_readinglist()

        assert [item["pid"] for item in result] == ["2301.00001"]


class TestReadingListLegacyAlignment:
    def test_readinglist_page_prefers_global_summary_status(self, app, monkeypatch):
        from backend import legacy

        monkeypatch.setattr(legacy, "default_context", lambda: {})
        monkeypatch.setattr(
            legacy,
            "get_readinglist",
            lambda: {
                "2301.00001": {
                    "added_time": 1,
                    "top_tags": [],
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "stale-task",
                }
            },
        )
        monkeypatch.setattr(
            legacy,
            "get_papers_bulk",
            lambda _pids: {"2301.00001": {"_rawid": "2301.00001", "title": "Paper", "authors": []}},
        )
        monkeypatch.setattr(legacy, "get_tags", lambda: {})
        monkeypatch.setattr(legacy, "get_neg_tags", lambda: {})
        monkeypatch.setattr(legacy, "render_pid", lambda pid, **_kwargs: {"id": pid})
        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status_many",
            lambda pids, model=None: {
                pid: {
                    "status": "ok",
                    "updated_time": 2,
                    "task_user": "alice",
                    "task_id": "fresh-task",
                }
                for pid in pids
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_render_snapshots",
            lambda pids, include_tldr=True, prefetched_status_rows=None: {
                pid: {"status": "ok", "last_error": None, "tldr": ""} for pid in pids
            },
        )
        captured = {}

        def _fake_render_template(_template, **context):
            captured["context"] = context
            return context

        monkeypatch.setattr(legacy, "render_template", _fake_render_template)

        with app.test_request_context("/reading-list"):
            from flask import g

            g.user = "alice"
            legacy.readinglist_page()

        papers = (captured.get("context") or {}).get("papers") or []
        assert papers[0]["summary_status"] == "ok"
        assert papers[0]["summary_task_id"] is None

    def test_list_readinglist_prefetches_upload_records_for_upload_items(self, app, monkeypatch):
        from backend.services.readinglist_service import list_readinglist

        monkeypatch.setattr(
            "backend.services.readinglist_service.get_user_readinglist",
            lambda user=None: {
                "up_abc123def456": {
                    "added_time": 1,
                    "top_tags": [],
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "task-upload",
                }
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.UploadedPaperRepository.get_by_owner_for_pids",
            lambda owner, pids: {
                "up_abc123def456": {
                    "owner": owner,
                    "parse_status": "ok",
                    "parse_error": "",
                }
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.UploadedPaperRepository.get",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("single upload get should not be used")),
        )
        monkeypatch.setattr(
            "backend.services.upload_service._normalize_upload_parse_status",
            lambda _pid, _record: ("ok", ""),
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_render_snapshots",
            lambda pids, include_tldr=False, prefetched_status_rows=None: {
                pid: {"status": "ok", "last_error": None} for pid in pids
            },
        )

        with app.app_context():
            from flask import g

            g.user = "alice"
            items = list_readinglist()

        assert items[0]["pid"] == "up_abc123def456"
        assert items[0]["summary_status"] == "ok"

    def test_list_readinglist_falls_back_when_upload_batch_lookup_fails(self, app, monkeypatch):
        from backend.services.readinglist_service import list_readinglist

        monkeypatch.setattr(
            "backend.services.readinglist_service.get_user_readinglist",
            lambda user=None: {
                "up_abc123def456": {
                    "added_time": 1,
                    "top_tags": [],
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "task-upload",
                }
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.UploadedPaperRepository.get_by_owner_for_pids",
            lambda owner, pids: (_ for _ in ()).throw(RuntimeError("db busy")),
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.UploadedPaperRepository.get",
            lambda pid: {
                "owner": "alice",
                "parse_status": "ok",
                "parse_error": "",
            },
        )
        monkeypatch.setattr(
            "backend.services.upload_service._normalize_upload_parse_status",
            lambda _pid, _record: ("ok", ""),
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_render_snapshots",
            lambda pids, include_tldr=False, prefetched_status_rows=None: {
                pid: {"status": "ok", "last_error": None} for pid in pids
            },
        )

        with app.app_context():
            from flask import g

            g.user = "alice"
            items = list_readinglist()

        assert items[0]["pid"] == "up_abc123def456"
        assert items[0]["summary_status"] == "ok"

    def test_overlay_summary_status_clears_hidden_foreign_task_id(self, app, monkeypatch):
        from backend.services.readinglist_service import overlay_summary_status_for_user

        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status",
            lambda pid, model=None: {
                "status": "queued",
                "updated_time": 2,
                "task_user": "other-user",
                "task_id": "foreign-task",
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )
        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_status",
            lambda pid, model=None: ("queued", None),
        )

        with app.app_context():
            item = overlay_summary_status_for_user(
                "alice",
                "2301.00001",
                {
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "stale-task",
                },
            )

        assert item["summary_status"] == "queued"
        assert item["summary_task_id"] is None

    def test_overlay_summary_status_uses_repaired_global_state(self, app, monkeypatch):
        from backend.services.readinglist_service import overlay_summary_status_for_user

        calls = []

        def _fake_get_summary_status(pid, model=None):
            calls.append((pid, model))
            return "failed", "stale_queued_repaired"

        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_status",
            _fake_get_summary_status,
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status",
            lambda pid, model=None: {
                "status": "queued",
                "updated_time": 2,
                "task_user": "alice",
                "task_id": "stale-task",
                "last_error": None,
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )

        with app.app_context():
            item = overlay_summary_status_for_user(
                "alice",
                "2301.00001",
                {
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "stale-task",
                },
            )

        assert calls == [("2301.00001", "gpt-5.4")]
        assert item["summary_status"] == "failed"
        assert item["summary_last_error"] == "stale_queued_repaired"
        assert item["summary_task_id"] is None

    def test_overlay_summary_status_keeps_existing_state_on_probe_failure(self, app, monkeypatch):
        from backend.services.readinglist_service import overlay_summary_status_for_user

        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status",
            lambda pid, model=None: {
                "status": "queued",
                "updated_time": 2,
                "task_user": "alice",
                "task_id": "task-current",
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )

        def _raise_summary_probe(pid, model=None):
            raise RuntimeError("transient summary probe failure")

        monkeypatch.setattr(
            "backend.services.summary_service.get_summary_status",
            _raise_summary_probe,
        )

        with app.app_context():
            item = overlay_summary_status_for_user(
                "alice",
                "2301.00001",
                {
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "task-current",
                },
            )

        assert item["summary_status"] == "queued"
        assert item["summary_last_error"] is None
        assert item["summary_task_id"] == "task-current"

    def test_overlay_summary_status_ignores_stale_summary_for_upload_waiting_parse(self, app, monkeypatch):
        from backend.services.readinglist_service import overlay_summary_status_for_user

        monkeypatch.setattr(
            "backend.services.readinglist_service.UploadedPaperRepository.get",
            lambda pid: {
                "owner": "alice",
                "parse_status": "pending",
                "parse_error": "",
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service.SummaryStatusRepository.get_status",
            lambda pid, model=None: {
                "status": "running",
                "updated_time": 2,
                "task_user": "alice",
                "task_id": "old-task",
            },
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )

        with app.app_context():
            item = overlay_summary_status_for_user(
                "alice",
                "up_waitparse001",
                {
                    "summary_status": "running",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "old-task",
                },
            )

        assert item["summary_status"] == ""
        assert item["summary_task_id"] is None

    def test_overlay_summary_status_for_upload_keeps_ok_state_via_single_record_fallback(self, app, monkeypatch):
        from backend.services.readinglist_service import overlay_summary_status_for_user

        monkeypatch.setattr(
            "backend.services.readinglist_service.UploadedPaperRepository.get",
            lambda pid: {
                "owner": "alice",
                "parse_status": "ok",
                "parse_error": "",
            },
        )
        monkeypatch.setattr(
            "backend.services.upload_service._normalize_upload_parse_status",
            lambda _pid, _record: ("ok", ""),
        )
        monkeypatch.setattr(
            "backend.services.readinglist_service._default_summary_model",
            lambda: "gpt-5.4",
        )

        with app.app_context():
            item = overlay_summary_status_for_user(
                "alice",
                "up_ready001",
                {
                    "summary_status": "queued",
                    "summary_last_error": None,
                    "summary_updated_time": 1,
                    "summary_task_id": "task-current",
                },
                prefetched_status_info={
                    "status": "ok",
                    "updated_time": 2,
                    "task_user": "alice",
                },
                prefetched_summary_snapshot={
                    "status": "ok",
                    "last_error": None,
                },
            )

        assert item["summary_status"] == "ok"
        assert item["summary_task_id"] is None
