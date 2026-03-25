"""Unit tests for cooperative summary cancellation."""

from __future__ import annotations

from contextlib import contextmanager
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

            @contextmanager
            def transaction(self, mode: str = "IMMEDIATE"):
                """No-op transaction for testing."""
                yield self

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
        mock_repo.get_status.return_value = {
            "status": "queued",
            "task_id": "t1",
            "task_user": "u",
        }
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

    @patch("tasks._revoke_task_by_id")
    @patch("tasks.UploadedPaperRepository")
    @patch("tasks.SummaryStatusRepository")
    def test_cancel_summary_tasks_clears_upload_summary_task_id(self, mock_repo, mock_upload_repo, mock_revoke):
        """Cancel should clear upload summary_task_id when it matches the canceled task."""
        mock_repo.get_status.return_value = {
            "status": "queued",
            "task_id": "t1",
            "task_user": "alice",
        }
        mock_repo.get_items_with_prefix.return_value = []
        mock_repo.bump_generation_epoch.return_value = 9
        mock_repo.get_task_status.return_value = {"user": "alice"}
        mock_upload_repo.get.return_value = {"summary_task_id": "t1"}

        import tasks

        res = tasks.cancel_summary_tasks("up_secret_cancel", "m", user="alice")

        assert res["canceled_task_ids"] == ["t1"]
        mock_upload_repo.update.assert_called_with("up_secret_cancel", {"summary_task_id": None})
        mock_revoke.assert_called_once_with("t1")


class TestStaleSummaryRepair:
    def test_repair_stale_summary_tasks_repairs_stale_queued_and_clears_upload_pointer(self, monkeypatch, tmp_path):
        import tasks

        monkeypatch.setattr(tasks.time, "time", lambda: 1000.0)
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "get_all_items",
            lambda: iter(
                [
                    (
                        "up_secret_repair::m",
                        {
                            "status": "queued",
                            "updated_time": 0.0,
                            "task_id": "task_stale_q1",
                        },
                    )
                ]
            ),
        )

        def _items_with_prefix(prefix):
            if prefix == "task::":
                return iter(
                    [
                        (
                            "task::task_stale_q1",
                            {
                                "status": "queued",
                                "pid": "up_secret_repair",
                                "model": "m",
                                "user": "alice",
                                "updated_time": 0.0,
                            },
                        )
                    ]
                )
            return iter([])

        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "get_items_with_prefix",
            _items_with_prefix,
        )

        status_updates = []
        task_updates = []
        upload_updates = []
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "update_status",
            lambda pid, model, updates: status_updates.append((pid, model, dict(updates))),
        )
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "set_task_status",
            lambda task_id, status, error=None, **extra: task_updates.append((task_id, status, error, dict(extra))),
        )
        monkeypatch.setattr(
            tasks.UploadedPaperRepository,
            "get",
            lambda _pid: {"summary_task_id": "task_stale_q1"},
        )
        monkeypatch.setattr(
            tasks.UploadedPaperRepository,
            "update",
            lambda pid, patch: upload_updates.append((pid, dict(patch))) or True,
        )
        monkeypatch.setattr(
            tasks,
            "summary_cache_paths",
            lambda *_a, **_k: (
                tmp_path / "body.md",
                tmp_path / "body.meta.json",
                tmp_path / ".model.lock",
                tmp_path / "legacy.md",
                tmp_path / "legacy.meta.json",
                tmp_path / ".legacy.lock",
            ),
        )
        monkeypatch.setattr(tasks, "_update_readinglist_summary_status", lambda *_a, **_k: None)

        repaired = tasks.repair_stale_summary_tasks(max_age_s=10, requeue=False)

        assert repaired == 1
        assert status_updates == [
            (
                "up_secret_repair",
                "m",
                {
                    "status": "failed",
                    "last_error": "stale_queued_repaired",
                    "task_id": None,
                    "task_user": None,
                    "resolved_model": None,
                    "updated_time": 1000.0,
                },
            )
        ]
        assert task_updates[0][0] == "task_stale_q1"
        assert task_updates[0][1] == "failed"
        assert task_updates[0][2] == "stale_queued_repaired"
        assert upload_updates == [("up_secret_repair", {"summary_task_id": None})]

    def test_repair_stale_summary_tasks_skips_active_lock(self, monkeypatch, tmp_path):
        import tasks

        monkeypatch.setattr(tasks.time, "time", lambda: 1000.0)
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "get_all_items",
            lambda: iter(
                [
                    (
                        "2301.00001::m",
                        {
                            "status": "queued",
                            "updated_time": 0.0,
                            "task_id": "task_still_valid",
                        },
                    )
                ]
            ),
        )
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "get_items_with_prefix",
            lambda _prefix: iter([]),
        )
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "update_status",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not repair")),
        )
        monkeypatch.setattr(
            tasks.SummaryStatusRepository,
            "set_task_status",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not repair")),
        )

        monkeypatch.setattr(
            "backend.services.summary_service.has_active_summary_lock",
            lambda *_a, **_k: True,
        )
        monkeypatch.setattr(tasks, "_update_readinglist_summary_status", lambda *_a, **_k: None)

        repaired = tasks.repair_stale_summary_tasks(max_age_s=10, requeue=False)

        assert repaired == 0


class TestSummaryStatusReadRepair:
    def test_get_summary_status_ignores_stale_orphan_lock(self, monkeypatch, tmp_path):
        from backend.services import summary_service

        pid = "2301.00001"
        model = "test-model"
        cache_dir = tmp_path / pid
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_file = cache_dir / f".{model}.lock"
        lock_file.write_text("999999\n0\n", encoding="utf-8")
        legacy_lock = tmp_path / f".{pid}.lock"

        def _paths(_pid, _model):
            return (
                cache_dir / f"{_model}.md",
                cache_dir / f"{_model}.meta.json",
                lock_file,
                tmp_path / f"{_pid}.md",
                tmp_path / f"{_pid}.meta.json",
                legacy_lock,
            )

        monkeypatch.setattr(summary_service, "summary_cache_paths", _paths)
        monkeypatch.setattr(
            summary_service.SummaryStatusRepository,
            "get_status",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(summary_service.settings.lock, "summary_lock_stale_sec", 1.0)
        monkeypatch.setattr(summary_service.time, "time", lambda: 1000.0)
        monkeypatch.setattr(
            summary_service.os,
            "kill",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("dead")),
        )

        lock_file.touch()
        import os as _os

        _os.utime(lock_file, (0.0, 0.0))

        status, last_error = summary_service.get_summary_status(pid, model)

        assert status == ""
        assert last_error is None
        assert not lock_file.exists()

    def test_get_summary_status_repairs_stale_queued_record(self, monkeypatch, tmp_path):
        from backend.services import summary_service

        pid = "up_secret_read_repair"
        model = "test-model"
        cache_dir = tmp_path / pid
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _paths(_pid, _model):
            return (
                cache_dir / f"{_model}.md",
                cache_dir / f"{_model}.meta.json",
                cache_dir / f".{_model}.lock",
                tmp_path / f"{_pid}.md",
                tmp_path / f"{_pid}.meta.json",
                tmp_path / f".{_pid}.lock",
            )

        monkeypatch.setattr(summary_service, "summary_cache_paths", _paths)
        monkeypatch.setattr(summary_service.time, "time", lambda: 1000.0)
        monkeypatch.setattr(summary_service.settings.huey, "summary_repair_ttl", 10)
        monkeypatch.setattr(
            summary_service.SummaryStatusRepository,
            "get_status",
            lambda *_a, **_k: {
                "status": "queued",
                "last_error": None,
                "task_id": "task_stale_q2",
                "task_user": "alice",
                "updated_time": 0.0,
            },
        )
        monkeypatch.setattr(
            summary_service.SummaryStatusRepository,
            "get_task_status",
            lambda *_a, **_k: {
                "status": "queued",
                "updated_time": 0.0,
            },
        )

        status_writes = []
        task_writes = []
        upload_updates = []
        monkeypatch.setattr(
            summary_service.SummaryStatusRepository,
            "set_status",
            lambda pid, model, status, error=None, **extra: status_writes.append(
                (pid, model, status, error, dict(extra))
            ),
        )
        monkeypatch.setattr(
            summary_service.SummaryStatusRepository,
            "set_task_status",
            lambda task_id, status, error=None, **extra: task_writes.append((task_id, status, error, dict(extra))),
        )
        monkeypatch.setattr(
            summary_service.UploadedPaperRepository,
            "get",
            lambda _pid: {"summary_task_id": "task_stale_q2"},
        )
        monkeypatch.setattr(
            summary_service.UploadedPaperRepository,
            "update",
            lambda pid, patch: upload_updates.append((pid, dict(patch))) or True,
        )

        status, last_error = summary_service.get_summary_status(pid, model)

        assert status == "failed"
        assert last_error == "stale_queued_repaired"
        assert status_writes == [
            (
                pid,
                model,
                "failed",
                "stale_queued_repaired",
                {"task_id": None, "task_user": None},
            )
        ]
        assert task_writes[0][0] == "task_stale_q2"
        assert task_writes[0][1] == "failed"
        assert task_writes[0][2] == "stale_queued_repaired"
        assert upload_updates == [(pid, {"summary_task_id": None})]
