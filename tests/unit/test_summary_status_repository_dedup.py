"""Unit tests for SummaryStatusRepository de-duplicated writes."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch


class _DummyDB(dict):
    def __init__(self):
        super().__init__()
        self.write_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __setitem__(self, key, value):
        self.write_count += 1
        return super().__setitem__(key, value)

    @contextmanager
    def transaction(self, mode: str = "IMMEDIATE"):
        yield self


def test_set_status_skips_write_when_no_effective_change():
    """set_status does not write when the stored fields do not change."""
    db = _DummyDB()

    def _get_db(flag="r", autocommit=True):
        return db

    with patch("aslite.repositories.get_summary_status_db", _get_db):
        from aslite.repositories import SummaryStatusRepository

        with patch("aslite.repositories.time.time", lambda: 100.0):
            SummaryStatusRepository.set_status("p1", "m1", "queued", None, task_id="t1")
        assert db.write_count == 1
        first = db.get("p1::m1") or {}
        assert first.get("updated_time") == 100.0

        # No-op update: same status + same extras -> should not write / bump updated_time.
        with patch("aslite.repositories.time.time", lambda: 200.0):
            SummaryStatusRepository.set_status("p1", "m1", "queued", None, task_id="t1")
        assert db.write_count == 1
        second = db.get("p1::m1") or {}
        assert second.get("updated_time") == 100.0

        # Same status but new field value -> should write.
        with patch("aslite.repositories.time.time", lambda: 300.0):
            SummaryStatusRepository.set_status("p1", "m1", "queued", None, task_id="t2")
        assert db.write_count == 2
        third = db.get("p1::m1") or {}
        assert third.get("task_id") == "t2"
        assert third.get("updated_time") == 300.0


def test_set_task_status_skips_write_when_no_effective_change():
    """set_task_status does not write when the stored fields do not change."""
    db = _DummyDB()

    def _get_db(flag="r", autocommit=True):
        return db

    with patch("aslite.repositories.get_summary_status_db", _get_db):
        from aslite.repositories import SummaryStatusRepository

        with patch("aslite.repositories.time.time", lambda: 10.0):
            SummaryStatusRepository.set_task_status("t1", "queued", None, pid="p1", model="m1", user="u1", priority=10)
        assert db.write_count == 1
        first = db.get("task::t1") or {}
        assert first.get("updated_time") == 10.0

        # No-op update: same payload -> should not write.
        with patch("aslite.repositories.time.time", lambda: 20.0):
            SummaryStatusRepository.set_task_status("t1", "queued", None, pid="p1", model="m1", user="u1", priority=10)
        assert db.write_count == 1
        second = db.get("task::t1") or {}
        assert second.get("updated_time") == 10.0

        # Same status but different error -> should write.
        with patch("aslite.repositories.time.time", lambda: 30.0):
            SummaryStatusRepository.set_task_status(
                "t1", "queued", "boom", pid="p1", model="m1", user="u1", priority=10
            )
        assert db.write_count == 2
        third = db.get("task::t1") or {}
        assert third.get("error") == "boom"
        assert third.get("updated_time") == 30.0


def test_set_status_backfills_missing_updated_time_even_on_noop_update():
    """Older entries without updated_time should be backfilled on the next set_status call."""
    db = _DummyDB()

    # Preload an "old" record without updated_time (avoid counting as a write).
    dict.__setitem__(db, "p1::m1", {"status": "queued", "task_id": "t1"})

    def _get_db(flag="r", autocommit=True):
        return db

    with patch("aslite.repositories.get_summary_status_db", _get_db):
        from aslite.repositories import SummaryStatusRepository

        with patch("aslite.repositories.time.time", lambda: 123.0):
            SummaryStatusRepository.set_status("p1", "m1", "queued", None, task_id="t1")
        assert db.write_count == 1
        out = db.get("p1::m1") or {}
        assert out.get("updated_time") == 123.0


def test_set_task_status_backfills_missing_updated_time_even_on_noop_update():
    """Older task:: entries without updated_time should be backfilled on the next set_task_status call."""
    db = _DummyDB()

    # Preload an "old" record without updated_time (avoid counting as a write).
    dict.__setitem__(db, "task::t1", {"status": "queued", "pid": "p1", "model": "m1", "user": "u1", "priority": 10})

    def _get_db(flag="r", autocommit=True):
        return db

    with patch("aslite.repositories.get_summary_status_db", _get_db):
        from aslite.repositories import SummaryStatusRepository

        with patch("aslite.repositories.time.time", lambda: 456.0):
            SummaryStatusRepository.set_task_status("t1", "queued", None, pid="p1", model="m1", user="u1", priority=10)
        assert db.write_count == 1
        out = db.get("task::t1") or {}
        assert out.get("updated_time") == 456.0
