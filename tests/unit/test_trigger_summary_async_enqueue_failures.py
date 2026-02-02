"""Unit tests for enqueue failure handling in trigger_summary_async."""

from __future__ import annotations


def test_trigger_summary_async_lock_timeout_marks_failed(monkeypatch):
    import backend.services.readinglist_service as rs
    import tasks

    # Avoid DB work in fast-path checks.
    monkeypatch.setattr(tasks.SummaryStatusRepository, "get_status", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "acquire_summary_lock", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "_find_active_task", lambda *_a, **_k: (None, None))

    calls = {"db": [], "rl": []}

    def update_db_fn(pid, model, status, error, **_extra):
        calls["db"].append((pid, model, status, error))

    def update_readinglist_fn(user, pid, status, error, task_id=None):
        calls["rl"].append((user, pid, status, error, task_id))

    task_id = rs.trigger_summary_async(
        user="u",
        pid="2301.00001",
        model="m",
        update_readinglist_fn=update_readinglist_fn,
        update_db_fn=update_db_fn,
        default_model="m",
    )

    assert task_id is None
    assert calls["db"]
    assert calls["db"][-1][2] == "failed"
    assert "enqueue_lock_timeout" in (calls["db"][-1][3] or "")


def test_trigger_summary_async_missing_task_id_marks_failed(monkeypatch):
    import backend.services.readinglist_service as rs
    import tasks

    monkeypatch.setattr(tasks.SummaryStatusRepository, "get_status", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks.SummaryStatusRepository, "get_generation_epoch", lambda *_a, **_k: 0)
    monkeypatch.setattr(tasks, "_find_active_task", lambda *_a, **_k: (None, None))
    monkeypatch.setattr(tasks, "acquire_summary_lock", lambda *_a, **_k: 1)
    monkeypatch.setattr(tasks, "release_summary_lock", lambda *_a, **_k: None)

    class DummyTask:
        id = None
        priority = None

    class DummyGenerate:
        @staticmethod
        def s(*_a, **_k):
            return DummyTask()

    class DummyResult:
        id = None

    monkeypatch.setattr(tasks, "generate_summary_task", DummyGenerate())
    monkeypatch.setattr(tasks.huey, "enqueue", lambda *_a, **_k: DummyResult())

    calls = {"db": [], "rl": []}

    def update_db_fn(pid, model, status, error, **_extra):
        calls["db"].append((pid, model, status, error))

    def update_readinglist_fn(user, pid, status, error, task_id=None):
        calls["rl"].append((user, pid, status, error, task_id))

    task_id = rs.trigger_summary_async(
        user="u",
        pid="2301.00001",
        model="m",
        update_readinglist_fn=update_readinglist_fn,
        update_db_fn=update_db_fn,
        default_model="m",
    )

    assert task_id is None
    assert calls["db"]
    assert calls["db"][-1][2] == "failed"
    assert "enqueue_task_id_missing" in (calls["db"][-1][3] or "")
