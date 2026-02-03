"""Unit tests for Huey summary task enqueue semantics."""

from __future__ import annotations


def test_enqueue_summary_task_force_refresh_calls_cancel(monkeypatch):
    """force_refresh should cancel existing tasks and still enqueue a new task."""
    import tasks

    canceled = {"called": False, "pid": None, "model": None}
    purged = {"called": False, "pid": None, "model": None}
    enqueued = {"kwargs": None}

    def _fake_cancel(pid, model=None, *, user=None, reason=None):
        canceled["called"] = True
        canceled["pid"] = pid
        canceled["model"] = model
        return {"canceled_task_ids": ["old_task"], "epoch": 123}

    def _fake_purge(pid, model):
        purged["called"] = True
        purged["pid"] = pid
        purged["model"] = model

    # Avoid touching filesystem locks and task scans.
    monkeypatch.setattr(tasks, "cancel_summary_tasks", _fake_cancel)
    monkeypatch.setattr(tasks, "_purge_summary_cache", _fake_purge)
    monkeypatch.setattr(tasks, "_enqueue_lock_path", lambda pid, model: None)
    monkeypatch.setattr(tasks, "acquire_summary_lock", lambda _path, timeout_s=0: object())
    monkeypatch.setattr(tasks, "release_summary_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tasks, "_find_active_task", lambda _pid, _model: (None, None))

    # Avoid repository I/O.
    monkeypatch.setattr(tasks.SummaryStatusRepository, "get_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tasks.SummaryStatusRepository, "get_generation_epoch", lambda *_args, **_kwargs: 123)
    monkeypatch.setattr(tasks.SummaryStatusRepository, "set_task_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tasks.SummaryStatusRepository, "set_status", lambda *_args, **_kwargs: None)

    # Avoid SSE in tests.
    monkeypatch.setattr(tasks, "_emit_all_event", lambda *_args, **_kwargs: None)

    # Make Huey enqueue deterministic.
    def _fake_enqueue(task_obj):
        enqueued["kwargs"] = dict(getattr(task_obj, "kwargs", {}) or {})
        task_obj.id = "task_new_1"

        class _Result:
            id = "task_new_1"

        return _Result()

    monkeypatch.setattr(tasks.huey, "enqueue", _fake_enqueue)

    task_id = tasks.enqueue_summary_task(
        "2301.00001",
        model="test-model",
        user=None,
        priority=1,
        force_refresh=True,
    )

    assert task_id == "task_new_1"
    assert canceled["called"] is True
    assert canceled["pid"] == "2301.00001"
    assert canceled["model"] == "test-model"
    assert purged["called"] is True
    assert purged["pid"] == "2301.00001"
    assert purged["model"] == "test-model"
    # Compatibility: do not pass new kwargs in Huey payload.
    assert "force_refresh" not in (enqueued["kwargs"] or {})
