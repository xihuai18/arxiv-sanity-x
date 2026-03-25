"""Unit tests for task-side summary status event payloads."""

from __future__ import annotations


def test_update_summary_status_db_redacts_global_error_payload(monkeypatch):
    import tasks

    emitted = {}
    monkeypatch.setattr(tasks.SummaryStatusRepository, "set_status", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "_emit_all_event", lambda payload: emitted.update(payload))

    tasks._update_summary_status_db("2301.00001", "test-model", "failed", "internal stack trace")

    assert emitted.get("status") == "failed"
    assert emitted.get("error") == "failed"


def test_update_summary_status_db_routes_upload_events_to_owner_only(monkeypatch):
    import tasks

    public_events = []
    private_events = []

    monkeypatch.setattr(tasks.SummaryStatusRepository, "set_status", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "_emit_all_event", lambda payload: public_events.append(dict(payload)))
    monkeypatch.setattr(
        tasks,
        "_emit_user_event",
        lambda user, payload: private_events.append((user, dict(payload))),
    )
    monkeypatch.setattr(
        tasks.UploadedPaperRepository,
        "get",
        lambda _pid: {"owner": "alice", "summary_task_id": "t1"},
    )
    monkeypatch.setattr(tasks.UploadedPaperRepository, "update", lambda *_a, **_k: True)

    tasks._update_summary_status_db("up_secret001", "test-model", "failed", "private detail", task_user="alice")

    assert public_events == []
    assert private_events == [
        (
            "alice",
            {
                "type": "summary_status",
                "pid": "up_secret001",
                "model": "test-model",
                "status": "failed",
                "error": "private detail",
            },
        )
    ]


def test_update_summary_status_db_emits_resolved_model(monkeypatch):
    import tasks

    emitted = {}
    monkeypatch.setattr(tasks.SummaryStatusRepository, "set_status", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "_emit_all_event", lambda payload: emitted.update(dict(payload)))

    tasks._update_summary_status_db(
        "2301.00001",
        "requested-model",
        "ok",
        None,
        task_id="task123",
        resolved_model="fallback-model",
    )

    assert emitted.get("status") == "ok"
    assert emitted.get("model") == "requested-model"
    assert emitted.get("resolved_model") == "fallback-model"


def test_update_summary_status_db_does_not_clear_upload_pointer_without_task_id(
    monkeypatch,
):
    import tasks

    upload_updates = []
    monkeypatch.setattr(tasks.SummaryStatusRepository, "set_status", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "_emit_user_event", lambda *_a, **_k: None)
    monkeypatch.setattr(tasks, "_emit_all_event", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tasks.UploadedPaperRepository,
        "get",
        lambda _pid: {"owner": "alice", "summary_task_id": "new_task_id"},
    )
    monkeypatch.setattr(
        tasks.UploadedPaperRepository,
        "update",
        lambda pid, patch: upload_updates.append((pid, dict(patch))) or True,
    )

    tasks._update_summary_status_db(
        "up_secret001",
        "test-model",
        "failed",
        "private detail",
        task_id=None,
        task_user="alice",
    )

    assert upload_updates == []
