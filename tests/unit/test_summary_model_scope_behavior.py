"""Behavior checks for default-model-only summary state on shared surfaces."""

from __future__ import annotations


def test_readinglist_service_persists_only_default_model(monkeypatch):
    import backend.services.readinglist_service as svc

    monkeypatch.setattr(svc, "_default_summary_model", lambda: "gpt-5.4")
    monkeypatch.setattr(
        svc.ReadingListRepository,
        "get_reading_list_item",
        lambda user, pid: {"id": pid},
    )

    persisted = []
    emitted = []

    monkeypatch.setattr(
        svc.ReadingListRepository,
        "update_reading_list_item",
        lambda user, pid, updates: persisted.append((user, pid, updates)),
    )
    monkeypatch.setattr(svc, "emit_user_event", lambda user, payload: emitted.append((user, payload)))

    svc.update_summary_status("alice", "2512.04388", "running", None, task_id="t1", model="qwen3.5-plus")

    assert persisted == []
    assert emitted == [
        (
            "alice",
            {
                "type": "summary_status",
                "pid": "2512.04388",
                "model": "qwen3.5-plus",
                "status": "running",
                "error": None,
                "task_id": "t1",
            },
        )
    ]


def test_readinglist_service_persists_default_model_state(monkeypatch):
    import backend.services.readinglist_service as svc

    monkeypatch.setattr(svc, "_default_summary_model", lambda: "gpt-5.4")
    monkeypatch.setattr(
        svc.ReadingListRepository,
        "get_reading_list_item",
        lambda user, pid: {"id": pid},
    )

    persisted = []
    emitted = []

    monkeypatch.setattr(
        svc.ReadingListRepository,
        "update_reading_list_item",
        lambda user, pid, updates: persisted.append((user, pid, updates)),
    )
    monkeypatch.setattr(svc, "emit_user_event", lambda user, payload: emitted.append((user, payload)))

    svc.update_summary_status("alice", "2512.04388", "queued", None, task_id="t2", model="gpt-5.4")

    assert persisted == [
        (
            "alice",
            "2512.04388",
            {
                "summary_status": "queued",
                "summary_last_error": None,
                "summary_updated_time": persisted[0][2]["summary_updated_time"],
                "summary_task_id": "t2",
            },
        )
    ]
    assert emitted[0][1]["model"] == "gpt-5.4"


def test_tasks_helper_matches_default_model_persistence(monkeypatch):
    import tasks

    monkeypatch.setattr(tasks, "_default_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(
        tasks.ReadingListRepository,
        "get_reading_list_item",
        lambda user, pid: {"id": pid},
    )

    persisted = []
    emitted = []

    monkeypatch.setattr(
        tasks.ReadingListRepository,
        "update_reading_list_item",
        lambda user, pid, updates: persisted.append((user, pid, updates)),
    )
    monkeypatch.setattr(tasks, "_emit_user_event", lambda user, payload: emitted.append((user, payload)))

    tasks._update_readinglist_summary_status("alice", "2512.04388", "ok", None, task_id=None, model="qwen3.5-plus")

    assert persisted == []
    assert emitted[0][1]["model"] == "qwen3.5-plus"
