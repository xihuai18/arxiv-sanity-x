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
