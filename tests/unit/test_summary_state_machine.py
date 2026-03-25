"""Unit tests for shared summary state helpers."""

from __future__ import annotations


def test_build_summary_status_transition_redacts_public_error_payload():
    from backend.services.summary_state_machine import build_summary_status_transition

    transition = build_summary_status_transition(
        "2301.00001",
        "test-model",
        "failed",
        "internal detail",
        task_id="task-1",
        task_user="alice",
    )

    assert transition is not None
    assert transition.repository_extra() == {
        "task_id": None,
        "task_user": None,
        "resolved_model": None,
    }
    assert transition.public_event_payload() == {
        "type": "summary_status",
        "pid": "2301.00001",
        "model": "test-model",
        "status": "failed",
        "error": "failed",
    }
    assert transition.private_event_payload()["error"] == "internal detail"


def test_build_summary_status_transition_keeps_active_task_fields():
    from backend.services.summary_state_machine import build_summary_status_transition

    transition = build_summary_status_transition(
        "2301.00001",
        "requested-model",
        "running",
        None,
        task_id="task-1",
        task_user="alice",
        resolved_model="fallback-model",
    )

    assert transition is not None
    assert transition.repository_extra() == {
        "task_id": "task-1",
        "task_user": "alice",
        "resolved_model": None,
    }


def test_build_summary_status_transition_emits_resolved_model_on_success():
    from backend.services.summary_state_machine import build_summary_status_transition

    transition = build_summary_status_transition(
        "2301.00001",
        "requested-model",
        "ok",
        None,
        task_id="task-1",
        resolved_model="fallback-model",
    )

    assert transition is not None
    assert transition.repository_extra() == {
        "task_id": None,
        "task_user": None,
        "resolved_model": "fallback-model",
    }
    assert transition.public_event_payload()["resolved_model"] == "fallback-model"


def test_build_readinglist_summary_transition_only_persists_default_model():
    from backend.services.summary_state_machine import (
        build_readinglist_summary_transition,
    )

    transition = build_readinglist_summary_transition(
        "alice",
        "2301.00001",
        "queued",
        None,
        task_id="task-1",
        model="other-model",
        default_model="default-model",
        updated_time=123.0,
    )

    assert transition is not None
    assert transition.persisted_updates() is None
    assert transition.event_payload() == {
        "type": "summary_status",
        "pid": "2301.00001",
        "model": "other-model",
        "status": "queued",
        "error": None,
        "task_id": "task-1",
    }


def test_build_readinglist_summary_transition_clears_task_id_for_terminal_state():
    from backend.services.summary_state_machine import (
        build_readinglist_summary_transition,
    )

    transition = build_readinglist_summary_transition(
        "alice",
        "2301.00001",
        "failed",
        "boom",
        task_id="task-1",
        model="default-model",
        default_model="default-model",
        updated_time=123.0,
    )

    assert transition is not None
    assert transition.persisted_updates() == {
        "summary_status": "failed",
        "summary_last_error": "boom",
        "summary_updated_time": 123.0,
        "summary_task_id": None,
    }
