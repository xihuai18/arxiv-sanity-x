from unittest.mock import patch

import pytest


def test_tasks_paper_exists_treats_deleting_upload_as_missing():
    import tasks

    with patch("aslite.repositories.UploadedPaperRepository.get") as mget:
        mget.return_value = {"owner": "u", "parse_status": "ok", "deleting": True}
        assert tasks._paper_exists("up_abc") is False


def test_process_uploaded_pdf_task_marks_upload_service_error_failed(monkeypatch):
    import tasks

    task_updates = []
    ref_updates = []

    monkeypatch.setattr(
        tasks,
        "_update_task_status",
        lambda *args, **kwargs: task_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "_update_upload_task_reference",
        lambda *args, **kwargs: ref_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(tasks, "_is_current_upload_task", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "backend.services.upload_service.process_uploaded_pdf",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("deleting")),
    )

    class DummyTask:
        id = "task_upload_process_1"

    with pytest.raises(RuntimeError, match="deleting"):
        tasks.process_uploaded_pdf_task.call_local("up_testpaper129", "test_user", task=DummyTask())

    assert task_updates[0][0][1] == "running"
    assert task_updates[1][0][1] == "failed"
    assert task_updates[1][1]["error"] == "deleting"
    assert ref_updates[-1][1]["clear"] is True


def test_process_uploaded_pdf_task_skips_superseded_task(monkeypatch):
    import tasks

    task_updates = []
    process_calls = []

    monkeypatch.setattr(
        tasks,
        "_is_current_upload_task",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        tasks,
        "_update_task_status",
        lambda *args, **kwargs: task_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "_maybe_mark_upload_task_canceled",
        lambda *args, **kwargs: task_updates.append(((None, "canceled"), {"error": "superseded_upload_task"})),
    )
    monkeypatch.setattr(
        "backend.services.upload_service.process_uploaded_pdf",
        lambda *args, **kwargs: process_calls.append((args, kwargs)),
    )

    class DummyTask:
        id = "task_upload_process_old"

    tasks.process_uploaded_pdf_task.call_local("up_testpaper132", "test_user", task=DummyTask())

    assert process_calls == []
    assert len(task_updates) == 1
    assert task_updates[0][0][1] == "canceled"
    assert task_updates[0][1]["error"] == "superseded_upload_task"


def test_parse_uploaded_pdf_task_skips_superseded_task(monkeypatch):
    import tasks

    task_updates = []
    parse_calls = []

    monkeypatch.setattr(tasks, "_is_current_upload_task", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        tasks,
        "_update_task_status",
        lambda *args, **kwargs: task_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "_maybe_mark_upload_task_canceled",
        lambda *args, **kwargs: task_updates.append(((None, "canceled"), {"error": "superseded_upload_task"})),
    )
    monkeypatch.setattr(
        "backend.services.upload_service.do_parse_only",
        lambda *args, **kwargs: parse_calls.append((args, kwargs)) or True,
    )

    class DummyTask:
        id = "task_upload_parse_old"

    tasks.parse_uploaded_pdf_task.call_local("up_testpaper133", "test_user", task=DummyTask())

    assert parse_calls == []
    assert len(task_updates) == 1
    assert task_updates[0][0][1] == "canceled"
    assert task_updates[0][1]["error"] == "superseded_upload_task"


def test_extract_info_task_skips_superseded_task(monkeypatch):
    import tasks

    task_updates = []
    extract_calls = []

    monkeypatch.setattr(tasks, "_is_current_upload_task", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        tasks,
        "_update_task_status",
        lambda *args, **kwargs: task_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "_maybe_mark_upload_task_canceled",
        lambda *args, **kwargs: task_updates.append(((None, "canceled"), {"error": "superseded_upload_task"})),
    )
    monkeypatch.setattr(
        "backend.services.upload_service.do_extract_metadata",
        lambda *args, **kwargs: extract_calls.append((args, kwargs)) or True,
    )

    class DummyTask:
        id = "task_upload_extract_old"

    tasks.extract_info_task.call_local("up_testpaper134", "test_user", task=DummyTask())

    assert extract_calls == []
    assert len(task_updates) == 1
    assert task_updates[0][0][1] == "canceled"
    assert task_updates[0][1]["error"] == "superseded_upload_task"


def test_process_uploaded_pdf_task_does_not_write_failed_after_delete(monkeypatch):
    import tasks

    task_updates = []
    ref_updates = []

    state = {"current": True}

    def _is_current(*args, **kwargs):
        return state["current"]

    def _process(*args, **kwargs):
        state["current"] = False
        raise RuntimeError("deleted during run")

    monkeypatch.setattr(tasks, "_is_current_upload_task", _is_current)
    monkeypatch.setattr(
        tasks,
        "_maybe_mark_upload_task_canceled",
        lambda *args, **kwargs: task_updates.append((("canceled",), {"error": "superseded_upload_task"})),
    )
    monkeypatch.setattr(
        tasks,
        "_update_task_status",
        lambda *args, **kwargs: task_updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "_update_upload_task_reference",
        lambda *args, **kwargs: ref_updates.append((args, kwargs)),
    )
    monkeypatch.setattr("backend.services.upload_service.process_uploaded_pdf", _process)

    class DummyTask:
        id = "task_upload_process_deleted"

    tasks.process_uploaded_pdf_task.call_local("up_testpaper137", "test_user", task=DummyTask())

    assert task_updates[0][0][1] == "running"
    assert not any(
        args and len(args) > 1 and args[1] == "failed" for args, _kwargs in task_updates[1:] if isinstance(args, tuple)
    )
