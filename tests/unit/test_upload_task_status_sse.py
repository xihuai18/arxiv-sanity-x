import types

import pytest


def test_process_uploaded_pdf_emits_sse_running_ok(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper123"
    user = "test_user"

    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
        "meta_override": {},
    }
    updates = []
    events = []

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))

    def fake_get(upload_pid):
        assert upload_pid == pid
        return dict(record)

    def fake_update(upload_pid, patch):
        assert upload_pid == pid
        updates.append(dict(patch))
        record.update(patch)
        return True

    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", fake_get)
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "update", fake_update)
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )

    pdf_path = upload_service.get_upload_pdf_path(pid, str(tmp_path))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    md_path = tmp_path / "parsed.md"
    md_path.write_text("# Test\n\n## Introduction\nBody", encoding="utf-8")

    class DummySummarizer:
        def parse_pdf_with_mineru(self, *_a, **_k):
            return md_path

    monkeypatch.setattr("tools.paper_summarizer.PaperSummarizer", DummySummarizer)
    monkeypatch.setattr(
        upload_service,
        "extract_metadata_with_llm",
        lambda _fm: {"title": "T", "authors": ["A"], "year": None, "abstract": "Abs"},
    )

    enqueue_calls = []

    def fake_enqueue_summary_task(upload_pid, model=None, user=None, priority=None, force_refresh=False):
        enqueue_calls.append((upload_pid, model, user, priority, force_refresh))
        return "sum_task_1"

    monkeypatch.setattr("tasks.enqueue_summary_task", fake_enqueue_summary_task)

    upload_service.process_uploaded_pdf(pid, user)

    assert any(e.get("type") == "upload_parse_status" and e.get("status") == "running" for e in events)
    assert any(e.get("type") == "upload_parse_status" and e.get("status") == "ok" for e in events)
    assert any(e.get("type") == "upload_extract_status" and e.get("status") == "running" for e in events)
    assert any(e.get("type") == "upload_extract_status" and e.get("status") == "ok" for e in events)
    assert record.get("summary_task_id") == "sum_task_1"
    assert enqueue_calls and enqueue_calls[0][0] == pid


def test_process_uploaded_pdf_emits_sse_failed_on_parse_error(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper124"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
        "meta_override": {},
    }
    events = []

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))

    def fake_update(_pid, patch):
        record.update(patch)
        return True

    monkeypatch.setattr(upload_service.UploadedPaperRepository, "update", fake_update)
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )

    # No PDF file -> FileNotFoundError in process_uploaded_pdf
    try:
        upload_service.process_uploaded_pdf(pid, user)
    except Exception:
        pass

    assert any(e.get("type") == "upload_parse_status" and e.get("status") == "running" for e in events)
    assert any(e.get("type") == "upload_parse_status" and e.get("status") == "failed" for e in events)


def test_register_upload_task_enqueue_writes_task_status(monkeypatch):
    from backend.services import upload_service

    calls = []

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda pid, patch: calls.append((pid, dict(patch))),
    )

    task_calls = []

    def fake_set_task_status(task_id, status, error=None, **extra):
        task_calls.append((task_id, status, error, dict(extra)))

    monkeypatch.setattr(upload_service.SummaryStatusRepository, "set_task_status", fake_set_task_status)

    task = types.SimpleNamespace(id="task_upload_status_1")

    task_id = upload_service.register_upload_task_enqueue(
        task_type="parse",
        pid="up_testpaper125",
        user="test_user",
        task=task,
        enqueue_result=None,
    )

    assert task_id == "task_upload_status_1"
    assert calls == [("up_testpaper125", {"parse_task_id": "task_upload_status_1"})]
    assert task_calls
    assert task_calls[0][0] == "task_upload_status_1"
    assert task_calls[0][1] == "queued"
    assert task_calls[0][3].get("model") == "upload_parse"
    assert task_calls[0][3].get("pid") == "up_testpaper125"


def test_process_uploaded_pdf_marks_summary_failed_when_enqueue_fails(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper126"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
        "meta_override": {},
    }

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))

    updates = []
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or record.update(patch) or True,
    )

    pdf_path = upload_service.get_upload_pdf_path(pid, str(tmp_path))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    md_path = tmp_path / "parsed-2.md"
    md_path.write_text("# Test\n\n## Introduction\nBody", encoding="utf-8")

    class DummySummarizer:
        def parse_pdf_with_mineru(self, *_a, **_k):
            return md_path

    monkeypatch.setattr("tools.paper_summarizer.PaperSummarizer", DummySummarizer)
    monkeypatch.setattr(
        upload_service,
        "extract_metadata_with_llm",
        lambda _fm: {"title": "T", "authors": ["A"], "year": None, "abstract": "Abs"},
    )

    summary_status_calls = []
    readinglist_calls = []

    def _raise_enqueue(*_a, **_k):
        raise RuntimeError("queue down")

    monkeypatch.setattr("tasks.enqueue_summary_task", _raise_enqueue)
    monkeypatch.setattr(
        "tasks._update_summary_status_db",
        lambda *args, **kwargs: summary_status_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "tasks._update_readinglist_summary_status",
        lambda *args, **kwargs: readinglist_calls.append((args, kwargs)),
    )

    upload_service.process_uploaded_pdf(pid, user)

    assert any(call[0][2] == "failed" for call in summary_status_calls)
    assert any(call[0][2] == "failed" for call in readinglist_calls)
    assert any(patch.get("summary_task_id") is None for patch in updates)


def test_process_uploaded_pdf_rejects_deleting_record_without_ghost_running(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper127"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "deleting": True,
        "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
        "meta_override": {},
    }
    updates = []
    events = []

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or record.update(patch) or True,
    )
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )

    with pytest.raises(upload_service.UploadServiceError) as exc_info:
        upload_service.process_uploaded_pdf(pid, user)

    assert exc_info.value.code == "deleting"
    assert updates == []
    assert events == []


def test_prepare_upload_parse_enqueue_repairs_stale_task(monkeypatch):
    from backend.services import upload_service

    pid = "up_testpaper128"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "running",
        "parse_task_id": "task_stale_parse_1",
    }
    status_writes = []
    task_status_state = {
        "status": "running",
        "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
        "updated_time": 1.0,
        "pid": pid,
        "user": user,
    }

    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))
    monkeypatch.setattr(upload_service, "_upload_task_repair_ttl", lambda: 10)
    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "get_task_status",
        lambda task_id: dict(task_status_state),
    )
    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "set_task_status",
        lambda task_id, status, error=None, **extra: status_writes.append((task_id, status, error, dict(extra)))
        or task_status_state.update({"status": status, "error": error, **extra}),
    )

    class DummyTxn:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyDb:
        def __init__(self):
            self.saved = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def transaction(self, mode=None):
            assert mode == "IMMEDIATE"
            return DummyTxn()

        def get(self, key):
            assert key == pid
            return dict(record)

        def __setitem__(self, key, value):
            assert key == pid
            self.saved = dict(value)

    db = DummyDb()
    monkeypatch.setattr("aslite.db.get_uploaded_papers_db", lambda **_kwargs: db)
    monkeypatch.setattr(upload_service.time, "time", lambda: 100.0)

    status, existing_task_id = upload_service._prepare_upload_parse_enqueue(
        pid=pid,
        user=user,
        require_failed=False,
        reject_if_ok=False,
        set_updated_time=True,
    )

    assert status == "ready"
    assert existing_task_id == ""
    assert status_writes == [
        (
            "task_stale_parse_1",
            "failed",
            "stale_running_repaired",
            {
                "pid": pid,
                "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
                "user": user,
            },
        )
    ]
    assert db.saved is not None
    assert db.saved["parse_status"] == "queued"
    assert db.saved["parse_error"] is None
    assert db.saved["parse_task_id"] is None
    assert db.saved["updated_time"] == 100.0


def test_normalize_upload_parse_status_repairs_terminal_task_pointer(monkeypatch):
    from backend.services import upload_service

    pid = "up_testpaper128b"
    record = {
        "pid": pid,
        "owner": "test_user",
        "parse_status": "running",
        "parse_task_id": "task_terminal_parse_1",
        "parse_error": "",
    }
    updates = []

    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda _task_id: (
            "terminal",
            "task_terminal_parse_1",
            {"status": "failed", "model": upload_service.UPLOAD_TASK_MODEL_PROCESS},
        ),
    )
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or True,
    )

    status, error = upload_service._normalize_upload_parse_status(pid, record)

    assert status == "failed"
    assert error == "stale_running_repaired"
    assert updates == [
        {
            "parse_status": "failed",
            "parse_error": "stale_running_repaired",
            "parse_task_id": None,
        }
    ]


def test_normalize_upload_parse_status_keeps_recent_missing_task_status_pointer(
    monkeypatch,
):
    from backend.services import upload_service

    pid = "up_testpaper128c"
    record = {
        "pid": pid,
        "owner": "test_user",
        "parse_status": "running",
        "parse_task_id": "task_recent_pointer_only",
        "parse_error": "",
        "updated_time": 100.0,
    }
    updates = []

    monkeypatch.setattr(upload_service.time, "time", lambda: 105.0)
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda _task_id: ("missing", "task_recent_pointer_only", None),
    )
    monkeypatch.setattr(
        upload_service,
        "_find_active_upload_task_for_record",
        lambda *_a, **_k: ("", None),
    )
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or True,
    )

    status, error = upload_service._normalize_upload_parse_status(pid, record)

    assert status == "running"
    assert error == ""
    assert updates == []


def test_normalize_upload_parse_status_recovers_missing_parse_task_pointer(
    monkeypatch,
):
    from backend.services import upload_service

    pid = "up_testpaper128d"
    record = {
        "pid": pid,
        "owner": "test_user",
        "parse_status": "queued",
        "parse_task_id": None,
        "parse_error": "",
        "updated_time": 50.0,
    }
    updates = []

    monkeypatch.setattr(upload_service.time, "time", lambda: 80.0)
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda _task_id: ("missing", "", None),
    )
    monkeypatch.setattr(
        upload_service,
        "_find_active_upload_task_for_record",
        lambda *_a, **_k: (
            "task_recovered_parse_1",
            {
                "status": "queued",
                "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
                "updated_time": 79.0,
            },
        ),
    )
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or True,
    )

    status, error = upload_service._normalize_upload_parse_status(pid, record)

    assert status == "queued"
    assert error == ""
    assert updates == [{"parse_task_id": "task_recovered_parse_1"}]


def test_normalize_upload_parse_status_repairs_old_missing_parse_task_pointer(
    monkeypatch,
):
    from backend.services import upload_service

    pid = "up_testpaper128e"
    record = {
        "pid": pid,
        "owner": "test_user",
        "parse_status": "queued",
        "parse_task_id": None,
        "parse_error": "",
        "updated_time": 10.0,
    }
    updates = []

    monkeypatch.setattr(upload_service.time, "time", lambda: 80.0)
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda _task_id: ("missing", "", None),
    )
    monkeypatch.setattr(
        upload_service,
        "_find_active_upload_task_for_record",
        lambda *_a, **_k: ("", None),
    )
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or True,
    )

    status, error = upload_service._normalize_upload_parse_status(pid, record)

    assert status == "failed"
    assert error == "stale_running_repaired"
    assert updates == [
        {
            "parse_status": "failed",
            "parse_error": "stale_running_repaired",
            "parse_task_id": None,
        }
    ]


def test_retry_parse_stale_task_is_recoverable(monkeypatch):
    from backend.services import upload_service

    prepare_calls = []
    enqueue_calls = []

    monkeypatch.setattr(
        upload_service,
        "_prepare_upload_parse_enqueue",
        lambda **kwargs: prepare_calls.append(dict(kwargs)) or ("ready", ""),
    )

    class DummyTasks:
        class process_uploaded_pdf_task:
            @staticmethod
            def s(pid, user):
                return (pid, user)

    monkeypatch.setattr(upload_service, "_get_tasks_module", lambda: DummyTasks())
    monkeypatch.setattr(
        upload_service,
        "_enqueue_upload_task",
        lambda **kwargs: enqueue_calls.append(dict(kwargs)) or "task_retry_after_stale",
    )

    result = upload_service.retry_parse_uploaded_paper("up_testpaper130", "test_user")

    assert result.status == "queued"
    assert result.task_id == "task_retry_after_stale"
    assert prepare_calls[0]["require_failed"] is True
    assert enqueue_calls[0]["task_type"] == "process"


def test_prepare_upload_parse_enqueue_retry_parse_accepts_stale_running(monkeypatch):
    from backend.services import upload_service

    pid = "up_testpaper131"
    user = "test_user"
    pre_record = {
        "pid": pid,
        "owner": user,
        "parse_status": "running",
        "parse_task_id": "task_stale_parse_retry",
    }
    tx_record = dict(pre_record)
    status_writes = []

    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(pre_record))
    monkeypatch.setattr(upload_service, "_upload_task_repair_ttl", lambda: 10)
    monkeypatch.setattr(upload_service.time, "time", lambda: 100.0)

    task_status_state = {
        "status": "running",
        "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
        "updated_time": 1.0,
        "pid": pid,
        "user": user,
    }

    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "get_task_status",
        lambda task_id: dict(task_status_state),
    )
    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "set_task_status",
        lambda task_id, status, error=None, **extra: status_writes.append((task_id, status, error, dict(extra)))
        or task_status_state.update({"status": status, "error": error, **extra}),
    )

    class DummyTxn:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyDb:
        def __init__(self):
            self.saved = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def transaction(self, mode=None):
            assert mode == "IMMEDIATE"
            return DummyTxn()

        def get(self, key):
            assert key == pid
            return dict(tx_record)

        def __setitem__(self, key, value):
            assert key == pid
            self.saved = dict(value)
            tx_record.update(value)

    monkeypatch.setattr("aslite.db.get_uploaded_papers_db", lambda **_kwargs: DummyDb())

    status, existing_task_id = upload_service._prepare_upload_parse_enqueue(
        pid=pid,
        user=user,
        require_failed=True,
        reject_if_ok=False,
        set_updated_time=False,
    )

    assert status == "ready"
    assert existing_task_id == ""
    assert len(status_writes) >= 1
    assert status_writes[0][1] == "failed"


def test_prepare_upload_parse_enqueue_keeps_recent_pending_registration_in_progress(
    monkeypatch,
):
    from backend.services import upload_service

    pid = "up_testpaper131b"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "parse_task_id": None,
        "updated_time": 100.0,
    }

    monkeypatch.setattr(upload_service.time, "time", lambda: 105.0)
    monkeypatch.setattr(upload_service, "_get_owned_upload_record", lambda _pid, _user: dict(record))
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda _task_id: ("missing", "", None),
    )
    monkeypatch.setattr(
        upload_service,
        "_find_active_upload_task_for_record",
        lambda *_a, **_k: ("", None),
    )

    status, existing_task_id = upload_service._prepare_upload_parse_enqueue(
        pid=pid,
        user=user,
        require_failed=False,
        reject_if_ok=False,
        set_updated_time=False,
    )

    assert status == "already_in_progress"
    assert existing_task_id == ""


def test_upload_task_repair_ttl_uses_upload_setting(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(upload_service.settings.huey, "upload_repair_ttl", 7200)
    monkeypatch.setattr(upload_service.settings.huey, "summary_repair_ttl", 60)

    assert upload_service._upload_task_repair_ttl() == 7200


def test_do_parse_only_stops_when_task_superseded_after_parse(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper135"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "parse_task_id": "task_parse_current",
    }
    events = []
    updates = []
    state = {"active": True}

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(upload_service, "validate_upload_pid", lambda _pid: True)
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or record.update(patch) or True,
    )
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )
    monkeypatch.setattr(
        upload_service,
        "_ensure_upload_record_active",
        lambda *_a, **_k: (
            (_ for _ in ()).throw(upload_service.UploadServiceError("superseded_task", "Task is no longer current"))
            if not state["active"]
            else dict(record)
        ),
    )

    pdf_path = upload_service.get_upload_pdf_path(pid, str(tmp_path))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    class DummySummarizer:
        def parse_pdf_with_mineru(self, *_a, **_k):
            state["active"] = False
            md_path = tmp_path / "parsed-stop.md"
            md_path.write_text("body", encoding="utf-8")
            return md_path

    monkeypatch.setattr("tools.paper_summarizer.PaperSummarizer", DummySummarizer)

    ok = upload_service.do_parse_only(pid, user, current_task_id="task_parse_current")

    assert ok is False
    assert updates[0]["parse_status"] == "running"
    assert not any(patch.get("parse_status") == "ok" for patch in updates[1:])
    assert not any(e.get("type") == "upload_parse_status" and e.get("status") == "ok" for e in events)


def test_do_extract_metadata_stops_when_task_superseded_after_llm(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper136"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "ok",
        "extract_task_id": "task_extract_current",
    }
    events = []
    updates = []
    state = {"active": True}

    monkeypatch.setattr(upload_service, "validate_upload_pid", lambda _pid: True)
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or record.update(patch) or True,
    )
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )
    monkeypatch.setattr(
        upload_service,
        "_ensure_upload_record_active",
        lambda *_a, **_k: (
            (_ for _ in ()).throw(upload_service.UploadServiceError("superseded_task", "Task is no longer current"))
            if not state["active"]
            else dict(record)
        ),
    )

    md_path = tmp_path / f"{pid}.md"
    md_path.write_text("# Title\n\nAbstract text", encoding="utf-8")

    class DummySummarizer:
        def _normalize_mineru_backend(self):
            return "api"

        def _find_mineru_markdown(self, _pid, backend=None):
            return md_path

    monkeypatch.setattr("tools.paper_summarizer.PaperSummarizer", DummySummarizer)

    def _extract(_front_matter):
        state["active"] = False
        return {"title": "T", "authors": ["A"]}

    monkeypatch.setattr(upload_service, "extract_metadata_with_llm", _extract)

    ok = upload_service.do_extract_metadata(pid, user, current_task_id="task_extract_current")

    assert ok is False
    assert updates == []
    assert any(e.get("type") == "upload_extract_status" and e.get("status") == "running" for e in events)
    assert not any(e.get("type") == "upload_extract_status" and e.get("status") == "ok" for e in events)


def test_process_uploaded_pdf_skips_failed_sse_after_superseded_exception(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper138"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "parse_task_id": "task_parse_current",
        "deleting": False,
        "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
        "meta_override": {},
    }
    events = []
    updates = []
    state = {"current": True}

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or record.update(patch) or True,
    )
    monkeypatch.setattr(upload_service, "_get_tasks_module", lambda: object())
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )

    def _ensure(*_a, **_k):
        if state["current"]:
            return dict(record)
        raise upload_service.UploadServiceError("superseded_task", "Task is no longer current")

    def _update_if_current(*_a, **_k):
        return state["current"]

    def _is_current(*_a, **_k):
        return state["current"]

    monkeypatch.setattr(upload_service, "_ensure_upload_record_active", _ensure)
    monkeypatch.setattr(upload_service, "_update_upload_record_if_current", _update_if_current)
    monkeypatch.setattr(upload_service, "_is_upload_record_current", _is_current)

    pdf_path = upload_service.get_upload_pdf_path(pid, str(tmp_path))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    class DummySummarizer:
        def parse_pdf_with_mineru(self, *_a, **_k):
            state["current"] = False
            raise RuntimeError("late parse failure")

    monkeypatch.setattr("tools.paper_summarizer.PaperSummarizer", DummySummarizer)

    with pytest.raises(RuntimeError, match="late parse failure"):
        upload_service.process_uploaded_pdf(pid, user, current_task_id="task_parse_current")

    assert any(e.get("type") == "upload_parse_status" and e.get("status") == "running" for e in events)
    assert not any(e.get("type") == "upload_parse_status" and e.get("status") == "failed" for e in events)


def test_process_uploaded_pdf_skips_summary_enqueue_after_superseded(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_testpaper139"
    user = "test_user"
    record = {
        "pid": pid,
        "owner": user,
        "parse_status": "queued",
        "parse_task_id": "task_parse_current",
        "deleting": False,
        "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
        "meta_override": {},
    }
    events = []
    updates = []
    state = {"current": True}
    enqueue_calls = []

    monkeypatch.setattr(upload_service, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or record.update(patch) or True,
    )
    monkeypatch.setattr(
        upload_service,
        "_emit_upload_event",
        lambda _u, payload: events.append(dict(payload)),
    )

    def _ensure(*_a, **_k):
        if state["current"]:
            return dict(record)
        raise upload_service.UploadServiceError("superseded_task", "Task is no longer current")

    def _update_if_current(*_a, **_k):
        return state["current"]

    def _is_current(*_a, **_k):
        return state["current"]

    monkeypatch.setattr(upload_service, "_ensure_upload_record_active", _ensure)
    monkeypatch.setattr(upload_service, "_update_upload_record_if_current", _update_if_current)
    monkeypatch.setattr(upload_service, "_is_upload_record_current", _is_current)

    pdf_path = upload_service.get_upload_pdf_path(pid, str(tmp_path))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    class DummySummarizer:
        def parse_pdf_with_mineru(self, *_a, **_k):
            md_path = tmp_path / "parsed-ok.md"
            md_path.write_text("body", encoding="utf-8")
            return md_path

    class DummyTasks:
        @staticmethod
        def enqueue_summary_task(*_a, **_k):
            enqueue_calls.append(True)
            return "summary_task_should_not_exist"

        @staticmethod
        def _update_summary_status_db(*_a, **_k):
            return None

        @staticmethod
        def _update_readinglist_summary_status(*_a, **_k):
            return None

        @staticmethod
        def cancel_paper_summary_tasks(*_a, **_k):
            return None

    monkeypatch.setattr("tools.paper_summarizer.PaperSummarizer", DummySummarizer)
    monkeypatch.setattr(
        upload_service,
        "extract_metadata_with_llm",
        lambda *_a, **_k: state.update({"current": False}) or {},
    )
    monkeypatch.setattr(upload_service, "_get_tasks_module", lambda: DummyTasks())

    upload_service.process_uploaded_pdf(pid, user, current_task_id="task_parse_current")

    assert not enqueue_calls
    assert any(e.get("type") == "upload_parse_status" and e.get("status") == "ok" for e in events)
    assert not any(patch.get("summary_task_id") for patch in updates)
