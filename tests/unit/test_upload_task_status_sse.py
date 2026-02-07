import types


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

    monkeypatch.setattr(upload_service, "DATA_DIR", str(tmp_path))

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
    monkeypatch.setattr(upload_service, "_emit_upload_event", lambda _u, payload: events.append(dict(payload)))

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

    monkeypatch.setattr(upload_service, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(record))

    def fake_update(_pid, patch):
        record.update(patch)
        return True

    monkeypatch.setattr(upload_service.UploadedPaperRepository, "update", fake_update)
    monkeypatch.setattr(upload_service, "_emit_upload_event", lambda _u, payload: events.append(dict(payload)))

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
        upload_service.UploadedPaperRepository, "update", lambda pid, patch: calls.append((pid, dict(patch)))
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
