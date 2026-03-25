from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def _response_with_json(payload: str):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=payload, reasoning_content=""),
        finish_reason="stop",
    )
    return SimpleNamespace(choices=[choice])


def _responses_output(payload: str):
    return SimpleNamespace(output_text=payload, status="completed", usage=None)


class _FakeResponses:
    def __init__(self, *, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeChatCompletions:
    def __init__(self, response=None):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeOpenAIClient:
    def __init__(self, *, responses=None, chat=None):
        self.responses = responses or _FakeResponses()
        self.chat = SimpleNamespace(completions=chat or _FakeChatCompletions())


def test_extract_metadata_falls_back_to_main_llm_when_route_differs(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(
        upload_service.paper_summarizer.PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(lambda cls, model_name: None),
    )

    extract_client = _FakeOpenAIClient(responses=_FakeResponses(error=RuntimeError("extract route failed")))
    main_client = _FakeOpenAIClient(
        responses=_FakeResponses(
            response=_responses_output('{"title": "Fallback Title", "authors": ["Alice"], "abstract": "A"}')
        )
    )

    monkeypatch.setattr(upload_service, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(upload_service, "_llm_base_url", lambda: "https://main.example/v1")
    monkeypatch.setattr(upload_service, "_llm_api_key", lambda: "main-key")
    monkeypatch.setattr(upload_service, "_get_extract_info_client", lambda: extract_client)
    monkeypatch.setattr(upload_service.openai, "OpenAI", lambda **_kwargs: main_client)
    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "gpt-5.4")
    monkeypatch.setattr(upload_service.settings.extract_info, "base_url", "https://extract.example/v1")
    monkeypatch.setattr(upload_service.settings.extract_info, "api_key", "extract-key")

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract text")

    assert result["title"] == "Fallback Title"
    assert len(extract_client.responses.calls) == 1
    assert len(main_client.responses.calls) == 1


def test_extract_metadata_does_not_duplicate_main_fallback_for_same_route(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(
        upload_service.paper_summarizer.PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(lambda cls, model_name: None),
    )

    extract_client = _FakeOpenAIClient(
        responses=_FakeResponses(
            response=_responses_output('{"title": "Primary Title", "authors": ["Bob"], "abstract": "B"}')
        )
    )

    openai_ctor = MagicMock(side_effect=AssertionError("main fallback should not be constructed"))

    monkeypatch.setattr(upload_service, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(upload_service, "_llm_base_url", lambda: "https://main.example/v1")
    monkeypatch.setattr(upload_service, "_llm_api_key", lambda: "main-key")
    monkeypatch.setattr(upload_service, "_get_extract_info_client", lambda: extract_client)
    monkeypatch.setattr(upload_service.openai, "OpenAI", openai_ctor)
    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "gpt-5.4")
    monkeypatch.setattr(upload_service.settings.extract_info, "base_url", "")
    monkeypatch.setattr(upload_service.settings.extract_info, "api_key", "")

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract text")

    assert result["title"] == "Primary Title"
    assert len(extract_client.responses.calls) == 1
    openai_ctor.assert_not_called()


def test_extract_metadata_falls_back_to_chat_when_responses_endpoint_is_missing(
    monkeypatch,
):
    from backend.services import upload_service

    monkeypatch.setattr(
        upload_service.paper_summarizer.PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(lambda cls, model_name: None),
    )

    extract_client = _FakeOpenAIClient(
        responses=_FakeResponses(error=RuntimeError("404 not found for /responses endpoint")),
        chat=_FakeChatCompletions(
            _response_with_json('{"title": "Chat Title", "authors": ["Carol"], "abstract": "C"}')
        ),
    )

    monkeypatch.setattr(upload_service, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(upload_service, "_llm_base_url", lambda: "https://main.example/v1")
    monkeypatch.setattr(upload_service, "_llm_api_key", lambda: "main-key")
    monkeypatch.setattr(upload_service, "_get_extract_info_client", lambda: extract_client)
    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "gpt-5.4")
    monkeypatch.setattr(upload_service.settings.extract_info, "base_url", "")
    monkeypatch.setattr(upload_service.settings.extract_info, "api_key", "")

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract text")

    assert result["title"] == "Chat Title"
    assert len(extract_client.responses.calls) == 1
    assert len(extract_client.chat.completions.calls) == 1


def test_extract_metadata_uses_direct_responses_route_on_shared_main_endpoint(
    monkeypatch,
):
    from backend.services import upload_service

    extract_client = _FakeOpenAIClient(
        responses=_FakeResponses(error=AssertionError("shared gateway responses client should be bypassed"))
    )
    direct_client = _FakeOpenAIClient(
        responses=_FakeResponses(
            response=_responses_output('{"title": "Direct Title", "authors": ["Dana"], "abstract": "D"}')
        )
    )

    def fake_openai_factory(*, api_key, base_url):
        assert api_key == "direct-key"
        assert base_url == "https://direct.example/v1"
        return direct_client

    monkeypatch.setattr(upload_service, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(upload_service, "_llm_base_url", lambda: "https://main.example/v1")
    monkeypatch.setattr(upload_service, "_llm_api_key", lambda: "main-key")
    monkeypatch.setattr(upload_service, "_get_extract_info_client", lambda: extract_client)
    monkeypatch.setattr(upload_service.openai, "OpenAI", fake_openai_factory)
    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "gpt-5.4")
    monkeypatch.setattr(upload_service.settings.extract_info, "base_url", "")
    monkeypatch.setattr(upload_service.settings.extract_info, "api_key", "")
    monkeypatch.setattr(
        upload_service.paper_summarizer.PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(
            lambda cls, model_name: {
                "base_url": "https://direct.example/v1",
                "api_key": "direct-key",
                "model": "gpt-5.4-upstream",
                "extra_body": {"reasoning": {"effort": "low"}},
            }
        ),
    )

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract text")

    assert result["title"] == "Direct Title"
    assert len(direct_client.responses.calls) == 1
    assert direct_client.responses.calls[0]["model"] == "gpt-5.4-upstream"
    assert direct_client.responses.calls[0]["reasoning"] == {"effort": "low"}
    assert len(extract_client.chat.completions.calls) == 0


def test_extract_metadata_respects_direct_route_max_output_tokens(monkeypatch):
    from backend.services import upload_service

    extract_client = _FakeOpenAIClient(
        responses=_FakeResponses(error=AssertionError("shared gateway responses client should be bypassed"))
    )
    direct_client = _FakeOpenAIClient(
        responses=_FakeResponses(
            response=_responses_output('{"title": "Direct Title", "authors": ["Dana"], "abstract": "D"}')
        )
    )

    monkeypatch.setattr(upload_service, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(upload_service, "_llm_base_url", lambda: "https://main.example/v1")
    monkeypatch.setattr(upload_service, "_llm_api_key", lambda: "main-key")
    monkeypatch.setattr(upload_service, "_get_extract_info_client", lambda: extract_client)
    monkeypatch.setattr(
        upload_service.openai,
        "OpenAI",
        lambda *, api_key, base_url: direct_client,
    )
    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "gpt-5.4")
    monkeypatch.setattr(upload_service.settings.extract_info, "base_url", "")
    monkeypatch.setattr(upload_service.settings.extract_info, "api_key", "")
    monkeypatch.setattr(upload_service.settings.extract_info, "max_tokens", 8192)
    monkeypatch.setattr(
        upload_service.paper_summarizer.PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(
            lambda cls, model_name: {
                "base_url": "https://direct.example/v1",
                "api_key": "direct-key",
                "model": "gpt-5.4-upstream",
                "extra_body": {},
                "max_output_tokens": 1024,
            }
        ),
    )

    upload_service.extract_metadata_with_llm("# Title\n\nAbstract text")

    assert direct_client.responses.calls[0]["max_output_tokens"] == 1024


def test_extract_metadata_falls_back_to_chat_when_responses_returns_empty_content(
    monkeypatch,
):
    from backend.services import upload_service

    extract_client = _FakeOpenAIClient(
        responses=_FakeResponses(response=_responses_output("")),
        chat=_FakeChatCompletions(
            _response_with_json('{"title": "Chat Title", "authors": ["Carol"], "abstract": "C"}')
        ),
    )

    monkeypatch.setattr(
        upload_service.paper_summarizer.PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(lambda cls, model_name: None),
    )
    monkeypatch.setattr(upload_service, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(upload_service, "_llm_base_url", lambda: "https://main.example/v1")
    monkeypatch.setattr(upload_service, "_llm_api_key", lambda: "main-key")
    monkeypatch.setattr(upload_service, "_get_extract_info_client", lambda: extract_client)
    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "gpt-5.4")
    monkeypatch.setattr(upload_service.settings.extract_info, "base_url", "https://extract.example/v1")
    monkeypatch.setattr(upload_service.settings.extract_info, "api_key", "extract-key")

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract text")

    assert result["title"] == "Chat Title"
    assert len(extract_client.responses.calls) == 1
    assert len(extract_client.chat.completions.calls) == 1


def test_trigger_extract_info_reuses_existing_active_task(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(upload_service, "validate_upload_pid", lambda _pid: True)
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get",
        lambda _pid: {
            "pid": _pid,
            "owner": "alice",
            "parse_status": "ok",
            "meta_extracted_ok": False,
            "extract_task_id": "task_extract_existing",
        },
    )
    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "get_task_status",
        lambda _task_id: {"status": "queued"},
    )
    monkeypatch.setattr(
        upload_service,
        "_get_tasks_module",
        lambda: (_ for _ in ()).throw(AssertionError("should not enqueue a duplicate extract task")),
    )

    task_id = upload_service.trigger_extract_info("up_extracttask1", "alice")

    assert task_id == "task_extract_existing"


def test_trigger_extract_info_persists_task_id_before_returning(monkeypatch):
    from backend.services import upload_service

    class _FakeTx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeDb:
        def __init__(self):
            self.record = {
                "pid": "up_extracttask2",
                "owner": "alice",
                "parse_status": "ok",
                "meta_extracted_ok": False,
                "extract_task_id": "",
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def transaction(self, mode=None):
            assert mode == "IMMEDIATE"
            return _FakeTx()

        def get(self, pid):
            assert pid == "up_extracttask2"
            return dict(self.record)

        def __setitem__(self, pid, value):
            assert pid == "up_extracttask2"
            self.record = dict(value)

    fake_db = _FakeDb()

    class _Task:
        id = "task_extract_new"

    class _Huey:
        def enqueue(self, task):
            assert task.id == "task_extract_new"
            return task

    class _Tasks:
        huey = _Huey()

        class extract_info_task:
            @staticmethod
            def s(pid, user):
                assert pid == "up_extracttask2"
                assert user == "alice"
                return _Task()

    status_calls = []

    monkeypatch.setattr(upload_service, "validate_upload_pid", lambda _pid: True)
    monkeypatch.setattr(upload_service.UploadedPaperRepository, "get", lambda _pid: dict(fake_db.record))
    monkeypatch.setattr(upload_service, "_get_tasks_module", lambda: _Tasks())
    monkeypatch.setattr(upload_service, "_get_active_upload_task_id", lambda _record, _field: "")
    monkeypatch.setattr(
        upload_service,
        "_normalize_upload_parse_status",
        lambda _pid, _record: ("ok", ""),
    )
    monkeypatch.setattr(upload_service, "_get_uploaded_papers_db", lambda **_k: fake_db)
    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "set_task_status",
        lambda task_id, status, error=None, **extra: status_calls.append((task_id, status, error, extra)),
    )

    task_id = upload_service.trigger_extract_info("up_extracttask2", "alice")

    assert task_id == "task_extract_new"
    assert fake_db.record["extract_task_id"] == "task_extract_new"
    assert status_calls[0][0] == "task_extract_new"
    assert status_calls[0][1] == "queued"


def test_do_extract_metadata_respects_normalized_parse_status(monkeypatch, tmp_path):
    from backend.services import upload_service

    pid = "up_extractmeta1"
    md_path = tmp_path / f"{pid}.md"
    md_path.write_text("# Title\n\nAbstract text", encoding="utf-8")

    monkeypatch.setattr(upload_service, "validate_upload_pid", lambda _pid: True)
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get",
        lambda _pid: {
            "pid": _pid,
            "owner": "alice",
            "parse_status": "ok",
            "meta_extracted_ok": False,
        },
    )
    monkeypatch.setattr(
        upload_service,
        "_normalize_upload_parse_status",
        lambda _pid, _record: ("ok", ""),
    )
    monkeypatch.setattr(
        upload_service,
        "extract_metadata_with_llm",
        lambda _fm: {"title": "T", "authors": ["A"], "abstract": "B"},
    )

    updates: list[dict] = []
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, payload: updates.append(payload) or True,
    )

    class DummySummarizer:
        def _normalize_mineru_backend(self):
            return "api"

        def _find_mineru_markdown(self, _pid, backend=None):
            assert backend == "api"
            return md_path

    monkeypatch.setattr(upload_service.paper_summarizer, "PaperSummarizer", DummySummarizer)

    assert upload_service.do_extract_metadata(pid, "alice") is True
    assert any(payload.get("meta_extracted_ok") is True for payload in updates)


def test_get_uploaded_papers_list_uses_normalized_parse_status(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get_by_owner",
        lambda _user: {
            "up_listparse001": {
                "original_filename": "paper.pdf",
                "parse_status": "",
                "parse_error": "",
                "meta_extracted": {},
                "meta_override": {},
            }
        },
    )
    monkeypatch.setattr(upload_service.TagRepository, "get_user_tags", lambda _user: {})
    monkeypatch.setattr(upload_service.NegativeTagRepository, "get_user_neg_tags", lambda _user: {})
    monkeypatch.setattr(
        upload_service,
        "_normalize_upload_parse_status",
        lambda _pid, _record: ("ok", ""),
    )

    from backend.services import summary_service

    monkeypatch.setattr(summary_service, "get_summary_render_snapshots", lambda _pids: {})

    items = upload_service.get_uploaded_papers_list("alice")

    assert items[0]["parse_status"] == "ok"


def test_get_uploaded_papers_list_hides_stale_task_ids(monkeypatch):
    from backend.services import summary_service, upload_service

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get_by_owner",
        lambda _user: {
            "up_listtasks001": {
                "original_filename": "paper.pdf",
                "parse_status": "running",
                "parse_error": "",
                "parse_task_id": "task_parse_active",
                "extract_task_id": "task_extract_stale",
                "summary_task_id": "task_summary_terminal",
                "meta_extracted": {},
                "meta_override": {},
            }
        },
    )
    monkeypatch.setattr(upload_service.TagRepository, "get_user_tags", lambda _user: {})
    monkeypatch.setattr(upload_service.NegativeTagRepository, "get_user_neg_tags", lambda _user: {})
    monkeypatch.setattr(
        upload_service,
        "_normalize_upload_parse_status",
        lambda _pid, _record: ("running", ""),
    )
    monkeypatch.setattr(summary_service, "get_summary_render_snapshots", lambda _pids: {})

    repairs = []

    def _fake_classify(task_id):
        task_id = str(task_id or "")
        if task_id == "task_parse_active":
            return (
                "active",
                task_id,
                {
                    "status": "running",
                    "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
                },
            )
        if task_id == "task_extract_stale":
            return (
                "stale",
                task_id,
                {
                    "status": "running",
                    "model": upload_service.UPLOAD_TASK_MODEL_EXTRACT,
                },
            )
        if task_id == "task_summary_terminal":
            return ("terminal", task_id, {"status": "failed", "model": "gpt-5.4"})
        return ("missing", task_id, None)

    monkeypatch.setattr(upload_service, "_classify_upload_task", _fake_classify)
    monkeypatch.setattr(
        upload_service,
        "_repair_upload_task_status",
        lambda task_id, **kwargs: repairs.append((task_id, kwargs["pid"], kwargs["user"])),
    )

    items = upload_service.get_uploaded_papers_list("alice")

    assert items[0]["parse_task_id"] == "task_parse_active"
    assert items[0]["extract_task_id"] == ""
    assert items[0]["summary_task_id"] == ""
    assert repairs == [("task_extract_stale", "up_listtasks001", "alice")]


def test_get_uploaded_papers_list_recovers_missing_parse_task_pointer(monkeypatch):
    from backend.services import summary_service, upload_service

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get_by_owner",
        lambda _user: {
            "up_listtasks002": {
                "pid": "up_listtasks002",
                "owner": "alice",
                "original_filename": "paper.pdf",
                "parse_status": "queued",
                "parse_error": "",
                "parse_task_id": None,
                "meta_extracted": {},
                "meta_override": {},
                "updated_time": 50.0,
            }
        },
    )
    monkeypatch.setattr(upload_service.TagRepository, "get_user_tags", lambda _user: {})
    monkeypatch.setattr(upload_service.NegativeTagRepository, "get_user_neg_tags", lambda _user: {})
    monkeypatch.setattr(summary_service, "get_summary_render_snapshots", lambda _pids: {})
    monkeypatch.setattr(upload_service.time, "time", lambda: 80.0)
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda _task_id: ("missing", "", None),
    )
    monkeypatch.setattr(
        upload_service,
        "_find_active_upload_task_for_record",
        lambda _pid, _user, record_field: (
            (
                "task_parse_recovered",
                {
                    "status": "queued",
                    "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
                    "updated_time": 79.0,
                },
            )
            if record_field == "parse_task_id"
            else ("", None)
        ),
    )

    updates = []
    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "update",
        lambda _pid, patch: updates.append(dict(patch)) or True,
    )

    items = upload_service.get_uploaded_papers_list("alice")

    assert items[0]["parse_status"] == "queued"
    assert items[0]["parse_task_id"] == "task_parse_recovered"
    assert updates == [{"parse_task_id": "task_parse_recovered"}]


def test_get_uploaded_papers_list_clears_summary_state_while_parse_pending(monkeypatch):
    from backend.services import summary_service, upload_service

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get_by_owner",
        lambda _user: {
            "up_listsummary001": {
                "original_filename": "paper.pdf",
                "parse_status": "queued",
                "parse_error": "",
                "parse_task_id": "task_parse_active",
                "summary_task_id": "task_summary_active",
                "meta_extracted": {},
                "meta_override": {},
            }
        },
    )
    monkeypatch.setattr(upload_service.TagRepository, "get_user_tags", lambda _user: {})
    monkeypatch.setattr(upload_service.NegativeTagRepository, "get_user_neg_tags", lambda _user: {})
    monkeypatch.setattr(
        upload_service,
        "_normalize_upload_parse_status",
        lambda _pid, _record: ("queued", ""),
    )
    monkeypatch.setattr(summary_service, "get_summary_render_snapshots", lambda _pids: {})
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda task_id: (
            "active",
            str(task_id or ""),
            {"status": "running", "model": upload_service.UPLOAD_TASK_MODEL_PROCESS},
        ),
    )

    items = upload_service.get_uploaded_papers_list("alice")

    assert items[0]["parse_status"] == "queued"
    assert items[0]["summary_status"] == ""
    assert items[0]["summary_last_error"] == ""
    assert items[0]["summary_task_id"] == ""


def test_get_uploaded_papers_list_reuses_task_snapshot_and_tag_reverse_index(
    monkeypatch,
):
    from backend.services import upload_service

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get_by_owner",
        lambda _user: {
            "up_batch001": {
                "pid": "up_batch001",
                "owner": "alice",
                "original_filename": "one.pdf",
                "parse_status": "queued",
                "parse_error": "",
                "parse_task_id": "task_batch_parse_1",
                "meta_extracted": {},
                "meta_override": {},
                "created_time": 10,
            },
            "up_batch002": {
                "pid": "up_batch002",
                "owner": "alice",
                "original_filename": "two.pdf",
                "parse_status": "queued",
                "parse_error": "",
                "parse_task_id": "task_batch_parse_2",
                "meta_extracted": {},
                "meta_override": {},
                "created_time": 20,
            },
        },
    )
    monkeypatch.setattr(
        upload_service.TagRepository,
        "get_user_tags",
        lambda _user: {
            "alpha": {"up_batch001", "up_batch002"},
            "beta": {"up_batch002"},
        },
    )
    monkeypatch.setattr(
        upload_service.NegativeTagRepository,
        "get_user_neg_tags",
        lambda _user: {"neg": {"up_batch001"}},
    )

    scan_calls = {"count": 0}
    fallback_calls = {"count": 0}

    def _fake_items_with_prefix(prefix):
        assert prefix == "task::"
        scan_calls["count"] += 1
        return [
            (
                "task::task_batch_parse_1",
                {
                    "pid": "up_batch001",
                    "user": "alice",
                    "model": upload_service.UPLOAD_TASK_MODEL_PARSE,
                    "status": "queued",
                    "updated_time": 100.0,
                },
            ),
            (
                "task::task_batch_parse_2",
                {
                    "pid": "up_batch002",
                    "user": "alice",
                    "model": upload_service.UPLOAD_TASK_MODEL_PROCESS,
                    "status": "running",
                    "updated_time": 101.0,
                },
            ),
        ]

    monkeypatch.setattr(
        upload_service.SummaryStatusRepository,
        "get_items_with_prefix",
        _fake_items_with_prefix,
    )

    def _fake_get_task_status(_task_id):
        fallback_calls["count"] += 1
        return None

    monkeypatch.setattr(upload_service.SummaryStatusRepository, "get_task_status", _fake_get_task_status)

    items = upload_service.get_uploaded_papers_list("alice")

    assert scan_calls["count"] == 1
    assert fallback_calls["count"] == 0
    assert [item["id"] for item in items] == ["up_batch002", "up_batch001"]
    assert items[0]["utags"] == ["alpha", "beta"]
    assert items[0]["ntags"] == []
    assert items[1]["utags"] == ["alpha"]
    assert items[1]["ntags"] == ["neg"]
    assert items[0]["parse_task_id"] == "task_batch_parse_2"
    assert items[1]["parse_task_id"] == "task_batch_parse_1"


def test_get_uploaded_papers_list_uses_summary_snapshots_for_parse_ok(monkeypatch):
    from backend.services import summary_service, upload_service

    monkeypatch.setattr(
        upload_service.UploadedPaperRepository,
        "get_by_owner",
        lambda _user: {
            "up_summary001": {
                "pid": "up_summary001",
                "owner": "alice",
                "original_filename": "paper.pdf",
                "parse_status": "ok",
                "parse_error": "",
                "summary_task_id": "task_summary_active",
                "meta_extracted": {},
                "meta_override": {},
            }
        },
    )
    monkeypatch.setattr(upload_service.TagRepository, "get_user_tags", lambda _user: {})
    monkeypatch.setattr(upload_service.NegativeTagRepository, "get_user_neg_tags", lambda _user: {})
    monkeypatch.setattr(
        upload_service,
        "_normalize_upload_parse_status",
        lambda _pid, _record: ("ok", ""),
    )
    monkeypatch.setattr(
        summary_service,
        "get_summary_render_snapshots",
        lambda pids: {pid: {"status": "ok", "last_error": None, "tldr": "Snapshot TLDR"} for pid in pids},
    )
    monkeypatch.setattr(
        upload_service,
        "_classify_upload_task",
        lambda task_id: ("active", str(task_id or ""), {"status": "running"}),
    )

    items = upload_service.get_uploaded_papers_list("alice")

    assert items[0]["summary_status"] == "ok"
    assert items[0]["summary_last_error"] == ""
    assert items[0]["tldr"] == "Snapshot TLDR"
    assert items[0]["summary_task_id"] == "task_summary_active"
