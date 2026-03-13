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
            "parse_status": "",
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
        lambda _pid, payload: updates.append(payload),
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

    monkeypatch.setattr(summary_service, "get_summary_status", lambda _pid: ("", ""))

    items = upload_service.get_uploaded_papers_list("alice")

    assert items[0]["parse_status"] == "ok"
