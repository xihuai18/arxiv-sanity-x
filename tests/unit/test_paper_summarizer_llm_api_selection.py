from __future__ import annotations

import json
from types import SimpleNamespace

from tools.paper_summarizer import PaperSummarizer


def _build_summary_text() -> str:
    return "# Title\n\n## TL;DR\n\nShort summary.\n\n## Body\n\nDetailed content."


def _build_chat_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


def _build_responses_response(text: str):
    return SimpleNamespace(
        output_text=text,
        status="completed",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )


def _build_proxy_wrapped_responses_response(text: str):
    payload = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                        }
                    ],
                }
            ],
        },
    }
    raw = (
        "event: response.created\n"
        'data: {"type":"response.created"}\n\n'
        "event: response.completed\n"
        f"data: {json.dumps(payload)}\n"
    )
    return SimpleNamespace(
        output_text=None,
        status=None,
        usage=None,
        error=SimpleNamespace(message=raw),
    )


class _FakeChatCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeResponses:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeOpenAIClient:
    def __init__(self, *, responses=None, chat=None):
        self.responses = responses or _FakeResponses()
        self.chat = chat or SimpleNamespace(completions=_FakeChatCompletions(None))


def _make_summarizer(monkeypatch, *, chat_response=None, responses_response=None, responses_error=None):
    import tools.paper_summarizer as ps

    original_data_dir = str(ps.settings.data_dir)
    monkeypatch.setattr(ps, "_data_dir", lambda: original_data_dir)
    monkeypatch.setattr(ps, "_llm_name", lambda: "")
    monkeypatch.setattr(ps, "_llm_timeout", lambda: 123)
    monkeypatch.setattr(ps, "_llm_summary_lang", lambda: "en")
    monkeypatch.setattr(
        PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(lambda cls, model_name: None),
    )
    monkeypatch.setattr(
        ps,
        "settings",
        SimpleNamespace(
            llm=SimpleNamespace(
                api_key="test-key",
                base_url="http://fake",
                fallback_models="",
                fallback_model_list=[],
            )
        ),
    )

    summarizer = PaperSummarizer()
    monkeypatch.setattr(summarizer, "_parse_summary_sections", lambda text: text)
    monkeypatch.setattr(summarizer, "_looks_like_valid_blog_summary", lambda text: True)

    chat = _FakeChatCompletions(chat_response or _build_chat_response(_build_summary_text()))
    responses = _FakeResponses(
        response=responses_response or _build_responses_response(_build_summary_text()),
        error=responses_error,
    )
    summarizer.client = SimpleNamespace(
        chat=SimpleNamespace(completions=chat),
        responses=responses,
    )
    return summarizer, chat, responses


def test_should_use_responses_api_for_versioned_gpt_models():
    assert PaperSummarizer._should_use_responses_api("gpt-5.4") is True
    assert PaperSummarizer._should_use_responses_api("gpt-5.4-mini") is True
    assert PaperSummarizer._should_use_responses_api("gpt-5.5") is True
    assert PaperSummarizer._should_use_responses_api("gpt-6") is True
    assert PaperSummarizer._should_use_responses_api("gpt-6-preview") is True
    assert PaperSummarizer._should_use_responses_api("gpt-5") is False
    assert PaperSummarizer._should_use_responses_api("gpt-5.3") is False
    assert PaperSummarizer._should_use_responses_api("gpt-4o") is False


def test_resolve_direct_responses_route_from_llm_yml(tmp_path, monkeypatch):
    llm_yml = tmp_path / "llm.yml"
    llm_yml.write_text(
        """
model_list:
  - model_name: gpt-5.4
    litellm_params:
      model: openai/gpt-5.4
      api_base: https://example.com/v1
      api_key: os.environ/RIGHTCODE_OPENAI_API_KEY
      max_tokens: 32000
      extra_body:
        reasoning:
          effort: low
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("RIGHTCODE_OPENAI_API_KEY", "test-key")

    import config.llm_model_order as llm_model_order

    monkeypatch.setattr(llm_model_order, "default_llm_yml_path", lambda: llm_yml)

    route = PaperSummarizer._resolve_direct_responses_route("gpt-5.4")

    assert route == {
        "base_url": "https://example.com/v1",
        "api_key": "test-key",
        "model": "gpt-5.4",
        "extra_body": {"reasoning": {"effort": "low"}},
        "max_output_tokens": 32000,
    }


def test_summarize_with_llm_uses_direct_responses_route_when_available(monkeypatch):
    summarizer, chat, responses = _make_summarizer(monkeypatch)
    direct_responses = _FakeResponses(response=_build_responses_response(_build_summary_text()))
    created_clients = []

    import tools.paper_summarizer as ps

    def fake_openai_factory(*, api_key, base_url):
        client = _FakeOpenAIClient(responses=direct_responses)
        created_clients.append({"api_key": api_key, "base_url": base_url, "client": client})
        return client

    monkeypatch.setattr(
        PaperSummarizer,
        "_resolve_direct_responses_route",
        classmethod(
            lambda cls, model_name: {
                "base_url": "https://example.com/v1",
                "api_key": "direct-key",
                "model": "gpt-5.4-upstream",
                "extra_body": {"reasoning": {"effort": "low"}},
                "max_output_tokens": 32000,
            }
        ),
    )
    monkeypatch.setattr(ps.openai, "OpenAI", fake_openai_factory)

    result = summarizer.summarize_with_llm("paper body", model="gpt-5.4", pid="1234.5678")

    assert result["content"].startswith("# Title")
    assert not responses.calls
    assert not chat.calls
    assert len(created_clients) == 1
    assert created_clients[0]["api_key"] == "direct-key"
    assert created_clients[0]["base_url"] == "https://example.com/v1"
    assert len(direct_responses.calls) == 1
    assert direct_responses.calls[0]["model"] == "gpt-5.4-upstream"
    assert direct_responses.calls[0]["reasoning"] == {"effort": "low"}
    assert direct_responses.calls[0]["max_output_tokens"] == 32000


def test_summarize_with_llm_uses_responses_api_for_gpt_5_4(monkeypatch):
    summarizer, chat, responses = _make_summarizer(monkeypatch)

    result = summarizer.summarize_with_llm("paper body", model="gpt-5.4", pid="1234.5678")

    assert result["content"].startswith("# Title")
    assert len(responses.calls) == 1
    assert not chat.calls
    assert responses.calls[0]["model"] == "gpt-5.4"
    assert isinstance(responses.calls[0]["input"], list)
    assert responses.calls[0]["input"][0]["role"] == "user"
    assert isinstance(responses.calls[0]["input"][0]["content"], str)
    assert responses.calls[0]["input"][0]["content"]
    assert responses.calls[0]["temperature"] == 0.3
    assert responses.calls[0]["top_p"] == 0.95
    assert responses.calls[0]["timeout"] == 123
    assert result["meta"]["llm"]["api"] == "responses"
    assert result["meta"]["llm"]["response_status"] == "completed"


def test_summarize_with_llm_uses_chat_completions_for_older_gpt(monkeypatch):
    summarizer, chat, responses = _make_summarizer(monkeypatch)

    result = summarizer.summarize_with_llm("paper body", model="gpt-5.3", pid="1234.5678")

    assert result["content"].startswith("# Title")
    assert len(chat.calls) == 1
    assert not responses.calls
    assert chat.calls[0]["model"] == "gpt-5.3"
    assert chat.calls[0]["temperature"] == 0.3
    assert chat.calls[0]["top_p"] == 0.95
    assert chat.calls[0]["tool_choice"] == "none"
    assert result["meta"]["llm"]["api"] == "chat_completions"
    assert result["meta"]["llm"]["finish_reason"] == "stop"


def test_summarize_with_llm_parses_proxy_wrapped_responses_payload(monkeypatch):
    summarizer, chat, responses = _make_summarizer(
        monkeypatch,
        responses_response=_build_proxy_wrapped_responses_response(_build_summary_text()),
    )

    result = summarizer.summarize_with_llm("paper body", model="gpt-5.4", pid="1234.5678")

    assert result["content"].startswith("# Title")
    assert len(responses.calls) == 1
    assert not chat.calls
    assert result["meta"]["llm"]["api"] == "responses"
    assert result["meta"]["llm"]["response_status"] == "completed"
    assert result["meta"]["llm"]["usage"]["total_tokens"] == 30


def test_summarize_with_llm_falls_back_to_chat_when_responses_endpoint_missing(
    monkeypatch,
):
    summarizer, chat, responses = _make_summarizer(
        monkeypatch,
        responses_error=RuntimeError("404 not found for /responses endpoint"),
    )

    result = summarizer.summarize_with_llm("paper body", model="gpt-5.4", pid="1234.5678")

    assert result["content"].startswith("# Title")
    assert len(responses.calls) == 1
    assert len(chat.calls) == 1
    assert chat.calls[0]["model"] == "gpt-5.4"
    assert chat.calls[0]["temperature"] == 0.3
    assert chat.calls[0]["top_p"] == 0.95
    assert result["meta"]["llm"]["api"] == "chat_completions"
    assert result["meta"]["llm"]["finish_reason"] == "stop"
