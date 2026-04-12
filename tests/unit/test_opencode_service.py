from __future__ import annotations


def test_generate_text_returns_text_and_resolved_model(monkeypatch):
    from backend.services import opencode_service as svc

    calls = []

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        calls.append((method, path, json_body, timeout))
        if path == "/session":
            return {"id": "ses_test"}
        if path == "/session/ses_test/message":
            return {
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-5.4",
                    "finish": "stop",
                    "id": "msg_test",
                    "tokens": {
                        "input": 11,
                        "output": 22,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [{"type": "text", "text": "hello world"}],
            }
        assert path == "/session/ses_test"
        return True

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.generate_text(
        model="openai/gpt-5.4",
        system="Return plain text only.",
        prompt="Say hello world.",
        timeout=12,
    )

    assert result["text"] == "hello world"
    assert result["resolved_model"] == "openai/gpt-5.4"
    assert result["provider"] == "openai"
    assert result["usage"]["input"] == 11
    assert calls[1][2]["format"] == {"type": "text"}


def test_generate_structured_json_prefers_structured_field(monkeypatch):
    from backend.services import opencode_service as svc

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        del method, json_body, timeout
        if path == "/session":
            return {"id": "ses_json"}
        if path == "/session/ses_json/message":
            return {
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-5.4",
                    "structured": {"answer": "ping"},
                    "tokens": {
                        "input": 1,
                        "output": 2,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [{"type": "tool", "tool": "StructuredOutput"}],
            }
        return True

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.generate_structured_json(
        model="openai/gpt-5.4",
        prompt="Return answer=ping",
        schema={"type": "object"},
    )

    assert result["json"] == {"answer": "ping"}
    assert result["resolved_model"] == "openai/gpt-5.4"


def test_generate_structured_json_falls_back_to_text_json(monkeypatch):
    from backend.services import opencode_service as svc

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        del method, json_body, timeout
        if path == "/session":
            return {"id": "ses_json_text"}
        if path == "/session/ses_json_text/message":
            return {
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-5.4-mini",
                    "tokens": {
                        "input": 1,
                        "output": 2,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [{"type": "text", "text": 'Here is JSON: {"answer": "pong"}'}],
            }
        return True

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.generate_structured_json(
        model="openai/gpt-5.4-mini",
        prompt="Return answer=pong",
        schema={"type": "object"},
    )

    assert result["json"] == {"answer": "pong"}
    assert result["text"] == 'Here is JSON: {"answer": "pong"}'


def test_generate_structured_json_prefers_last_json_block(monkeypatch):
    from backend.services import opencode_service as svc

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        del method, json_body, timeout
        if path == "/session":
            return {"id": "ses_json_last"}
        if path == "/session/ses_json_last/message":
            return {
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-5.4-mini",
                    "tokens": {
                        "input": 1,
                        "output": 2,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [
                    {
                        "type": "text",
                        "text": 'Example: {"answer": "draft"}\nFinal: {"answer": "final"}',
                    }
                ],
            }
        return True

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.generate_structured_json(
        model="openai/gpt-5.4-mini",
        prompt="Return answer=final",
        schema={"type": "object"},
    )

    assert result["json"] == {"answer": "final"}


def test_generate_structured_json_ignores_nested_json_candidates(monkeypatch):
    from backend.services import opencode_service as svc

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        del method, json_body, timeout
        if path == "/session":
            return {"id": "ses_json_nested"}
        if path == "/session/ses_json_nested/message":
            return {
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-5.4-mini",
                    "tokens": {
                        "input": 1,
                        "output": 2,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [
                    {
                        "type": "text",
                        "text": 'Example: {"outer": {"inner": 1}, "items": [1, 2]}\nFinal: {"answer": "final"}',
                    }
                ],
            }
        return True

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.generate_structured_json(
        model="openai/gpt-5.4-mini",
        prompt="Return answer=final",
        schema={"type": "object"},
    )

    assert result["json"] == {"answer": "final"}


def test_list_models_handles_string_error_list(monkeypatch):
    from backend.services import opencode_service as svc

    class _FakeResponse:
        status_code = 400
        text = ""

    error = svc._classify_http_error(_FakeResponse(), {"error": ["bad request", "invalid model"]})

    assert error.kind == "http_error"
    assert str(error) == "bad request; invalid model"


def test_list_models_parses_provider_payload(monkeypatch):
    from backend.services import opencode_service as svc

    monkeypatch.setattr(
        svc,
        "_request_json",
        lambda method, path, *, json_body=None, timeout=None: {
            "providers": [
                {
                    "id": "openai",
                    "name": "OpenAI",
                    "source": "env",
                    "models": {
                        "gpt-5.4": {"id": "gpt-5.4", "name": "GPT-5.4"},
                        "gpt-5.4-mini": {"id": "gpt-5.4-mini", "name": "GPT-5.4 Mini"},
                    },
                }
            ],
            "default": {"openai": "gpt-5.4"},
        },
    )

    models = svc.list_models()

    assert [item["id"] for item in models] == [
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
    ]
    assert models[0]["default"] is True


def test_build_llm_models_api_payload_collapses_alias_models(monkeypatch):
    from backend.services import opencode_service as svc

    monkeypatch.setattr(svc.settings.llm, "name", "openai/gpt-5.4")

    payload = svc.build_llm_models_api_payload(
        [
            {"id": "openai/gpt-5.4"},
            {"id": "rightcode-openai/gpt-5.4"},
            {"id": "anthropic/claude-sonnet-4-6"},
        ]
    )

    assert payload == {
        "models": [{"id": "gpt-5.4"}],
        "default": "gpt-5.4",
    }


def test_build_llm_models_api_payload_ignores_non_alias_default(monkeypatch):
    from backend.services import opencode_service as svc

    monkeypatch.setattr(svc.settings.llm, "name", "anthropic/claude-sonnet-4-6")

    payload = svc.build_llm_models_api_payload(
        [
            {"id": "openai/gpt-5.4"},
            {"id": "anthropic/claude-sonnet-4-6"},
        ]
    )

    assert payload == {
        "models": [{"id": "gpt-5.4"}],
        "default": "",
    }


def test_list_models_skips_invalid_provider_model_ids(monkeypatch):
    from backend.services import opencode_service as svc

    monkeypatch.setattr(
        svc,
        "_request_json",
        lambda method, path, *, json_body=None, timeout=None: {
            "providers": [
                {
                    "id": "openai",
                    "name": "OpenAI",
                    "source": "env",
                    "models": {"gpt-5.4": {"id": "gpt-5.4", "name": "GPT-5.4"}},
                },
                {
                    "id": "modelscope/Qwen",
                    "name": "ModelScope",
                    "source": "env",
                    "models": {
                        "Qwen3-30B-A3B-Thinking-2507": {
                            "id": "Qwen3-30B-A3B-Thinking-2507",
                            "name": "Qwen3-30B-A3B-Thinking-2507",
                        }
                    },
                },
            ],
            "default": {"openai": "gpt-5.4"},
        },
    )

    models = svc.list_models()

    assert [item["id"] for item in models] == ["openai/gpt-5.4"]


def test_healthcheck_reports_missing_required_models(monkeypatch):
    from backend.services import opencode_service as svc

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        del method, json_body, timeout
        if path == "/global/health":
            return {"healthy": True, "version": "1.4.3"}
        if path == "/config/providers":
            return {
                "providers": [
                    {
                        "id": "openai",
                        "name": "OpenAI",
                        "source": "env",
                        "models": {"gpt-5.4": {"id": "gpt-5.4", "name": "GPT-5.4"}},
                    }
                ],
                "default": {"openai": "gpt-5.4"},
            }
        raise AssertionError(path)

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.healthcheck(default_model="openai/gpt-5.4", probe=False)

    assert result["ok"] is True
    assert result["required"]["missing"] == []


def test_healthcheck_accepts_alias_when_any_member_exists(monkeypatch):
    from backend.services import opencode_service as svc

    def fake_request_json(method, path, *, json_body=None, timeout=None):
        del method, json_body, timeout
        if path == "/global/health":
            return {"healthy": True, "version": "1.4.3"}
        if path == "/config/providers":
            return {
                "providers": [
                    {
                        "id": "openai",
                        "name": "OpenAI",
                        "source": "env",
                        "models": {"gpt-5.4": {"id": "gpt-5.4", "name": "GPT-5.4"}},
                    }
                ],
                "default": {"openai": "gpt-5.4"},
            }
        raise AssertionError(path)

    monkeypatch.setattr(svc, "_request_json", fake_request_json)

    result = svc.healthcheck(default_model="gpt-5.4", probe=False)

    assert result["ok"] is True
    assert result["required"]["required"] == ["gpt-5.4"]
    assert result["required"]["missing"] == []
