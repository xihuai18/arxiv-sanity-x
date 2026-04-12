from __future__ import annotations


def test_extract_metadata_uses_structured_opencode_result(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "openai/gpt-5.4-mini")
    monkeypatch.setattr(upload_service.settings.llm, "name", "openai/gpt-5.4")
    monkeypatch.setattr(
        upload_service,
        "generate_structured_json",
        lambda **kwargs: {
            "json": {
                "title": "Paper Title",
                "authors": ["Alice", "Bob"],
                "year": 2026,
                "abstract": "Abstract",
            },
            "resolved_model": kwargs["model"],
        },
    )

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract")

    assert result == {
        "title": "Paper Title",
        "authors": ["Alice", "Bob"],
        "year": None,
        "abstract": "Abstract",
    }


def test_extract_metadata_falls_back_from_extract_model_to_default(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "openai/gpt-5.4-mini")
    monkeypatch.setattr(upload_service.settings.llm, "name", "openai/gpt-5.4")

    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai/gpt-5.4-mini":
            raise RuntimeError("temporary extract failure")
        return {
            "json": {
                "title": "Fallback Title",
                "authors": ["Carol"],
                "year": None,
                "abstract": "Fallback abstract",
            },
            "resolved_model": kwargs["model"],
        }

    monkeypatch.setattr(upload_service, "generate_structured_json", fake_generate)

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract")

    assert calls == [
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4",
    ]
    assert result["title"] == "Fallback Title"


def test_extract_metadata_returns_empty_when_all_models_fail(monkeypatch):
    from backend.services import upload_service

    monkeypatch.setattr(upload_service.settings.extract_info, "model_name", "openai/gpt-5.4-mini")
    monkeypatch.setattr(upload_service.settings.llm, "name", "openai/gpt-5.4")
    monkeypatch.setattr(
        upload_service,
        "generate_structured_json",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(f"failed: {kwargs['model']}")),
    )

    result = upload_service.extract_metadata_with_llm("# Title\n\nAbstract")

    assert result == {"title": "", "authors": [], "year": None, "abstract": None}
