from __future__ import annotations

from types import SimpleNamespace


def test_collect_validation_errors_on_noncanonical_models(monkeypatch):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.llm, "name", "not-a-valid-model")
    monkeypatch.setattr(settings.extract_info, "model_name", "not-a-valid-model")
    errors, warnings = cli._collect_validation_messages(settings)

    assert any(
        "LLM model must use an alias or OpenCode provider/model format" in item
        for item in errors
    )
    assert any(
        "Extract model must use an alias or OpenCode provider/model format" in item
        for item in errors
    )
    assert warnings == [] or isinstance(warnings, list)


def test_collect_validation_accepts_alias_model_names(monkeypatch):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.llm, "name", "gpt-5.4")
    monkeypatch.setattr(settings.extract_info, "model_name", "gpt-5.4-mini")
    errors, _warnings = cli._collect_validation_messages(settings)

    assert not any("provider/model format" in item for item in errors)


def test_collect_validation_requires_embed_api_base_when_llm_api_enabled(monkeypatch):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.embedding, "use_llm_api", True)
    monkeypatch.setattr(settings.embedding, "api_base", "")

    errors, _warnings = cli._collect_validation_messages(settings)

    assert any("Embedding API base URL is required" in item for item in errors)


def test_collect_validation_warns_when_managed_opencode_binary_missing(monkeypatch):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.opencode, "managed", True)
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: None if name == "opencode" else "/bin/true"
    )

    _errors, warnings = cli._collect_validation_messages(settings)

    assert any("opencode` binary is not on PATH" in item for item in warnings)


def test_cmd_env_includes_opencode_and_extract_variables(capsys):
    from config import cli

    cli.cmd_env(SimpleNamespace(include_secrets=False))
    out = capsys.readouterr().out

    assert "ARXIV_SANITY_OPENCODE_BASE_URL=" in out
    assert "ARXIV_SANITY_OPENCODE_MANAGED=" in out
    assert "ARXIV_SANITY_EXTRACT_MODEL_NAME=" in out
    assert "ARXIV_SANITY_HUEY_WORKERS=" in out
    assert "ARXIV_SANITY_SSE_ENABLED=" in out
    assert "ARXIV_SANITY_LOG_FORMAT=" in out


def test_print_env_var_preserves_empty_secret_semantics(capsys):
    from config import cli

    cli._print_env_var(
        "ARXIV_SANITY_OPENCODE_PASSWORD",
        "",
        include_secrets=False,
        secret=True,
    )

    assert capsys.readouterr().out == "ARXIV_SANITY_OPENCODE_PASSWORD=\n"


def test_cmd_show_human_readable_displays_alias_models(monkeypatch, capsys):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.llm, "name", "openai/gpt-5.4")
    monkeypatch.setattr(
        settings.extract_info, "model_name", "rightcode-openai/gpt-5.4-mini"
    )

    cli.cmd_show(SimpleNamespace(json=False, include_secrets=False))
    out = capsys.readouterr().out

    assert "model:        gpt-5.4" in out
    assert "model_name:   gpt-5.4-mini" in out
    assert "openai/gpt-5.4" not in out
    assert "rightcode-openai/gpt-5.4-mini" not in out


def test_cmd_doctor_displays_alias_models(monkeypatch, capsys):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.opencode, "managed", False)
    monkeypatch.setattr(settings.opencode, "base_url", "http://127.0.0.1:53000")
    monkeypatch.setattr(settings.llm, "name", "openai/gpt-5.4")
    monkeypatch.setattr(
        settings.extract_info, "model_name", "rightcode-openai/gpt-5.4-mini"
    )

    cli.cmd_doctor(SimpleNamespace())
    out = capsys.readouterr().out

    assert "model=gpt-5.4" in out
    assert "model=gpt-5.4-mini" in out
    assert "model=openai/gpt-5.4" not in out
    assert "model=rightcode-openai/gpt-5.4-mini" not in out
