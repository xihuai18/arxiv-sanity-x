from __future__ import annotations

from types import SimpleNamespace


def test_collect_validation_warns_when_extract_alias_assumes_nonlocal_gateway(
    monkeypatch,
):
    from config import cli
    from config.settings import settings

    monkeypatch.setattr(settings.llm, "base_url", "https://api.example.com/v1")
    monkeypatch.setattr(settings.llm, "name", "gpt-5.4")
    monkeypatch.setattr(settings.extract_info, "model_name", "qwen3.5-plus")
    monkeypatch.setattr(settings.extract_info, "base_url", "")

    _errors, warnings = cli._collect_validation_messages(settings)

    assert any("ARXIV_SANITY_EXTRACT_BASE_URL is empty" in item for item in warnings)


def test_collect_validation_warns_when_local_gateway_lacks_model_aliases(monkeypatch):
    from config import cli, llm_model_order
    from config.settings import settings

    monkeypatch.setattr(settings.llm, "base_url", "http://localhost:53000")
    monkeypatch.setattr(settings.llm, "name", "gpt-5.4")
    monkeypatch.setattr(settings.extract_info, "model_name", "qwen3.5-plus")
    monkeypatch.setattr(settings.extract_info, "base_url", "")
    monkeypatch.setattr(llm_model_order, "read_llm_yml_model_order", lambda path=None: ["or-mimo"])

    _errors, warnings = cli._collect_validation_messages(settings)

    assert any("Main LLM model `gpt-5.4` is not declared" in item for item in warnings)
    assert any("Extract model `qwen3.5-plus` is not declared" in item for item in warnings)


def test_collect_validation_warns_when_local_gateway_alias_validation_is_skipped(
    monkeypatch,
):
    from config import cli, llm_model_order
    from config.settings import settings

    monkeypatch.setattr(settings.llm, "base_url", "http://localhost:53000")
    monkeypatch.setattr(settings.llm, "name", "gpt-5.4")
    monkeypatch.setattr(settings.extract_info, "model_name", "qwen3.5-plus")
    monkeypatch.setattr(settings.extract_info, "base_url", "")
    monkeypatch.setattr(
        llm_model_order,
        "read_llm_yml_model_order",
        lambda path=None: (_ for _ in ()).throw(FileNotFoundError("missing config/llm.yml")),
    )

    _errors, warnings = cli._collect_validation_messages(settings)

    assert any("model alias validation is skipped" in item for item in warnings)
    assert any("config/llm.yml" in item for item in warnings)


def test_cmd_env_includes_extract_variables(capsys):
    from config import cli

    cli.cmd_env(SimpleNamespace(include_secrets=False))
    out = capsys.readouterr().out

    assert "ARXIV_SANITY_EXTRACT_MODEL_NAME=" in out
    assert "ARXIV_SANITY_EXTRACT_BASE_URL=" in out
    assert "ARXIV_SANITY_EXTRACT_API_KEY=" in out
    assert "ARXIV_SANITY_EXTRACT_TEMPERATURE=" in out
    assert "ARXIV_SANITY_LLM_FALLBACK_MODELS=" in out
    assert "ARXIV_SANITY_HUEY_WORKERS=" in out
    assert "ARXIV_SANITY_SSE_ENABLED=" in out
    assert "ARXIV_SANITY_GUNICORN_WORKER_CLASS=" in out
    assert "ARXIV_SANITY_LOG_FORMAT=" in out
    assert "ARXIV_SANITY_ENABLE_SWAGGER=" in out


def test_print_env_var_preserves_empty_secret_semantics(capsys):
    from config import cli

    cli._print_env_var(
        "ARXIV_SANITY_EXTRACT_API_KEY",
        "",
        include_secrets=False,
        secret=True,
    )

    assert capsys.readouterr().out == "ARXIV_SANITY_EXTRACT_API_KEY=\n"
