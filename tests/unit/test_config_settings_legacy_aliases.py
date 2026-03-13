from __future__ import annotations


def test_summary_settings_accept_legacy_alias_env(monkeypatch):
    from config.settings_services import SummarySettings

    monkeypatch.delenv("ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE", raising=False)
    monkeypatch.delenv("ARXIV_SANITY_SUMMARY_HTML_SOURCES", raising=False)
    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_SOURCE", "mineru")
    monkeypatch.setenv("ARXIV_SANITY_HTML_SOURCES", "arxiv")

    settings = SummarySettings(_env_file=None)

    assert settings.markdown_source == "mineru"
    assert settings.html_sources == "arxiv"


def test_mineru_settings_accept_legacy_alias_env(monkeypatch):
    from config.settings_services import MinerUSettings

    monkeypatch.delenv("ARXIV_SANITY_MINERU_API_KEY", raising=False)
    monkeypatch.delenv("ARXIV_SANITY_MINERU_API_POLL_INTERVAL", raising=False)
    monkeypatch.delenv("ARXIV_SANITY_MINERU_API_TIMEOUT", raising=False)
    monkeypatch.setenv("MINERU_API_KEY", "legacy-mineru-key")
    monkeypatch.setenv("MINERU_API_POLL_INTERVAL", "11")
    monkeypatch.setenv("MINERU_API_TIMEOUT", "22")

    settings = MinerUSettings(_env_file=None)

    assert settings.api_key == "legacy-mineru-key"
    assert settings.api_poll_interval == 11
    assert settings.api_timeout == 22


def test_email_settings_accept_legacy_alias_env(monkeypatch):
    from config.settings_services import EmailSettings

    monkeypatch.delenv("ARXIV_SANITY_EMAIL_API_WORKERS", raising=False)
    monkeypatch.setenv("ARXIV_SANITY_DAEMON_EMAIL_API_WORKERS", "13")

    settings = EmailSettings(_env_file=None)

    assert settings.api_workers == 13
