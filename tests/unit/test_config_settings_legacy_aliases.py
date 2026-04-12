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


def test_summary_image_compression_settings_accept_env(monkeypatch):
    from config.settings_services import SummarySettings

    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_IMAGE_COMPRESSION_ENABLED", "false")
    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_IMAGE_MAX_LONG_EDGE", "2048")
    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_IMAGE_WEBP_QUALITY", "90")
    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_IMAGE_MIN_SAVINGS_BYTES", "4096")
    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_IMAGE_SKIP_BELOW_BYTES", "65536")

    settings = SummarySettings(_env_file=None)

    assert settings.image_compression_enabled is False
    assert settings.image_max_long_edge == 2048
    assert settings.image_webp_quality == 90
    assert settings.image_min_savings_bytes == 4096
    assert settings.image_skip_below_bytes == 65536


def test_summary_image_max_long_edge_defaults_to_disabled(monkeypatch):
    from config.settings_services import SummarySettings

    monkeypatch.delenv("ARXIV_SANITY_SUMMARY_IMAGE_MAX_LONG_EDGE", raising=False)

    settings = SummarySettings(_env_file=None)

    assert settings.image_max_long_edge == 0
