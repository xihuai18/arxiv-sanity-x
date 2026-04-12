from __future__ import annotations


def test_opencode_resolved_base_url_prefers_explicit_base_url():
    from config.settings_services import OpenCodeSettings

    cfg = OpenCodeSettings(base_url="https://opencode.example/api", host="127.0.0.1", port=53000)

    assert cfg.resolved_base_url == "https://opencode.example/api"


def test_opencode_resolved_base_url_falls_back_to_host_port():
    from config.settings_services import OpenCodeSettings

    cfg = OpenCodeSettings(base_url="", host="127.0.0.1", port=53000)

    assert cfg.resolved_base_url == "http://127.0.0.1:53000"


def test_opencode_managed_defaults_to_true():
    from config.settings_services import OpenCodeSettings

    cfg = OpenCodeSettings()

    assert cfg.managed is True
