"""Unit tests for config reload behavior."""

from __future__ import annotations

import importlib


def test_reload_settings_updates_module_binding(monkeypatch):
    """reload_settings should update both `config.settings` and `config.settings.settings`."""
    import config

    settings_module = importlib.import_module("config.settings")

    old = config.settings

    monkeypatch.setenv("ARXIV_SANITY_LLM_BASE_URL", "http://example.invalid:1234")
    new = config.reload_settings()

    assert new is not old
    assert new is config.settings
    assert new is settings_module.settings
    assert new.llm.base_url == "http://example.invalid:1234"
