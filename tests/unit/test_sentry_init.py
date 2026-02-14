from __future__ import annotations

import importlib
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _reload_sentry_module():
    import config.sentry as sentry_mod

    return importlib.reload(sentry_mod)


def test_initialize_sentry_disabled_is_noop():
    sentry_mod = _reload_sentry_module()

    settings_obj = SimpleNamespace(
        sentry=SimpleNamespace(
            enabled=False,
            dsn="https://public@example.invalid/1",
            environment="test",
            release="unit-test",
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
    )

    # When disabled, this should be a pure no-op (even if sentry_sdk is not installed).
    assert sentry_mod.initialize_sentry(settings_obj=settings_obj) is False


def test_initialize_sentry_enabled_calls_sdk_init(monkeypatch):
    sentry_mod = _reload_sentry_module()

    fake_sentry_sdk = types.ModuleType("sentry_sdk")
    fake_sentry_sdk.init = MagicMock()

    fake_integrations = types.ModuleType("sentry_sdk.integrations")
    fake_logging = types.ModuleType("sentry_sdk.integrations.logging")
    fake_flask = types.ModuleType("sentry_sdk.integrations.flask")

    class LoggingIntegration:
        def __init__(self, level=None, event_level=None):
            self.level = level
            self.event_level = event_level

    class FlaskIntegration:
        pass

    fake_logging.LoggingIntegration = LoggingIntegration
    fake_flask.FlaskIntegration = FlaskIntegration

    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", fake_sentry_sdk)
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk.integrations", fake_integrations)
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk.integrations.logging", fake_logging)
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk.integrations.flask", fake_flask)

    settings_obj = SimpleNamespace(
        sentry=SimpleNamespace(
            enabled=True,
            dsn="https://public@example.invalid/1",
            environment="test",
            release="unit-test",
            traces_sample_rate=0.25,
            profiles_sample_rate=0.0,
        )
    )

    assert sentry_mod.initialize_sentry(settings_obj=settings_obj) is True
    assert fake_sentry_sdk.init.call_count == 1
    kwargs = fake_sentry_sdk.init.call_args.kwargs
    assert kwargs.get("dsn") == "https://public@example.invalid/1"
    assert kwargs.get("environment") == "test"
    assert kwargs.get("release") == "unit-test"
    assert kwargs.get("traces_sample_rate") == 0.25
    assert kwargs.get("send_default_pii") is False
