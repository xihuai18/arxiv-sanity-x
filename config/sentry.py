"""Sentry initialization helper (optional).

This module intentionally does not import Flask at import time to keep startup
robust across different entrypoints (web, huey workers, daemon scripts).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

_SENTRY_INITIALIZED = False


def initialize_sentry(*, settings_obj: Any | None = None) -> bool:
    """Initialize Sentry if configured.

    This is a no-op unless both:
    - settings.sentry.enabled is True
    - settings.sentry.dsn is a non-empty string
    """
    global _SENTRY_INITIALIZED
    if _SENTRY_INITIALIZED:
        return True

    try:
        from config import settings as _settings

        settings = settings_obj or _settings
        sentry_cfg = getattr(settings, "sentry", None)
        enabled = bool(getattr(sentry_cfg, "enabled", False))
        dsn = str(getattr(sentry_cfg, "dsn", "") or "").strip()
        if not enabled or not dsn:
            return False

        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        integrations: list[Any] = [
            LoggingIntegration(level=None, event_level="ERROR"),
        ]

        # Best-effort: enable Flask integration when available (web context).
        try:
            from sentry_sdk.integrations.flask import FlaskIntegration

            integrations.append(FlaskIntegration())
        except Exception:
            pass

        init_kwargs: dict[str, Any] = {
            "dsn": dsn,
            "integrations": integrations,
            "send_default_pii": False,
            "attach_stacktrace": True,
        }

        environment = str(getattr(sentry_cfg, "environment", "") or "").strip()
        release = str(getattr(sentry_cfg, "release", "") or "").strip()
        traces_sample_rate = float(getattr(sentry_cfg, "traces_sample_rate", 0.0) or 0.0)
        profiles_sample_rate = float(getattr(sentry_cfg, "profiles_sample_rate", 0.0) or 0.0)

        if environment:
            init_kwargs["environment"] = environment
        if release:
            init_kwargs["release"] = release
        if traces_sample_rate > 0:
            init_kwargs["traces_sample_rate"] = traces_sample_rate
        if profiles_sample_rate > 0:
            init_kwargs["profiles_sample_rate"] = profiles_sample_rate

        sentry_sdk.init(**init_kwargs)
        _SENTRY_INITIALIZED = True
        logger.info("Sentry initialized")
        return True

    except Exception:
        logger.opt(exception=True).warning("Failed to initialize Sentry (ignored)")
        return False

