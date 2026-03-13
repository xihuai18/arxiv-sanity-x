"""Public configuration module."""

from . import settings_main as _settings_main
from .settings_base import PROJECT_ROOT
from .settings_features import (
    ArxivSettings,
    RecommendationSettings,
    SearchSettings,
    SentrySettings,
    SVMSettings,
)
from .settings_main import Settings, get_settings
from .settings_runtime import (
    DaemonSettings,
    DatabaseSettings,
    GunicornSettings,
    HueySettings,
    LockSettings,
    SSESettings,
    WebSettings,
)
from .settings_services import (
    EmailSettings,
    EmbeddingSettings,
    ExtractInfoSettings,
    LLMSettings,
    MinerUSettings,
    SummarySettings,
)

settings = _settings_main.settings


def reload_settings() -> Settings:
    global settings
    settings = _settings_main.reload_settings()
    return settings


__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "reload_settings",
    "settings",
    "EmailSettings",
    "LLMSettings",
    "ExtractInfoSettings",
    "EmbeddingSettings",
    "MinerUSettings",
    "SummarySettings",
    "SVMSettings",
    "DaemonSettings",
    "HueySettings",
    "SSESettings",
    "GunicornSettings",
    "WebSettings",
    "LockSettings",
    "DatabaseSettings",
    "SearchSettings",
    "RecommendationSettings",
    "SentrySettings",
    "ArxivSettings",
]
