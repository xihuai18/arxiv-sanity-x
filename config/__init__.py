# Config package
# Contains all configuration files for the project
#
# Usage:
#   from config import settings
#   print(settings.data_dir)
#   print(settings.llm.base_url)
#
# CLI tool:
#   python -m config.cli show      # Show configuration
#   python -m config.cli validate  # Validate configuration

from typing import TYPE_CHECKING, cast

from config.settings import Settings, get_settings, reload_settings
from config.settings import settings as _settings

# Use cast to help IDE infer the type of settings correctly
# This does not affect runtime behavior, only helps the type checker
if TYPE_CHECKING:
    settings: Settings = _settings
else:
    settings = cast(Settings, _settings)

__all__ = ["Settings", "get_settings", "reload_settings", "settings"]
