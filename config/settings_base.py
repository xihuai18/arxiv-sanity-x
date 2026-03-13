from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_FILE_ENCODING = "utf-8"


def group_model_config(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )


def root_model_config() -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix="ARXIV_SANITY_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        env_nested_delimiter="__",
        extra="ignore",
    )


class SettingsGroup(BaseSettings):
    pass
