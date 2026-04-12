from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from .settings_base import PROJECT_ROOT, root_model_config
from .settings_features import (
    ArxivSettings,
    RecommendationSettings,
    SearchSettings,
    SentrySettings,
    SVMSettings,
)
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
    OpenCodeSettings,
    SummarySettings,
)


class Settings(BaseSettings):
    """Main configuration class."""

    model_config = root_model_config()

    data_dir: Path = PROJECT_ROOT / "data"
    summary_dir: Path | None = None
    log_dir: Path | None = None
    host: str = "http://localhost:55555"
    serve_port: int = 55555
    process_role: Literal["", "web", "worker"] = Field(
        default="",
        description="Optional process role override for SQLite timeout/retry tuning",
    )
    log_level: str = "WARNING"
    log_format: Literal["text", "json"] = Field(default="text", description="Log output format")
    enable_swagger: bool = False
    main_content_min_ratio: float = 0.1

    email: EmailSettings = Field(default_factory=EmailSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    opencode: OpenCodeSettings = Field(default_factory=OpenCodeSettings)
    extract_info: ExtractInfoSettings = Field(default_factory=ExtractInfoSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)
    summary: SummarySettings = Field(default_factory=SummarySettings)
    svm: SVMSettings = Field(default_factory=SVMSettings)
    daemon: DaemonSettings = Field(default_factory=DaemonSettings)
    huey: HueySettings = Field(default_factory=HueySettings)
    sse: SSESettings = Field(default_factory=SSESettings)
    gunicorn: GunicornSettings = Field(default_factory=GunicornSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    lock: LockSettings = Field(default_factory=LockSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    reco: RecommendationSettings = Field(default_factory=RecommendationSettings)
    sentry: SentrySettings = Field(default_factory=SentrySettings)
    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)

    @property
    def access_log(self) -> bool:  # pragma: no cover
        return bool(self.web.access_log)

    @model_validator(mode="after")
    def set_defaults(self) -> Settings:
        if self.summary_dir is None:
            self.summary_dir = self.data_dir / "summary"
        if self.log_dir is None:
            self.log_dir = self.data_dir / "logs"
        if not self.huey.db_path:
            self.huey.db_path = str(self.data_dir / "huey.db")
        if not self.sse.db_path:
            self.sse.db_path = str(self.data_dir / "sse_events.db")
        if not self.reco.api_base_url:
            self.reco.api_base_url = f"http://localhost:{self.serve_port}"
        if not self.extract_info.model_name:
            self.extract_info.model_name = self.llm.name
        return self

    @field_validator("data_dir", "summary_dir", "log_dir", mode="before")
    @classmethod
    def resolve_path(cls, value):
        if value is None:
            return value
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve()
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()


def reload_settings() -> Settings:
    global settings
    get_settings.cache_clear()
    settings = get_settings()
    return settings
