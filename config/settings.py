"""
Arxiv Sanity Configuration Management Module

Provides type-safe configuration management using pydantic-settings.
Supports loading configuration from environment variables and .env files.

Usage:
    from config.settings import settings
    print(settings.data_dir)
    print(settings.llm.base_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EmailSettings(BaseSettings):
    """Email configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_EMAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    from_email: str = Field(default="", description="Sender email address")
    smtp_server: str = Field(default="", description="SMTP server")
    smtp_port: int = Field(default=465, description="SMTP port (25=public, 465=SSL)")
    username: str = Field(default="", description="SMTP username")
    password: str = Field(default="", description="SMTP password")
    # Concurrent recommendation API calls for send_emails.py.
    # Compatible with legacy config: some docs/templates use ARXIV_SANITY_DAEMON_EMAIL_API_WORKERS.
    api_workers: int = Field(
        default=8,
        description="Email recommendation API concurrent requests",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_EMAIL_API_WORKERS",
            "ARXIV_SANITY_DAEMON_EMAIL_API_WORKERS",
        ),
    )


class LLMSettings(BaseSettings):
    """LLM configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(default="http://localhost:53000", description="LLM API base URL")
    api_key: str = Field(default="no-key", description="LLM API key")
    name: str = Field(default="deepseek-v3.2", description="Default LLM model name")
    summary_lang: str = Field(default="zh", description="Summary language (zh/en)")
    fallback_models: str = Field(default="glm-4.7,mimo-v2-flash", description="Fallback models (comma-separated)")
    # LiteLLM verbose logging switch (for bin/litellm.sh)
    litellm_verbose: bool = Field(default=False, description="LiteLLM verbose logging mode")

    @property
    def fallback_model_list(self) -> list[str]:
        """Get fallback model list"""
        models: str = self.fallback_models  # type: ignore[assignment]
        return [m.strip() for m in models.split(",") if m.strip()]


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_EMBED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    port: int = Field(default=54000, description="Ollama embedding service port")
    use_llm_api: bool = Field(default=False, description="Whether to use LLM API for embedding")
    model_name: str = Field(default="qwen3-embedding:0.6b", description="Embedding model name")
    api_base: str = Field(default="", description="Embedding API base URL (empty uses LLM_BASE_URL)")
    api_key: str = Field(default="", description="Embedding API key (empty uses LLM_API_KEY)")


class MinerUSettings(BaseSettings):
    """MinerU PDF parsing configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_MINERU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Whether to enable MinerU")
    port: int = Field(default=52000, description="MinerU vLLM service port")
    backend: Literal["pipeline", "vlm-http-client", "api"] = Field(default="api", description="MinerU backend type")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Compute device (pipeline backend)")
    max_workers: int = Field(default=2, description="Max concurrent processes (pipeline backend)")
    max_vram: int = Field(default=4, description="Max VRAM per process GB (pipeline+cuda)")
    # Compatible: supports both structured prefix variables and legacy MINERU_API_KEY.
    api_key: str = Field(
        default="",
        description="MinerU API key",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_MINERU_API_KEY",
            "MINERU_API_KEY",
        ),
    )
    # Compatible: README/.env.example has variables without ARXIV_SANITY_MINERU_ prefix.
    api_poll_interval: int = Field(
        default=5,
        description="API polling interval in seconds",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_MINERU_API_POLL_INTERVAL",
            "MINERU_API_POLL_INTERVAL",
        ),
    )
    api_timeout: int = Field(
        default=600,
        description="API timeout in seconds",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_MINERU_API_TIMEOUT",
            "MINERU_API_TIMEOUT",
        ),
    )


class SummarySettings(BaseSettings):
    """Summary generation configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_SUMMARY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    min_chinese_ratio: float = Field(default=0.25, description="Minimum Chinese character ratio threshold")
    default_semantic_weight: float = Field(default=0.5, description="Semantic search weight in hybrid search")
    # Compatible with legacy environment variable: ARXIV_SANITY_SUMMARY_SOURCE.
    # Also supports more consistent naming: ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE.
    markdown_source: Literal["html", "mineru"] = Field(
        default="html",
        description="Markdown source",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_SUMMARY_SOURCE",
            "ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE",
        ),
    )
    # Compatible with both ARXIV_SANITY_SUMMARY_HTML_SOURCES and ARXIV_SANITY_HTML_SOURCES.
    html_sources: str = Field(
        default="ar5iv,arxiv",
        description="HTML source order (comma-separated)",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_SUMMARY_HTML_SOURCES",
            "ARXIV_SANITY_HTML_SOURCES",
        ),
    )
    batch_num: int = Field(default=500, description="Batch summary generation count")

    @property
    def html_source_list(self) -> list[str]:
        """Get HTML source list"""
        sources: str = self.html_sources  # type: ignore[assignment]
        return [s.strip() for s in sources.split(",") if s.strip()]


class SVMSettings(BaseSettings):
    """SVM classifier configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_SVM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    c: float = Field(default=0.02, description="SVM C parameter")
    max_iter: int = Field(default=5000, description="Maximum iterations")
    tol: float = Field(default=1e-3, description="Convergence tolerance")
    neg_weight: float = Field(default=5.0, description="Negative feedback sample weight")


class DaemonSettings(BaseSettings):
    """Daemon scheduled task configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_DAEMON_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paper fetching configuration
    fetch_num: int = Field(default=2000, description="Papers to fetch per run")
    fetch_max: int = Field(default=1000, description="Max papers per API query")

    # Summary generation configuration
    summary_num: int = Field(default=200, description="Summaries to generate per run")
    summary_workers: int = Field(default=2, description="Summary generation concurrent workers")
    enable_summary: bool = Field(default=True, description="Enable summary generation")
    enable_embeddings: bool = Field(default=True, description="Enable embedding computation")

    # Priority queue configuration
    enable_priority_queue: bool = Field(default=True, description="Enable priority queue")
    enable_summary_queue: bool = Field(default=True, description="Enable summary queue")
    priority_days: float = Field(default=2.0, description="Priority process papers from last N days")
    priority_limit: int = Field(default=100, description="Priority queue max size")

    # Email configuration
    email_dry_run: bool = Field(default=False, description="Email dry-run mode (no actual sending)")

    # Backup configuration
    enable_git_backup: bool = Field(default=True, description="Enable git backup")

    # Scheduler timezone
    timezone: str = Field(default="Asia/Shanghai", description="Scheduler timezone")


class HueySettings(BaseSettings):
    """Huey task queue configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_HUEY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: str = Field(default="", description="Huey database path (default data_dir/huey.db)")
    workers: int = Field(default=4, description="Huey worker count")
    worker_type: str = Field(default="thread", description="Huey worker type (thread/process/greenlet)")

    # Summary task priority
    summary_priority_high: int = Field(default=100, description="High priority summary task")
    summary_priority_low: int = Field(default=10, description="Low priority summary task")

    # Summary repair configuration
    summary_repair_on_start: bool = Field(default=True, description="Repair stuck tasks on startup")
    summary_repair_ttl: int = Field(default=3600, description="Repair task TTL (seconds)")
    summary_repair_requeue: bool = Field(default=False, description="Requeue on repair")
    summary_repair_enable: bool = Field(default=True, description="Enable periodic repair")
    summary_repair_interval: int = Field(default=900, description="Repair check interval (seconds)")
    force_repair: bool = Field(default=False, description="Force repair")

    # SSE configuration
    tasks_sse_enabled: bool = Field(default=True, description="Enable task SSE event push")


class GunicornSettings(BaseSettings):
    """Gunicorn server configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_GUNICORN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workers: int = Field(default=4, description="Gunicorn worker processes")
    threads: int = Field(default=0, description="Threads per worker (0=auto)")
    preload: bool = Field(default=True, description="Preload app (copy-on-write shared memory)")
    extra_args: str = Field(default="--reload", description="Extra Gunicorn arguments")


class WebSettings(BaseSettings):
    """Web application configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cache configuration
    cache_papers: bool = Field(default=False, description="Cache papers in memory")
    warmup_data: bool = Field(default=True, description="Warm up data cache on startup")
    warmup_ml: bool = Field(default=True, description="Warm up ML models on startup")
    enable_scheduler: bool = Field(default=True, description="Enable APScheduler tasks")

    # Development configuration
    reload: bool = Field(default=False, description="Development hot reload mode")
    access_log: bool = Field(default=False, description="Enable access logging")

    # Security configuration
    secret_key: str = Field(default="", description="Flask session secret key")
    cookie_samesite: str = Field(default="Lax", description="Cookie SameSite policy")
    cookie_secure: bool = Field(default=False, description="Cookie Secure flag")
    max_content_length: int = Field(default=1048576, description="Max request body size (bytes)")

    # Cache statistics
    summary_cache_stats_refresh: int = Field(default=1800, description="Summary cache stats refresh interval (seconds)")

    # Debug switch
    enable_cache_status: bool = Field(default=False, description="Enable /cache_status debug page")


class LockSettings(BaseSettings):
    """Lock configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    summary_lock_stale_sec: float = Field(default=600, description="Summary lock stale time (seconds)")
    mineru_lock_stale_sec: float = Field(default=3600, description="MinerU lock stale time (seconds)")


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    timeout: int = Field(default=120, description="SQLite connection timeout (seconds)")
    max_retries: int = Field(default=5, description="Database operation max retries")
    retry_base_sleep: float = Field(default=0.2, description="Retry base sleep time (seconds)")


class SearchSettings(BaseSettings):
    """Search configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ret_num: int = Field(default=100, description="Papers returned per page")
    max_results: int = Field(default=1000, description="Search max results (default ret_num * 10)")

    @model_validator(mode="after")
    def set_max_results_default(self) -> SearchSettings:
        """If max_results is not set, default to ret_num * 10"""
        # Note: user may explicitly set max_results=1000; don't override in that case.
        if "max_results" not in self.model_fields_set:
            self.max_results = self.ret_num * 10
        return self


class RecommendationSettings(BaseSettings):
    """Recommendation system configuration (email recommendations, etc.)"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_RECO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_base_url: str = Field(default="", description="Recommendation API base URL (empty uses local service)")
    api_timeout: float = Field(default=120.0, description="Recommendation API timeout (seconds)")
    api_limit: int = Field(default=1000, description="Candidate papers limit per query")
    model_c: float = Field(default=0.1, description="Recommendation model C parameter")
    num_threads: int = Field(default=0, description="Thread count (0=auto)")
    max_threads: int = Field(default=192, description="Max thread count limit")
    web_name: str = Field(default="Arxiv Sanity X", description="Brand name in email templates")


class ArxivSettings(BaseSettings):
    """arXiv data collection configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_ARXIV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # arXiv category tags
    core_tags: str = Field(default="cs.AI,cs.LG,stat.ML", description="Core AI tags (comma-separated)")
    lang_tags: str = Field(default="cs.CL,cs.IR,cs.CV", description="Language/vision tags (comma-separated)")
    agent_tags: str = Field(default="cs.MA,cs.RO,cs.HC,cs.GT,cs.NE", description="Agent-related tags (comma-separated)")
    app_tags: str = Field(default="cs.SE,cs.CY", description="Application tags (comma-separated)")
    empty_response_fallback: int = Field(default=3, description="Empty response fallback count")
    # API request configuration
    api_timeout: int = Field(default=20, description="arXiv API request timeout (seconds)")

    @property
    def all_tags(self) -> list[str]:
        """Get all tags list"""
        tags = []
        for tag_str in [self.core_tags, self.lang_tags, self.agent_tags, self.app_tags]:
            tags.extend([t.strip() for t in tag_str.split(",") if t.strip()])
        return tags


class Settings(BaseSettings):
    """Main configuration class"""

    model_config = SettingsConfigDict(
        env_prefix="ARXIV_SANITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Data directories
    data_dir: Path = PROJECT_ROOT / "data"
    summary_dir: Path | None = None
    log_dir: Path | None = None  # Log directory (default data_dir/logs)

    # Service configuration
    host: str = "http://localhost:55555"
    serve_port: int = 55555
    litellm_port: int = 53000

    # Log configuration
    log_level: str = "WARNING"

    # Feature switches
    enable_swagger: bool = False

    # Other configuration
    main_content_min_ratio: float = 0.1

    # Nested configuration - use type annotation + default value directly, Pydantic v2 handles instantiation
    # This approach is more concise than Field(default_factory=...) and provides better type inference
    email: EmailSettings = EmailSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    mineru: MinerUSettings = MinerUSettings()
    summary: SummarySettings = SummarySettings()
    svm: SVMSettings = SVMSettings()
    daemon: DaemonSettings = DaemonSettings()
    huey: HueySettings = HueySettings()
    gunicorn: GunicornSettings = GunicornSettings()
    web: WebSettings = WebSettings()
    lock: LockSettings = LockSettings()
    db: DatabaseSettings = DatabaseSettings()
    search: SearchSettings = SearchSettings()
    reco: RecommendationSettings = RecommendationSettings()
    arxiv: ArxivSettings = ArxivSettings()

    # Backward-compatible alias: older code referenced settings.access_log.
    @property
    def access_log(self) -> bool:  # pragma: no cover
        return bool(self.web.access_log)

    @model_validator(mode="after")
    def set_defaults(self) -> Settings:
        """Set dependent default values"""
        if self.summary_dir is None:
            self.summary_dir = self.data_dir / "summary"
        # Set log directory default path
        if self.log_dir is None:
            self.log_dir = self.data_dir / "logs"
        # Set Huey database default path
        if not self.huey.db_path:
            self.huey.db_path = str(self.data_dir / "huey.db")
        # Set recommendation API base URL default value
        if not self.reco.api_base_url:
            self.reco.api_base_url = f"http://localhost:{self.serve_port}"
        return self

    @field_validator("data_dir", "summary_dir", "log_dir", mode="before")
    @classmethod
    def resolve_path(cls, v):
        """Resolve path"""
        if v is None:
            return v
        if isinstance(v, str):
            return Path(v)
        return v


@lru_cache
def get_settings() -> Settings:
    """Get settings singleton (with cache)"""
    return Settings()


# Global settings instance - explicit type annotation ensures IDE correctly infers type
settings: Settings = get_settings()


def reload_settings() -> Settings:
    """Reload settings (clear cache)"""
    get_settings.cache_clear()
    # Note: this does not update module-level settings variable, caller should use return value
    # To update global settings, use config package's reload_settings
    return get_settings()
