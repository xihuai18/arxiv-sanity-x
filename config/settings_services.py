from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field

from .settings_base import SettingsGroup, group_model_config


class EmailSettings(SettingsGroup):
    """Email configuration."""

    model_config = group_model_config("ARXIV_SANITY_EMAIL_")

    from_email: str = Field(default="", description="Sender email address")
    smtp_server: str = Field(default="", description="SMTP server")
    smtp_port: int = Field(default=465, description="SMTP port (25=public, 465=SSL)")
    username: str = Field(default="", description="SMTP username")
    password: str = Field(default="", description="SMTP password")
    api_workers: int = Field(
        default=8,
        description="Email recommendation API concurrent requests",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_EMAIL_API_WORKERS",
            "ARXIV_SANITY_DAEMON_EMAIL_API_WORKERS",
        ),
    )


class LLMSettings(SettingsGroup):
    """LLM configuration."""

    model_config = group_model_config("ARXIV_SANITY_LLM_")

    name: str = Field(
        default="openai/gpt-5.4",
        description="Default OpenCode text model id (provider/model)",
    )
    summary_lang: str = Field(default="zh", description="Summary language (zh/en)")
    timeout: int = Field(default=600, description="LLM request timeout (seconds)")


class OpenCodeSettings(SettingsGroup):
    """OpenCode HTTP server configuration."""

    model_config = group_model_config("ARXIV_SANITY_OPENCODE_")

    base_url: str = Field(default="", description="OpenCode server base URL")
    managed: bool = Field(
        default=True,
        description="Whether bin/run_services.py should launch OpenCode locally",
    )
    host: str = Field(default="127.0.0.1", description="Managed OpenCode host")
    port: int = Field(default=53000, description="Managed OpenCode port")
    username: str = Field(default="", description="Optional OpenCode basic-auth username")
    password: str = Field(default="", description="Optional OpenCode basic-auth password")
    timeout: int = Field(default=600, description="OpenCode request timeout (seconds)")

    @property
    def resolved_base_url(self) -> str:
        base = str(self.base_url or "").strip().rstrip("/")
        if base:
            return base
        host = str(self.host or "").strip() or "127.0.0.1"
        return f"http://{host}:{int(self.port)}"


class ExtractInfoSettings(SettingsGroup):
    """Metadata extraction model configuration."""

    model_config = group_model_config("ARXIV_SANITY_EXTRACT_")

    model_name: str = Field(
        default="",
        description="OpenCode text model for metadata extraction (empty uses llm.name)",
    )
    timeout: int = Field(default=600, description="Request timeout in seconds")


class EmbeddingSettings(SettingsGroup):
    """Embedding service configuration."""

    model_config = group_model_config("ARXIV_SANITY_EMBED_")

    port: int = Field(default=54000, description="Ollama embedding service port")
    use_llm_api: bool = Field(default=False, description="Whether to use LLM API for embedding")
    model_name: str = Field(default="qwen3-embedding:0.6b", description="Embedding model name")
    api_base: str = Field(default="", description="Embedding API base URL")
    api_key: str = Field(default="", description="Embedding API key")


class MinerUSettings(SettingsGroup):
    """MinerU PDF parsing configuration."""

    model_config = group_model_config("ARXIV_SANITY_MINERU_")

    enabled: bool = Field(default=False, description="Whether to enable MinerU")
    port: int = Field(default=52000, description="MinerU vLLM service port")
    backend: Literal["pipeline", "vlm-http-client", "api"] = Field(default="api", description="MinerU backend type")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Compute device (pipeline backend)")
    max_workers: int = Field(default=2, description="Max concurrent processes (pipeline backend)")
    max_vram: int = Field(default=4, description="Max VRAM per process GB (pipeline+cuda)")
    api_key: str = Field(
        default="",
        description="MinerU API key",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_MINERU_API_KEY",
            "MINERU_API_KEY",
        ),
    )
    api_poll_interval: int = Field(
        default=5,
        description="API polling interval in seconds",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_MINERU_API_POLL_INTERVAL",
            "MINERU_API_POLL_INTERVAL",
        ),
    )
    api_timeout: int = Field(
        default=900,
        description="API timeout in seconds",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_MINERU_API_TIMEOUT",
            "MINERU_API_TIMEOUT",
        ),
    )


class SummarySettings(SettingsGroup):
    """Summary generation configuration."""

    model_config = group_model_config("ARXIV_SANITY_SUMMARY_")

    min_chinese_ratio: float = Field(default=0.25, description="Minimum Chinese character ratio threshold")
    default_semantic_weight: float = Field(default=0.5, description="Semantic search weight in hybrid search")
    markdown_source: Literal["html", "mineru"] = Field(
        default="html",
        description="Markdown source",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE",
            "ARXIV_SANITY_SUMMARY_SOURCE",
        ),
    )
    html_sources: str = Field(
        default="ar5iv,arxiv",
        description="HTML source order (comma-separated)",
        validation_alias=AliasChoices(
            "ARXIV_SANITY_SUMMARY_HTML_SOURCES",
            "ARXIV_SANITY_HTML_SOURCES",
        ),
    )
    batch_num: int = Field(default=500, description="Batch summary generation count")
    force_cache_only: bool = Field(
        default=True,
        description="Force /api/get_paper_summary to be cache-only (regeneration must go through /api/trigger_paper_summary)",
    )
    image_compression_enabled: bool = Field(
        default=True,
        description="Whether to compress cached summary images after download or MinerU extraction",
    )
    image_max_long_edge: int = Field(
        default=0,
        description="Maximum long edge for cached summary raster images (0 disables resizing)",
    )
    image_webp_quality: int = Field(
        default=86,
        description="Target quality for lossy WebP compression of photo-like summary images",
    )
    image_min_savings_bytes: int = Field(
        default=8192,
        description="Minimum byte savings before rewriting an already-sized cached summary image",
    )
    image_skip_below_bytes: int = Field(
        default=32768,
        description="Skip compression for cached summary images smaller than this size in bytes",
    )

    @property
    def html_source_list(self) -> list[str]:
        return [s.strip() for s in str(self.html_sources or "").split(",") if s.strip()]
