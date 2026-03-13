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

    base_url: str = Field(default="http://localhost:53000", description="LLM API base URL")
    api_key: str = Field(default="no-key", description="LLM API key")
    name: str = Field(default="gpt-5.4", description="Default LLM model name")
    summary_lang: str = Field(default="zh", description="Summary language (zh/en)")
    fallback_models: str = Field(
        default="auto",
        description='Fallback models (comma-separated) or "auto" to fallback backwards from the current model based on config/llm.yml order',
    )
    timeout: int = Field(default=600, description="LLM request timeout (seconds)")
    litellm_verbose: bool = Field(default=False, description="LiteLLM verbose logging mode")

    @property
    def fallback_model_list(self) -> list[str]:
        raw = str(self.fallback_models or "").strip()
        mode = raw.lower()

        from .llm_model_order import (
            compute_auto_fallback_models,
            read_llm_yml_model_order,
        )

        yml_order = read_llm_yml_model_order()
        default_fallback = ["glm-4.7"]

        def _parse_list(value: str) -> list[str]:
            return [m.strip() for m in str(value or "").split(",") if m.strip()]

        if mode in ("", "auto", "yml"):
            if not yml_order:
                return default_fallback
            return compute_auto_fallback_models(yml_order=yml_order, anchor=str(self.name or "").strip())

        allowlist = _parse_list(raw)
        if not yml_order:
            return allowlist

        allowed = set(allowlist)
        in_order = [m for m in yml_order if m in allowed]
        extras = [m for m in allowlist if m not in set(yml_order)]
        return in_order + extras


class ExtractInfoSettings(SettingsGroup):
    """Metadata extraction model configuration."""

    model_config = group_model_config("ARXIV_SANITY_EXTRACT_")

    model_name: str = Field(default="qwen3.5-plus", description="LLM model for metadata extraction")
    base_url: str = Field(default="", description="Extract Info API base URL (empty uses LLM_BASE_URL)")
    api_key: str = Field(default="", description="Extract Info API key (empty uses LLM_API_KEY)")
    temperature: float = Field(default=0.1, description="LLM temperature for extraction")
    max_tokens: int = Field(default=8192, description="Max tokens for extraction response")
    timeout: int = Field(default=600, description="Request timeout in seconds")


class EmbeddingSettings(SettingsGroup):
    """Embedding service configuration."""

    model_config = group_model_config("ARXIV_SANITY_EMBED_")

    port: int = Field(default=54000, description="Ollama embedding service port")
    use_llm_api: bool = Field(default=False, description="Whether to use LLM API for embedding")
    model_name: str = Field(default="qwen3-embedding:0.6b", description="Embedding model name")
    api_base: str = Field(default="", description="Embedding API base URL (empty uses LLM_BASE_URL)")
    api_key: str = Field(default="", description="Embedding API key (empty uses LLM_API_KEY)")


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

    @property
    def html_source_list(self) -> list[str]:
        return [s.strip() for s in str(self.html_sources or "").split(",") if s.strip()]
