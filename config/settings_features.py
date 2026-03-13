from __future__ import annotations

from pydantic import Field, model_validator

from .settings_base import SettingsGroup, group_model_config


class SVMSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_SVM_")

    c: float = Field(default=0.02, description="SVM C parameter")
    max_iter: int = Field(default=5000, description="Maximum iterations")
    tol: float = Field(default=1e-3, description="Convergence tolerance")
    neg_weight: float = Field(default=5.0, description="Negative feedback sample weight")


class SearchSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_SEARCH_")

    ret_num: int = Field(default=100, description="Papers returned per page")
    max_results: int = Field(default=1000, description="Search max results (default ret_num * 10)")
    disable_fullscan: bool = Field(default=False, description="Disable keyword-search fullscan fallback")
    semantic_disabled: bool = Field(
        default=False,
        description="Disable semantic/hybrid search; downgrade to keyword-only",
    )

    @model_validator(mode="after")
    def set_max_results_default(self) -> SearchSettings:
        if "max_results" not in self.model_fields_set:
            self.max_results = self.ret_num * 10
        return self


class RecommendationSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_RECO_")

    api_base_url: str = Field(default="", description="Recommendation API base URL (empty uses local service)")
    api_key: str = Field(default="", description="Internal API key for service-to-service calls")
    api_timeout: float = Field(default=45.0, description="Recommendation API timeout (seconds)")
    api_limit: int = Field(default=1000, description="Candidate papers limit per query")
    model_c: float = Field(default=0.1, description="Recommendation model C parameter")
    num_threads: int = Field(default=0, description="Thread count (0=auto)")
    max_threads: int = Field(default=192, description="Max thread count limit")
    web_name: str = Field(default="Arxiv Sanity X", description="Brand name in email templates")


class SentrySettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_SENTRY_")

    enabled: bool = Field(default=False, description="Enable Sentry initialization when DSN is set")
    dsn: str = Field(default="", description="Sentry DSN")
    environment: str = Field(default="", description="Sentry environment (optional)")
    release: str = Field(default="", description="Sentry release (optional)")
    traces_sample_rate: float = Field(default=0.0, description="Sentry tracing sample rate (0 disables)")
    profiles_sample_rate: float = Field(default=0.0, description="Sentry profiling sample rate (0 disables)")


class ArxivSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_ARXIV_")

    core_tags: str = Field(default="cs.AI,cs.LG,stat.ML", description="Core AI tags (comma-separated)")
    lang_tags: str = Field(
        default="cs.CL,cs.IR,cs.CV",
        description="Language/vision tags (comma-separated)",
    )
    agent_tags: str = Field(
        default="cs.MA,cs.RO,cs.HC,cs.GT,cs.NE",
        description="Agent-related tags (comma-separated)",
    )
    app_tags: str = Field(default="cs.SE,cs.CY", description="Application tags (comma-separated)")
    empty_response_fallback: int = Field(default=3, description="Empty response fallback count")
    api_timeout: int = Field(default=30, description="arXiv API request timeout (seconds)")

    @property
    def all_tags(self) -> list[str]:
        tags: list[str] = []
        for tag_str in [self.core_tags, self.lang_tags, self.agent_tags, self.app_tags]:
            tags.extend([t.strip() for t in tag_str.split(",") if t.strip()])
        return tags
