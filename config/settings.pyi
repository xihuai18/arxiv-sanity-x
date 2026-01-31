"""
Type stub file - Helps Pylance correctly infer types for config.settings module
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

class GunicornSettings(BaseSettings):
    workers: int
    threads: int
    preload: bool
    extra_args: str

class EmailSettings(BaseSettings):
    from_email: str
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    api_workers: int

class LLMSettings(BaseSettings):
    base_url: str
    api_key: str
    name: str
    summary_lang: str
    fallback_models: str
    timeout: int
    litellm_verbose: bool
    @property
    def fallback_model_list(self) -> list[str]: ...

class ExtractInfoSettings(BaseSettings):
    model_name: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: int

class EmbeddingSettings(BaseSettings):
    port: int
    use_llm_api: bool
    model_name: str
    api_base: str
    api_key: str

class MinerUSettings(BaseSettings):
    enabled: bool
    port: int
    backend: Literal["pipeline", "vlm-http-client", "api"]
    device: Literal["cuda", "cpu"]
    max_workers: int
    max_vram: int
    api_key: str
    api_poll_interval: int
    api_timeout: int

class SummarySettings(BaseSettings):
    min_chinese_ratio: float
    default_semantic_weight: float
    markdown_source: Literal["html", "mineru"]
    html_sources: str
    batch_num: int
    @property
    def html_source_list(self) -> list[str]: ...

class SVMSettings(BaseSettings):
    c: float
    max_iter: int
    tol: float
    neg_weight: float

class DaemonSettings(BaseSettings):
    fetch_num: int
    fetch_max: int
    summary_num: int
    summary_workers: int
    enable_summary: bool
    enable_embeddings: bool
    enable_priority_queue: bool
    enable_summary_queue: bool
    priority_days: float
    priority_limit: int
    email_dry_run: bool
    enable_git_backup: bool
    timezone: str

class HueySettings(BaseSettings):
    db_path: str
    workers: int
    worker_type: str
    max_memory_mb: int
    summary_priority_high: int
    summary_priority_low: int
    summary_repair_on_start: bool
    summary_repair_ttl: int
    summary_repair_requeue: bool
    summary_repair_enable: bool
    summary_repair_interval: int
    force_repair: bool
    tasks_sse_enabled: bool
    sqlite_timeout_web: float
    sqlite_timeout_worker: float
    allow_thread_fallback: bool

class WebSettings(BaseSettings):
    cache_papers: bool
    warmup_data: bool
    warmup_ml: bool
    enable_scheduler: bool
    reload: bool
    access_log: bool
    secret_key: str
    cookie_samesite: str
    cookie_secure: bool
    max_content_length: int
    summary_cache_stats_refresh: int
    enable_cache_status: bool

class LockSettings(BaseSettings):
    summary_lock_stale_sec: float
    mineru_lock_stale_sec: float

class DatabaseSettings(BaseSettings):
    timeout: int
    max_retries: int
    retry_base_sleep: float
    timeout_web: int
    timeout_worker: int
    max_retries_web: int
    max_retries_worker: int

class SearchSettings(BaseSettings):
    ret_num: int
    max_results: int

class RecommendationSettings(BaseSettings):
    api_base_url: str
    api_key: str
    api_timeout: float
    api_limit: int
    model_c: float
    num_threads: int
    max_threads: int
    web_name: str

class ArxivSettings(BaseSettings):
    core_tags: str
    lang_tags: str
    agent_tags: str
    app_tags: str
    empty_response_fallback: int
    api_timeout: int
    @property
    def all_tags(self) -> list[str]: ...

PROJECT_ROOT: Path

class Settings(BaseSettings):
    data_dir: Path
    summary_dir: Path | None
    log_dir: Path | None
    host: str
    serve_port: int
    litellm_port: int
    log_level: str
    enable_swagger: bool
    main_content_min_ratio: float

    # Nested configuration - explicit types
    email: EmailSettings
    llm: LLMSettings
    extract_info: ExtractInfoSettings
    embedding: EmbeddingSettings
    mineru: MinerUSettings
    summary: SummarySettings
    svm: SVMSettings
    daemon: DaemonSettings
    huey: HueySettings
    gunicorn: GunicornSettings
    web: WebSettings
    lock: LockSettings
    db: DatabaseSettings
    search: SearchSettings
    reco: RecommendationSettings
    arxiv: ArxivSettings

    @property
    def access_log(self) -> bool: ...

def get_settings() -> Settings: ...
def reload_settings() -> Settings: ...

settings: Settings
