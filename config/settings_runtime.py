from __future__ import annotations

from pydantic import Field, model_validator

from .settings_base import SettingsGroup, group_model_config


class DaemonSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_DAEMON_")

    fetch_num: int = Field(default=2000, description="Papers to fetch per run")
    fetch_max: int = Field(default=1000, description="Max papers per API query")
    summary_num: int = Field(default=250, description="Summaries to generate per run")
    summary_workers: int = Field(default=2, description="Summary generation concurrent workers")
    enable_summary: bool = Field(default=True, description="Enable summary generation")
    enable_embeddings: bool = Field(default=True, description="Enable embedding computation")
    enable_priority_queue: bool = Field(default=True, description="Enable priority queue")
    enable_summary_queue: bool = Field(default=True, description="Enable summary queue")
    priority_days: float = Field(default=2.0, description="Priority process papers from last N days")
    priority_limit: int = Field(default=200, description="Priority queue max size")
    email_dry_run: bool = Field(default=False, description="Email dry-run mode (no actual sending)")
    enable_git_backup: bool = Field(default=True, description="Enable git backup")
    backup_repo_dir: str = Field(default="data-repo", description="Directory used for git backups of dict.db")
    backup_push: bool = Field(default=True, description="Whether to push backup commits to remote")
    backup_push_remote: str = Field(default="", description="Optional git remote name to push to")
    backup_push_branch: str = Field(default="", description="Optional git branch to push to")
    backup_push_retries: int = Field(default=3, description="git push retry count")
    backup_git_user_name: str = Field(default="arxiv-sanity-daemon", description="git user.name for backup commits")
    backup_git_user_email: str = Field(default="daemon@localhost", description="git user.email for backup commits")
    subprocess_timeout_s: int = Field(default=7200, description="Max seconds for each daemon subprocess call")
    web_cache_warmup_on_update: bool = Field(
        default=True, description="Ping web /health once after successful fetch/compute"
    )
    timezone: str = Field(default="Asia/Shanghai", description="Scheduler timezone")


class HueySettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_HUEY_")

    db_path: str = Field(default="", description="Huey database path (default data_dir/huey.db)")
    workers: int = Field(default=4, description="Huey worker count")
    worker_type: str = Field(default="thread", description="Huey worker type (thread/process/greenlet)")
    max_memory_mb: int = Field(default=0, description="Max memory for Huey consumer (MB, 0=unlimited)")
    summary_priority_high: int = Field(default=200, description="High priority summary task")
    summary_priority_low: int = Field(default=10, description="Low priority summary task")
    summary_repair_on_start: bool = Field(default=True, description="Repair stuck tasks on startup")
    summary_repair_ttl: int = Field(default=3600, description="Repair task TTL (seconds)")
    upload_repair_ttl: int = Field(
        default=21600,
        description="Repair TTL for upload parse/extract tasks (seconds)",
    )
    summary_repair_requeue: bool = Field(default=False, description="Requeue on repair")
    summary_repair_enable: bool = Field(default=True, description="Enable periodic repair")
    summary_repair_interval: int = Field(default=900, description="Repair check interval (seconds)")
    force_repair: bool = Field(default=False, description="Force repair")
    tasks_sse_enabled: bool = Field(default=True, description="Enable task SSE event push")
    sqlite_timeout_web: float = Field(default=2.0, description="Huey SQLite timeout for web process (seconds)")
    sqlite_timeout_worker: float = Field(default=10.0, description="Huey SQLite timeout for Huey consumer (seconds)")
    allow_thread_fallback: bool = Field(
        default=False,
        description="Allow in-process thread fallback when Huey enqueue fails",
    )


class SSESettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_SSE_")

    enabled: bool = Field(default=True, description="Enable SQLite-backed SSE IPC")
    db_path: str = Field(
        default="",
        description="SSE event bus SQLite path (default data_dir/sse_events.db)",
    )
    poll_interval: float = Field(default=0.05, description="SQLite poll interval (seconds)")
    batch_size: int = Field(default=500, description="Max events fetched per poll")
    retention_seconds: int = Field(default=86400, description="Event retention (seconds)")
    cleanup_interval: float = Field(default=60.0, description="Cleanup interval (seconds)")
    queue_maxsize: int = Field(default=200, description="Per-client in-process SSE queue size")
    max_connections_per_user: int = Field(
        default=2, description="Max concurrent SSE connections per user (0=unlimited)"
    )
    connection_lease_ttl_s: float = Field(default=90.0, description="TTL for per-user SSE connection leases (seconds)")
    strict_worker_class: bool = Field(
        default=False,
        description="Fail startup if SSE enabled but gunicorn worker_class is not gevent",
    )
    publish_retry_queue_maxsize: int = Field(default=2000, description="Async publish retry queue size (per process)")
    publish_retry_backoff_max_s: float = Field(default=1.0, description="Async publish max backoff (seconds)")
    publish_async: bool = Field(default=True, description="Publish to SQLite asynchronously")


class GunicornSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_GUNICORN_")

    workers: int = Field(default=2, description="Gunicorn worker processes")
    threads: int = Field(default=2, description="Threads per worker")
    preload: bool = Field(default=True, description="Preload app (copy-on-write shared memory)")
    preload_caches: bool = Field(default=True, description="Preload data caches in master process")
    worker_class: str = Field(default="", description="Optional worker class override")
    force_workers: bool = Field(default=False, description="Disable safety clamp for high gevent worker counts")
    extra_args: str = Field(default="", description="Extra Gunicorn arguments")
    max_memory_mb: int = Field(default=0, description="Max memory per worker (MB, 0=unlimited)")


class WebSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_")

    cache_papers: bool = Field(default=False, description="Cache papers in memory")
    warmup_data: bool = Field(default=True, description="Warm up data cache on startup")
    warmup_ml: bool = Field(default=True, description="Warm up ML models on startup")
    enable_scheduler: bool = Field(default=True, description="Enable APScheduler tasks")
    reload: bool = Field(default=False, description="Development hot reload mode")
    access_log: bool = Field(default=False, description="Enable access logging")
    secret_key: str = Field(default="", description="Flask session secret key")
    cookie_samesite: str = Field(default="Lax", description="Cookie SameSite policy")
    cookie_secure: bool = Field(default=False, description="Cookie Secure flag")
    max_content_length: int = Field(default=52428800, description="Max request body size (bytes)")
    summary_cache_stats_refresh: int = Field(default=1800, description="Summary cache stats refresh interval (seconds)")
    data_cache_refresh_min_interval: int = Field(
        default=60,
        description="Minimum interval between background refreshes when papers.db changes (seconds)",
    )
    features_cache_refresh_min_interval: int = Field(
        default=300,
        description="Minimum interval between background refreshes when features files change (seconds)",
    )
    enable_cache_status: bool = Field(default=False, description="Enable /cache_status debug page")
    ready_require_embedding: bool = Field(
        default=True, description="Require embedding backend for strict /ready checks"
    )
    ready_require_mineru: bool = Field(
        default=True,
        description="Require MinerU backend for strict /ready checks when enabled",
    )
    enable_metrics: bool = Field(default=False, description="Enable Prometheus /metrics endpoint")
    metrics_key: str = Field(default="", description="Optional shared secret for /metrics")
    asset_cdn_enabled: bool = Field(default=True, description="Enable public CDN for third-party frontend assets")
    asset_npm_cdn_base: str = Field(
        default="https://cdn.jsdelivr.net/npm",
        description="Base URL for npm package CDN",
    )


class LockSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_")

    summary_lock_stale_sec: float = Field(default=3600, description="Summary lock stale time (seconds)")
    mineru_lock_stale_sec: float = Field(default=3600, description="MinerU lock stale time (seconds)")


class DatabaseSettings(SettingsGroup):
    model_config = group_model_config("ARXIV_SANITY_DB_")

    timeout: int = Field(default=120, description="SQLite connection timeout (seconds)")
    max_retries: int = Field(default=5, description="Database operation max retries")
    retry_base_sleep: float = Field(default=0.2, description="Retry base sleep time (seconds)")
    timeout_web: int = Field(default=2, description="SQLite timeout for web process (seconds)")
    timeout_worker: int = Field(default=120, description="SQLite timeout for worker process (seconds)")
    max_retries_web: int = Field(default=3, description="Database operation max retries for web process")
    max_retries_worker: int = Field(default=5, description="Database operation max retries for worker process")

    @model_validator(mode="after")
    def set_split_defaults(self) -> DatabaseSettings:
        if "timeout" in self.model_fields_set:
            if "timeout_web" not in self.model_fields_set:
                self.timeout_web = self.timeout
            if "timeout_worker" not in self.model_fields_set:
                self.timeout_worker = self.timeout
        if "max_retries" in self.model_fields_set:
            if "max_retries_web" not in self.model_fields_set:
                self.max_retries_web = self.max_retries
            if "max_retries_worker" not in self.model_fields_set:
                self.max_retries_worker = self.max_retries
        return self
