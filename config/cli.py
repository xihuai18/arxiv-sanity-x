#!/usr/bin/env python3
"""
Configuration Management CLI Tool

Usage:
    python -m config.cli show          # Show current configuration
    python -m config.cli show --json   # JSON format output
    python -m config.cli validate      # Validate configuration
    python -m config.cli doctor        # Diagnose common operator mistakes
    python -m config.cli env           # Generate environment variable template
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

_SENSITIVE_FIELDS = {
    ("llm", "api_key"),
    ("extract_info", "api_key"),
    ("embedding", "api_key"),
    ("mineru", "api_key"),
    ("email", "password"),
    ("reco", "api_key"),
    ("sentry", "dsn"),
    ("web", "secret_key"),
    ("web", "metrics_key"),
}

_ENV_SECTION_TITLES = {
    "__root__": "Main configuration",
    "email": "Email configuration",
    "llm": "LLM configuration",
    "extract_info": "Extract info configuration",
    "embedding": "Embedding configuration",
    "mineru": "MinerU configuration",
    "summary": "Summary configuration",
    "svm": "SVM configuration",
    "daemon": "Daemon configuration",
    "huey": "Huey configuration",
    "sse": "SSE configuration",
    "gunicorn": "Gunicorn configuration",
    "web": "Web configuration",
    "lock": "Lock configuration",
    "db": "Database configuration",
    "search": "Search configuration",
    "reco": "Recommendation configuration",
    "sentry": "Sentry configuration",
    "arxiv": "arXiv configuration",
}


def _bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def _mask_secret(value) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}...{text[-4:]}"


def _redact_json_data(obj, path=(), *, include_secrets: bool = False):
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            next_path = path + (str(key),)
            if not include_secrets and next_path in _SENSITIVE_FIELDS:
                redacted[key] = _mask_secret(value)
            else:
                redacted[key] = _redact_json_data(value, next_path, include_secrets=include_secrets)
        return redacted
    if isinstance(obj, list):
        return [_redact_json_data(v, path, include_secrets=include_secrets) for v in obj]
    return obj


def _format_secret(value, *, include_secrets: bool) -> str:
    if not value:
        return "(not set)"
    return str(value) if include_secrets else _mask_secret(value)


def _print_env_var(name: str, value, *, include_secrets: bool, secret: bool = False) -> None:
    if secret and not include_secrets:
        if value in (None, ""):
            print(f"{name}=")
            return
        print(f"{name}=<redacted>")
        return
    print(f"{name}={value}")


def _normalize_env_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_text(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _iter_env_items(model, *, path=()):
    env_prefix = str(getattr(model.__class__, "model_config", {}).get("env_prefix") or "")

    for field_name in model.__class__.model_fields:
        value = getattr(model, field_name)
        next_path = path + (field_name,)
        if isinstance(value, BaseSettings):
            yield from _iter_env_items(value, path=next_path)
            continue
        yield (
            next_path,
            f"{env_prefix}{field_name.upper()}",
            _normalize_env_value(value),
        )


def _env_section_key(path: tuple[str, ...]) -> str:
    if len(path) <= 1:
        return "__root__"
    return path[0]


def _env_section_title(path: tuple[str, ...]) -> str:
    key = _env_section_key(path)
    if key in _ENV_SECTION_TITLES:
        return _ENV_SECTION_TITLES[key]
    if key == "__root__":
        return _ENV_SECTION_TITLES[key]
    return key.replace("_", " ").title()


def _collect_validation_messages(settings) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not settings.data_dir.exists():
        warnings.append(f"Data directory does not exist: {settings.data_dir}")

    llm_base_url = str(settings.llm.base_url or "").strip().lower()
    llm_is_local_gateway = llm_base_url.startswith("http://localhost:") or llm_base_url.startswith("http://127.0.0.1:")

    llm_yml_order: list[str] = []
    llm_yml_read_error = ""
    if llm_is_local_gateway:
        try:
            from config.llm_model_order import read_llm_yml_model_order

            llm_yml_order = read_llm_yml_model_order()
        except Exception as exc:
            llm_yml_read_error = str(exc or exc.__class__.__name__)
            llm_yml_order = []

    if llm_is_local_gateway and not llm_yml_order:
        detail = f" (last read error: {llm_yml_read_error})" if llm_yml_read_error else ""
        warnings.append(
            "Local LLM gateway detected but config/llm.yml could not be read or contains no model aliases; "
            f"model alias validation is skipped{detail}. Ensure your gateway serves "
            f"`{settings.llm.name}` and `{settings.extract_info.model_name}`, or add routes to config/llm.yml"
        )

    if settings.llm.api_key == "no-key" and not llm_is_local_gateway:
        warnings.append("LLM API key not set (using default value 'no-key')")

    if llm_is_local_gateway and llm_yml_order and settings.llm.name not in llm_yml_order:
        warnings.append(
            f"Main LLM model `{settings.llm.name}` is not declared in config/llm.yml; "
            "use a configured alias or add a matching route"
        )

    if settings.mineru.enabled and settings.mineru.backend == "api" and not settings.mineru.api_key:
        errors.append(
            "MinerU is enabled and using API backend, but API key is not set "
            "(set ARXIV_SANITY_MINERU_API_KEY, or disable MinerU via ARXIV_SANITY_MINERU_ENABLED=false)"
        )

    if settings.email.from_email and not settings.email.smtp_server:
        warnings.append("From email is set but SMTP server is not configured")
    if settings.email.smtp_server and not settings.email.from_email:
        warnings.append("SMTP server is set but from_email is not configured")

    if (
        settings.extract_info.model_name != settings.llm.name
        and not str(settings.extract_info.base_url or "").strip()
        and not llm_is_local_gateway
    ):
        warnings.append(
            "Extract model differs from main LLM model, but ARXIV_SANITY_EXTRACT_BASE_URL is empty; "
            "this assumes your main LLM endpoint can route that extract model alias"
        )

    if (
        llm_is_local_gateway
        and llm_yml_order
        and not str(settings.extract_info.base_url or "").strip()
        and settings.extract_info.model_name not in llm_yml_order
    ):
        warnings.append(
            f"Extract model `{settings.extract_info.model_name}` is not declared in config/llm.yml; "
            "set ARXIV_SANITY_EXTRACT_MODEL_NAME to a configured alias or add a matching route"
        )

    if not settings.reco.api_base_url:
        warnings.append("Recommendation API base URL is empty; local service URL will be inferred from serve_port")

    if settings.daemon.enable_git_backup and not getattr(settings.daemon, "backup_repo_dir", ""):
        errors.append("Git backup is enabled but daemon.backup_repo_dir is empty")

    return errors, warnings


def _doctor_status(ok: bool) -> str:
    return "OK" if ok else "WARN"


def cmd_doctor(_args):
    """Show operator-focused configuration diagnosis."""
    from config.settings import settings

    errors, warnings = _collect_validation_messages(settings)

    llm_ready = bool(str(settings.llm.base_url or "").strip()) and bool(str(settings.llm.name or "").strip())
    summary_ready = llm_ready and bool(settings.daemon.enable_summary)
    extract_ready = bool(str(settings.extract_info.model_name or "").strip())
    email_ready = bool(
        settings.email.from_email and settings.email.smtp_server and settings.email.username and settings.email.password
    )
    mineru_ready = (not settings.mineru.enabled) or settings.mineru.backend != "api" or bool(settings.mineru.api_key)

    print("=" * 60)
    print("Arxiv Sanity Config Doctor")
    print("=" * 60)
    print(f"- data_dir: {settings.data_dir}")
    print(f"- summary_dir: {settings.summary_dir}")
    print(f"- host: {settings.host}")
    print(f"- daemon_schedule_tz: {settings.daemon.timezone}")
    print()
    print("Core")
    print(
        f"- [{_doctor_status(llm_ready)}] LLM configured: base_url={settings.llm.base_url or '(empty)'} model={settings.llm.name or '(empty)'}"
    )
    print(
        f"- [{_doctor_status(summary_ready)}] Batch summary path: daemon_enable_summary={settings.daemon.enable_summary} huey_workers={settings.huey.workers}"
    )
    print(
        f"- [{_doctor_status(extract_ready)}] Upload metadata extraction: model={settings.extract_info.model_name or '(empty)'} base_url={settings.extract_info.base_url or '(uses main LLM)'}"
    )
    print(
        f"- [{_doctor_status(email_ready)}] Email delivery: from={settings.email.from_email or '(empty)'} smtp={settings.email.smtp_server or '(empty)'}"
    )
    print(
        f"- [{_doctor_status(mineru_ready)}] MinerU backend: enabled={settings.mineru.enabled} backend={settings.mineru.backend}"
    )
    print()
    print("Behavior Notes")
    print(
        "- Daemon is not auto-started by the web server; run `python -m tools daemon` or `python bin/run_services.py --with-daemon` for automated fetch/summary/email."
    )
    print(
        f"- Strict /ready checks: embedding_required={getattr(settings.web, 'ready_require_embedding', True)} mineru_required={getattr(settings.web, 'ready_require_mineru', True)}"
    )
    print(
        f"- Summary source: {settings.summary.markdown_source} | HTML fallback order: {settings.summary.html_sources}"
    )
    print(
        f"- arXiv tags: core={settings.arxiv.core_tags} lang={settings.arxiv.lang_tags} agent={settings.arxiv.agent_tags} app={settings.arxiv.app_tags}"
    )

    if errors:
        print()
        print("Errors")
        for item in errors:
            print(f"- {item}")

    if warnings:
        print()
        print("Warnings")
        for item in warnings:
            print(f"- {item}")

    if not errors and not warnings:
        print()
        print("- No configuration problems detected.")

    return 1 if errors else 0


def cmd_show(args):
    """Show current configuration"""
    from config.settings import settings

    if args.json:
        # JSON format output
        data = settings.model_dump(mode="json")

        # Convert Path to string
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = convert_paths(data)
        data = _redact_json_data(data, include_secrets=args.include_secrets)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        # Human-readable format
        print("=" * 60)
        print("Arxiv Sanity Configuration")
        print("=" * 60)

        print("\n📁 Data Directories:")
        print(f"  data_dir:    {settings.data_dir}")
        print(f"  summary_dir: {settings.summary_dir}")

        print("\n🌐 Service Configuration:")
        print(f"  host:         {settings.host}")
        print(f"  serve_port:   {settings.serve_port}")
        print(f"  litellm_port: {settings.litellm_port}")

        print("\n🤖 LLM Configuration:")
        print(f"  base_url:     {settings.llm.base_url}")
        print(f"  api_key:      {_format_secret(settings.llm.api_key, include_secrets=args.include_secrets)}")
        print(f"  model:        {settings.llm.name}")
        print(f"  summary_lang: {settings.llm.summary_lang}")
        print(f"  fallback:     {settings.llm.fallback_models}")
        print(f"  timeout:      {settings.llm.timeout}s")

        print("\n🧾 Extract Info LLM Configuration:")
        print(f"  model_name:   {settings.extract_info.model_name}")
        print(f"  base_url:     {settings.extract_info.base_url or '(uses LLM base_url)'}")
        if settings.extract_info.api_key:
            extract_api_key = _format_secret(settings.extract_info.api_key, include_secrets=args.include_secrets)
        else:
            extract_api_key = "(uses LLM api_key)"
        print(f"  api_key:      {extract_api_key}")
        print(f"  temperature:  {settings.extract_info.temperature}")
        print(f"  max_tokens:   {settings.extract_info.max_tokens}")
        print(f"  timeout:      {settings.extract_info.timeout}s")

        print("\n🔢 Embedding Configuration:")
        print(f"  port:         {settings.embedding.port}")
        print(f"  use_llm_api:  {settings.embedding.use_llm_api}")
        print(f"  model_name:   {settings.embedding.model_name}")

        print("\n📄 MinerU Configuration:")
        print(f"  enabled:      {settings.mineru.enabled}")
        print(f"  backend:      {settings.mineru.backend}")
        print(f"  port:         {settings.mineru.port}")
        print(f"  api_key:      {_format_secret(settings.mineru.api_key, include_secrets=args.include_secrets)}")

        print("\n📝 Summary Configuration:")
        print(f"  markdown_source:  {settings.summary.markdown_source}")
        print(f"  html_sources:     {settings.summary.html_sources}")
        print(f"  batch_num:        {settings.summary.batch_num}")
        print(f"  force_cache_only: {getattr(settings.summary, 'force_cache_only')}")

        print("\n📧 Email Configuration:")
        print(f"  from_email:   {settings.email.from_email or '(not set)'}")
        print(f"  smtp_server:  {settings.email.smtp_server or '(not set)'}")
        print(f"  smtp_port:    {settings.email.smtp_port}")
        print(f"  password:     {_format_secret(settings.email.password, include_secrets=args.include_secrets)}")

        print("\n📊 SVM Configuration:")
        print(f"  C:            {settings.svm.c}")
        print(f"  max_iter:     {settings.svm.max_iter}")
        print(f"  neg_weight:   {settings.svm.neg_weight}")

        print("\n⏰ Daemon Configuration:")
        print(f"  fetch_num:        {settings.daemon.fetch_num}")
        print(f"  fetch_max:        {settings.daemon.fetch_max}")
        print(f"  summary_num:      {settings.daemon.summary_num}")
        print(f"  summary_workers:  {settings.daemon.summary_workers}")
        print(f"  enable_summary:   {settings.daemon.enable_summary}")
        print(f"  enable_embeddings:{settings.daemon.enable_embeddings}")
        print(f"  priority_queue:   {settings.daemon.enable_priority_queue}")
        print(f"  priority_days:    {settings.daemon.priority_days}")
        print(f"  git_backup:       {settings.daemon.enable_git_backup}")
        print(f"  timezone:         {settings.daemon.timezone}")

        print("\n⚙️  Huey Configuration:")
        print(f"  db_path:          {settings.huey.db_path}")
        print(f"  workers:          {settings.huey.workers}")
        print(f"  worker_type:      {settings.huey.worker_type}")
        print(f"  priority_high:    {settings.huey.summary_priority_high}")
        print(f"  priority_low:     {settings.huey.summary_priority_low}")
        print(f"  repair_on_start:  {settings.huey.summary_repair_on_start}")
        print(f"  tasks_sse:        {settings.huey.tasks_sse_enabled}")

        print("\n📡 SSE Configuration:")
        print(f"  enabled:          {settings.sse.enabled}")
        print(f"  db_path:          {settings.sse.db_path}")
        print(f"  poll_interval:    {settings.sse.poll_interval}")
        print(f"  batch_size:       {settings.sse.batch_size}")
        print(f"  queue_maxsize:    {settings.sse.queue_maxsize}")
        print(f"  max_conn/user:    {getattr(settings.sse, 'max_connections_per_user')}")
        print(f"  strict_worker:    {getattr(settings.sse, 'strict_worker_class')}")

        print("\n🚀 Gunicorn Configuration:")
        print(f"  workers:          {settings.gunicorn.workers}")
        print(f"  threads:          {settings.gunicorn.threads}")
        print(f"  preload:          {settings.gunicorn.preload}")
        print(f"  preload_caches:   {getattr(settings.gunicorn, 'preload_caches')}")
        print(f"  extra_args:       {settings.gunicorn.extra_args or '(none)'}")
        print(f"  max_memory_mb:    {getattr(settings.gunicorn, 'max_memory_mb')}")

        print("\n🌐 Web Configuration:")
        print(f"  cache_papers:     {settings.web.cache_papers}")
        print(f"  warmup_data:      {settings.web.warmup_data}")
        print(f"  warmup_ml:        {settings.web.warmup_ml}")
        print(f"  enable_scheduler: {settings.web.enable_scheduler}")
        print(f"  ready_embed:      {getattr(settings.web, 'ready_require_embedding', True)}")
        print(f"  ready_mineru:     {getattr(settings.web, 'ready_require_mineru', True)}")
        print(f"  reload:           {settings.web.reload}")
        print(f"  access_log:       {settings.web.access_log}")
        print(f"  secret_key:       {_format_secret(settings.web.secret_key, include_secrets=args.include_secrets)}")
        print(
            f"  metrics_key:      {_format_secret(getattr(settings.web, 'metrics_key'), include_secrets=args.include_secrets)}"
        )

        print("\n🔒 Lock Configuration:")
        print(f"  summary_lock:     {settings.lock.summary_lock_stale_sec}s")
        print(f"  mineru_lock:      {settings.lock.mineru_lock_stale_sec}s")

        print("\n🗄️ Database Configuration:")
        print(f"  timeout:          {settings.db.timeout}s")
        print(f"  max_retries:      {settings.db.max_retries}")
        print(f"  retry_base_sleep: {settings.db.retry_base_sleep}s")
        print(f"  timeout_web:      {settings.db.timeout_web}s")
        print(f"  timeout_worker:   {settings.db.timeout_worker}s")
        print(f"  retries_web:      {settings.db.max_retries_web}")
        print(f"  retries_worker:   {settings.db.max_retries_worker}")

        print("\n🔍 Search Configuration:")
        print(f"  ret_num:          {settings.search.ret_num}")
        print(f"  max_results:      {settings.search.max_results}")
        print(f"  disable_fullscan: {getattr(settings.search, 'disable_fullscan')}")
        print(f"  semantic_disabled:{getattr(settings.search, 'semantic_disabled')}")

        print("\n📬 Recommendation Configuration:")
        print(f"  api_base_url:     {settings.reco.api_base_url}")
        reco_api_key = str(getattr(settings.reco, "api_key", "") or "")
        print(f"  api_key:          {_format_secret(reco_api_key, include_secrets=args.include_secrets)}")
        print(f"  api_timeout:      {settings.reco.api_timeout}s")
        print(f"  api_limit:        {settings.reco.api_limit}")
        print(f"  model_c:          {settings.reco.model_c}")
        print(f"  num_threads:      {settings.reco.num_threads}")
        print(f"  max_threads:      {settings.reco.max_threads}")
        print(f"  web_name:         {settings.reco.web_name}")

        print("\n📚 arXiv Configuration:")
        print(f"  core_tags:        {settings.arxiv.core_tags}")
        print(f"  lang_tags:        {settings.arxiv.lang_tags}")
        print(f"  agent_tags:       {settings.arxiv.agent_tags}")
        print(f"  app_tags:         {settings.arxiv.app_tags}")
        print(f"  empty_fallback:   {settings.arxiv.empty_response_fallback}")
        print(f"  api_timeout:      {settings.arxiv.api_timeout}s")

        print("\n📋 Log Configuration:")
        print(f"  log_level:    {settings.log_level}")

        print("\n" + "=" * 60)


def cmd_validate(args):
    """Validate configuration"""
    from config.settings import settings

    errors, warnings = _collect_validation_messages(settings)

    # Output results
    if errors:
        print("❌ Configuration validation failed:")
        for e in errors:
            print(f"  - {e}")
        print()

    if warnings:
        print("⚠️  Configuration warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    if not errors and not warnings:
        print("✅ Configuration validation passed")
    elif not errors:
        print("✅ Configuration validation passed (with warnings)")

    return 1 if errors else 0


def cmd_env(args):
    """Generate environment variable template"""
    from config.settings import settings

    print("# Environment variable representation of current configuration")
    print("# Can be copied to .env file")
    if not args.include_secrets:
        print("# Secret values are redacted by default. Use --include-secrets to print them.")
    print()

    current_section = None
    for path, env_name, value in _iter_env_items(settings):
        next_section = _env_section_key(path)
        if next_section != current_section:
            if current_section is not None:
                print()
            print(f"# {_env_section_title(path)}")
            current_section = next_section
        _print_env_var(
            env_name,
            value,
            include_secrets=args.include_secrets,
            secret=path in _SENSITIVE_FIELDS,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Arxiv Sanity Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m config.cli show          Show current configuration
  python -m config.cli show --json   JSON format output
  python -m config.cli validate      Validate configuration
  python -m config.cli doctor        Diagnose common operator mistakes
  python -m config.cli env           Generate environment variables
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # show command
    show_parser = subparsers.add_parser("show", help="Show current configuration")
    show_parser.add_argument("--json", action="store_true", help="JSON format output")
    show_parser.add_argument("--include-secrets", action="store_true", help="Include secret values in output")

    # validate command
    subparsers.add_parser("validate", help="Validate configuration")

    # doctor command
    subparsers.add_parser("doctor", help="Diagnose common operator mistakes")

    # env command
    env_parser = subparsers.add_parser("env", help="Generate environment variable template")
    env_parser.add_argument("--include-secrets", action="store_true", help="Include secret values in output")

    args = parser.parse_args()

    if args.command == "show":
        cmd_show(args)
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "env":
        cmd_env(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
