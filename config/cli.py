#!/usr/bin/env python3
"""
Configuration Management CLI Tool

Usage:
    python -m config.cli show          # Show current configuration
    python -m config.cli show --json   # JSON format output
    python -m config.cli validate      # Validate configuration
    python -m config.cli env           # Generate environment variable template
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
        print(f"  api_key:      {'*' * 8 if settings.llm.api_key else '(not set)'}")
        print(f"  model:        {settings.llm.name}")
        print(f"  summary_lang: {settings.llm.summary_lang}")
        print(f"  fallback:     {settings.llm.fallback_models}")

        print("\n� Extract Info LLM Configuration:")
        print(f"  model_name:   {settings.extract_info.model_name}")
        print(f"  base_url:     {settings.extract_info.base_url or '(uses LLM base_url)'}")
        print(f"  api_key:      {'*' * 8 if settings.extract_info.api_key else '(uses LLM api_key)'}")
        print(f"  temperature:  {settings.extract_info.temperature}")
        print(f"  max_tokens:   {settings.extract_info.max_tokens}")
        print(f"  timeout:      {settings.extract_info.timeout}s")

        print("\n�🔢 Embedding Configuration:")
        print(f"  port:         {settings.embedding.port}")
        print(f"  use_llm_api:  {settings.embedding.use_llm_api}")
        print(f"  model_name:   {settings.embedding.model_name}")

        print("\n📄 MinerU Configuration:")
        print(f"  enabled:      {settings.mineru.enabled}")
        print(f"  backend:      {settings.mineru.backend}")
        print(f"  port:         {settings.mineru.port}")
        print(f"  api_key:      {'*' * 8 if settings.mineru.api_key else '(not set)'}")

        print("\n📝 Summary Configuration:")
        print(f"  markdown_source:  {settings.summary.markdown_source}")
        print(f"  html_sources:     {settings.summary.html_sources}")
        print(f"  batch_num:        {settings.summary.batch_num}")
        print(f"  force_cache_only: {settings.summary.force_cache_only}")

        print("\n📧 Email Configuration:")
        print(f"  from_email:   {settings.email.from_email or '(not set)'}")
        print(f"  smtp_server:  {settings.email.smtp_server or '(not set)'}")
        print(f"  smtp_port:    {settings.email.smtp_port}")

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

        print("\n🌐 Web Configuration:")
        print(f"  cache_papers:     {settings.web.cache_papers}")
        print(f"  warmup_data:      {settings.web.warmup_data}")
        print(f"  warmup_ml:        {settings.web.warmup_ml}")
        print(f"  enable_scheduler: {settings.web.enable_scheduler}")
        print(f"  reload:           {settings.web.reload}")
        print(f"  access_log:       {settings.web.access_log}")

        print("\n🔒 Lock Configuration:")
        print(f"  summary_lock:     {settings.lock.summary_lock_stale_sec}s")
        print(f"  mineru_lock:      {settings.lock.mineru_lock_stale_sec}s")

        print("\n🗄️ Database Configuration:")
        print(f"  timeout:          {settings.db.timeout}s")
        print(f"  max_retries:      {settings.db.max_retries}")
        print(f"  retry_base_sleep: {settings.db.retry_base_sleep}s")

        print("\n🔍 Search Configuration:")
        print(f"  ret_num:          {settings.search.ret_num}")
        print(f"  max_results:      {settings.search.max_results}")
        print(f"  disable_fullscan: {settings.search.disable_fullscan}")
        print(f"  semantic_disabled:{settings.search.semantic_disabled}")

        print("\n📬 Recommendation Configuration:")
        print(f"  api_base_url:     {settings.reco.api_base_url}")
        reco_api_key = str(getattr(settings.reco, "api_key", "") or "")
        if reco_api_key:
            masked = f"{reco_api_key[:4]}...{reco_api_key[-4:]}" if len(reco_api_key) > 8 else "***"
            print(f"  api_key:          {masked}")
        else:
            print("  api_key:          (not set)")
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

        print("\n📋 Log Configuration:")
        print(f"  log_level:    {settings.log_level}")

        print("\n" + "=" * 60)


def cmd_validate(args):
    """Validate configuration"""
    from config.settings import settings

    errors = []
    warnings = []

    # Check data directory
    if not settings.data_dir.exists():
        warnings.append(f"Data directory does not exist: {settings.data_dir}")

    # Check LLM configuration
    if settings.llm.api_key == "no-key":
        warnings.append("LLM API key not set (using default value 'no-key')")

    # Check MinerU configuration
    if settings.mineru.enabled and settings.mineru.backend == "api":
        if not settings.mineru.api_key:
            errors.append(
                "MinerU is enabled and using API backend, but API key is not set "
                "(set MINERU_API_KEY / ARXIV_SANITY_MINERU_API_KEY, or disable MinerU via ARXIV_SANITY_MINERU_ENABLED=false)"
            )

    # Check email configuration
    if settings.email.from_email and not settings.email.smtp_server:
        warnings.append("From email is set but SMTP server is not configured")

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
    print()

    # Main configuration
    print(f"ARXIV_SANITY_DATA_DIR={settings.data_dir}")
    print(f"ARXIV_SANITY_HOST={settings.host}")
    print(f"ARXIV_SANITY_SERVE_PORT={settings.serve_port}")
    print(f"ARXIV_SANITY_LITELLM_PORT={settings.litellm_port}")
    print(f"ARXIV_SANITY_LOG_LEVEL={settings.log_level}")
    print(f"ARXIV_SANITY_ENABLE_CACHE_STATUS={str(settings.web.enable_cache_status).lower()}")
    print()

    # LLM configuration
    print(f"ARXIV_SANITY_LLM_BASE_URL={settings.llm.base_url}")
    print(f"ARXIV_SANITY_LLM_API_KEY={settings.llm.api_key}")
    print(f"ARXIV_SANITY_LLM_NAME={settings.llm.name}")
    print(f"ARXIV_SANITY_LLM_SUMMARY_LANG={settings.llm.summary_lang}")
    print()

    # Embedding configuration
    print(f"ARXIV_SANITY_EMBED_PORT={settings.embedding.port}")
    print(f"ARXIV_SANITY_EMBED_USE_LLM_API={str(settings.embedding.use_llm_api).lower()}")
    print(f"ARXIV_SANITY_EMBED_MODEL_NAME={settings.embedding.model_name}")
    print()

    # Search configuration
    print(f"ARXIV_SANITY_SEARCH_RET_NUM={settings.search.ret_num}")
    print(f"ARXIV_SANITY_SEARCH_MAX_RESULTS={settings.search.max_results}")
    print(f"ARXIV_SANITY_SEARCH_DISABLE_FULLSCAN={str(settings.search.disable_fullscan).lower()}")
    print(f"ARXIV_SANITY_SEARCH_SEMANTIC_DISABLED={str(settings.search.semantic_disabled).lower()}")
    print()

    # MinerU configuration
    print(f"ARXIV_SANITY_MINERU_ENABLED={str(settings.mineru.enabled).lower()}")
    print(f"ARXIV_SANITY_MINERU_BACKEND={settings.mineru.backend}")
    print(f"ARXIV_SANITY_MINERU_PORT={settings.mineru.port}")
    print(f"ARXIV_SANITY_MINERU_API_POLL_INTERVAL={settings.mineru.api_poll_interval}")
    print(f"ARXIV_SANITY_MINERU_API_TIMEOUT={settings.mineru.api_timeout}")
    print()

    # Email concurrency configuration
    print(f"ARXIV_SANITY_EMAIL_API_WORKERS={settings.email.api_workers}")


def main():
    parser = argparse.ArgumentParser(
        description="Arxiv Sanity Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m config.cli show          Show current configuration
  python -m config.cli show --json   JSON format output
  python -m config.cli validate      Validate configuration
  python -m config.cli env           Generate environment variables
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # show command
    show_parser = subparsers.add_parser("show", help="Show current configuration")
    show_parser.add_argument("--json", action="store_true", help="JSON format output")

    # validate command
    subparsers.add_parser("validate", help="Validate configuration")

    # env command
    subparsers.add_parser("env", help="Generate environment variable template")

    args = parser.parse_args()

    if args.command == "show":
        cmd_show(args)
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "env":
        cmd_env(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
