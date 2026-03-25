#!/usr/bin/env python3
"""
One-command launcher for arxiv-sanity-X services.

Starts (optionally):
- Ollama embedding server (embedding_serve.sh)
- minerU vLLM server (mineru_serve.sh)
- LiteLLM gateway (litellm.sh)
- Web app (serve.py or up.sh)
- Scheduler (daemon.py)

All logs are multiplexed into the current terminal with a prefix.
Press Ctrl+C to stop everything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

# Ensure repository root is importable when executing this file directly.
# When running `python bin/run_services.py`, sys.path[0] points to bin/, so
# top-level packages like `config/` are not visible unless we add the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import settings early for configuration access
try:
    from config import settings
except Exception as e:
    # Keep launcher robust, but make the root cause visible.
    import traceback

    print(
        f"[launcher] Failed to import config.settings: {e!r}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
    settings = None


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    cmd: list[str]
    cwd: Path
    health_url: str | None = None


_TASK_TERMINAL_STATUSES = ("ok", "failed", "canceled")
_TASK_ACTIVE_STATUSES = ("queued", "running")
_TASK_CATEGORY_ORDER = {
    "summary": 0,
    "upload_process": 1,
    "upload_parse": 2,
    "upload_extract": 3,
}


@dataclass
class _AggregatedLogState:
    count: int = 0
    total_ms: float = 0.0
    last_report_at: float = 0.0
    last_message: str = ""
    statuses: dict[str, int] | None = None


_LOG_STATE_LOCK = threading.Lock()
_LOG_STATE: dict[tuple[str, str], _AggregatedLogState] = {}
_LOGURU_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[^|]*\|\s*(?P<level>[A-Z]+)\s*\|\s*(?P<message>.*)$")
_GIN_RE = re.compile(
    r'^\[GIN\]\s+[^|]+\|\s*(?P<status>\d{3})\s*\|\s*(?P<duration>[^|]+)\|\s*(?P<client>[^|]+)\|\s*(?P<method>[A-Z]+)\s+"(?P<path>[^"]+)"$'
)
_MISFIRE_RE = re.compile(r'^Run time of job "(?P<job>.+?)\s+\(trigger:.*?\)" was missed by (?P<delay>.+)$')
_SUMMARY_COUNT_THRESHOLD = 20
_SUMMARY_TIME_THRESHOLD_S = 60.0
_MISFIRE_SUMMARY_COUNT_THRESHOLD = 3
_MISFIRE_SUMMARY_TIME_THRESHOLD_S = 10 * 60.0
_USE_RAW_LOG_STREAM = False


def _launcher_sigterm_handler(_signum, _frame):
    raise SystemExit(0)


def _print_launcher_line(prefix: str, message: str) -> None:
    sys.stdout.write(f"[{prefix}] {message}\n")
    sys.stdout.flush()


def _normalize_level(level: str | None) -> str:
    level_upper = (level or "INFO").strip().upper()
    if level_upper in {"WARNING", "WARN"}:
        return "Warning"
    if level_upper in {"ERROR", "CRITICAL", "FATAL"}:
        return "Error"
    return "Info"


def _sanitize_message(message: str) -> str:
    return " ".join((message or "").strip().split())


def _parse_duration_to_ms(raw: str) -> float | None:
    text = (raw or "").strip()
    m = re.match(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ns|µs|us|ms|s)$", text)
    if not m:
        return None
    value = float(m.group("value"))
    unit = m.group("unit")
    if unit == "ns":
        return value / 1_000_000.0
    if unit in {"µs", "us"}:
        return value / 1000.0
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    return None


def _summarize_statuses(statuses: dict[str, int] | None) -> str:
    if not statuses:
        return ""
    parts = [f"{code}x{count}" for code, count in sorted(statuses.items())]
    return ", ".join(parts)


def _update_aggregate(
    key: tuple[str, str],
    *,
    count_increment: int = 1,
    total_ms_increment: float = 0.0,
    message: str = "",
    statuses: dict[str, int] | None = None,
) -> _AggregatedLogState:
    with _LOG_STATE_LOCK:
        state = _LOG_STATE.setdefault(key, _AggregatedLogState(last_report_at=time.time()))
        state.count += count_increment
        state.total_ms += total_ms_increment
        if message:
            state.last_message = message
        if statuses:
            if state.statuses is None:
                state.statuses = {}
            for code, value in statuses.items():
                state.statuses[code] = state.statuses.get(code, 0) + value
        snapshot = _AggregatedLogState(
            count=state.count,
            total_ms=state.total_ms,
            last_report_at=state.last_report_at,
            last_message=state.last_message,
            statuses=dict(state.statuses) if state.statuses else None,
        )
    return snapshot


def _should_report_summary(key: tuple[str, str], *, threshold_count: int, threshold_seconds: float) -> bool:
    now = time.time()
    with _LOG_STATE_LOCK:
        state = _LOG_STATE.get(key)
        if state is None:
            return False
        if state.count < threshold_count and now - state.last_report_at < threshold_seconds:
            return False
        state.last_report_at = now
        return True


def _flush_log_summaries(service: str | None = None) -> None:
    with _LOG_STATE_LOCK:
        items = list(_LOG_STATE.items())
        if service is not None:
            items = [item for item in items if item[0][0] == service]
        for key, state in items:
            state.last_report_at = time.time()
            snapshot = _AggregatedLogState(
                count=state.count,
                total_ms=state.total_ms,
                last_report_at=state.last_report_at,
                last_message=state.last_message,
                statuses=dict(state.statuses) if state.statuses else None,
            )
            kind = key[1]
            if kind == "gin":
                avg_ms = snapshot.total_ms / snapshot.count if snapshot.count else 0.0
                statuses = _summarize_statuses(snapshot.statuses)
                _print_launcher_line(
                    key[0],
                    f"HTTP summary: {snapshot.last_message}; requests={snapshot.count}; avg_latency={avg_ms:.2f} ms"
                    + (f"; statuses={statuses}" if statuses else ""),
                )
            elif kind == "misfire":
                _print_launcher_line(
                    key[0],
                    f"Scheduler summary: {snapshot.last_message}; missed_runs={snapshot.count}",
                )


def _emit_normalized_line(service: str, line: str) -> None:
    text = _sanitize_message(line)
    if not text:
        return

    gin_match = _GIN_RE.match(text)
    if gin_match:
        method = gin_match.group("method")
        path = gin_match.group("path")
        status = gin_match.group("status")
        duration_ms = _parse_duration_to_ms(gin_match.group("duration")) or 0.0
        key = (service, f"gin:{method}:{path}")
        snapshot = _update_aggregate(
            key,
            total_ms_increment=duration_ms,
            message=f"{method} {path}",
            statuses={status: 1},
        )
        if _should_report_summary(
            key,
            threshold_count=_SUMMARY_COUNT_THRESHOLD,
            threshold_seconds=_SUMMARY_TIME_THRESHOLD_S,
        ):
            avg_ms = snapshot.total_ms / snapshot.count if snapshot.count else 0.0
            statuses = _summarize_statuses(snapshot.statuses)
            _print_launcher_line(
                service,
                f"HTTP summary: {method} {path}; requests={snapshot.count}; avg_latency={avg_ms:.2f} ms"
                + (f"; statuses={statuses}" if statuses else ""),
            )
        return

    misfire_match = _MISFIRE_RE.match(text)
    if misfire_match:
        job = misfire_match.group("job")
        delay = misfire_match.group("delay")
        key = (service, f"misfire:{job}")
        _update_aggregate(key, message=f"job={job}; last_delay={delay}")
        if _should_report_summary(
            key,
            threshold_count=_MISFIRE_SUMMARY_COUNT_THRESHOLD,
            threshold_seconds=_MISFIRE_SUMMARY_TIME_THRESHOLD_S,
        ):
            with _LOG_STATE_LOCK:
                snapshot = _LOG_STATE.get(key)
                count = snapshot.count if snapshot else 0
                last_message = snapshot.last_message if snapshot else f"job={job}"
            _print_launcher_line(service, f"Scheduler summary: {last_message}; missed_runs={count}")
        return

    loguru_match = _LOGURU_RE.match(text)
    if loguru_match:
        level = _normalize_level(loguru_match.group("level"))
        text = _sanitize_message(loguru_match.group("message"))
    else:
        level = "Info"

    nested_match = re.match(r"^\[(?P<source>[^\]]+)\]\s*(?P<body>.*)$", text)
    if nested_match:
        source = nested_match.group("source").strip().lower()
        body = _sanitize_message(nested_match.group("body"))
        if body.lower().startswith("warning:"):
            level = "Warning"
            body = _sanitize_message(body.split(":", 1)[1])
        elif body.lower().startswith("error:"):
            level = "Error"
            body = _sanitize_message(body.split(":", 1)[1])
        elif body.lower().startswith("info:"):
            level = "Info"
            body = _sanitize_message(body.split(":", 1)[1])
        source_label = {
            "build": "Build",
            "gunicorn": "Gunicorn",
            "pipeline": "Pipeline",
            "preload": "Preload",
            "huey_wrapper": "Huey",
        }.get(source, source.replace("_", " ").title())
        _print_launcher_line(service, f"{level}: {source_label} - {body}")
        return

    _print_launcher_line(service, f"{level}: {text}")


def _emit_raw_line(service: str, line: str) -> None:
    text = line.rstrip("\n")
    if not text:
        return
    _print_launcher_line(service, text)


def _http_ok(url: str, timeout_s: float = 1.0) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "arxiv-sanity-x-launcher"})
        # Ignore env proxies (HTTP(S)_PROXY) for local health checks.
        opener = build_opener(ProxyHandler({}))
        with opener.open(req, timeout=timeout_s) as resp:  # nosec - local health checks only
            return 200 <= resp.status < 300
    except Exception:
        return False


def _http_get_json(url: str, timeout_s: float = 2.0) -> dict | list | None:
    """Best-effort JSON GET for local admin endpoints.

    - Ignores env proxies to avoid localhost issues.
    - Never raises; returns None on any error.
    """

    try:
        req = Request(url, headers={"User-Agent": "arxiv-sanity-x-launcher"})
        opener = build_opener(ProxyHandler({}))
        with opener.open(req, timeout=timeout_s) as resp:  # nosec - local endpoints only
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)
    except Exception:
        return None


def _mask_secret(value: str | None, keep_tail: int = 4) -> str:
    if not value:
        return "(empty)"
    v = value.strip()
    if not v:
        return "(empty)"
    if len(v) <= keep_tail:
        return "***"
    return "***" + v[-keep_tail:]


def _get_command_version(cmd: list[str], timeout_s: float = 2.0) -> str | None:
    """Best-effort `--version` probing.

    Returns a single-line version string, or None if the command is unavailable.
    """

    if not cmd:
        return None
    exe = cmd[0]
    if shutil.which(exe) is None:
        return None

    try:
        p = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        out = (p.stdout or "").strip()
        if not out:
            return None
        # Keep it to one line to avoid noisy banners.
        return out.splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def _print_versions(*, args) -> None:
    """Log key binary versions (best-effort)."""

    # Always print Python version (it helps a lot when env is wrong).
    py_v = sys.version.split()[0]
    print(f"[launcher] - python_version: {py_v}", flush=True)

    if not args.no_litellm:
        v = _get_command_version(["litellm", "--version"])
        if v:
            print(f"[launcher] - litellm_version: {v}", flush=True)

    if not args.no_embed:
        v = _get_command_version(["ollama", "--version"])
        if v:
            print(f"[launcher] - ollama_version: {v}", flush=True)

    if not args.no_mineru:
        v = _get_command_version(["mineru-vllm-server", "--version"])
        if v:
            print(f"[launcher] - mineru_vllm_version: {v}", flush=True)

    if args.web == "gunicorn":
        v = _get_command_version(["gunicorn", "--version"])
        if v:
            print(f"[launcher] - gunicorn_version: {v}", flush=True)
        # Frontend build happens in bin/up.sh; log Node tooling versions if available.
        node_v = _get_command_version(["node", "--version"])
        if node_v:
            print(f"[launcher] - node_version: {node_v}", flush=True)
        npm_v = _get_command_version(["npm", "--version"])
        if npm_v:
            print(f"[launcher] - npm_version: {npm_v}", flush=True)


def _print_startup_context(*, repo_root: Path, args, verbose: bool) -> None:
    """Print the effective runtime context that matters for debugging."""

    if settings is None:
        return

    # Effective markdown source: CLI override has priority over settings.
    effective_summary_source = args.summary_source or settings.summary.markdown_source

    print("[launcher] Effective configuration:", flush=True)
    print(f"[launcher] - repo_root: {repo_root}", flush=True)
    print(f"[launcher] - python: {sys.executable}", flush=True)
    print(f"[launcher] - log_level: {settings.log_level}", flush=True)
    print(
        f"[launcher] - data_dir: {settings.data_dir} | log_dir: {settings.log_dir} | summary_dir: {settings.summary_dir}",
        flush=True,
    )
    print(
        "[launcher] - ports: "
        f"web={settings.serve_port} litellm={settings.litellm_port} embed={settings.embedding.port} mineru={settings.mineru.port}",
        flush=True,
    )

    # Versions are extremely helpful for diagnosing incompatible binaries.
    _print_versions(args=args)

    # LLM / LiteLLM (no secrets).
    print(
        "[launcher] - llm: "
        f"base_url={settings.llm.base_url} default_model={settings.llm.name} "
        f"fallback_models={settings.llm.fallback_model_list} litellm_verbose={settings.llm.litellm_verbose}",
        flush=True,
    )

    # LiteLLM config + log file locations (no secrets).
    litellm_cfg = repo_root / "config" / "llm.yml"
    litellm_log = (settings.log_dir / "litellm.log") if settings.log_dir else None
    print(f"[launcher] - litellm: config={litellm_cfg}", flush=True)
    if litellm_log is not None:
        print(f"[launcher] - litellm: log_file={litellm_log}", flush=True)

    # Embedding.
    print(
        "[launcher] - embedding: "
        f"use_llm_api={settings.embedding.use_llm_api} model_name={settings.embedding.model_name} "
        f"api_base={(settings.embedding.api_base or '(inherit LLM_BASE_URL)')}",
        flush=True,
    )

    # MinerU (mask secrets).
    mineru_backend = (settings.mineru.backend or "pipeline").strip().lower()
    api_key_hint = "(not used)" if mineru_backend != "api" else _mask_secret(settings.mineru.api_key)
    print(
        "[launcher] - mineru: " f"enabled={settings.mineru.enabled} backend={mineru_backend} api_key={api_key_hint}",
        flush=True,
    )

    # Summary behavior.
    print(
        "[launcher] - summary: "
        f"markdown_source={effective_summary_source} html_sources={settings.summary.html_source_list}",
        flush=True,
    )

    # Huey.
    print(
        "[launcher] - huey: "
        f"enabled={not args.no_huey} workers={settings.huey.workers} worker_type={settings.huey.worker_type} db_path={settings.huey.db_path}",
        flush=True,
    )

    # Web server details.
    if args.web == "gunicorn":
        print(
            "[launcher] - gunicorn: "
            f"workers={settings.gunicorn.workers} threads={settings.gunicorn.threads} preload={settings.gunicorn.preload} extra_args={settings.gunicorn.extra_args!r}",
            flush=True,
        )
        if verbose:
            print(
                "[launcher] - gunicorn: env overrides supported: GUNICORN_WORKERS, GUNICORN_THREADS, GUNICORN_EXTRA_ARGS, ARXIV_SANITY_GUNICORN_PRELOAD",
                flush=True,
            )

    if verbose:
        print(
            "[launcher] - web: "
            f"mode={args.web} with_daemon={args.with_daemon} no_wait={args.no_wait} wait_timeout={args.wait_timeout}",
            flush=True,
        )


def _log_litellm_models(port: int, *, verbose: bool) -> None:
    """Log which OpenAI-compatible models LiteLLM exposes."""

    url = f"http://localhost:{port}/v1/models"
    data = _http_get_json(url, timeout_s=2.5)
    if not isinstance(data, dict):
        print(f"[launcher] LiteLLM models: unable to read {url}", flush=True)
        return

    models = data.get("data")
    if not isinstance(models, list):
        print(
            f"[launcher] LiteLLM models: unexpected response schema from {url}",
            flush=True,
        )
        return

    ids: list[str] = []
    for m in models:
        if isinstance(m, dict) and isinstance(m.get("id"), str):
            ids.append(m["id"])

    if not ids:
        print("[launcher] LiteLLM models: (none)", flush=True)
        return

    ids_sorted = sorted(set(ids))
    if verbose or len(ids_sorted) <= 20:
        print(f"[launcher] LiteLLM models ({len(ids_sorted)}): {ids_sorted}", flush=True)
    else:
        head = ids_sorted[:20]
        print(
            f"[launcher] LiteLLM models ({len(ids_sorted)}): {head} ... (+{len(ids_sorted) - len(head)} more)",
            flush=True,
        )


def _log_ollama_models(port: int, *, verbose: bool) -> None:
    """Log which models are present in Ollama (best-effort)."""

    url = f"http://localhost:{port}/api/tags"
    data = _http_get_json(url, timeout_s=2.5)
    if not isinstance(data, dict):
        # Not all setups have /api/tags ready; keep it best-effort.
        return

    models = data.get("models")
    if not isinstance(models, list):
        return

    names: list[str] = []
    for m in models:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            names.append(m["name"])

    if not names:
        print("[launcher] Ollama models: (none pulled yet)", flush=True)
        return

    names_sorted = sorted(set(names))
    if verbose or len(names_sorted) <= 20:
        print(
            f"[launcher] Ollama models ({len(names_sorted)}): {names_sorted}",
            flush=True,
        )
    else:
        head = names_sorted[:20]
        print(
            f"[launcher] Ollama models ({len(names_sorted)}): {head} ... (+{len(names_sorted) - len(head)} more)",
            flush=True,
        )


def _wait_for_http(url: str, timeout_s: float, name: str, verbose: bool = False) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _http_ok(url, timeout_s=1.0):
            if verbose:
                print(f"[launcher] {name} ready: {url}", flush=True)
            return True
        time.sleep(0.4)
    print(f"[launcher] {name} not ready after {timeout_s:.1f}s: {url}", flush=True)
    return False


def _wait_for_all_services(
    services_to_wait: list[tuple[str, str]],  # [(name, health_url), ...]
    timeout_s: float,
    verbose: bool = False,
) -> dict[str, bool]:
    """Wait for all services to become ready.

    Args:
        services_to_wait: List of (name, health_url) tuples
        timeout_s: Maximum time to wait for each service
        verbose: Whether to show verbose output

    Returns:
        Dict mapping service name to ready status
    """
    if not services_to_wait:
        return {}

    deadline = time.time() + timeout_s
    status: dict[str, bool] = {name: False for name, _ in services_to_wait}

    try:
        while time.time() < deadline:
            # Check all pending services
            all_ready = True
            for name, url in services_to_wait:
                if status[name]:
                    continue
                if _http_ok(url, timeout_s=0.5):
                    status[name] = True
                    ready_count = sum(1 for ready in status.values() if ready)
                    total = len(status)
                    print(f"[launcher] ({ready_count}/{total}) {name} ready", flush=True)
                else:
                    all_ready = False

            if all_ready:
                break

            time.sleep(0.5)

        # Final status for any services that didn't become ready
        for name, url in services_to_wait:
            if not status[name]:
                print(
                    f"[launcher] {name} not ready after {timeout_s:.1f}s: {url}",
                    flush=True,
                )

    except KeyboardInterrupt:
        raise

    return status


def _check_mineru_api(api_key: str | None, verbose: bool = False) -> bool:
    """Check if MinerU API is available and key is valid."""
    if not api_key or not api_key.strip():
        print(
            "[launcher] Error: ARXIV_SANITY_MINERU_API_KEY is not set. API backend requires a valid API key.",
            file=sys.stderr,
            flush=True,
        )
        return False

    api_url = "https://mineru.net/api/v4/extract/task"
    try:
        import requests

        # Test API key validity with a lightweight request
        headers = {"Authorization": f"Bearer {api_key.strip()}"}
        response = requests.get(api_url, headers=headers, timeout=5)

        if response.status_code == 401:
            print(
                "[launcher] Error: MinerU API key is invalid or expired. Please check your ARXIV_SANITY_MINERU_API_KEY.",
                file=sys.stderr,
                flush=True,
            )
            return False
        elif response.status_code == 403:
            print(
                "[launcher] Error: MinerU API access denied. Your API key may have expired or lacks permissions.",
                file=sys.stderr,
                flush=True,
            )
            return False
        elif response.status_code >= 500:
            print(
                f"[launcher] Warning: MinerU API server error (status {response.status_code}). Service may be temporarily unavailable.",
                file=sys.stderr,
                flush=True,
            )
            return True  # Don't block startup for temporary server issues

        # 200, 400, 404 etc. are acceptable - means API is reachable
        if verbose:
            print(f"[launcher] MinerU API is available (endpoint: {api_url})", flush=True)
        return True

    except ImportError:
        print(
            "[launcher] Error: 'requests' library not found. Install with: pip install requests",
            file=sys.stderr,
            flush=True,
        )
        return False
    except Exception as e:
        print(
            f"[launcher] Warning: Failed to verify MinerU API availability: {e}",
            file=sys.stderr,
            flush=True,
        )
        return True  # Don't block startup for network issues


def _task_category_from_model(model: str | None) -> str:
    normalized = str(model or "").strip().lower()
    if normalized in {"upload_process", "upload_parse", "upload_extract"}:
        return normalized
    return "summary"


def _sort_task_categories(categories: list[str] | set[str]) -> list[str]:
    return sorted(categories, key=lambda item: (_TASK_CATEGORY_ORDER.get(item, 99), item))


def _compact_duration_label(seconds: float) -> str:
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        total = 0
    if total < 60:
        return f"{total}s"
    if total % 60 == 0:
        return f"{total // 60}m"
    return f"{total}s"


def _summarize_task_infos(
    task_infos: list[dict],
    *,
    window_start: float,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    terminal_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    active_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for info in task_infos:
        if not isinstance(info, dict):
            continue
        status = str(info.get("status") or "").strip().lower()
        if not status:
            continue
        category = _task_category_from_model(info.get("model"))

        if status in _TASK_ACTIVE_STATUSES:
            active_counts[category][status] += 1
            continue

        if status not in _TASK_TERMINAL_STATUSES:
            continue

        try:
            updated_time = float(info.get("updated_time") or 0.0)
        except Exception:
            updated_time = 0.0
        if updated_time > float(window_start):
            terminal_counts[category][status] += 1

    return (
        {category: dict(counts) for category, counts in terminal_counts.items()},
        {category: dict(counts) for category, counts in active_counts.items()},
    )


def _format_task_summary_lines(
    terminal_counts: dict[str, dict[str, int]],
    active_counts: dict[str, dict[str, int]],
    *,
    interval_s: float,
) -> list[str]:
    lines: list[str] = []

    terminal_categories = _sort_task_categories(set(terminal_counts.keys()))
    if terminal_categories:
        terminal_parts: list[str] = []
        for category in terminal_categories:
            counts = terminal_counts.get(category) or {}
            status_parts: list[str] = []
            ok = int(counts.get("ok") or 0)
            failed = int(counts.get("failed") or 0)
            canceled = int(counts.get("canceled") or 0)
            if ok:
                status_parts.append(f"ok={ok}")
            if failed:
                status_parts.append(f"fail={failed}")
            if canceled:
                status_parts.append(f"cancel={canceled}")
            if status_parts:
                terminal_parts.append(f"{category} {' '.join(status_parts)}")
        if terminal_parts:
            lines.append(f"[huey] past {_compact_duration_label(interval_s)}: {' | '.join(terminal_parts)}")

    active_categories = _sort_task_categories(set(active_counts.keys()))
    if active_categories:
        active_parts: list[str] = []
        for category in active_categories:
            counts = active_counts.get(category) or {}
            status_parts: list[str] = []
            queued = int(counts.get("queued") or 0)
            running = int(counts.get("running") or 0)
            if queued:
                status_parts.append(f"queued={queued}")
            if running:
                status_parts.append(f"running={running}")
            if status_parts:
                active_parts.append(f"{category} {' '.join(status_parts)}")
        if active_parts:
            lines.append(f"[huey] active: {' | '.join(active_parts)}")

    return lines


def _read_task_infos() -> list[dict]:
    from aslite.repositories import SummaryStatusRepository, safe_closing

    infos: list[dict] = []
    with safe_closing(SummaryStatusRepository.get_items_with_prefix("task::")) as items:
        for _key, info in items:
            if isinstance(info, dict):
                infos.append(info)
    return infos


def _task_summary_loop(*, repo_root: Path, interval_s: float, stop_event: threading.Event, verbose: bool) -> None:
    if interval_s <= 0:
        return

    import_failure_logged = False
    window_start = time.time()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    while not stop_event.wait(interval_s):
        now = time.time()
        advance_window = False
        try:
            task_infos = _read_task_infos()
            terminal_counts, active_counts = _summarize_task_infos(task_infos, window_start=window_start)
            for line in _format_task_summary_lines(terminal_counts, active_counts, interval_s=now - window_start):
                print(line, flush=True)
            import_failure_logged = False
            advance_window = True
        except Exception as e:
            if verbose and not import_failure_logged:
                print(f"[launcher] Task summary reporter unavailable: {e}", flush=True)
                import_failure_logged = True
        finally:
            if advance_window:
                window_start = now


def _build_huey_command(
    *,
    python_executable: str,
    consumer_script: Path,
    workers: int,
    worker_type: str,
    show_task_logs: bool,
) -> list[str]:
    cmd = [
        python_executable,
        str(consumer_script),
        "tasks.huey",
        "-w",
        str(workers),
        "-k",
        worker_type,
    ]
    if not show_task_logs:
        cmd.append("-q")
    return cmd


def _stream_lines(prefix: str, pipe):
    """Stream subprocess output into our terminal.

    Important: when using text mode + universal newlines, Python treats '\r' as a
    line terminator. tqdm uses '\r' to update progress in-place, so reading with
    readline() would turn progress updates into many lines (flood screen).

    We therefore read bytes and handle '\r' (in-place update) separately from
    '\n' (real new line).
    """

    buffer = ""
    in_place = False
    try:
        fd = pipe.fileno()
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break

            buffer += chunk.decode("utf-8", errors="replace")

            while True:
                idx_r = buffer.find("\r")
                idx_n = buffer.find("\n")
                if idx_r == -1 and idx_n == -1:
                    break

                use_r = idx_r != -1 and (idx_n == -1 or idx_r < idx_n)
                if use_r:
                    part = buffer[:idx_r]
                    buffer = buffer[idx_r + 1 :]
                    sys.stdout.write(f"\r[{prefix}] {_sanitize_message(part)}")
                    sys.stdout.flush()
                    in_place = True
                else:
                    part = buffer[:idx_n]
                    buffer = buffer[idx_n + 1 :]
                    if in_place:
                        sys.stdout.write("\n")
                        in_place = False
                    if _USE_RAW_LOG_STREAM:
                        _emit_raw_line(prefix, part)
                    else:
                        _emit_normalized_line(prefix, part)

        # Flush any remaining buffered output.
        if buffer:
            if in_place:
                sys.stdout.write(f"\r[{prefix}] {_sanitize_message(buffer)}")
                sys.stdout.flush()
            else:
                if _USE_RAW_LOG_STREAM:
                    _emit_raw_line(prefix, buffer)
                else:
                    _emit_normalized_line(prefix, buffer)

    finally:
        if not _USE_RAW_LOG_STREAM:
            _flush_log_summaries(prefix)
        try:
            pipe.close()
        except Exception:
            pass


def _start_service(spec: ServiceSpec) -> subprocess.Popen:
    env = os.environ.copy()
    proc = subprocess.Popen(
        spec.cmd,
        cwd=str(spec.cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        start_new_session=hasattr(os, "setsid"),
    )
    assert proc.stdout is not None
    t = threading.Thread(target=_stream_lines, args=(spec.name, proc.stdout), daemon=True)
    t.start()
    return proc


def _stop_process(proc: subprocess.Popen, _name: str, timeout_s: float = 10.0):
    if proc.poll() is not None:
        return
    _ = _name
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def _run_fetch_compute(repo_root: Path, tools_dir: Path, num_papers: int, max_r: int) -> int:
    print(f"[launcher] Fetching latest {num_papers} papers...", flush=True)
    fetch_cmd = [
        sys.executable,
        str(tools_dir / "arxiv_daemon.py"),
        "-n",
        str(num_papers),
        "--num-total",
        str(num_papers),
        "-m",
        str(max_r),
        "--break-after",
        "0",
    ]
    fetch_rc = subprocess.call(fetch_cmd, cwd=str(repo_root))
    if fetch_rc != 0:
        print(f"[launcher] Fetch failed with code {fetch_rc}", flush=True)
        return fetch_rc

    print("[launcher] Computing features...", flush=True)
    compute_cmd = [sys.executable, str(tools_dir / "compute.py"), "--use_embeddings"]
    compute_rc = subprocess.call(compute_cmd, cwd=str(repo_root))
    if compute_rc != 0:
        print(f"[launcher] Compute failed with code {compute_rc}", flush=True)
        return compute_rc

    print("[launcher] Fetch + compute complete.", flush=True)
    return 0


def _configure_web_readiness_env(*, no_embed: bool, no_mineru: bool) -> None:
    """Align web /ready strict checks with launcher-managed services."""
    os.environ.setdefault("ARXIV_SANITY_READY_REQUIRE_EMBEDDING", "0" if no_embed else "1")
    os.environ.setdefault("ARXIV_SANITY_READY_REQUIRE_MINERU", "0" if no_mineru else "1")


def main() -> int:
    global _USE_RAW_LOG_STREAM

    parser = argparse.ArgumentParser(description="Run arxiv-sanity-X services in one terminal.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose launcher logs and restore raw Huey task logs.",
    )
    parser.add_argument(
        "--verbose-raw-logs",
        action="store_true",
        help="Pass child process logs through without launcher summarization.",
    )
    parser.add_argument("--no-embed", action="store_true", help="Do not start Ollama embedding service.")
    parser.add_argument("--no-mineru", action="store_true", help="Do not start minerU service.")
    parser.add_argument("--no-litellm", action="store_true", help="Do not start LiteLLM gateway.")
    parser.add_argument(
        "--web",
        choices=["python", "gunicorn", "none"],
        default="gunicorn",
        help="How to start the web server (default: gunicorn).",
    )
    parser.add_argument("--with-daemon", action="store_true", help="Also start scheduler daemon.py.")
    parser.add_argument(
        "--no-huey",
        action="store_true",
        help="Disable Huey task worker (enabled by default).",
    )
    parser.add_argument(
        "--huey-workers",
        type=int,
        default=None,
        help="Number of Huey workers (default: env ARXIV_SANITY_HUEY_WORKERS or 4).",
    )
    parser.add_argument(
        "--huey-worker-type",
        type=str,
        default=None,
        choices=["thread", "process", "greenlet"],
        help="Huey worker type (default: env ARXIV_SANITY_HUEY_WORKER_TYPE or thread).",
    )
    parser.add_argument("--no-wait", action="store_true", help="Skip health-check waits.")
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=60.0,
        help="Health-check wait timeout seconds.",
    )
    parser.add_argument(
        "--task-summary-interval",
        type=float,
        default=180.0,
        help="Seconds between aggregated Huey task summaries (default: 180). Use 0 to disable.",
    )
    parser.add_argument(
        "--fetch-compute",
        type=int,
        nargs="?",
        const=10000,
        default=None,
        help="Run one-shot fetch of latest N papers and compute features, then exit (default: 10000).",
    )
    parser.add_argument(
        "--summary-source",
        choices=["html", "mineru"],
        default=None,
        help="Markdown source for paper summaries (default: html).",
    )
    args = parser.parse_args()
    user_disabled_mineru = bool(args.no_mineru)
    _USE_RAW_LOG_STREAM = bool(args.verbose_raw_logs)

    if settings is None:
        print("[launcher] Failed to import config.settings", file=sys.stderr)
        return 2

    previous_sigterm_handler = None
    try:
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _launcher_sigterm_handler)
    except Exception:
        previous_sigterm_handler = None

    verbose = args.verbose or settings.log_level.upper() in ("DEBUG", "INFO")
    show_huey_task_logs = bool(args.verbose or args.verbose_raw_logs)

    # Get project root (parent of bin/ directory)
    repo_root = Path(__file__).resolve().parent.parent
    bin_dir = repo_root / "bin"
    tools_dir = repo_root / "tools"

    # Allow choosing summary source when launching the web service.
    if args.summary_source:
        os.environ["ARXIV_SANITY_SUMMARY_MARKDOWN_SOURCE"] = args.summary_source

    EMBED_PORT = settings.embedding.port
    LITELLM_PORT = settings.litellm_port
    MINERU_PORT = settings.mineru.port
    MINERU_ENABLED = settings.mineru.enabled
    MINERU_BACKEND = settings.mineru.backend
    mineru_api_key = settings.mineru.api_key
    SERVE_PORT = settings.serve_port

    if args.fetch_compute is not None:
        return _run_fetch_compute(
            repo_root=repo_root,
            tools_dir=tools_dir,
            num_papers=args.fetch_compute,
            max_r=1000,
        )

    # Check if MinerU is disabled globally
    if not MINERU_ENABLED and not args.no_mineru:
        print(
            "[launcher] MinerU is disabled (ARXIV_SANITY_MINERU_ENABLED=false), skip starting minerU service.",
            file=sys.stderr,
            flush=True,
        )
        args.no_mineru = True

    mineru_backend = (MINERU_BACKEND or "pipeline").strip().lower()
    if mineru_backend == "api" and user_disabled_mineru and MINERU_ENABLED:
        print(
            "[launcher] Warning: --no-mineru skips MinerU API preflight checks while backend=api.",
            file=sys.stderr,
            flush=True,
        )
    if mineru_backend == "pipeline" and not args.no_mineru:
        print(
            "[launcher] Warning: ARXIV_SANITY_MINERU_BACKEND=pipeline, skip starting minerU vLLM service. "
            "Set ARXIV_SANITY_MINERU_BACKEND=vlm-http-client if you need the VLM backend.",
            file=sys.stderr,
            flush=True,
        )
        args.no_mineru = True
    elif mineru_backend == "api" and not args.no_mineru:
        # For API backend, check API availability instead of starting local service
        api_key = mineru_api_key
        if not _check_mineru_api(api_key, verbose=verbose):
            print(
                "[launcher] Error: MinerU API backend is not available. Fix the API key issue or disable MinerU.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        if verbose:
            print(
                "[launcher] MinerU API backend configured, skip starting local minerU service.",
                flush=True,
            )
        args.no_mineru = True

    _configure_web_readiness_env(no_embed=bool(args.no_embed), no_mineru=bool(args.no_mineru))

    services: list[ServiceSpec] = []

    if not args.no_mineru:
        services.append(
            ServiceSpec(
                name="mineru",
                cmd=["bash", str(bin_dir / "mineru_serve.sh")],
                cwd=repo_root,
                health_url=f"http://localhost:{MINERU_PORT}/health",
            )
        )

    if not args.no_embed:
        services.append(
            ServiceSpec(
                name="embed",
                cmd=["bash", str(bin_dir / "embedding_serve.sh")],
                cwd=repo_root,
                health_url=f"http://localhost:{EMBED_PORT}/api/version",
            )
        )

    if not args.no_litellm:
        services.append(
            ServiceSpec(
                name="litellm",
                cmd=["bash", str(bin_dir / "litellm.sh")],
                cwd=repo_root,
                # LiteLLM exposes an OpenAI-compatible API; /v1/models is a stable readiness check.
                health_url=f"http://localhost:{LITELLM_PORT}/v1/models",
            )
        )

    if args.web == "python":
        services.append(
            ServiceSpec(
                name="web",
                cmd=[sys.executable, str(repo_root / "serve.py")],
                cwd=repo_root,
                health_url=f"http://localhost:{SERVE_PORT}/ready",
            )
        )
    elif args.web == "gunicorn":
        services.append(
            ServiceSpec(
                name="web",
                cmd=["bash", str(bin_dir / "up.sh")],
                cwd=repo_root,
                health_url=f"http://localhost:{SERVE_PORT}/ready",
            )
        )

    if args.with_daemon:
        services.append(
            ServiceSpec(
                name="daemon",
                cmd=[sys.executable, str(tools_dir / "daemon.py")],
                cwd=repo_root,
            )
        )

    # Huey task worker (default enabled unless --no-huey)
    if not args.no_huey:
        huey_workers = args.huey_workers
        if huey_workers is None:
            huey_workers = settings.huey.workers

        huey_worker_type = args.huey_worker_type or settings.huey.worker_type
        huey_worker_type = (huey_worker_type or "thread").strip().lower()
        if huey_worker_type not in {"thread", "process", "greenlet"}:
            huey_worker_type = "thread"

        # Use wrapper script for memory limit support
        huey_consumer_script = bin_dir / "huey_consumer.py"
        services.append(
            ServiceSpec(
                name="huey",
                cmd=_build_huey_command(
                    python_executable=sys.executable,
                    consumer_script=huey_consumer_script,
                    workers=huey_workers,
                    worker_type=huey_worker_type,
                    show_task_logs=show_huey_task_logs,
                ),
                cwd=repo_root,
            )
        )

    if not services:
        print("[launcher] Nothing to start (all services disabled).")
        return 0

    procs: list[tuple[ServiceSpec, subprocess.Popen]] = []

    _print_startup_context(repo_root=repo_root, args=args, verbose=verbose)

    # Helpful warning: git backup is configured in settings.daemon but the daemon
    # process is not started unless --with-daemon is passed.
    try:
        if settings is not None and getattr(settings.daemon, "enable_git_backup", False) and not args.with_daemon:
            print(
                "[launcher] Warning: git backup is enabled but daemon is not started; pass --with-daemon",
                flush=True,
            )
    except Exception:
        pass

    print("[launcher] Starting services:", flush=True)
    if verbose:
        for spec in services:
            print(f"[launcher] - {spec.name}: {' '.join(spec.cmd)}", flush=True)

    task_summary_stop = threading.Event()
    enable_task_summary = (
        not args.no_huey and not args.verbose_raw_logs and float(args.task_summary_interval or 0.0) > 0
    )
    if enable_task_summary:
        if not show_huey_task_logs:
            print(
                "[launcher] Huey task logs are condensed; periodic task summaries are enabled "
                f"every {_compact_duration_label(float(args.task_summary_interval))}.",
                flush=True,
            )
        threading.Thread(
            target=_task_summary_loop,
            kwargs={
                "repo_root": repo_root,
                "interval_s": float(args.task_summary_interval),
                "stop_event": task_summary_stop,
                "verbose": verbose,
            },
            name="launcher-task-summary",
            daemon=True,
        ).start()

    try:
        for spec in services:
            procs.append((spec, _start_service(spec)))

        if not args.no_wait:
            # Collect all services that need health checks
            services_to_wait = [(spec.name, spec.health_url) for spec, _ in procs if spec.health_url]

            if services_to_wait:
                print(
                    f"[launcher] Waiting for {len(services_to_wait)} service(s)...",
                    flush=True,
                )
                status = _wait_for_all_services(services_to_wait, timeout_s=args.wait_timeout, verbose=verbose)

                # Log additional info for ready services
                for spec, _ in procs:
                    if spec.health_url and status.get(spec.name):
                        if spec.name == "litellm":
                            _log_litellm_models(LITELLM_PORT, verbose=verbose)
                        elif spec.name == "embed":
                            _log_ollama_models(EMBED_PORT, verbose=verbose)

                # Check if all services are ready
                all_ready = all(status.values())
                if all_ready:
                    # Find web service URL for final message
                    web_url = None
                    for spec, _ in procs:
                        if spec.name == "web" and spec.health_url:
                            web_url = spec.health_url.replace("/ready", "/").replace("/health", "/")
                            break
                    if web_url:
                        print(f"[launcher] All services ready: {web_url}", flush=True)
                    else:
                        print("[launcher] All services ready", flush=True)
                else:
                    not_ready = [name for name, ok in status.items() if not ok]
                    print(
                        f"[launcher] Readiness check failed for: {', '.join(not_ready)}",
                        flush=True,
                    )
                    raise SystemExit(2)

        # main loop
        while True:
            for spec, proc in procs:
                rc = proc.poll()
                if rc is not None:
                    print(f"[launcher] {spec.name} exited with code {rc}", flush=True)
                    raise SystemExit(rc if rc != 0 else 0)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[launcher] Stopping...", flush=True)
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    finally:
        task_summary_stop.set()
        _flush_log_summaries()
        if previous_sigterm_handler is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
            except Exception:
                pass
        for spec, proc in reversed(procs):
            _stop_process(proc, spec.name)


if __name__ == "__main__":
    raise SystemExit(main())
