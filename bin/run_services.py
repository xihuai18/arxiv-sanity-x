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
import shutil
import signal
import subprocess
import sys
import threading
import time
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

    print(f"[launcher] Failed to import config.settings: {e!r}", file=sys.stderr, flush=True)
    traceback.print_exc()
    settings = None


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    cmd: list[str]
    cwd: Path
    health_url: str | None = None


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
        print(f"[launcher] LiteLLM models: unexpected response schema from {url}", flush=True)
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
        print(f"[launcher] Ollama models ({len(names_sorted)}): {names_sorted}", flush=True)
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


def _check_mineru_api(api_key: str | None, verbose: bool = False) -> bool:
    """Check if MinerU API is available and key is valid."""
    if not api_key or not api_key.strip():
        print(
            "[launcher] Error: MINERU_API_KEY is not set. API backend requires a valid API key.",
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
                "[launcher] Error: MinerU API key is invalid or expired. Please check your MINERU_API_KEY.",
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
        print(f"[launcher] Warning: Failed to verify MinerU API availability: {e}", file=sys.stderr, flush=True)
        return True  # Don't block startup for network issues


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
                    sys.stdout.write(f"\r[{prefix}] {part}")
                    sys.stdout.flush()
                    in_place = True
                else:
                    part = buffer[:idx_n]
                    buffer = buffer[idx_n + 1 :]
                    if in_place:
                        sys.stdout.write("\n")
                        in_place = False
                    sys.stdout.write(f"[{prefix}] {part}\n")
                    sys.stdout.flush()

        # Flush any remaining buffered output.
        if buffer:
            if in_place:
                sys.stdout.write(f"\r[{prefix}] {buffer}")
            else:
                sys.stdout.write(f"[{prefix}] {buffer}")
            sys.stdout.flush()

    finally:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run arxiv-sanity-X services in one terminal.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose launcher logs.")
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
    parser.add_argument("--no-huey", action="store_true", help="Disable Huey task worker (enabled by default).")
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
    parser.add_argument("--wait-timeout", type=float, default=60.0, help="Health-check wait timeout seconds.")
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

    if settings is None:
        print("[launcher] Failed to import config.settings", file=sys.stderr)
        return 2

    verbose = args.verbose or settings.log_level.upper() in ("DEBUG", "INFO")

    # Get project root (parent of bin/ directory)
    repo_root = Path(__file__).resolve().parent.parent
    bin_dir = repo_root / "bin"
    tools_dir = repo_root / "tools"

    # Allow choosing summary source when launching the web service.
    if args.summary_source:
        os.environ["ARXIV_SANITY_SUMMARY_SOURCE"] = args.summary_source

    EMBED_PORT = settings.embedding.port
    LITELLM_PORT = settings.litellm_port
    MINERU_PORT = settings.mineru.port
    MINERU_ENABLED = settings.mineru.enabled
    MINERU_BACKEND = settings.mineru.backend
    MINERU_API_KEY = settings.mineru.api_key
    SERVE_PORT = settings.serve_port

    if args.fetch_compute is not None:
        return _run_fetch_compute(repo_root=repo_root, tools_dir=tools_dir, num_papers=args.fetch_compute, max_r=1000)

    # Check if MinerU is disabled globally
    if not MINERU_ENABLED and not args.no_mineru:
        print(
            "[launcher] MinerU is disabled (ARXIV_SANITY_MINERU_ENABLED=false), skip starting minerU service.",
            file=sys.stderr,
            flush=True,
        )
        args.no_mineru = True

    mineru_backend = (MINERU_BACKEND or "pipeline").strip().lower()
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
        api_key = MINERU_API_KEY
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
                health_url=f"http://localhost:{SERVE_PORT}/",
            )
        )
    elif args.web == "gunicorn":
        services.append(
            ServiceSpec(
                name="web",
                cmd=["bash", str(bin_dir / "up.sh")],
                cwd=repo_root,
                health_url=f"http://localhost:{SERVE_PORT}/",
            )
        )

    if args.with_daemon:
        services.append(ServiceSpec(name="daemon", cmd=[sys.executable, str(tools_dir / "daemon.py")], cwd=repo_root))

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
                cmd=[
                    sys.executable,
                    str(huey_consumer_script),
                    "tasks.huey",
                    "-w",
                    str(huey_workers),
                    "-k",
                    huey_worker_type,
                ],
                cwd=repo_root,
            )
        )

    if not services:
        print("[launcher] Nothing to start (all services disabled).")
        return 0

    procs: list[tuple[ServiceSpec, subprocess.Popen]] = []

    _print_startup_context(repo_root=repo_root, args=args, verbose=verbose)

    print("[launcher] Starting services:", flush=True)
    if verbose:
        for spec in services:
            print(f"[launcher] - {spec.name}: {' '.join(spec.cmd)}", flush=True)

    try:
        for spec in services:
            procs.append((spec, _start_service(spec)))

        if not args.no_wait:
            for spec, proc in procs:
                if spec.health_url:
                    _wait_for_http(spec.health_url, timeout_s=args.wait_timeout, name=spec.name, verbose=verbose)
                    # After ready, log runtime-discovered info that helps debugging.
                    if spec.name == "litellm":
                        _log_litellm_models(LITELLM_PORT, verbose=verbose)
                    elif spec.name == "embed":
                        _log_ollama_models(EMBED_PORT, verbose=verbose)

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
        for spec, proc in reversed(procs):
            _stop_process(proc, spec.name)


if __name__ == "__main__":
    raise SystemExit(main())
