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
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


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


def _wait_for_http(url: str, timeout_s: float, name: str) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _http_ok(url, timeout_s=1.0):
            print(f"[launcher] {name} ready: {url}", flush=True)
            return True
        time.sleep(0.4)
    print(f"[launcher] {name} not ready after {timeout_s:.1f}s: {url}", flush=True)
    return False


def _check_mineru_api(api_key: str | None) -> bool:
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
    readline() would turn progress updates into many lines ("刷屏").

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
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    assert proc.stdout is not None
    t = threading.Thread(target=_stream_lines, args=(spec.name, proc.stdout), daemon=True)
    t.start()
    return proc


def _stop_process(proc: subprocess.Popen, name: str, timeout_s: float = 10.0):
    if proc.poll() is not None:
        return
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


def _run_fetch_compute(repo_root: Path, num_papers: int, max_r: int) -> int:
    print(f"[launcher] Fetching latest {num_papers} papers...", flush=True)
    fetch_cmd = [
        sys.executable,
        "arxiv_daemon.py",
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
    compute_cmd = [sys.executable, "compute.py", "--use_embeddings"]
    compute_rc = subprocess.call(compute_cmd, cwd=str(repo_root))
    if compute_rc != 0:
        print(f"[launcher] Compute failed with code {compute_rc}", flush=True)
        return compute_rc

    print("[launcher] Fetch + compute complete.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run arxiv-sanity-X services in one terminal.")
    parser.add_argument("--no-embed", action="store_true", help="Do not start Ollama embedding service.")
    parser.add_argument("--no-mineru", action="store_true", help="Do not start minerU service.")
    parser.add_argument("--no-litellm", action="store_true", help="Do not start LiteLLM gateway.")
    parser.add_argument(
        "--web",
        choices=["python", "gunicorn", "none"],
        default="python",
        help="How to start the web server (default: python).",
    )
    parser.add_argument("--with-daemon", action="store_true", help="Also start scheduler daemon.py.")
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

    repo_root = Path(__file__).resolve().parent

    # Allow choosing summary source when launching the web service.
    if args.summary_source:
        os.environ["ARXIV_SANITY_SUMMARY_SOURCE"] = args.summary_source

    try:
        from vars import (
            EMBED_PORT,
            LITELLM_PORT,
            MINERU_API_KEY,
            MINERU_BACKEND,
            MINERU_ENABLED,
            MINERU_PORT,
            SERVE_PORT,
        )
    except Exception as e:
        print(f"[launcher] Failed to import vars.py: {e}", file=sys.stderr)
        return 2

    if args.fetch_compute is not None:
        return _run_fetch_compute(repo_root=repo_root, num_papers=args.fetch_compute, max_r=1000)

    # Check if MinerU is disabled globally
    if not MINERU_ENABLED and not args.no_mineru:
        print(
            "[launcher] MinerU is disabled (ARXIV_SANITY_MINERU_ENABLED=false), skip starting minerU service.",
            file=sys.stderr,
            flush=True,
        )
        args.no_mineru = True

    mineru_backend = (os.environ.get("ARXIV_SANITY_MINERU_BACKEND") or MINERU_BACKEND or "pipeline").strip().lower()
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
        api_key = os.environ.get("ARXIV_SANITY_MINERU_API_KEY") or MINERU_API_KEY
        if not _check_mineru_api(api_key):
            print(
                "[launcher] Error: MinerU API backend is not available. Fix the API key issue or disable MinerU.",
                file=sys.stderr,
                flush=True,
            )
            return 3
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
                cmd=["bash", "mineru_serve.sh"],
                cwd=repo_root,
                health_url=f"http://localhost:{MINERU_PORT}/health",
            )
        )

    if not args.no_embed:
        services.append(
            ServiceSpec(
                name="embed",
                cmd=["bash", "embedding_serve.sh"],
                cwd=repo_root,
                health_url=f"http://localhost:{EMBED_PORT}/api/version",
            )
        )

    if not args.no_litellm:
        services.append(
            ServiceSpec(
                name="litellm",
                cmd=["bash", "litellm.sh"],
                cwd=repo_root,
                # LiteLLM exposes an OpenAI-compatible API; /v1/models is a stable readiness check.
                health_url=f"http://localhost:{LITELLM_PORT}/v1/models",
            )
        )

    if args.web == "python":
        services.append(
            ServiceSpec(
                name="web",
                cmd=[sys.executable, "serve.py"],
                cwd=repo_root,
                health_url=f"http://localhost:{SERVE_PORT}/",
            )
        )
    elif args.web == "gunicorn":
        services.append(
            ServiceSpec(
                name="web",
                cmd=["bash", "up.sh"],
                cwd=repo_root,
                health_url=f"http://localhost:{SERVE_PORT}/",
            )
        )

    if args.with_daemon:
        services.append(ServiceSpec(name="daemon", cmd=[sys.executable, "daemon.py"], cwd=repo_root))

    if not services:
        print("[launcher] Nothing to start (all services disabled).")
        return 0

    procs: list[tuple[ServiceSpec, subprocess.Popen]] = []

    print("[launcher] Starting services:", flush=True)
    for spec in services:
        print(f"[launcher] - {spec.name}: {' '.join(spec.cmd)}", flush=True)

    try:
        for spec in services:
            procs.append((spec, _start_service(spec)))

        if not args.no_wait:
            for spec, proc in procs:
                if spec.health_url:
                    _wait_for_http(spec.health_url, timeout_s=args.wait_timeout, name=spec.name)

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
