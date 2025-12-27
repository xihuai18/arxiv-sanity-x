#!/usr/bin/env python3
"""
One-command launcher for arxiv-sanity-X services.

Starts (optionally):
- vLLM embedding server (embedding_serve.sh)
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
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    cmd: list[str]
    cwd: Path
    health_url: str | None = None


def _http_ok(url: str, timeout_s: float = 1.0) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "arxiv-sanity-x-launcher"})
        with urlopen(req, timeout=timeout_s) as resp:  # nosec - local health checks only
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


def _stream_lines(prefix: str, pipe):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sys.stdout.write(f"[{prefix}] {line}")
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
        text=True,
        bufsize=1,
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
    parser.add_argument("--no-embed", action="store_true", help="Do not start vLLM embedding service.")
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
            LITELLM_PORT,
            MINERU_BACKEND,
            SERVE_PORT,
            VLLM_EMBED_PORT,
            VLLM_MINERU_PORT,
        )
    except Exception as e:
        print(f"[launcher] Failed to import vars.py: {e}", file=sys.stderr)
        return 2

    if args.fetch_compute is not None:
        return _run_fetch_compute(repo_root=repo_root, num_papers=args.fetch_compute, max_r=1000)

    mineru_backend = (os.environ.get("ARXIV_SANITY_MINERU_BACKEND") or MINERU_BACKEND or "pipeline").strip().lower()
    if mineru_backend == "pipeline" and not args.no_mineru:
        print(
            "[launcher] Warning: ARXIV_SANITY_MINERU_BACKEND=pipeline, skip starting minerU vLLM service. "
            "Set ARXIV_SANITY_MINERU_BACKEND=vlm-http-client if you need the VLM backend.",
            file=sys.stderr,
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
                health_url=f"http://localhost:{VLLM_MINERU_PORT}/health",
            )
        )

    if not args.no_embed:
        services.append(
            ServiceSpec(
                name="embed",
                cmd=["bash", "embedding_serve.sh"],
                cwd=repo_root,
                health_url=f"http://localhost:{VLLM_EMBED_PORT}/health",
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
