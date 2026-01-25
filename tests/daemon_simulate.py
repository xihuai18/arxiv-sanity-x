#!/usr/bin/env python3
"""Quick simulator for tools.daemon behavior.

Daemon normally blocks in a scheduler and may not print immediately.
This script runs the underlying job functions once with a fake runner so you can
see what would be executed.

Usage examples:
  conda run -n sanity python tests/daemon_simulate.py --fetch-ok
  conda run -n sanity python tests/daemon_simulate.py --fetch-ok --no-embeddings
  conda run -n sanity python tests/daemon_simulate.py --no-fetch-ok

Notes:
- No external commands are executed; we only print what would run.
- This is intentionally placed under tests/ for easy local debugging.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description="Simulate tools.daemon job execution once (no scheduler).")
    p.add_argument("--fetch-ok", action="store_true", help="Simulate fetch step succeeds.")
    p.add_argument("--no-fetch-ok", action="store_true", help="Simulate fetch step fails (default if neither set).")
    p.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Simulate ENABLE_EMBEDDINGS=false (skip --use_embeddings for compute).",
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Simulate ENABLE_SUMMARY=false (skip summary generation).",
    )
    args = p.parse_args(argv)

    # Ensure repository root is importable when executing this file directly.
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    # Import inside main so import-time side effects are visible during debugging.
    import tools.daemon as d

    calls: list[tuple[str, list[str]]] = []

    def fake_run_cmd(cmd, name: str) -> bool:
        calls.append((name, list(cmd)))
        print(f"[simulate] {name}: {' '.join(map(str, cmd))}")
        if name == "fetch":
            return bool(args.fetch_ok) and not bool(args.no_fetch_ok)
        return True

    # Patch module knobs for this run.
    d._run_cmd = fake_run_cmd  # type: ignore[assignment]
    d.ENABLE_EMBEDDINGS = not args.no_embeddings
    d.ENABLE_SUMMARY = not args.no_summary

    # If neither flag is set, default to fetch failing (to show skip path).
    if not args.fetch_ok and not args.no_fetch_ok:
        args.no_fetch_ok = True

    print(
        "[simulate] config: "
        f"fetch_ok={args.fetch_ok and not args.no_fetch_ok} "
        f"enable_embeddings={d.ENABLE_EMBEDDINGS} "
        f"enable_summary={d.ENABLE_SUMMARY}"
    )

    # Run one "tick".
    d.fetch_compute()

    print(f"[simulate] done. steps={len(calls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
