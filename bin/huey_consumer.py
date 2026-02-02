#!/usr/bin/env python3
"""
Huey consumer wrapper with memory limit support.

This script wraps the huey consumer to apply memory limits configured
in settings.huey.max_memory_mb.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    # Mark this process as worker for DB lock tolerance tuning.
    os.environ.setdefault("ARXIV_SANITY_PROCESS_ROLE", "worker")
    # Mark this process explicitly as a Huey consumer to avoid argv-based misdetection.
    os.environ.setdefault("ARXIV_SANITY_HUEY_CONSUMER", "1")

    # Apply memory limit before importing heavy modules
    try:
        from config import settings

        max_mb = settings.huey.max_memory_mb
        if max_mb > 0:
            from backend.utils.memory_limit import set_memory_limit

            set_memory_limit(max_mb)
    except Exception as e:
        print(f"[huey_wrapper] Warning: Failed to apply memory limit: {e}", file=sys.stderr)

    # Import and run huey consumer
    from huey.bin.huey_consumer import consumer_main

    consumer_main()


if __name__ == "__main__":
    main()
