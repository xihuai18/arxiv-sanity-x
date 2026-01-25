#!/usr/bin/env python3
"""
Task Cleanup Utility for Development

This script cleans up task entries in the summary status DB (task::*)
that may be stale or incorrect after crashes or manual restarts.

Usage:
    python cleanup_tasks.py                          # Dry run (shows what would be deleted)
    python cleanup_tasks.py --force                  # Actually delete matching tasks
    python cleanup_tasks.py --all                    # Delete ALL task entries (dangerous!)
    python cleanup_tasks.py --status running         # Only clean tasks with status=running
    python cleanup_tasks.py --pid 2601.14256         # Only clean tasks for a pid
    python cleanup_tasks.py --model glm-4.7          # Only clean tasks for a model
    python cleanup_tasks.py --task-id <id>           # Only clean a specific task id
    python cleanup_tasks.py --stale-time 3600        # Only clean tasks older than N seconds
    python cleanup_tasks.py --keep-summary-status    # Do not delete pid::model summary status
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Ensure repository root is importable when executing this file directly.
# e.g. `python scripts/cleanup_tasks.py` would otherwise miss top-level packages like `aslite/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aslite.db import get_summary_status_db
from aslite.repositories import SummaryStatusRepository
from config import settings


class TaskCleaner:
    def __init__(self):
        self.stats = {
            "total": 0,
            "matched": 0,
            "removed": 0,
        }

    @staticmethod
    def _normalize_status_filter(status_filter: Optional[str]) -> Optional[List[str]]:
        if not status_filter:
            return None
        if "," in status_filter:
            return [s.strip() for s in status_filter.split(",") if s.strip()]
        return [status_filter.strip()]

    @staticmethod
    def _task_age_s(updated_time: Optional[float]) -> Optional[int]:
        if not updated_time:
            return None
        return int(max(0.0, time.time() - float(updated_time)))

    def list_tasks(self):
        return list(SummaryStatusRepository.get_items_with_prefix("task::"))

    def clean_tasks(
        self,
        force: bool = False,
        delete_all: bool = False,
        stale_time: Optional[float] = None,
        status_filter: Optional[List[str]] = None,
        pid_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        task_id_filter: Optional[str] = None,
        keep_summary_status: bool = False,
    ) -> None:
        tasks = self.list_tasks()
        self.stats["total"] = len(tasks)

        if not tasks:
            logger.debug("No task entries found")
            return

        for key, info in tasks:
            if not isinstance(info, dict):
                continue

            task_id = key.replace("task::", "")
            status = info.get("status")
            pid = info.get("pid")
            model = info.get("model")
            updated_time = info.get("updated_time") or 0

            if not delete_all:
                if status_filter and status not in status_filter:
                    continue
                if pid_filter and pid != pid_filter:
                    continue
                if model_filter and model != model_filter:
                    continue
                if task_id_filter and task_id != task_id_filter:
                    continue
                if stale_time is not None:
                    if not updated_time:
                        continue
                    if (time.time() - float(updated_time)) < stale_time:
                        continue

            self.stats["matched"] += 1
            age_s = self._task_age_s(updated_time)
            label = f"task::{task_id} pid={pid} model={model} status={status} age={age_s}s"

            if force:
                try:
                    with get_summary_status_db(flag="c") as sdb:
                        if key in sdb:
                            del sdb[key]
                    self.stats["removed"] += 1
                    logger.success(f"Removed: {label}")
                except Exception as e:
                    logger.error(f"Failed to remove {label}: {e}")
                    continue

                # Optionally clear summary status if it is queued/running
                if not keep_summary_status and pid and model and status in ("queued", "running"):
                    try:
                        SummaryStatusRepository.delete_status(pid, model)
                    except Exception:
                        pass
            else:
                logger.warning(f"Would remove: {label}")

        if force:
            logger.success(f"Removed tasks: {self.stats['removed']} / matched {self.stats['matched']}")
        else:
            logger.warning(f"Would remove: {self.stats['matched']} task(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up task entries in summary status DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--force", action="store_true", help="Actually delete tasks (default: dry run)")
    parser.add_argument("--all", action="store_true", help="Delete ALL task entries (dangerous!)")
    parser.add_argument("--stale-time", type=float, default=None, help="Only delete tasks older than N seconds")
    parser.add_argument("--status", type=str, default=None, help="Filter by status (e.g. queued,running,failed)")
    parser.add_argument("--pid", type=str, default=None, help="Filter by pid")
    parser.add_argument("--model", type=str, default=None, help="Filter by model")
    parser.add_argument("--task-id", type=str, default=None, help="Filter by task id")
    parser.add_argument(
        "--keep-summary-status",
        action="store_true",
        help="Do not delete pid::model summary status when removing queued/running tasks",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger.remove()
    base_level = settings.log_level.upper()
    log_level = "DEBUG" if args.verbose else base_level
    logger.add(sys.stderr, level=log_level)

    if args.all:
        logger.warning("⚠️  --all will delete ALL task entries!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            logger.debug("Aborted")
            return

    status_filter = TaskCleaner._normalize_status_filter(args.status)

    cleaner = TaskCleaner()
    cleaner.clean_tasks(
        force=args.force,
        delete_all=args.all,
        stale_time=args.stale_time,
        status_filter=status_filter,
        pid_filter=args.pid,
        model_filter=args.model,
        task_id_filter=args.task_id,
        keep_summary_status=args.keep_summary_status,
    )


if __name__ == "__main__":
    main()
