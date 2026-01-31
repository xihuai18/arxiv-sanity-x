#!/usr/bin/env python3
"""
Task Cleanup Utility for Development

This script cleans up task entries in the summary status DB (task::*)
and revokes corresponding Huey tasks. Optionally flushes the entire Huey queue.

Usage:
    python cleanup_tasks.py                          # Dry run (shows what would be deleted)
    python cleanup_tasks.py --force                  # Actually delete matching tasks + revoke Huey tasks
    python cleanup_tasks.py --all                    # Delete ALL task entries (dangerous!)
    python cleanup_tasks.py --status running         # Only clean tasks with status=running
    python cleanup_tasks.py --pid 2601.14256         # Only clean tasks for a pid
    python cleanup_tasks.py --model glm-4.7          # Only clean tasks for a model
    python cleanup_tasks.py --task-id <id>           # Only clean a specific task id
    python cleanup_tasks.py --stale-time 3600        # Only clean tasks older than N seconds
    python cleanup_tasks.py --keep-summary-status    # Do not delete pid::model summary status
    python cleanup_tasks.py --no-revoke-huey         # Do not revoke Huey tasks when cleaning
    python cleanup_tasks.py --flush-huey             # Also flush entire Huey task queue (huey.db)
    python cleanup_tasks.py --all --force            # Complete cleanup (recommended after crashes)
"""

import argparse
import os
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
            "revoked": 0,
        }
        self._huey = None

    def _get_huey(self):
        """Lazy load Huey instance."""
        if self._huey is None:
            try:
                from tasks import huey

                self._huey = huey
            except Exception as e:
                logger.debug(f"Failed to load Huey: {e}")
        return self._huey

    def _revoke_huey_task(self, task_id: str, model: str) -> bool:
        """Revoke a Huey task by task_id.

        Args:
            task_id: The task ID to revoke
            model: Model name (used to construct dummy task)

        Returns:
            True if revoke was attempted, False otherwise
        """
        huey = self._get_huey()
        if huey is None:
            return False

        try:
            from tasks import generate_summary_task

            # Construct a dummy task with matching revoke_id
            dummy = generate_summary_task.s("_", model=model or "default", user=None)
            dummy.revoke_id = f"r:{task_id}"
            huey.revoke(dummy, revoke_once=True)
            return True
        except Exception as e:
            logger.debug(f"Failed to revoke Huey task {task_id}: {e}")
            return False

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
        revoke_huey: bool = True,
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

                # Revoke corresponding Huey task (best-effort)
                if revoke_huey and status in ("queued", "running"):
                    if self._revoke_huey_task(task_id, model):
                        self.stats["revoked"] += 1
                        logger.debug(f"Revoked Huey task: {task_id}")

                # Optionally clear summary status if it is queued/running
                if not keep_summary_status and pid and model and status in ("queued", "running"):
                    try:
                        SummaryStatusRepository.delete_status(pid, model)
                    except Exception:
                        pass
            else:
                revoke_hint = " (+ revoke Huey)" if revoke_huey and status in ("queued", "running") else ""
                logger.warning(f"Would remove: {label}{revoke_hint}")

        if force:
            revoke_msg = f", revoked {self.stats['revoked']}" if self.stats["revoked"] > 0 else ""
            logger.success(f"Removed tasks: {self.stats['removed']} / matched {self.stats['matched']}{revoke_msg}")
        else:
            logger.warning(f"Would remove: {self.stats['matched']} task(s)")

    def flush_huey_queue(self, force: bool = False) -> dict:
        """Flush all pending tasks from the Huey queue (huey.db).

        Args:
            force: If False, only show what would be flushed (dry run).

        Returns:
            dict with 'pending_count', 'flushed', 'huey_db_path'
        """
        huey_db_path = settings.huey.db_path or str(settings.data_dir / "huey.db")
        result = {
            "pending_count": 0,
            "flushed": False,
            "huey_db_path": huey_db_path,
        }

        if not os.path.exists(huey_db_path):
            logger.info(f"Huey database not found: {huey_db_path}")
            return result

        try:
            from tasks import huey

            # Get pending task count
            pending_count = huey.pending_count()
            result["pending_count"] = pending_count

            if pending_count == 0:
                logger.info("Huey queue is empty, nothing to flush")
                return result

            if force:
                # Flush all pending tasks
                huey.flush()
                result["flushed"] = True
                logger.success(f"Flushed {pending_count} pending task(s) from Huey queue")
            else:
                logger.warning(f"Would flush {pending_count} pending task(s) from Huey queue")

        except Exception as e:
            logger.error(f"Failed to flush Huey queue: {e}")

        return result


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
    parser.add_argument(
        "--no-revoke-huey",
        action="store_true",
        help="Do not revoke Huey tasks when cleaning (by default, queued/running tasks are revoked)",
    )
    parser.add_argument(
        "--flush-huey",
        action="store_true",
        help="Also flush entire Huey task queue (huey.db) to clear ALL pending tasks",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger.remove()
    base_level = settings.log_level.upper()
    log_level = "DEBUG" if args.verbose else base_level
    logger.add(sys.stderr, level=log_level)

    if args.all:
        logger.warning("⚠️  --all will delete ALL task entries!")
        if args.flush_huey:
            logger.warning("⚠️  --flush-huey will also flush ALL pending Huey tasks!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            logger.debug("Aborted")
            return
    elif args.flush_huey and args.force:
        logger.warning("⚠️  --flush-huey will flush ALL pending Huey tasks!")
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
        revoke_huey=not args.no_revoke_huey,
    )

    # Flush Huey queue if requested
    if args.flush_huey:
        cleaner.flush_huey_queue(force=args.force)


if __name__ == "__main__":
    main()
