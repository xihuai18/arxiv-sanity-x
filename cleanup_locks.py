#!/usr/bin/env python3
"""
Lock Cleanup Utility for Development

This script cleans up orphaned lock files left by crashed or killed processes.
It checks:
1. MinerU parsing locks (.{paper_id}.lock)
2. GPU slot locks (slot_*.lock)
3. Summary cache locks (.{model}.lock)

Usage:
    python cleanup_locks.py                    # Dry run (shows what would be deleted)
    python cleanup_locks.py --force            # Actually delete orphan locks
    python cleanup_locks.py --all              # Delete all locks (dangerous!)
    python cleanup_locks.py --stale-time 300   # Custom stale time (seconds)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from loguru import logger

# Import configuration
try:
    from vars import DATA_DIR, SUMMARY_DIR
except ImportError:
    logger.error("Failed to import vars.py, using default paths")
    DATA_DIR = "data"
    SUMMARY_DIR = "data/summary"


class LockCleaner:
    def __init__(self, data_dir: str = DATA_DIR, summary_dir: str = SUMMARY_DIR):
        self.data_dir = Path(data_dir)
        self.summary_dir = Path(summary_dir)
        self.mineru_dir = self.data_dir / "mineru"
        self.stats = {
            "total": 0,
            "orphan": 0,
            "stale": 0,
            "alive": 0,
            "removed": 0,
        }

    @staticmethod
    def is_process_alive(pid: int) -> bool:
        """Check if a process with given PID is alive."""
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            return True
        except OSError:
            return False

    @staticmethod
    def read_lock_pid(lock_path: Path) -> Tuple[int | None, float | None]:
        """
        Read PID and timestamp from lock file.

        Returns:
            Tuple of (pid, timestamp). Both can be None if unable to read.
        """
        try:
            content = lock_path.read_text().strip()
            lines = content.split("\n")
            pid = int(lines[0]) if lines and lines[0] else None
            timestamp = float(lines[1]) if len(lines) > 1 and lines[1] else None
            return pid, timestamp
        except (ValueError, IndexError, OSError):
            return None, None

    def is_lock_orphan(self, lock_path: Path) -> Tuple[bool, str]:
        """
        Check if lock file is orphaned (owner process dead).

        Returns:
            Tuple of (is_orphan, reason)
        """
        pid, timestamp = self.read_lock_pid(lock_path)

        if pid is None:
            # No PID in lock file, can't determine
            return False, "no_pid_info"

        if not self.is_process_alive(pid):
            age = time.time() - lock_path.stat().st_mtime
            return True, f"dead_pid_{pid}_age_{age:.0f}s"

        return False, f"alive_pid_{pid}"

    def is_lock_stale(self, lock_path: Path, stale_time: float) -> Tuple[bool, str]:
        """
        Check if lock file is stale (too old).

        Returns:
            Tuple of (is_stale, reason)
        """
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_time:
                return True, f"stale_{age:.0f}s"
            return False, f"fresh_{age:.0f}s"
        except Exception as e:
            return False, f"check_failed_{e}"

    def find_mineru_locks(self) -> List[Path]:
        """Find all MinerU parsing lock files."""
        if not self.mineru_dir.exists():
            return []

        locks = []
        # Pattern: .{paper_id}.lock
        for item in self.mineru_dir.iterdir():
            if item.is_file() and item.name.startswith(".") and item.name.endswith(".lock"):
                # Exclude GPU slot locks
                if item.name.startswith(".gpu_slots"):
                    continue
                locks.append(item)

        return locks

    def find_gpu_slot_locks(self) -> List[Path]:
        """Find all GPU slot lock files."""
        gpu_slots_dir = self.mineru_dir / ".gpu_slots"
        if not gpu_slots_dir.exists():
            return []

        locks = []
        # Pattern: slot_*.lock
        for item in gpu_slots_dir.iterdir():
            if item.is_file() and item.name.startswith("slot_") and item.name.endswith(".lock"):
                locks.append(item)

        return locks

    def find_summary_locks(self) -> List[Path]:
        """Find all summary cache lock files."""
        if not self.summary_dir.exists():
            return []

        locks = []
        # Pattern: .{model}.lock in subdirectories and root
        for root, dirs, files in os.walk(self.summary_dir):
            root_path = Path(root)
            for file in files:
                if file.startswith(".") and file.endswith(".lock"):
                    locks.append(root_path / file)

        return locks

    def scan_all_locks(self) -> List[Path]:
        """Scan and return all lock files."""
        locks = []
        locks.extend(self.find_mineru_locks())
        locks.extend(self.find_gpu_slot_locks())
        locks.extend(self.find_summary_locks())
        return locks

    def clean_locks(self, force: bool = False, delete_all: bool = False, stale_time: float = 600) -> None:
        """
        Clean lock files.

        Args:
            force: Actually delete files (dry run if False)
            delete_all: Delete all locks regardless of status
            stale_time: Time in seconds after which a lock is considered stale
        """
        locks = self.scan_all_locks()
        self.stats["total"] = len(locks)

        if not locks:
            logger.debug("No lock files found")
            return

        logger.debug(f"Found {len(locks)} lock file(s)")

        for lock_path in locks:
            relative_path = lock_path.relative_to(Path.cwd()) if lock_path.is_relative_to(Path.cwd()) else lock_path

            if delete_all:
                # Delete all locks
                self.stats["removed"] += 1
                if force:
                    try:
                        lock_path.unlink()
                        logger.success(f"Removed: {relative_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove {relative_path}: {e}")
                else:
                    logger.warning(f"Would remove: {relative_path}")
                continue

            # Check if orphan
            is_orphan, orphan_reason = self.is_lock_orphan(lock_path)
            if is_orphan:
                self.stats["orphan"] += 1
                if force:
                    try:
                        lock_path.unlink()
                        logger.success(f"Removed orphan lock: {relative_path} ({orphan_reason})")
                        self.stats["removed"] += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {relative_path}: {e}")
                else:
                    logger.warning(f"Orphan lock: {relative_path} ({orphan_reason})")
                continue

            # Check if stale
            is_stale, stale_reason = self.is_lock_stale(lock_path, stale_time)
            if is_stale:
                self.stats["stale"] += 1
                if force:
                    try:
                        lock_path.unlink()
                        logger.success(f"Removed stale lock: {relative_path} ({stale_reason})")
                        self.stats["removed"] += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {relative_path}: {e}")
                else:
                    logger.warning(f"Stale lock: {relative_path} ({stale_reason})")
                continue

            # Lock is alive and fresh
            self.stats["alive"] += 1
            logger.debug(f"Active lock: {relative_path} ({orphan_reason})")

        # Print summary
        logger.debug("=" * 60)
        logger.debug(f"Total locks: {self.stats['total']}")
        logger.debug(f"Orphan locks (dead process): {self.stats['orphan']}")
        logger.debug(f"Stale locks (too old): {self.stats['stale']}")
        logger.debug(f"Active locks: {self.stats['alive']}")
        if force:
            logger.success(f"Removed locks: {self.stats['removed']}")
        else:
            logger.warning(f"Would remove: {self.stats['orphan'] + self.stats['stale']} lock(s)")
            logger.warning("Run with --force to actually delete locks")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up orphaned lock files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete lock files (default: dry run)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL locks (dangerous! Use with caution)",
    )
    parser.add_argument(
        "--stale-time",
        type=float,
        default=600,
        help="Time in seconds after which a lock is considered stale (default: 600)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    base_level = os.environ.get("ARXIV_SANITY_LOG_LEVEL", "WARNING").upper()
    log_level = "DEBUG" if args.verbose else base_level
    logger.add(sys.stderr, level=log_level)

    if args.all:
        logger.warning("⚠️  --all flag will delete ALL locks, including active ones!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            logger.debug("Aborted")
            return

    cleaner = LockCleaner()
    cleaner.clean_locks(force=args.force, delete_all=args.all, stale_time=args.stale_time)


if __name__ == "__main__":
    main()
