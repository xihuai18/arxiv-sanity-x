import datetime
import shutil
import subprocess
import sys
from pathlib import Path

import holidays
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

# Get project root (parent of tools/ directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

PYTHON = sys.executable
TOOLS_DIR = PROJECT_ROOT / "tools"

logger.remove()
logger.add(sys.stdout, level=settings.log_level.upper())

# Read daemon settings from configuration
FETCH_NUM = settings.daemon.fetch_num
FETCH_MAX = settings.daemon.fetch_max
SUMMARY_NUM = settings.daemon.summary_num
SUMMARY_WORKERS = settings.daemon.summary_workers
ENABLE_SUMMARY = settings.daemon.enable_summary
ENABLE_EMBEDDINGS = settings.daemon.enable_embeddings
ENABLE_PRIORITY_QUEUE = settings.daemon.enable_priority_queue
ENABLE_SUMMARY_QUEUE = settings.daemon.enable_summary_queue
PRIORITY_DAYS = settings.daemon.priority_days
PRIORITY_LIMIT = settings.daemon.priority_limit


def _run_cmd(cmd, name: str) -> bool:
    logger.debug(f"{name}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{name} failed with code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"{name} failed: {e}")
        return False


def _get_email_time_delta() -> float:
    """
    Calculate the time_delta that will be used for email recommendations.
    Matches the logic in send_email() to ensure priority papers align with email content.
    """
    us_holidays = holidays.UnitedStates()
    now = datetime.datetime.now()
    weekday_int = now.weekday() + 1  # 1=Monday, 7=Sunday

    # Base: 4 days for Mon/Tue (covers weekend), 2 days otherwise
    time_delta = 4.0 if weekday_int in [1, 2] else 2.0

    # Add holiday duration if today is post-holiday (same logic as send_email)
    def is_post_holiday(date):
        return date not in us_holidays and (date - datetime.timedelta(days=1)) in us_holidays

    def count_holiday_duration(date):
        duration = 0
        current_date = date - datetime.timedelta(days=1)
        while current_date in us_holidays:
            duration += 1
            current_date -= datetime.timedelta(days=1)
        return duration

    if is_post_holiday(now):
        time_delta += count_holiday_duration(now)

    return time_delta


def gen_summary():
    if not ENABLE_SUMMARY:
        logger.debug("Summary generation disabled (set ARXIV_SANITY_DAEMON_ENABLE_SUMMARY=false)")
        return True

    cmd = [PYTHON, str(TOOLS_DIR / "batch_paper_summarizer.py"), "-n", str(SUMMARY_NUM), "-w", str(SUMMARY_WORKERS)]

    if ENABLE_SUMMARY_QUEUE:
        cmd.append("--queue")

    # Add priority queue arguments if enabled
    if ENABLE_PRIORITY_QUEUE:
        # Use dynamic time_delta matching email recommendations
        priority_days = _get_email_time_delta() if PRIORITY_DAYS == 2.0 else PRIORITY_DAYS
        cmd.extend(["--priority", "--priority-days", str(priority_days), "--priority-limit", str(PRIORITY_LIMIT)])

    return _run_cmd(cmd, "generate_summary")


def fetch_compute():
    logger.debug("Fetch and compute")
    fetch_ok = _run_cmd(
        [PYTHON, str(TOOLS_DIR / "arxiv_daemon.py"), "-n", str(FETCH_NUM), "-m", str(FETCH_MAX)], "fetch"
    )

    # arxiv_daemon returns exit code 1 if no new papers, skip compute and summary
    if not fetch_ok:
        logger.debug("No new papers fetched, skipping compute and summary")
        return

    compute_cmd = [PYTHON, str(TOOLS_DIR / "compute.py")]
    if ENABLE_EMBEDDINGS:
        compute_cmd.append("--use_embeddings")
    else:
        compute_cmd.append("--no-embeddings")
    _run_cmd(compute_cmd, "compute")

    gen_summary()


def send_email():
    us_holidays = holidays.UnitedStates()

    def is_post_holiday(date):
        # Check if today is the first day after a holiday
        if date not in us_holidays and (date - datetime.timedelta(days=1)) in us_holidays:
            return True
        return False

    def count_holiday_duration(date):
        # Calculate the duration of the holiday
        duration = 0
        current_date = date - datetime.timedelta(days=1)
        while current_date in us_holidays:
            duration += 1
            current_date -= datetime.timedelta(days=1)
        return duration

    now = datetime.datetime.now()
    weekday_int = now.weekday() + 1
    logger.debug("Send emails")

    time_param = "2" if weekday_int not in [1, 2] else "4"
    if is_post_holiday(now):
        holiday_duration = count_holiday_duration(now)
        time_param = str(float(time_param) + holiday_duration)

    subprocess.call(
        [
            PYTHON,
            str(TOOLS_DIR / "send_emails.py"),
            "-t",
            time_param,
            *(["--dry-run"] if settings.daemon.email_dry_run else []),
        ]
    )


def backup_user_data():
    logger.debug("Backup user data to submodule")

    if not settings.daemon.enable_git_backup:
        logger.debug("Git backup disabled; set ARXIV_SANITY_DAEMON_ENABLE_GIT_BACKUP=true to enable")
        return

    # Source: data/dict.db (actual file used by the application)
    # Destination: data-repo/dict.db (submodule for backup)
    src_file = PROJECT_ROOT / "data" / "dict.db"
    data_repo_dir = PROJECT_ROOT / "data-repo"
    dst_file = data_repo_dir / "dict.db"

    if not src_file.is_file():
        logger.warning(f"Source dict.db not found: {src_file}")
        return

    if not data_repo_dir.is_dir():
        logger.warning(f"Submodule directory not found: {data_repo_dir}")
        return

    # Copy dict.db to submodule
    try:
        shutil.copy2(src_file, dst_file)
    except Exception as e:
        logger.warning(f"Failed to copy dict.db to submodule: {e}")
        return

    # Stage dict.db in submodule
    add_result = subprocess.run(["git", "add", "dict.db"], cwd=data_repo_dir, capture_output=True, text=True)
    if add_result.returncode != 0:
        logger.warning(f"git add failed in submodule: {add_result.stderr.strip()}")
        return

    # Check if there are staged changes
    diff_result = subprocess.run(["git", "diff", "--cached", "--quiet", "--", "dict.db"], cwd=data_repo_dir)
    if diff_result.returncode == 0:
        logger.debug("No changes to back up")
        return
    if diff_result.returncode not in (0, 1):
        logger.warning("git diff failed while checking staged changes in submodule")
        return

    # Commit in submodule
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_result = subprocess.run(
        ["git", "commit", "-m", f"backup dict.db ({timestamp})"],
        cwd=data_repo_dir,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        logger.warning(f"git commit failed in submodule: {commit_result.stderr.strip()}")
        return

    # Push submodule
    push_result = subprocess.run(["git", "push"], cwd=data_repo_dir, capture_output=True, text=True)
    if push_result.returncode != 0:
        logger.warning(f"git push failed in submodule: {push_result.stderr.strip()}")


def create_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=settings.daemon.timezone)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=8)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=12)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=16)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=20)
    scheduler.add_job(send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
    scheduler.add_job(backup_user_data, "cron", hour=20)
    return scheduler


def _log_startup_info():
    """Print daemon configuration and schedule on startup (always visible)."""
    lines = [
        "=" * 60,
        "Arxiv-Sanity Daemon Starting",
        "=" * 60,
        "",
        "Configuration:",
        f"  Timezone: {settings.daemon.timezone}",
        f"  Log level: {settings.log_level.upper()}",
        "",
        "Fetch Settings:",
        f"  Papers per fetch: {FETCH_NUM}",
        f"  Max per API call: {FETCH_MAX}",
        "",
        "Summary Settings:",
        f"  Summary generation: {'enabled' if ENABLE_SUMMARY else 'disabled'}",
    ]

    if ENABLE_SUMMARY:
        lines.extend(
            [
                f"  Papers per batch: {SUMMARY_NUM}",
                f"  Workers: {SUMMARY_WORKERS}",
                f"  Summary queue: {'enabled' if ENABLE_SUMMARY_QUEUE else 'disabled'}",
                f"  Priority queue: {'enabled' if ENABLE_PRIORITY_QUEUE else 'disabled'}",
            ]
        )
        if ENABLE_PRIORITY_QUEUE:
            if PRIORITY_DAYS == 2.0:
                # Show dynamic rule explanation
                current_days = _get_email_time_delta()
                lines.extend(
                    [
                        f"    Priority days: {PRIORITY_DAYS} (dynamic mode)",
                        f"      Current value: {current_days} days",
                        "      Rule: Mon/Tue=4d, Wed-Fri=2d, +holidays",
                    ]
                )
            else:
                lines.append(f"    Priority days: {PRIORITY_DAYS} (fixed)")
            lines.append(f"    Priority limit: {PRIORITY_LIMIT}")

    lines.extend(
        [
            "",
            "Feature Flags:",
            f"  Embeddings: {'enabled' if ENABLE_EMBEDDINGS else 'disabled'}",
            f"  Email dry-run: {'yes' if settings.daemon.email_dry_run else 'no'}",
            f"  Git backup: {'enabled' if settings.daemon.enable_git_backup else 'disabled'}",
            "",
            "Scheduled Tasks:",
            "  fetch_compute:    Mon-Fri at 08:00, 12:00, 16:00, 20:00",
            "  send_email:       Mon-Fri at 18:00",
            "  backup_user_data: Daily at 20:00",
            "",
            "=" * 60,
            "Daemon is now running. Press Ctrl+C to stop.",
            "=" * 60,
        ]
    )

    print("\n".join(lines), flush=True)


def main(argv: list[str] | None = None) -> int:
    # argv is reserved for future flags; keep signature stable for reuse.
    _ = argv
    _log_startup_info()
    scheduler = create_scheduler()
    scheduler.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
