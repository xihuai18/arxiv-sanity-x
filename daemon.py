import datetime
import os
import subprocess
import sys

import holidays
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

PYTHON = sys.executable

logger.remove()
logger.add(sys.stdout, level=os.environ.get("ARXIV_SANITY_LOG_LEVEL", "WARNING").upper())


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


FETCH_NUM = _env_int("ARXIV_SANITY_FETCH_NUM", 2000)
FETCH_MAX = _env_int("ARXIV_SANITY_FETCH_MAX", 1000)
SUMMARY_NUM = _env_int("ARXIV_SANITY_SUMMARY_NUM", 200)
SUMMARY_WORKERS = _env_int("ARXIV_SANITY_SUMMARY_WORKERS", 2)
ENABLE_SUMMARY = _truthy_env("ARXIV_SANITY_DAEMON_SUMMARY", "1")
ENABLE_EMBEDDINGS = _truthy_env("ARXIV_SANITY_DAEMON_EMBEDDINGS", "1")
ENABLE_PRIORITY_QUEUE = _truthy_env("ARXIV_SANITY_PRIORITY_QUEUE", "1")
PRIORITY_DAYS = float(os.environ.get("ARXIV_SANITY_PRIORITY_DAYS", "2"))
PRIORITY_LIMIT = _env_int("ARXIV_SANITY_PRIORITY_LIMIT", 100)


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
        logger.debug("Summary generation disabled (ARXIV_SANITY_DAEMON_SUMMARY=0)")
        return True

    cmd = [PYTHON, "batch_paper_summarizer.py", "-n", str(SUMMARY_NUM), "-w", str(SUMMARY_WORKERS)]

    # Add priority queue arguments if enabled
    if ENABLE_PRIORITY_QUEUE:
        # Use dynamic time_delta matching email recommendations
        priority_days = _get_email_time_delta() if PRIORITY_DAYS == 2.0 else PRIORITY_DAYS
        cmd.extend(["--priority", "--priority-days", str(priority_days), "--priority-limit", str(PRIORITY_LIMIT)])

    return _run_cmd(cmd, "generate_summary")


def fetch_compute():
    logger.debug("Fetch and compute")
    _run_cmd([PYTHON, "arxiv_daemon.py", "-n", str(FETCH_NUM), "-m", str(FETCH_MAX)], "fetch")

    compute_cmd = [PYTHON, "compute.py"]
    if ENABLE_EMBEDDINGS:
        compute_cmd.append("--use_embeddings")
    if not _run_cmd(compute_cmd, "compute"):
        return

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
            "send_emails.py",
            "-t",
            time_param,
        ]
    )


def backup_user_data():
    logger.debug("Backup user data")

    enable_git = os.environ.get("ARXIV_SANITY_ENABLE_GIT_BACKUP", "1").lower() in ("1", "true", "yes")
    if not enable_git:
        logger.debug("Git backup disabled; set ARXIV_SANITY_ENABLE_GIT_BACKUP=1 to enable")
        return

    try:
        from aslite.db import DICT_DB_FILE
    except Exception as e:
        logger.warning(f"Failed to resolve dict.db path: {e}")
        return

    if not os.path.isfile(DICT_DB_FILE):
        logger.warning(f"dict.db not found: {DICT_DB_FILE}")
        return

    add_result = subprocess.run(["git", "add", DICT_DB_FILE], capture_output=True, text=True)
    if add_result.returncode != 0:
        logger.warning(f"git add failed: {add_result.stderr.strip()}")
        return

    diff_result = subprocess.run(["git", "diff", "--cached", "--quiet", "--", DICT_DB_FILE])
    if diff_result.returncode == 0:
        logger.debug("No changes to back up")
        return
    if diff_result.returncode not in (0, 1):
        logger.warning("git diff failed while checking staged changes")
        return

    commit_result = subprocess.run(["git", "commit", "-m", "backup dict.db"], capture_output=True, text=True)
    if commit_result.returncode != 0:
        logger.warning(f"git commit failed: {commit_result.stderr.strip()}")
        return

    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    if push_result.returncode != 0:
        logger.warning(f"git push failed: {push_result.stderr.strip()}")


scheduler = BlockingScheduler(timezone="Asia/Shanghai")
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=8)
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=12)
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=16)
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=20)
scheduler.add_job(send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
scheduler.add_job(backup_user_data, "cron", hour=20)
scheduler.start()
