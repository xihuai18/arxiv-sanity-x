import datetime
import os
import subprocess
import sys

import holidays
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

PYTHON = sys.executable


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


FETCH_NUM = _env_int("ARXIV_SANITY_FETCH_NUM", 1000)
FETCH_MAX = _env_int("ARXIV_SANITY_FETCH_MAX", 200)
SUMMARY_NUM = _env_int("ARXIV_SANITY_SUMMARY_NUM", 100)
SUMMARY_WORKERS = _env_int("ARXIV_SANITY_SUMMARY_WORKERS", 2)
ENABLE_SUMMARY = _truthy_env("ARXIV_SANITY_DAEMON_SUMMARY", "1")
ENABLE_EMBEDDINGS = _truthy_env("ARXIV_SANITY_DAEMON_EMBEDDINGS", "1")


def _run_cmd(cmd, name: str) -> bool:
    logger.info(f"{name}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{name} failed with code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"{name} failed: {e}")
        return False


def gen_summary():
    if not ENABLE_SUMMARY:
        logger.info("Summary generation disabled (ARXIV_SANITY_DAEMON_SUMMARY=0)")
        return True
    return _run_cmd(
        [PYTHON, "batch_paper_summarizer.py", "-n", str(SUMMARY_NUM), "-w", str(SUMMARY_WORKERS)],
        "generate_summary",
    )


def fetch_compute():
    logger.info("Fetch and compute")
    if not _run_cmd([PYTHON, "arxiv_daemon.py", "-n", str(FETCH_NUM), "-m", str(FETCH_MAX)], "fetch"):
        return

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
    logger.info("Send emails")

    time_param = "2" if weekday_int not in [1, 2] else "5"
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
    logger.info("Backup user data")

    enable_git = os.environ.get("ARXIV_SANITY_ENABLE_GIT_BACKUP", "1").lower() in ("1", "true", "yes")
    if not enable_git:
        logger.info("Git backup disabled; set ARXIV_SANITY_ENABLE_GIT_BACKUP=1 to enable")
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
        logger.info("No changes to back up")
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
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=9)
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=13)
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=17)
scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=21)
scheduler.add_job(send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
scheduler.add_job(backup_user_data, "cron", hour=20)
scheduler.start()
