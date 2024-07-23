import datetime
import os
import subprocess

import holidays
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger


def fetch_compute():
    logger.info("featch and compute")
    subprocess.call(["python", "arxiv_daemon.py", "-n", "2000", "-m", "1000"])
    subprocess.call(["python", "compute.py"])


def send_email():
    us_holidays = holidays.UnitedStates()

    def is_post_holiday(date):
        # 检查今天是否是节假日后的第一天
        if date not in us_holidays and (date - datetime.timedelta(days=1)) in us_holidays:
            return True
        return False

    def count_holiday_duration(date):
        # 计算节假日持续的天数
        duration = 0
        current_date = date - datetime.timedelta(days=1)
        while current_date in us_holidays:
            duration += 1
            current_date -= datetime.timedelta(days=1)
        return duration

    now = datetime.datetime.now()
    weekday_int = now.weekday() + 1
    logger.info("send emails")

    time_param = "1.5" if weekday_int not in [1, 2] else "3.5"
    if is_post_holiday(now):
        holiday_duration = count_holiday_duration(now)
        time_param = str(float(time_param) + holiday_duration)

    subprocess.call(
        [
            "python",
            "send_emails.py",
            "-t",
            time_param,
        ]
    )


def backup_user_data():
    logger.info("backup user data")
    import shutil

    from vars import DATA_DIR

    DICT_DB_FILE = os.path.join(DATA_DIR, "dict.db")
    BACKUP_DIR = "./data"
    BACKUP_FILE = os.path.join(BACKUP_DIR, "dict.db")

    # Ensure the backup directory exists
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Copy the database file to the backup directory
    shutil.copy2(DICT_DB_FILE, BACKUP_FILE)

    # Now we'll use git to commit and push the changes to the backup directory
    subprocess.call(["git", "add", "."])
    subprocess.call(["git", "commit", "-am", '"backup"'])
    subprocess.call(["git", "push"])


scheduler = BlockingScheduler(timezone="Asia/Shanghai")
scheduler.add_job(fetch_compute, "cron", day_of_week="tue,wed,thu,fri,mon", hour=17)
scheduler.add_job(send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
scheduler.add_job(backup_user_data, "cron", hour=19)
scheduler.start()
