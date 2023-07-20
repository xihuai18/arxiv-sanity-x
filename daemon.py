from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess


def fetch_compute():
    subprocess.call(["python", "arxiv_daemon.py", "-n", "2000", "-m", "1000"])
    subprocess.call(["python", "compute.py"])


def send_email():
    subprocess.call(["python", "send_emails.py", "-t", "1.5"])


scheduler = BlockingScheduler(timezone="Asia/Shanghai")
scheduler.add_job(fetch_compute, "cron", day_of_week="tue,wed,thu,fri,mon", hour=14)
scheduler.add_job(
    send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=15, minute=30
)
scheduler.start()
