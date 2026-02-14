import datetime
import os
import re
import shutil

# SQLite is used for user state (dict.db). Use sqlite3 backup API to snapshot safely.
import sqlite3
import subprocess
import subprocess as _real_subprocess
import sys
import time
from pathlib import Path

import holidays
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

# Get project root (parent of tools/ directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from config.sentry import initialize_sentry

PYTHON = sys.executable
TOOLS_DIR = PROJECT_ROOT / "tools"

logger.remove()
logger.add(sys.stdout, level=settings.log_level.upper())

# Optional Sentry error reporting (no-op unless configured).
initialize_sentry()

# Guardrail: avoid daemon getting stuck forever on a hung child process.
SUBPROCESS_TIMEOUT_S = int(getattr(settings.daemon, "subprocess_timeout_s", 7200) or 7200)

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

_LAST_RUN_REASON: dict[str, str] = {}
_LAST_RUN_OUTPUT: dict[str, str] = {}


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file (best-effort)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sqlite_snapshot_copy(src_file: Path, dst_file: Path) -> bool:
    """Create a consistent snapshot copy of an SQLite database file.

    Writes to a temp file then atomically replaces dst_file.
    Falls back to shutil.copy2 on any failure.
    """

    try:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Do not fail early; fallback copy might still work.
        pass

    tmp_file = dst_file.with_name(dst_file.name + ".tmp")

    try:
        # Use SQLite backup API for consistency even while the DB is being written.
        # Use read-only URI to avoid creating new DB files accidentally.
        src_uri = f"file:{src_file.as_posix()}?mode=ro"
        src_conn = sqlite3.connect(src_uri, uri=True, timeout=30)
        try:
            tmp_file.unlink(missing_ok=True)  # py3.11+; guarded by try
        except Exception:
            pass
        dst_conn = sqlite3.connect(str(tmp_file), timeout=30)
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            try:
                dst_conn.close()
            except Exception:
                pass
            try:
                src_conn.close()
            except Exception:
                pass

        # If snapshot matches existing dst_file, skip replacing to avoid noisy commits.
        try:
            if dst_file.exists() and _sha256_file(tmp_file) == _sha256_file(dst_file):
                return False
        except Exception:
            pass

        # Ensure data is on disk before replace (best-effort).
        try:
            fd = os.open(str(tmp_file), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass

        os.replace(tmp_file, dst_file)
        return True
    except Exception as e:
        logger.warning(f"SQLite snapshot copy failed; falling back to raw copy: {e}")
        try:
            shutil.copy2(src_file, tmp_file)
            try:
                if dst_file.exists() and _sha256_file(tmp_file) == _sha256_file(dst_file):
                    return False
            except Exception:
                pass
            os.replace(tmp_file, dst_file)
            return True
        except Exception as e2:
            logger.warning(f"Failed to copy dict.db: {e2}")
            return False
    finally:
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except Exception:
            pass


def _is_git_repo(repo_dir: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        return r.returncode == 0 and (r.stdout or "").strip() == "true"
    except Exception:
        return False


def _ensure_git_identity(repo_dir: Path) -> None:
    """Ensure git user.name/user.email exist (local repo config)."""
    name = str(getattr(settings.daemon, "backup_git_user_name", "") or "").strip()
    email = str(getattr(settings.daemon, "backup_git_user_email", "") or "").strip()
    if not name:
        name = "arxiv-sanity-daemon"
    if not email:
        email = "daemon@localhost"

    def _get(key: str) -> str:
        try:
            r = subprocess.run(
                ["git", "config", "--get", key],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_S,
            )
            if r.returncode != 0:
                return ""
            return (r.stdout or "").strip()
        except Exception:
            return ""

    if not _get("user.name"):
        subprocess.run(
            ["git", "config", "user.name", name],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    if not _get("user.email"):
        subprocess.run(
            ["git", "config", "user.email", email],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )


def _git_current_branch(repo_dir: Path) -> str:
    """Return current branch name, or 'HEAD' when detached."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _ensure_on_branch(repo_dir: Path, branch: str) -> bool:
    """Ensure repo is attached to a branch (best-effort)."""
    b = (branch or "").strip()
    if not b:
        return True
    try:
        r = subprocess.run(
            ["git", "checkout", "-B", b],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        if r.returncode != 0:
            logger.warning(f"git checkout -B {b} failed in backup repo: {(r.stderr or '').strip()}")
            return False
        return True
    except _real_subprocess.TimeoutExpired:
        logger.warning(f"[pipeline] backup: git checkout timed out after {SUBPROCESS_TIMEOUT_S}s")
        return False
    except Exception as e:
        logger.warning(f"git checkout failed in backup repo: {e}")
        return False


def _run_cmd(cmd: list[str], name: str, fail_level: str = "error", capture: bool = False) -> bool:
    logger.info(f"[pipeline] {name}: starting")
    logger.debug(f"{name}: {' '.join(cmd)}")
    log_fn = getattr(logger, fail_level, logger.error)
    _LAST_RUN_OUTPUT.pop(name, None)
    try:
        t0 = time.time()
        kw: dict = dict(check=True, timeout=SUBPROCESS_TIMEOUT_S)
        if capture:
            kw.update(capture_output=True, text=True)
        result = subprocess.run(cmd, **kw)
        elapsed = time.time() - t0
        if capture and result.stdout:
            _LAST_RUN_OUTPUT[name] = result.stdout
        logger.info(f"[pipeline] {name}: ok ({elapsed:.1f}s)")
        _LAST_RUN_REASON[name] = "ok"
        return True
    except _real_subprocess.CalledProcessError as e:
        if capture and getattr(e, "stdout", None):
            _LAST_RUN_OUTPUT[name] = e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
        # arxiv_daemon returns exit code 1 if no new papers (not a real error).
        if name == "fetch" and e.returncode == 1:
            logger.info("[pipeline] fetch: no new papers (exit 1)")
            _LAST_RUN_REASON[name] = "no_new_papers"
            return False
        log_fn(f"[pipeline] {name}: failed (exit {e.returncode})")
        _LAST_RUN_REASON[name] = f"exit_{int(e.returncode)}"
        return False
    except _real_subprocess.TimeoutExpired as e:
        if capture and getattr(e, "stdout", None):
            _LAST_RUN_OUTPUT[name] = e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
        log_fn(f"[pipeline] {name}: timed out after {SUBPROCESS_TIMEOUT_S}s")
        _LAST_RUN_REASON[name] = "timeout"
        return False
    except Exception as e:
        log_fn(f"[pipeline] {name}: failed: {e}")
        _LAST_RUN_REASON[name] = "error"
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
    logger.info("[pipeline] === fetch_compute started ===")
    fetch_ok = _run_cmd(
        [PYTHON, str(TOOLS_DIR / "arxiv_daemon.py"), "-n", str(FETCH_NUM), "-m", str(FETCH_MAX)],
        "fetch",
        capture=True,
    )

    # Log paper stats parsed from captured stdout
    fetch_output = _LAST_RUN_OUTPUT.get("fetch", "")
    if fetch_output:
        total_new = total_replaced = 0
        for line in fetch_output.splitlines():
            m = re.match(r"\s*\S+:\s*\+(\d+)\s+new,\s*~(\d+)\s+replaced", line)
            if m:
                total_new += int(m.group(1))
                total_replaced += int(m.group(2))
        if total_new or total_replaced:
            logger.info(f"[pipeline] fetch: +{total_new} new, ~{total_replaced} replaced")

    if not fetch_ok:
        reason = _LAST_RUN_REASON.get("fetch") or "unknown"
        if reason != "no_new_papers":
            logger.warning(f"[pipeline] Fetch failed ({reason}); skipping compute and summary")
        return

    compute_cmd = [PYTHON, str(TOOLS_DIR / "compute.py")]
    if ENABLE_EMBEDDINGS:
        compute_cmd.append("--use_embeddings")
    else:
        compute_cmd.append("--no-embeddings")
    _run_cmd(compute_cmd, "compute")

    gen_summary()
    logger.info("[pipeline] === fetch_compute completed ===")


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
    logger.info("[pipeline] send_email started")

    time_param = "2" if weekday_int not in [1, 2] else "4"
    if is_post_holiday(now):
        holiday_duration = count_holiday_duration(now)
        time_param = str(float(time_param) + holiday_duration)

    t0 = time.time()
    try:
        rc = subprocess.call(
            [
                PYTHON,
                str(TOOLS_DIR / "send_emails.py"),
                "-t",
                time_param,
                *(["--dry-run"] if settings.daemon.email_dry_run else []),
            ],
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        if rc not in (0, 1):
            logger.warning(f"[pipeline] send_email: returned code {rc}")
    except _real_subprocess.TimeoutExpired:
        logger.warning(f"[pipeline] send_email: timed out after {SUBPROCESS_TIMEOUT_S}s")
    logger.info(f"[pipeline] send_email: done ({time.time() - t0:.1f}s)")


def backup_user_data():
    logger.info("[pipeline] backup_user_data started")

    if not settings.daemon.enable_git_backup:
        logger.debug("Git backup disabled; set ARXIV_SANITY_DAEMON_ENABLE_GIT_BACKUP=true to enable")
        return

    # Source: <data_dir>/dict.db (actual file used by the application)
    # Destination: <backup_repo_dir>/dict.db (git repo for backup; can be a submodule or standalone repo)
    try:
        src_file = Path(settings.data_dir) / "dict.db"
    except Exception:
        src_file = PROJECT_ROOT / "data" / "dict.db"
    backup_repo_dir_name = str(getattr(settings.daemon, "backup_repo_dir", "") or "data-repo").strip() or "data-repo"
    data_repo_dir = PROJECT_ROOT / backup_repo_dir_name
    dst_file = data_repo_dir / "dict.db"

    # NOTE: Use os.path.isfile so unit tests can patch tools.daemon.os reliably.
    if not os.path.isfile(src_file):
        logger.warning(f"Source dict.db not found: {src_file}")
        return

    if not data_repo_dir.is_dir():
        logger.warning(f"Backup repo directory not found: {data_repo_dir}")
        return

    if not _is_git_repo(data_repo_dir):
        logger.warning(f"Backup repo is not a git repository: {data_repo_dir}")
        return

    # Snapshot copy dict.db to backup repo
    copied = _sqlite_snapshot_copy(src_file, dst_file)
    if not copied:
        logger.debug("No changes to back up (dict.db unchanged)")
        return

    # Stage dict.db in submodule
    t0 = time.time()
    try:
        _ensure_git_identity(data_repo_dir)
        # Submodules are often checked out in detached HEAD state. If a push branch
        # is configured, attach HEAD to it to avoid "not currently on a branch".
        configured_branch = str(getattr(settings.daemon, "backup_push_branch", "") or "").strip()
        if configured_branch:
            cur = _git_current_branch(data_repo_dir)
            if cur == "HEAD":
                _ensure_on_branch(data_repo_dir, configured_branch)
        add_result = subprocess.run(
            ["git", "add", "dict.db"],
            cwd=data_repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except _real_subprocess.TimeoutExpired:
        logger.warning(f"[pipeline] backup: git add timed out after {SUBPROCESS_TIMEOUT_S}s")
        return
    logger.debug(f"[pipeline] backup: git add ({time.time() - t0:.1f}s)")
    if add_result.returncode != 0:
        logger.warning(f"git add failed in submodule: {add_result.stderr.strip()}")
        return

    # Check if there are staged changes
    t0 = time.time()
    try:
        diff_result = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", "dict.db"],
            cwd=data_repo_dir,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except _real_subprocess.TimeoutExpired:
        logger.warning(f"[pipeline] backup: git diff timed out after {SUBPROCESS_TIMEOUT_S}s")
        return
    logger.debug(f"[pipeline] backup: git diff ({time.time() - t0:.1f}s)")
    if diff_result.returncode == 0:
        logger.debug("No changes to back up")
        return
    if diff_result.returncode not in (0, 1):
        logger.warning("git diff failed while checking staged changes in submodule")
        return

    # Commit in submodule
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    t0 = time.time()
    try:
        commit_result = subprocess.run(
            ["git", "commit", "-m", f"backup dict.db ({timestamp})"],
            cwd=data_repo_dir,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except _real_subprocess.TimeoutExpired:
        logger.warning(f"[pipeline] backup: git commit timed out after {SUBPROCESS_TIMEOUT_S}s")
        return
    logger.debug(f"[pipeline] backup: git commit ({time.time() - t0:.1f}s)")
    if commit_result.returncode != 0:
        logger.warning(f"git commit failed in submodule: {commit_result.stderr.strip()}")
        return

    # Push backup commits (optional)
    if not bool(getattr(settings.daemon, "backup_push", True)):
        logger.info("[pipeline] backup: push disabled (backup_push=false)")
        return

    remote = str(getattr(settings.daemon, "backup_push_remote", "") or "").strip()
    branch = str(getattr(settings.daemon, "backup_push_branch", "") or "").strip()
    retries = int(getattr(settings.daemon, "backup_push_retries", 3) or 3)
    retries = max(0, retries)

    # Default: respect upstream configured in the backup repo.
    # For detached HEAD (common in submodules), pushing needs an explicit refspec.
    push_cmd = ["git", "push"]
    if remote and branch:
        push_cmd = ["git", "push", remote, f"HEAD:{branch}"]
    elif remote:
        push_cmd = ["git", "push", remote]
    elif branch:
        push_cmd = ["git", "push", "origin", f"HEAD:{branch}"]

    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            push_result = subprocess.run(
                push_cmd,
                cwd=data_repo_dir,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_S,
            )
        except _real_subprocess.TimeoutExpired:
            logger.warning(f"[pipeline] backup: git push timed out after {SUBPROCESS_TIMEOUT_S}s")
            return
        logger.debug(f"[pipeline] backup: git push ({time.time() - t0:.1f}s) attempt={attempt+1}")
        if push_result.returncode == 0:
            return
        err = (push_result.stderr or push_result.stdout or "").strip()
        logger.warning(f"git push failed in backup repo: {err}")
        if attempt < retries:
            sleep_s = min(60.0, 2.0**attempt)
            time.sleep(sleep_s)
        else:
            logger.warning("[pipeline] backup: giving up on push; will retry next schedule")


def cleanup_task_records():
    """Cleanup old terminal task records to keep summary_status DB small."""
    try:
        from tasks import cleanup_tasks

        # Keep active tasks; delete old terminal records only.
        res = cleanup_tasks(
            status_filter=["ok", "failed", "canceled"],
            max_age_s=7 * 24 * 3600,
            dry_run=False,
        )
        cleaned = int(res.get("cleaned") or 0) if isinstance(res, dict) else 0
        if cleaned:
            logger.info(f"Cleaned {cleaned} old task records")
        else:
            logger.debug("No old task records to clean")
    except Exception as e:
        logger.warning(f"Task record cleanup failed: {e}")


def create_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=settings.daemon.timezone)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=8)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=12)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=16)
    scheduler.add_job(fetch_compute, "cron", day_of_week="mon,tue,wed,thu,fri", hour=20)
    scheduler.add_job(send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
    scheduler.add_job(backup_user_data, "cron", hour=20)
    scheduler.add_job(cleanup_task_records, "cron", hour=3)
    return scheduler


def _log_startup_info():
    """Print daemon configuration summary on startup (always visible)."""
    summary = "off"
    if ENABLE_SUMMARY:
        parts = [f"{SUMMARY_NUM}/batch", f"{SUMMARY_WORKERS}w"]
        if ENABLE_SUMMARY_QUEUE:
            parts.append("queue")
        if ENABLE_PRIORITY_QUEUE:
            parts.append(f"priority(limit={PRIORITY_LIMIT})")
        summary = ", ".join(parts)

    flags = []
    if ENABLE_EMBEDDINGS:
        flags.append("embeddings")
    if settings.daemon.email_dry_run:
        flags.append("email-dry-run")
    if settings.daemon.enable_git_backup:
        flags.append("git-backup")

    lines = [
        "=" * 50,
        "Arxiv-Sanity Daemon",
        f"  TZ={settings.daemon.timezone}  Log={settings.log_level.upper()}",
        f"  Fetch: {FETCH_NUM} papers (max {FETCH_MAX}/query)",
        f"  Summary: {summary}",
        f"  Flags: {', '.join(flags) or 'none'}",
        "Schedule:",
        "  fetch_compute  Mon-Fri 08,12,16,20",
        "  send_email     Mon-Fri 18:00",
        "  backup         Daily   20:00",
        "  cleanup        Daily   03:00",
    ]
    if settings.log_level.upper() in ("WARNING", "ERROR", "CRITICAL"):
        lines.append("  (set ARXIV_SANITY_LOG_LEVEL=INFO to see pipeline activity)")
    lines.append("=" * 50)
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
