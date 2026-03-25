"""Background services and scheduler management."""

from __future__ import annotations

import logging
import threading
import time

from loguru import logger

from config import settings

try:
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
except Exception:  # pragma: no cover - APScheduler is a runtime dependency
    EVENT_JOB_EXECUTED = None
    EVENT_JOB_ERROR = None
    EVENT_JOB_MISSED = None

_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_STARTED = False
_SCHEDULER = None
_SUMMARY_REPAIR_JOB = False
_SCHEDULER_STATS_LOCK = threading.Lock()
_SCHEDULER_JOB_LABELS: dict[str, str] = {}
_SCHEDULER_JOB_STATS: dict[str, dict[str, int | float]] = {}
_SCHEDULER_SUMMARY_EVERY_RUNS = 12
_SCHEDULER_SUMMARY_EVERY_SECONDS = 30 * 60


def _configure_apscheduler_logging() -> None:
    """Keep APScheduler internal warnings from flooding launcher output."""
    logging.getLogger("apscheduler").setLevel(logging.ERROR)


def _scheduler_misfire_grace_time(interval_seconds: int) -> int:
    """Allow minor scheduling delays for low-priority maintenance jobs."""
    interval = max(1, int(interval_seconds or 1))
    return max(60, min(interval, 30 * 60))


def _scheduler_job_label(job_id: str) -> str:
    return _SCHEDULER_JOB_LABELS.get(job_id, job_id)


def _record_scheduler_job_event(job_id: str, field: str, *, result: int | None = None) -> None:
    now = time.time()
    with _SCHEDULER_STATS_LOCK:
        stats = _SCHEDULER_JOB_STATS.setdefault(
            job_id,
            {
                "runs": 0,
                "missed": 0,
                "errors": 0,
                "result_total": 0,
                "last_result": 0,
                "last_report_at": now,
                "last_run_at": 0,
            },
        )
        stats[field] = int(stats.get(field, 0)) + 1
        if field == "runs":
            stats["last_run_at"] = now
            if result is not None:
                stats["last_result"] = int(result)
                stats["result_total"] = int(stats.get("result_total", 0)) + int(result)
        should_report = (
            field != "errors"
            and int(stats.get("runs", 0)) > 0
            and (
                int(stats.get("runs", 0)) % _SCHEDULER_SUMMARY_EVERY_RUNS == 0
                or now - float(stats.get("last_report_at", now)) >= _SCHEDULER_SUMMARY_EVERY_SECONDS
            )
        )
        if should_report:
            stats["last_report_at"] = now
        snapshot = dict(stats)

    label = _scheduler_job_label(job_id)
    if field == "errors":
        logger.warning(f"Scheduler job failed: {label}")
        return
    if field == "runs" and result and result > 0:
        logger.info(f"Scheduler job result: {label} repaired {result} stale item(s)")
    if should_report:
        logger.info(
            "Scheduler job summary: "
            f"{label} ran {int(snapshot.get('runs', 0))} time(s), "
            f"missed {int(snapshot.get('missed', 0))} time(s), "
            f"failed {int(snapshot.get('errors', 0))} time(s), "
            f"cumulative_result={int(snapshot.get('result_total', 0))}, "
            f"last_result={int(snapshot.get('last_result', 0))}"
        )


def _on_scheduler_event(event) -> None:
    job_id = getattr(event, "job_id", None)
    if not job_id:
        return
    code = getattr(event, "code", None)
    if EVENT_JOB_EXECUTED is not None and code == EVENT_JOB_EXECUTED:
        result = getattr(event, "retval", None)
        _record_scheduler_job_event(job_id, "runs", result=result if isinstance(result, int) else None)
    elif EVENT_JOB_ERROR is not None and code == EVENT_JOB_ERROR:
        _record_scheduler_job_event(job_id, "errors")
    elif EVENT_JOB_MISSED is not None and code == EVENT_JOB_MISSED:
        _record_scheduler_job_event(job_id, "missed")


def _get_native_thread_class():
    """Return a real OS Thread class even under gevent monkey-patching."""
    try:
        import gevent.monkey

        if gevent.monkey.is_module_patched("threading"):
            return gevent.monkey.get_original("threading", "Thread")
    except Exception:
        return threading.Thread
    return threading.Thread


def _start_daemon_thread(*, target, name: str) -> None:
    Thread = _get_native_thread_class()
    t = Thread(target=target, name=name, daemon=True)
    t.start()


def _is_data_cache_loaded() -> bool:
    """Check if data cache (metas/pids) is already loaded."""
    try:
        from .data_service import _METAS_CACHE, _PIDS_CACHE

        return _METAS_CACHE is not None and _PIDS_CACHE is not None
    except Exception:
        return False


def _is_features_cache_loaded() -> bool:
    """Check if features cache is already loaded."""
    try:
        from .data_service import _FEATURES_CACHE

        return _FEATURES_CACHE is not None
    except Exception:
        return False


def _warmup_data_cache():
    """Warm up data cache in background."""
    # Skip if already loaded (e.g., by preload in master process)
    if _is_data_cache_loaded():
        logger.debug("Data cache already loaded, skipping warmup")
        return

    try:
        from .data_service import warmup_data_cache

        logger.debug("Warming metas/pids cache in background...")
        warmup_data_cache()
    except Exception as e:
        logger.warning(f"Data cache warmup failed: {e}")


def _warmup_ml_cache():
    """Warm up ML-related caches in background."""
    # Skip if already loaded (e.g., by preload in master process)
    if _is_features_cache_loaded():
        logger.debug("Features cache already loaded, skipping warmup")
        return

    try:
        from .data_service import get_features_cached
        from .semantic_service import get_paper_embeddings, get_semantic_model

        logger.debug("Warming features/embeddings/model in background...")
        get_features_cached()
        get_paper_embeddings()
        get_semantic_model()
    except Exception as e:
        logger.warning(f"ML cache warmup failed: {e}")


def ensure_background_services_started():
    """Start background threads/schedulers lazily (per worker process)."""
    global _BACKGROUND_STARTED, _SCHEDULER, _SUMMARY_REPAIR_JOB

    if _BACKGROUND_STARTED:
        return

    with _BACKGROUND_LOCK:
        if _BACKGROUND_STARTED:
            return

        if settings.web.warmup_data:
            _start_daemon_thread(target=_warmup_data_cache, name="warmup-data-cache")

        if settings.web.warmup_ml:
            _start_daemon_thread(target=_warmup_ml_cache, name="warmup-ml-cache")

        if settings.web.enable_scheduler:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler

                from .data_service import warmup_data_cache
                from .summary_service import refresh_summary_cache_stats_full

                _configure_apscheduler_logging()
                _SCHEDULER = BackgroundScheduler(timezone=settings.daemon.timezone)
                _SCHEDULER.add_listener(
                    _on_scheduler_event,
                    EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
                )
                _SCHEDULER_JOB_LABELS["warmup_data_cache"] = "warmup_data_cache"
                _SCHEDULER.add_job(
                    warmup_data_cache,
                    "cron",
                    id="warmup_data_cache",
                    day_of_week="tue,wed,thu,fri,mon",
                    hour=18,
                )

                # Periodically refresh summary cache stats to avoid expensive on-demand full scans.
                refresh_interval = settings.web.summary_cache_stats_refresh
                if refresh_interval > 0:

                    def _refresh_summary_cache_stats():
                        try:
                            refresh_summary_cache_stats_full()
                        except Exception as e:
                            logger.warning(f"Failed to refresh summary cache stats: {e}")

                    _SCHEDULER.add_job(
                        _refresh_summary_cache_stats,
                        "interval",
                        id="refresh_summary_cache_stats",
                        seconds=refresh_interval,
                        max_instances=1,
                        coalesce=True,
                        misfire_grace_time=_scheduler_misfire_grace_time(refresh_interval),
                    )
                    _SCHEDULER_JOB_LABELS["refresh_summary_cache_stats"] = "refresh_summary_cache_stats"

                if settings.huey.summary_repair_enable:
                    try:
                        from tasks import repair_stale_summary_tasks

                        repair_interval = settings.huey.summary_repair_interval
                        repair_requeue = settings.huey.summary_repair_requeue
                        if repair_interval > 0:
                            _SCHEDULER.add_job(
                                repair_stale_summary_tasks,
                                "interval",
                                id="repair_stale_summary_tasks",
                                seconds=repair_interval,
                                kwargs={"requeue": repair_requeue},
                                max_instances=1,
                                coalesce=True,
                                misfire_grace_time=_scheduler_misfire_grace_time(repair_interval),
                            )
                            _SCHEDULER_JOB_LABELS["repair_stale_summary_tasks"] = "repair_stale_summary_tasks"
                            _SUMMARY_REPAIR_JOB = True
                    except Exception as e:
                        logger.warning(f"Failed to schedule summary repair job: {e}")

                _SCHEDULER.start()
            except Exception as e:
                logger.warning(f"Failed to start scheduler: {e}")

        _BACKGROUND_STARTED = True


def is_scheduler_running() -> bool:
    """Check if scheduler is running."""
    return _SCHEDULER is not None and _SCHEDULER.running


def is_summary_repair_enabled() -> bool:
    """Check if summary repair job is enabled."""
    return _SUMMARY_REPAIR_JOB
