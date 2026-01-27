"""Background services and scheduler management."""

from __future__ import annotations

import threading

from loguru import logger

from config import settings

_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_STARTED = False
_SCHEDULER = None
_SUMMARY_REPAIR_JOB = False


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
            threading.Thread(target=_warmup_data_cache, daemon=True).start()

        if settings.web.warmup_ml:
            threading.Thread(target=_warmup_ml_cache, daemon=True).start()

        if settings.web.enable_scheduler:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler

                from .data_service import warmup_data_cache
                from .summary_service import refresh_summary_cache_stats_full

                _SCHEDULER = BackgroundScheduler(timezone=settings.daemon.timezone)
                _SCHEDULER.add_job(warmup_data_cache, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)

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
                        seconds=refresh_interval,
                        max_instances=1,
                        coalesce=True,
                    )

                if settings.huey.summary_repair_enable:
                    try:
                        from tasks import repair_stale_summary_tasks

                        repair_interval = settings.huey.summary_repair_interval
                        repair_requeue = settings.huey.summary_repair_requeue
                        if repair_interval > 0:
                            _SCHEDULER.add_job(
                                repair_stale_summary_tasks,
                                "interval",
                                seconds=repair_interval,
                                kwargs={"requeue": repair_requeue},
                                max_instances=1,
                                coalesce=True,
                            )
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
