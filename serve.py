"""
Flask server backend - thin wrapper for modular backend.

This file maintains backward compatibility with existing deployment scripts
while delegating all functionality to the modular backend package.
"""

import os
import sys

os.environ.setdefault("ARXIV_SANITY_PROCESS_ROLE", "web")

from backend.app import create_app


# Apply memory limit if configured (for gunicorn workers)
def _apply_memory_limit():
    """Apply memory limit to prevent OOM."""
    try:
        from config import settings

        max_mb = settings.gunicorn.max_memory_mb
        if max_mb > 0:
            from backend.utils.memory_limit import set_memory_limit

            set_memory_limit(max_mb)
    except Exception:
        pass


_apply_memory_limit()

# Create app instance for gunicorn/deployment
app = create_app()


def _is_gunicorn_master() -> bool:
    """Best-effort detection for gunicorn preload import.

    When gunicorn runs with `--preload`, this module is imported in the master
    process once, then workers are forked without re-importing the app module.
    Running cache warmup here therefore naturally happens in the master only.
    """
    try:
        argv0 = os.path.basename(sys.argv[0] or "").lower()
    except Exception:
        argv0 = ""
    if "gunicorn" not in argv0:
        return False

    cmd = " ".join(sys.argv[1:]) + " " + os.environ.get("GUNICORN_CMD_ARGS", "")
    return "--preload" in cmd or "preload_app" in cmd


def _preload_caches():
    """Preload data caches in master process for copy-on-write sharing.

    This should be called during gunicorn --preload, before workers fork.
    All workers will share the same memory pages (CoW) until they modify them.
    """
    from loguru import logger

    from config import settings

    if not settings.gunicorn.preload_caches:
        logger.info("[preload] Skipping cache preload (preload_caches=False)")
        return

    logger.info("[preload] Preloading data caches in master process...")

    # 1. Load metas/pids (always needed, relatively small)
    if settings.web.warmup_data:
        try:
            from backend.services.data_service import warmup_data_cache

            logger.info("[preload] Loading metas/pids...")
            warmup_data_cache()
            logger.info("[preload] Metas/pids loaded")
        except Exception as e:
            logger.warning(f"[preload] Failed to load metas/pids: {e}")

    # 2. Load features/embeddings (large, but shared via CoW)
    if settings.web.warmup_ml:
        try:
            from backend.services.data_service import get_features_cached

            logger.info("[preload] Loading features (this may take a few seconds)...")
            features = get_features_cached()
            logger.info(f"[preload] Features loaded: {len(features.get('pids', []))} papers")
        except Exception as e:
            logger.warning(f"[preload] Failed to load features: {e}")

        try:
            from backend.services.semantic_service import get_paper_embeddings

            logger.info("[preload] Loading embeddings...")
            emb_data = get_paper_embeddings()
            if emb_data:
                logger.info(f"[preload] Embeddings loaded: {len(emb_data.get('pids', []))} papers")
            else:
                logger.info("[preload] No embeddings available")
        except Exception as e:
            logger.warning(f"[preload] Failed to load embeddings: {e}")

    logger.info("[preload] Cache preload complete")


# Preload caches if running in gunicorn master with --preload
if _is_gunicorn_master():
    _preload_caches()

if __name__ == "__main__":
    from loguru import logger

    from config import settings

    logger.remove()
    logger.add(sys.stdout, level=settings.log_level.upper())
    enable_reload = settings.web.reload
    app.run(
        host="0.0.0.0",
        port=settings.serve_port,
        threaded=True,
        debug=enable_reload,
        use_reloader=enable_reload,
    )
