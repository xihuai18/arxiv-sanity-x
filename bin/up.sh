# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version

# Build static assets (fast, ensures latest JS with cache-busting hashes)
echo "[build] Building static assets..." 1>&2
npm run build:static --silent || {
  echo "[build] Warning: npm build failed, continuing with existing dist files" 1>&2
}

# Get configuration from config.settings
SERVE_PORT=$(python3 -c "from config import settings; print(settings.serve_port)")
GUNICORN_WORKERS_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.workers)")
GUNICORN_THREADS_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.threads)")
GUNICORN_PRELOAD_CFG=$(python3 -c "from config import settings; print('1' if settings.gunicorn.preload else '0')")
GUNICORN_EXTRA_ARGS_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.extra_args)")

# Tunables (safe defaults)
#
# For large corpora, each gunicorn worker process will load its own copy of
# in-memory caches / feature matrices. More workers can therefore *hurt*
# performance due to memory pressure and duplicated initialization.
#
# Default strategy:
# - Use moderate workers (default 4) for balanced concurrency
# - Prefer more threads per worker for concurrent requests

_detect_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null && return
  fi
  python3 - <<'PY'
import os
print(os.cpu_count() or 2)
PY
}

CORES="$(_detect_cores)"
if [ -z "$CORES" ]; then
  CORES=2
fi

# Use config value, allow env override for backward compatibility
WORKERS="${GUNICORN_WORKERS:-$GUNICORN_WORKERS_CFG}"

# Threads: use config if set (non-zero), otherwise auto-calculate
if [ "$GUNICORN_THREADS_CFG" -gt 0 ] 2>/dev/null; then
  DEFAULT_THREADS="$GUNICORN_THREADS_CFG"
else
  DEFAULT_THREADS=$(( CORES * 2 ))
  if [ "$DEFAULT_THREADS" -lt 4 ]; then DEFAULT_THREADS=4; fi
  if [ "$DEFAULT_THREADS" -gt 16 ]; then DEFAULT_THREADS=16; fi
fi
THREADS="${GUNICORN_THREADS:-$DEFAULT_THREADS}"

# Preload app in master for copy-on-write sharing (recommended for large feature matrices).
# Requires the app to avoid starting background threads at import-time (serve.py does this).
PRELOAD="${ARXIV_SANITY_GUNICORN_PRELOAD:-$GUNICORN_PRELOAD_CFG}"
PRELOAD_ARG=""
case "${PRELOAD}" in
  1|true|TRUE|yes|YES) PRELOAD_ARG="--preload" ;;
esac

EXTRA_ARGS="${GUNICORN_EXTRA_ARGS:-$GUNICORN_EXTRA_ARGS_CFG}"

echo "[gunicorn] port=${SERVE_PORT} workers=${WORKERS} threads=${THREADS} preload=${PRELOAD_ARG:-0} extra_args='${EXTRA_ARGS}'" 1>&2

gunicorn -w "${WORKERS}" --threads "${THREADS}" -k gthread ${PRELOAD_ARG} ${EXTRA_ARGS} -b 0.0.0.0:${SERVE_PORT} serve:app
