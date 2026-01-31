# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version

# Build static assets (fast, ensures latest JS with cache-busting hashes)
echo "[build] Building static assets..." 1>&2
npm run build:static --silent || {
  echo "[build] Warning: npm build failed, continuing with existing dist files if present" 1>&2
  if [ ! -f "static/dist/manifest.json" ]; then
    echo "[build] Warning: manifest.json not found, will use fallback resolution (may be slower)" 1>&2
    echo "[build] For optimal performance, fix the build: npm install && npm run build:static" 1>&2
  fi
}

# Get configuration from config.settings
SERVE_PORT=$(python3 -c "from config import settings; print(settings.serve_port)")
GUNICORN_WORKERS_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.workers)")
GUNICORN_THREADS_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.threads)")
GUNICORN_PRELOAD_CFG=$(python3 -c "from config import settings; print('1' if settings.gunicorn.preload else '0')")
GUNICORN_EXTRA_ARGS_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.extra_args)")
GUNICORN_MAX_MEMORY_CFG=$(python3 -c "from config import settings; print(settings.gunicorn.max_memory_mb)")

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

# Debug/dev convenience: if Flask/web reload is enabled, also enable gunicorn --reload
# unless the user explicitly provided it already.
WEB_RELOAD_CFG=$(python3 -c "from config import settings; print('1' if settings.web.reload else '0')")
if [ "${WEB_RELOAD_CFG}" = "1" ]; then
  case " ${EXTRA_ARGS} " in
    *" --reload "*) : ;; # already set
    *) EXTRA_ARGS="${EXTRA_ARGS} --reload" ;;
  esac
fi

# Memory limit: use --max-requests to restart workers periodically (helps with memory leaks)
# and set RLIMIT via Python if max_memory_mb is configured
MAX_MEMORY="${ARXIV_SANITY_GUNICORN_MAX_MEMORY_MB:-$GUNICORN_MAX_MEMORY_CFG}"
MEMORY_ARGS=""
if [ "$MAX_MEMORY" -gt 0 ] 2>/dev/null; then
  # Restart workers after 1000 requests (with jitter) to prevent memory accumulation
  MEMORY_ARGS="--max-requests 1000 --max-requests-jitter 100"
  echo "[gunicorn] Memory limit: ${MAX_MEMORY}MB per worker (with periodic restart)" 1>&2
fi

echo "[gunicorn] port=${SERVE_PORT} workers=${WORKERS} threads=${THREADS} preload=${PRELOAD_ARG:-0} extra_args='${EXTRA_ARGS}'" 1>&2

# Set memory limit environment variable for serve.py to apply
export ARXIV_SANITY_GUNICORN_MAX_MEMORY_MB="${MAX_MEMORY}"
# Mark this process as web for DB fail-fast tuning.
export ARXIV_SANITY_PROCESS_ROLE="web"

gunicorn -w "${WORKERS}" --threads "${THREADS}" -k gthread ${PRELOAD_ARG} ${MEMORY_ARGS} ${EXTRA_ARGS} -b 0.0.0.0:${SERVE_PORT} serve:app
