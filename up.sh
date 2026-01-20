# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version
# Get port from vars.py
SERVE_PORT=$(python3 -c "from vars import SERVE_PORT; print(SERVE_PORT)")

# Tunables (safe defaults)
#
# For large corpora, each gunicorn worker process will load its own copy of
# in-memory caches / feature matrices. More workers can therefore *hurt*
# performance due to memory pressure and duplicated initialization.
#
# Default strategy:
# - Prefer fewer workers (default 1)
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

# Default: 1 worker. Override via GUNICORN_WORKERS.
WORKERS="${GUNICORN_WORKERS:-4}"

# Default threads: 2x cores, capped to avoid oversubscription.
DEFAULT_THREADS=$(( CORES * 2 ))
if [ "$DEFAULT_THREADS" -lt 4 ]; then DEFAULT_THREADS=4; fi
if [ "$DEFAULT_THREADS" -gt 16 ]; then DEFAULT_THREADS=16; fi
THREADS="${GUNICORN_THREADS:-$DEFAULT_THREADS}"

# Preload app in master for copy-on-write sharing (recommended for large feature matrices).
# Requires the app to avoid starting background threads at import-time (serve.py does this).
PRELOAD="${ARXIV_SANITY_GUNICORN_PRELOAD:-1}"
PRELOAD_ARG=""
case "${PRELOAD}" in
  1|true|TRUE|yes|YES) PRELOAD_ARG="--preload" ;;
esac

EXTRA_ARGS="${GUNICORN_EXTRA_ARGS:-}"

echo "[gunicorn] port=${SERVE_PORT} workers=${WORKERS} threads=${THREADS} preload=${PRELOAD_ARG:-0} extra_args='${EXTRA_ARGS}'" 1>&2

gunicorn -w "${WORKERS}" --threads "${THREADS}" -k gthread ${PRELOAD_ARG} ${EXTRA_ARGS} -b 0.0.0.0:${SERVE_PORT} serve:app
