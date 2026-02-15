#!/usr/bin/env sh

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

EXTRA_ARGS_RAW="${GUNICORN_EXTRA_ARGS:-$GUNICORN_EXTRA_ARGS_CFG}"
EXTRA_ARGS_WORKER_CLASS=""
EXTRA_ARGS_HAS_TIMEOUT=0
EXTRA_ARGS_HAS_GRACEFUL_TIMEOUT=0
EXTRA_ARGS_HAS_RELOAD=0
EXTRA_ARGS_HAS_THREADS=0
EXTRA_ARGS_TOKENS=""

EXTRA_ARGS_PARSE=$(
  EXTRA_ARGS_RAW="${EXTRA_ARGS_RAW}" python3 - <<'PY'
import os
import shlex

raw = os.environ.get("EXTRA_ARGS_RAW", "") or ""
try:
    tokens = shlex.split(raw)
except Exception:
    tokens = raw.split()

wc = ""
stripped = []
has_timeout = 0
has_graceful_timeout = 0
has_reload = 0
has_threads = 0

i = 0
while i < len(tokens):
    t = tokens[i]

    if t == "--timeout" and i + 1 < len(tokens):
        has_timeout = 1
    elif t.startswith("--timeout="):
        has_timeout = 1

    if t == "--graceful-timeout" and i + 1 < len(tokens):
        has_graceful_timeout = 1
    elif t.startswith("--graceful-timeout="):
        has_graceful_timeout = 1

    if t == "--reload":
        has_reload = 1

    if t == "--threads" and i + 1 < len(tokens):
        has_threads = 1
    elif t.startswith("--threads="):
        has_threads = 1

    if t in ("-k", "--worker-class"):
        if i + 1 < len(tokens):
            wc = tokens[i + 1]
            i += 2
            continue
        i += 1
        continue
    if t.startswith("--worker-class="):
        wc = t.split("=", 1)[1]
        i += 1
        continue
    if t.startswith("-k") and len(t) > 2:
        wc = t[2:]
        i += 1
        continue

    stripped.append(t)
    i += 1

print(f"WORKER_CLASS={wc}")
print(f"HAS_TIMEOUT={has_timeout}")
print(f"HAS_GRACEFUL_TIMEOUT={has_graceful_timeout}")
print(f"HAS_RELOAD={has_reload}")
print(f"HAS_THREADS={has_threads}")
print("")
for t in stripped:
    print(t)
PY
)

_parse_state="header"
while IFS= read -r _line; do
  if [ "${_parse_state}" = "header" ]; then
    if [ -z "${_line}" ]; then
      _parse_state="tokens"
      continue
    fi
    case "${_line}" in
      WORKER_CLASS=*) EXTRA_ARGS_WORKER_CLASS="${_line#WORKER_CLASS=}" ;;
      HAS_TIMEOUT=*) EXTRA_ARGS_HAS_TIMEOUT="${_line#HAS_TIMEOUT=}" ;;
      HAS_GRACEFUL_TIMEOUT=*) EXTRA_ARGS_HAS_GRACEFUL_TIMEOUT="${_line#HAS_GRACEFUL_TIMEOUT=}" ;;
      HAS_RELOAD=*) EXTRA_ARGS_HAS_RELOAD="${_line#HAS_RELOAD=}" ;;
      HAS_THREADS=*) EXTRA_ARGS_HAS_THREADS="${_line#HAS_THREADS=}" ;;
    esac
  else
    EXTRA_ARGS_TOKENS="${EXTRA_ARGS_TOKENS}${_line}
"
  fi
done <<EOF
${EXTRA_ARGS_PARSE}
EOF
unset _parse_state _line EXTRA_ARGS_PARSE

# Decide worker class.
# Default strategy:
# - If SSE is enabled, prefer an async worker to avoid tying up threads per connection.
# - Allow explicit override via ARXIV_SANITY_GUNICORN_WORKER_CLASS / EXTRA_ARGS (-k/--worker-class).
SSE_ENABLED_CFG=$(python3 -c "from config import settings; print('1' if getattr(settings, 'sse', None) and settings.sse.enabled else '0')")
SSE_STRICT_WORKER_CLASS_CFG=$(python3 -c "from config import settings; print('1' if getattr(settings, 'sse', None) and getattr(settings.sse, 'strict_worker_class', False) else '0')")

WORKER_CLASS_OVERRIDE="${ARXIV_SANITY_GUNICORN_WORKER_CLASS:-}"
WORKER_CLASS=""
if [ -n "${WORKER_CLASS_OVERRIDE}" ]; then
  WORKER_CLASS="${WORKER_CLASS_OVERRIDE}"
elif [ -n "${EXTRA_ARGS_WORKER_CLASS}" ]; then
  WORKER_CLASS="${EXTRA_ARGS_WORKER_CLASS}"
else
  if [ "${SSE_ENABLED_CFG}" = "1" ]; then
    # Prefer gevent if available; otherwise fall back to gthread.
    if python3 -c "import gevent" >/dev/null 2>&1; then
      WORKER_CLASS="gevent"
    else
      echo "[gunicorn] Warning: SSE enabled but gevent not available; falling back to gthread" 1>&2
      WORKER_CLASS="gthread"
    fi
  else
    WORKER_CLASS="gthread"
  fi
fi

# Export selected worker class so the app can make safe early-runtime decisions.
# (e.g., gevent monkey-patching before any ssl/urllib3 imports when using --preload.)
export ARXIV_SANITY_GUNICORN_WORKER_CLASS="${WORKER_CLASS}"

# SSE + non-gevent is a common footgun: each SSE connection ties up a thread/worker.
# Always warn; optionally fail-fast via settings.sse.strict_worker_class (or env var override).
if [ "${SSE_ENABLED_CFG}" = "1" ] && [ "${WORKER_CLASS}" != "gevent" ]; then
  echo "[gunicorn] Warning: SSE enabled but worker_class=${WORKER_CLASS}; recommend gevent for SSE" 1>&2
  if [ "${SSE_STRICT_WORKER_CLASS_CFG}" = "1" ]; then
    echo "[gunicorn] Error: SSE strict worker class enabled; refusing to start without gevent (set ARXIV_SANITY_SSE_STRICT_WORKER_CLASS=false to override)" 1>&2
    exit 1
  fi
fi

# Use config value, allow env override for backward compatibility
WORKERS="${GUNICORN_WORKERS:-$GUNICORN_WORKERS_CFG}"

# Safety guardrail: gevent workers are processes and can easily OOM when combined with
# large in-memory caches (e.g., cache_papers) and warmup on startup. If the user wants
# the full worker count anyway, set ARXIV_SANITY_GUNICORN_FORCE_WORKERS=1.
if [ "${WORKER_CLASS}" = "gevent" ]; then
  FORCE_WORKERS="${ARXIV_SANITY_GUNICORN_FORCE_WORKERS:-}"
  if [ -z "${FORCE_WORKERS}" ]; then
    CACHE_PAPERS_CFG=$(python3 -c "from config import settings; print('1' if settings.web.cache_papers else '0')")
    WARMUP_DATA_CFG=$(python3 -c "from config import settings; print('1' if settings.web.warmup_data else '0')")
    WARMUP_ML_CFG=$(python3 -c "from config import settings; print('1' if settings.web.warmup_ml else '0')")
    PRELOAD_CACHES_CFG=$(python3 -c "from config import settings; print('1' if settings.gunicorn.preload_caches else '0')")
    if [ "${WORKERS}" -gt 4 ] 2>/dev/null; then
      if [ "${CACHE_PAPERS_CFG}" = "1" ] || [ "${WARMUP_DATA_CFG}" = "1" ] || [ "${WARMUP_ML_CFG}" = "1" ] || [ "${PRELOAD_CACHES_CFG}" = "1" ]; then
        echo "[gunicorn] Warning: gevent workers=${WORKERS} with cache/warmup enabled can OOM; clamping to 2 (set ARXIV_SANITY_GUNICORN_FORCE_WORKERS=1 to override)" 1>&2
        WORKERS=2
      fi
    fi
  fi
fi

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

# Memory limit: use --max-requests to restart workers periodically (helps with memory leaks)
# and set RLIMIT via Python if max_memory_mb is configured
MAX_MEMORY="${ARXIV_SANITY_GUNICORN_MAX_MEMORY_MB:-$GUNICORN_MAX_MEMORY_CFG}"
MEMORY_ARGS=""
if [ "$MAX_MEMORY" -gt 0 ] 2>/dev/null; then
  # Restart workers after 1000 requests (with jitter) to prevent memory accumulation
  MEMORY_ARGS="--max-requests 1000 --max-requests-jitter 100"
  echo "[gunicorn] Memory limit: ${MAX_MEMORY}MB per worker (with periodic restart)" 1>&2
fi

# Build gunicorn argv without relying on shell word-splitting of EXTRA_ARGS.
set -- gunicorn -w "${WORKERS}" -k "${WORKER_CLASS}"

# Threads: only meaningful for gthread; let EXTRA_ARGS override if it sets --threads itself.
if [ "${WORKER_CLASS}" = "gthread" ] && [ "${EXTRA_ARGS_HAS_THREADS}" != "1" ]; then
  set -- "$@" --threads "${THREADS}"
fi

# Preload app in master for copy-on-write sharing (recommended for large feature matrices).
if [ -n "${PRELOAD_ARG}" ]; then
  set -- "$@" "${PRELOAD_ARG}"
fi

# Gunicorn timeouts:
# - Default timeout (30s) is often too small for cold-start / preload / cache warmup on large corpora.
# - gthread fallback (when gevent isn't available) is especially prone to worker timeouts under heavy DB scans.
# - Only set defaults when the user did not specify them explicitly.
if [ "${EXTRA_ARGS_HAS_TIMEOUT}" != "1" ]; then
  # SSE connections and cold-start I/O can legitimately take >30s; keep this generous.
  set -- "$@" --timeout 600
fi
if [ "${EXTRA_ARGS_HAS_GRACEFUL_TIMEOUT}" != "1" ]; then
  set -- "$@" --graceful-timeout 600
fi

# Debug/dev convenience: if Flask/web reload is enabled, also enable gunicorn --reload
# unless the user explicitly provided it already.
WEB_RELOAD_CFG=$(python3 -c "from config import settings; print('1' if settings.web.reload else '0')")
if [ "${WEB_RELOAD_CFG}" = "1" ] && [ "${EXTRA_ARGS_HAS_RELOAD}" != "1" ]; then
  set -- "$@" --reload
fi

# Memory limit: use --max-requests to restart workers periodically (helps with memory leaks)
if [ -n "${MEMORY_ARGS}" ]; then
  set -- "$@" --max-requests 1000 --max-requests-jitter 100
fi

# Append remaining EXTRA_ARGS tokens (worker-class flags were stripped during parsing).
while IFS= read -r _token; do
  [ -n "${_token}" ] || continue
  set -- "$@" "${_token}"
done <<EOF
${EXTRA_ARGS_TOKENS}
EOF
unset _token EXTRA_ARGS_TOKENS

# Warn about common OOM footgun: too many worker processes with large caches/models.
if [ "${WORKER_CLASS}" = "gevent" ]; then
  if [ "${WORKERS}" -gt 4 ] 2>/dev/null; then
    echo "[gunicorn] Warning: workers=${WORKERS} with gevent can OOM on large corpora; consider ARXIV_SANITY_GUNICORN_WORKERS=1-2" 1>&2
  fi
fi

_threads_log=0
if [ "${WORKER_CLASS}" = "gthread" ]; then
  _threads_log="${THREADS}"
fi
echo "[gunicorn] port=${SERVE_PORT} workers=${WORKERS} worker_class=${WORKER_CLASS} threads=${_threads_log} preload=${PRELOAD_ARG:-0} extra_args_raw='${EXTRA_ARGS_RAW}'" 1>&2
unset _threads_log

# Set memory limit environment variable for serve.py to apply
export ARXIV_SANITY_GUNICORN_MAX_MEMORY_MB="${MAX_MEMORY}"
# Mark this process as web for DB fail-fast tuning.
export ARXIV_SANITY_PROCESS_ROLE="web"

exec "$@" -b "0.0.0.0:${SERVE_PORT}" serve:app
