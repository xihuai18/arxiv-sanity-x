# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version
# Get port from vars.py
SERVE_PORT=$(python3 -c "from vars import SERVE_PORT; print(SERVE_PORT)")

# Tunables (safe defaults)
WORKERS="${GUNICORN_WORKERS:-2}"
THREADS="${GUNICORN_THREADS:-4}"

# Preload app in master for copy-on-write sharing (recommended for large feature matrices).
# Requires the app to avoid starting background threads at import-time (serve.py does this).
PRELOAD="${ARXIV_SANITY_GUNICORN_PRELOAD:-1}"
PRELOAD_ARG=""
case "${PRELOAD}" in
  1|true|TRUE|yes|YES) PRELOAD_ARG="--preload" ;;
esac

EXTRA_ARGS="${GUNICORN_EXTRA_ARGS:-}"

gunicorn -w "${WORKERS}" --threads "${THREADS}" -k gthread ${PRELOAD_ARG} ${EXTRA_ARGS} -b 0.0.0.0:${SERVE_PORT} serve:app
