# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version
# Get port from vars.py
SERVE_PORT=$(python3 -c "from vars import SERVE_PORT; print(SERVE_PORT)")
gunicorn -w 2 --threads 4 -k gthread -b 0.0.0.0:${SERVE_PORT} serve:app
