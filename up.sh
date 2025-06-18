# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version
gunicorn -w 2 --threads 4 -k gthread -b 0.0.0.0:55555 serve:app
