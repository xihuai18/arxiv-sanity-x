# Development version
# export FLASK_APP=serve.py; flask run --host 0.0.0.0
# Production version
gunicorn -w 8 -b 0.0.0.0:5000 serve:app
