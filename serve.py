"""
Flask server backend - thin wrapper for modular backend.

This file maintains backward compatibility with existing deployment scripts
while delegating all functionality to the modular backend package.
"""

import sys

from backend.app import create_app

# Create app instance for gunicorn/deployment
app = create_app()

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
