"""Flask application factory."""

from __future__ import annotations

import logging
import os
import sys

from flask import Flask, request
from loguru import logger

from config import settings

from .blueprints import (
    api_papers,
    api_readinglist,
    api_search,
    api_sse,
    api_summary,
    api_tags,
    api_user,
    web,
)
from .utils.manifest import static_url as manifest_static_url
from .utils.validation import get_or_set_csrf_token

logger.remove()
logger.add(sys.stdout, level=settings.log_level.upper())

if not settings.access_log:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _load_secret_key() -> str:
    sk = (settings.web.secret_key or "").strip()
    if sk:
        return sk

    if os.path.isfile("secret_key.txt"):
        try:
            with open("secret_key.txt", encoding="utf-8") as f:
                sk = f.read().strip()
            if sk:
                return sk
        except Exception as exc:
            logger.warning(f"Failed to read secret_key.txt: {exc}")

    logger.warning(
        "No secret key found (ARXIV_SANITY_SECRET_KEY/secret_key.txt); generating a random key (sessions reset on restart)"
    )
    import secrets

    return secrets.token_urlsafe(32)


def create_app() -> Flask:
    import os

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(
        __name__,
        template_folder=os.path.join(root_dir, "templates"),
        static_folder=os.path.join(root_dir, "static"),
    )
    app.secret_key = _load_secret_key()

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE=settings.web.cookie_samesite,
        SESSION_COOKIE_SECURE=settings.web.cookie_secure,
        MAX_CONTENT_LENGTH=settings.web.max_content_length,
    )

    # Initialize Swagger/OpenAPI documentation (disabled by default for security)
    if settings.enable_swagger:
        try:
            from flasgger import Swagger

            app.config["SWAGGER"] = {
                "title": "Arxiv Sanity API",
                "uiversion": 3,
                "description": "API documentation for arxiv-sanity paper management system",
                "version": "1.0.0",
                "termsOfService": "",
                "specs_route": "/apidocs/",
            }
            Swagger(app)
        except ImportError:
            logger.warning("flasgger not installed, Swagger UI disabled")

    @app.after_request
    def add_security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        try:
            host = (request.host or "").split(":", 1)[0].lower()
            is_localhost = host in ("localhost", "127.0.0.1", "::1")
            if request.is_secure or is_localhost:
                resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        except Exception:
            pass
        resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        resp.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), interest-cohort=()",
        )

        # Add long cache headers for hashed static assets
        # Files with hash in name (e.g., paper_list-ABC123.js) can be cached forever
        if request.path.startswith("/static/dist/"):
            import re

            # Check if filename contains a hash pattern (name-HASH.ext)
            if re.search(r"-[a-zA-Z0-9]{8,}\.(js|css|map)$", request.path):
                # Immutable cache for 1 year
                resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                # Short cache for non-hashed files (fallback)
                resp.headers.setdefault("Cache-Control", "public, max-age=3600")

        return resp

    @app.context_processor
    def inject_csrf_token():
        return {"csrf_token": get_or_set_csrf_token()}

    @app.context_processor
    def inject_static_helpers():
        """Inject static asset helpers for cache-busting."""

        def hashed_static(filename: str) -> str:
            """Generate URL for static file with cache-busting hash."""
            from flask import url_for

            resolved = manifest_static_url(filename)
            return url_for("static", filename=resolved)

        return {"hashed_static": hashed_static}

    app.register_blueprint(web.bp)
    app.register_blueprint(api_user.bp)
    app.register_blueprint(api_sse.bp)
    app.register_blueprint(api_summary.bp)
    app.register_blueprint(api_search.bp)
    app.register_blueprint(api_tags.bp)
    app.register_blueprint(api_papers.bp)
    app.register_blueprint(api_readinglist.bp)

    from . import legacy

    app.before_request(legacy.before_request)
    app.teardown_request(legacy.close_connection)

    return app
