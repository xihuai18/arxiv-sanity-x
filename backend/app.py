"""Flask application factory."""

from __future__ import annotations

import logging
import mimetypes
import os
import sys

from flask import Flask, render_template, request
from loguru import logger

from config import settings

from .blueprints import (
    api_papers,
    api_readinglist,
    api_search,
    api_sse,
    api_summary,
    api_tags,
    api_uploads,
    api_user,
    metrics,
    web,
)
from .utils.manifest import static_url as manifest_static_url
from .utils.validation import get_or_set_csrf_token

logger.remove()
_serialize_logs = str(getattr(settings, "log_format", "text") or "text").strip().lower() == "json"
logger.add(sys.stdout, level=settings.log_level.upper(), serialize=_serialize_logs)

if not settings.access_log:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _register_static_mimetypes() -> None:
    """Register missing MIME types for static assets.

    We set `X-Content-Type-Options: nosniff` for security. If font files are
    served as `application/octet-stream` (Python's default when the type is
    unknown), browsers may refuse to load them, which breaks MathJax CHTML
    glyphs (e.g. calligraphic / Greek symbols).
    """

    # Fonts
    mimetypes.add_type("font/woff", ".woff")
    mimetypes.add_type("font/woff2", ".woff2")
    mimetypes.add_type("font/ttf", ".ttf")
    mimetypes.add_type("font/otf", ".otf")
    mimetypes.add_type("application/vnd.ms-fontobject", ".eot")


_register_static_mimetypes()


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

    # Optional Sentry error reporting (no-op unless configured).
    try:
        from config.sentry import initialize_sentry

        initialize_sentry()
    except Exception:
        pass

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

    @app.errorhandler(404)
    def _handle_404(_err):
        try:
            return (
                render_template(
                    "error.html",
                    title="Not Found",
                    status_code=404,
                    message="The requested page does not exist.",
                ),
                404,
            )
        except Exception:
            return "Not Found", 404

    @app.errorhandler(500)
    def _handle_500(_err):
        try:
            return (
                render_template(
                    "error.html",
                    title="Server Error",
                    status_code=500,
                    message="An unexpected error occurred.",
                ),
                500,
            )
        except Exception:
            return "Internal Server Error", 500

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
        except Exception as e:
            logger.debug(f"Failed to set COOP header: {e}")
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
        else:
            # HTML pages must not be cached:
            # - prevents stale CSRF meta after server restart
            # - prevents stale hashed asset references after deploy
            try:
                if resp.mimetype == "text/html":
                    resp.headers.setdefault("Cache-Control", "no-store")
                    resp.headers.setdefault("Pragma", "no-cache")
            except Exception as e:
                logger.debug(f"Failed to set cache headers for HTML: {e}")

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

    @app.context_processor
    def inject_asset_cdn_config():
        """Inject frontend asset CDN config and helpers.

        Templates can use these values to load third-party libraries from a CDN
        with a deterministic local fallback.
        """

        def npm_cdn_url(path: str) -> str:
            base = (getattr(settings.web, "asset_npm_cdn_base", "") or "").strip()
            if not base:
                base = "https://cdn.jsdelivr.net/npm"
            base = base.rstrip("/")
            rel = (path or "").lstrip("/")
            return f"{base}/{rel}"

        return {
            # Plain values for inline JS.
            "asset_cdn_enabled": bool(getattr(settings.web, "asset_cdn_enabled", True)),
            "asset_npm_cdn_base": (getattr(settings.web, "asset_npm_cdn_base", "") or "").strip(),
            # Helper for templates.
            "npm_cdn_url": npm_cdn_url,
        }

    app.register_blueprint(web.bp)
    app.register_blueprint(api_user.bp)
    app.register_blueprint(api_sse.bp)
    app.register_blueprint(api_summary.bp)
    app.register_blueprint(api_search.bp)
    app.register_blueprint(api_tags.bp)
    app.register_blueprint(api_papers.bp)
    app.register_blueprint(api_readinglist.bp)
    app.register_blueprint(api_uploads.bp)
    app.register_blueprint(metrics.bp)

    from . import legacy

    app.before_request(legacy.before_request)
    app.teardown_request(legacy.close_connection)

    # Prometheus request metrics (no-op unless enabled).
    app.before_request(metrics.before_request_hook)
    app.after_request(metrics.after_request_hook)

    # Start SSE IPC runtime lazily (safe under gunicorn --preload).
    @app.before_request
    def _ensure_sse_ipc_runtime():
        try:
            from .utils.sse import ensure_sse_runtime_started

            ensure_sse_runtime_started()
        except Exception:
            pass

    return app
