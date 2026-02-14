"""Prometheus metrics endpoint (optional)."""

from __future__ import annotations

import hmac
import time

from flask import Blueprint, Response, abort, g, request
from loguru import logger

bp = Blueprint("metrics", __name__)


def _is_enabled() -> bool:
    try:
        import config

        return bool(getattr(config.settings.web, "enable_metrics", False))
    except Exception:
        return False


def _check_key() -> None:
    try:
        import config

        key = str(getattr(config.settings.web, "metrics_key", "") or "").strip()
    except Exception:
        key = ""

    if not key:
        return

    provided = (request.headers.get("X-ARXIV-SANITY-METRICS-KEY") or "").strip()
    if not hmac.compare_digest(provided, key):
        abort(403)


@bp.route("/metrics", methods=["GET"])
def metrics():
    if not _is_enabled():
        abort(404)
    _check_key()

    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        data = generate_latest()
        return Response(data, mimetype=CONTENT_TYPE_LATEST)
    except Exception:
        logger.opt(exception=True).warning("Failed to generate Prometheus metrics")
        abort(503)


def before_request_hook() -> None:
    """Record request start time (only when metrics are enabled)."""
    if not _is_enabled():
        return
    try:
        g._metrics_start_time = time.perf_counter()
    except Exception:
        return


def after_request_hook(response):
    """Update request counters/histograms (only when metrics are enabled)."""
    if not _is_enabled():
        return response

    try:
        from prometheus_client import Counter, Histogram

        start = getattr(g, "_metrics_start_time", None)
        if start is None:
            return response
        duration = max(0.0, float(time.perf_counter() - start))

        endpoint = request.endpoint or "unknown"
        method = request.method or "UNKNOWN"
        status = str(getattr(response, "status_code", 0) or 0)

        # Lazy-init metric objects to keep import overhead low.
        # Note: default registry is process-local; under gunicorn this is per-worker.
        if not hasattr(after_request_hook, "_req_total"):
            after_request_hook._req_total = Counter(  # type: ignore[attr-defined]
                "http_requests_total",
                "Total HTTP requests",
                ["method", "endpoint", "status"],
            )
            after_request_hook._req_seconds = Histogram(  # type: ignore[attr-defined]
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "endpoint"],
            )

        after_request_hook._req_total.labels(method=method, endpoint=endpoint, status=status).inc()  # type: ignore[attr-defined]
        after_request_hook._req_seconds.labels(method=method, endpoint=endpoint).observe(duration)  # type: ignore[attr-defined]
    except Exception:
        # Avoid breaking responses on metrics failures.
        pass

    return response

