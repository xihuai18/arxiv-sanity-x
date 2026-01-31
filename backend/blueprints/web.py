"""Web page routes."""

from __future__ import annotations

from flask import Blueprint, jsonify

from .. import legacy

bp = Blueprint("web", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint that verifies data is loaded.

    Returns 200 only when the database and features are ready.
    Used by launcher to determine when the service is truly ready.
    """
    from backend.services.data_service import get_data_cached

    try:
        data = get_data_cached()
        pids = data.get("pids", [])
        if not pids:
            return jsonify({"status": "loading", "message": "No papers loaded yet"}), 503
        return jsonify({"status": "ok", "papers": len(pids)}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@bp.route("/", methods=["GET"])
def main():
    return legacy.main()


@bp.route("/inspect", methods=["GET"])
def inspect():
    return legacy.inspect()


@bp.route("/summary", methods=["GET"])
def summary():
    return legacy.summary()


@bp.route("/profile")
def profile():
    return legacy.profile()


@bp.route("/stats")
def stats():
    return legacy.stats()


@bp.route("/about")
def about():
    return legacy.about()


@bp.route("/readinglist", methods=["GET"])
def readinglist_page():
    return legacy.readinglist_page()
