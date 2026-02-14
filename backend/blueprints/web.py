"""Web page routes."""

from __future__ import annotations

import os

from flask import Blueprint, jsonify

from config import settings

from .. import legacy

bp = Blueprint("web", __name__)


def _http_probe(url: str, *, timeout_s: float = 1.0) -> dict:
    """Probe an HTTP endpoint quickly (best-effort)."""
    u = (url or "").strip()
    if not u:
        return {"reachable": False, "error": "empty_url"}
    try:
        import requests

        # Local service calls should not be routed via HTTP(S)_PROXY.
        s = requests.Session()
        s.trust_env = False
        resp = s.get(u, timeout=(min(0.5, timeout_s), timeout_s))
        return {"reachable": True, "status_code": int(resp.status_code)}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint that verifies data is loaded.

    Returns 200 only when the database and features are ready.
    Used by launcher to determine when the service is truly ready.
    """
    from backend.services.data_service import get_data_cached

    try:
        # Non-blocking peek: avoid /health being stuck on cold-start cache loading.
        data = get_data_cached(wait=False)
        pids = data.get("pids", [])
        if not pids:
            return jsonify({"status": "loading", "message": "No papers loaded yet"}), 503

        deps: dict[str, object] = {}

        # DB file presence is an operator hint.
        try:
            from aslite.db import PAPERS_DB_FILE

            deps["papers_db_file"] = {
                "exists": bool(PAPERS_DB_FILE and os.path.exists(PAPERS_DB_FILE)),
            }
        except Exception as e:
            deps["papers_db_file"] = {"error": str(e)}

        # Optional dependency probes (best-effort, should not block readiness).
        try:
            llm_base = (settings.llm.base_url or "").rstrip("/")
            if llm_base:
                deps["llm"] = _http_probe(f"{llm_base}/v1/models")
        except Exception as e:
            deps["llm"] = {"reachable": False, "error": str(e)}

        try:
            if getattr(settings.embedding, "use_llm_api", False):
                embed_base = (settings.embedding.api_base or settings.llm.base_url or "").rstrip("/")
                if embed_base:
                    deps["embedding"] = _http_probe(f"{embed_base}/v1/models")
            else:
                deps["embedding"] = _http_probe(f"http://localhost:{int(settings.embedding.port)}/api/version")
        except Exception as e:
            deps["embedding"] = {"reachable": False, "error": str(e)}

        try:
            if getattr(settings.mineru, "enabled", False):
                deps["mineru"] = _http_probe(f"http://localhost:{int(settings.mineru.port)}/health")
        except Exception as e:
            deps["mineru"] = {"reachable": False, "error": str(e)}

        return jsonify({"status": "ok", "papers": len(pids), "deps": deps}), 200
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
