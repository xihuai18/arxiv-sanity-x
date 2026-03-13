"""Web page routes."""

from __future__ import annotations

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


def _health_response(*, strict_ready: bool) -> tuple[object, int]:
    """Build health/readiness response.

    - `/health`: non-strict liveness/degraded check (`strict_ready=False`)
    - `/ready`: strict readiness for launchers (`strict_ready=True`)
    """
    from aslite.db import PAPERS_DB_FILE
    from backend.services.data_service import get_data_cached

    try:
        require_embedding_ready = bool(getattr(settings.web, "ready_require_embedding", True))
        require_mineru_ready = bool(getattr(settings.web, "ready_require_mineru", True))
        mineru_backend = str(getattr(settings.mineru, "backend", "pipeline") or "pipeline").strip().lower()
        mineru_enabled = bool(getattr(settings.mineru, "enabled", False))
        # Local MinerU HTTP health probe only applies to the vLLM backend.
        require_local_mineru_probe = mineru_enabled and mineru_backend == "vlm-http-client"

        # Non-blocking peek: avoid /health being stuck on cold-start cache loading.
        data = get_data_cached(wait=False)
        pids = data.get("pids", [])

        deps: dict[str, object] = {}
        warnings: list[str] = []
        strict_dep_issues: list[str] = []

        if not pids:
            msg = "No papers loaded yet"
            if strict_ready:
                return jsonify({"status": "loading", "message": msg}), 503
            warnings.append(msg)

        # DB file presence is an operator hint.
        try:
            deps["papers_db_file"] = {
                "exists": bool(PAPERS_DB_FILE and PAPERS_DB_FILE.exists()),
            }
        except Exception as e:
            deps["papers_db_file"] = {"error": str(e)}

        # Optional dependency probes. In strict readiness mode, enforce that
        # default/fallback models exist in the /models list.
        try:
            llm_base = (settings.llm.base_url or "").rstrip("/")
            if llm_base:
                from backend.services.health_service import (
                    fetch_llm_model_ids,
                    validate_required_models,
                )

                result = fetch_llm_model_ids(llm_base, settings.llm.api_key, timeout_s=1.0)
                deps["llm"] = {
                    "reachable": bool(result.get("ok")),
                    "url": result.get("url"),
                    "error": result.get("error"),
                }
                deps["llm_models"] = result
                llm_issue: str | None = None
                if not result.get("ok"):
                    llm_issue = "Failed to fetch LLM model list for fallback validation"
                else:
                    model_ids = result.get("model_ids") or []
                    required = []
                    try:
                        required.append(str(settings.llm.name or "").strip())
                    except Exception:
                        pass
                    try:
                        required.extend(list(settings.llm.fallback_model_list or []))
                    except Exception:
                        pass
                    check = validate_required_models(required, list(model_ids))
                    deps["llm_fallback_check"] = check
                    if not check.get("ok"):
                        llm_issue = "Fallback model(s) missing from LLM model list"

                if llm_issue and strict_ready:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": llm_issue,
                                "deps": deps,
                            }
                        ),
                        503,
                    )
                if llm_issue:
                    warnings.append(llm_issue)
        except Exception as e:
            deps["llm"] = {"reachable": False, "error": str(e)}
            if strict_ready:
                strict_dep_issues.append("LLM probe failed")
            else:
                warnings.append("LLM probe failed")

        try:
            if getattr(settings.embedding, "use_llm_api", False):
                embed_base = (settings.embedding.api_base or settings.llm.base_url or "").rstrip("/")
                if embed_base:
                    deps["embedding"] = _http_probe(f"{embed_base}/v1/models")
                else:
                    deps["embedding"] = {
                        "reachable": False,
                        "error": "missing_embed_api_base",
                    }
            else:
                deps["embedding"] = _http_probe(f"http://localhost:{int(settings.embedding.port)}/api/version")
        except Exception as e:
            deps["embedding"] = {"reachable": False, "error": str(e)}
        embed_dep = deps.get("embedding")
        if require_embedding_ready and isinstance(embed_dep, dict) and embed_dep.get("reachable") is False:
            if strict_ready:
                strict_dep_issues.append("Embedding service is unreachable")
            else:
                warnings.append("Embedding service is unreachable")

        try:
            if require_local_mineru_probe:
                deps["mineru"] = _http_probe(f"http://localhost:{int(settings.mineru.port)}/health")
            elif mineru_enabled:
                if mineru_backend == "api":
                    api_key = str(getattr(settings.mineru, "api_key", "") or "").strip()
                    deps["mineru"] = {"reachable": bool(api_key), "mode": "api"}
                    if not api_key:
                        deps["mineru"]["error"] = "missing_api_key"
                elif mineru_backend == "pipeline":
                    deps["mineru"] = {"reachable": True, "mode": "pipeline"}
                else:
                    deps["mineru"] = {
                        "reachable": False,
                        "mode": mineru_backend,
                        "error": "unsupported_backend",
                    }
        except Exception as e:
            deps["mineru"] = {"reachable": False, "error": str(e)}
        mineru_dep = deps.get("mineru")
        if require_mineru_ready and isinstance(mineru_dep, dict) and mineru_dep.get("reachable") is False:
            if strict_ready:
                strict_dep_issues.append("MinerU service is unreachable")
            else:
                warnings.append("MinerU service is unreachable")

        if strict_ready and strict_dep_issues:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Dependency readiness checks failed",
                        "deps": deps,
                        "issues": strict_dep_issues,
                    }
                ),
                503,
            )

        status = "ok"
        if not pids:
            status = "loading"
        elif warnings:
            status = "degraded"
        payload: dict[str, object] = {
            "status": status,
            "papers": len(pids),
            "deps": deps,
        }
        if warnings:
            payload["warnings"] = warnings
        return jsonify(payload), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@bp.route("/health", methods=["GET"])
def health():
    """Non-strict health endpoint (liveness + degraded signals)."""
    return _health_response(strict_ready=False)


@bp.route("/ready", methods=["GET"])
def ready():
    """Strict readiness endpoint for launchers/automation."""
    return _health_response(strict_ready=True)


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
