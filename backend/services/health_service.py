"""Health-check helpers (LLM model list validation, etc.)."""

from __future__ import annotations

from typing import Any


def fetch_llm_model_ids(base_url: str, api_key: str | None = None, *, timeout_s: float = 1.0) -> dict[str, Any]:
    """Fetch LLM model ids from an OpenAI-compatible /models endpoint.

    Returns:
      {
        "ok": bool,
        "url": str | None,
        "model_ids": list[str] | None,
        "error": str | None,
      }
    """
    u = (base_url or "").rstrip("/")
    if not u:
        return {"ok": False, "url": None, "model_ids": None, "error": "empty_base_url"}

    candidate_urls: list[str] = []
    if u.endswith("/v1"):
        candidate_urls.append(f"{u}/models")
    else:
        candidate_urls.append(f"{u}/v1/models")
        candidate_urls.append(f"{u}/models")

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import requests

        s = requests.Session()
        s.trust_env = False
        last_err: str | None = None
        for url in candidate_urls:
            try:
                resp = s.get(url, headers=headers, timeout=(min(0.5, timeout_s), timeout_s))
                if resp.status_code == 404 and url != candidate_urls[-1]:
                    last_err = f"HTTP 404 for {url}"
                    continue
                resp.raise_for_status()
                payload = resp.json()
                items = payload.get("data", []) if isinstance(payload, dict) else []
                ids: list[str] = []
                for it in items:
                    try:
                        mid = (it or {}).get("id")
                    except Exception:
                        mid = None
                    if mid:
                        ids.append(str(mid))
                return {"ok": True, "url": url, "model_ids": ids, "error": None}
            except Exception as e:
                last_err = str(e)
                continue
        return {"ok": False, "url": None, "model_ids": None, "error": last_err or "unknown_error"}
    except Exception as e:
        return {"ok": False, "url": None, "model_ids": None, "error": str(e)}


def validate_required_models(required: list[str], model_ids: list[str]) -> dict[str, Any]:
    """Validate required models are present in the LLM model list."""
    req = [str(m or "").strip() for m in (required or [])]
    req = [m for m in req if m]
    ids = [str(m or "").strip() for m in (model_ids or [])]
    ids_set = set(ids)

    missing = [m for m in req if m not in ids_set]
    return {
        "ok": len(missing) == 0,
        "required": req,
        "missing": missing,
        "available_count": len(ids),
    }
