from __future__ import annotations


def test_health_fails_when_fallback_missing(client, monkeypatch):
    # Ensure papers are "loaded" so /health reaches dependency checks.
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )
    # Avoid real network probes in /health.
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": True, "status_code": 200, "url": url},
    )

    # Fake LLM model list: missing glm-4.7
    monkeypatch.setattr(
        "backend.services.health_service.fetch_llm_model_ids",
        lambda base_url, api_key=None, timeout_s=1.0: {
            "ok": True,
            "url": "http://fake/v1/models",
            "model_ids": ["deepseek-v3.2"],
            "error": None,
        },
    )

    import backend.blueprints.web as web

    old_base = str(web.settings.llm.base_url or "")
    old_name = str(web.settings.llm.name or "")
    old_fb = str(web.settings.llm.fallback_models or "")
    try:
        web.settings.llm.base_url = "http://fake"
        web.settings.llm.name = "deepseek-v3.2"
        web.settings.llm.fallback_models = "glm-4.7"

        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.get_json() or {}
        assert data.get("status") == "error"
        deps = data.get("deps") or {}
        check = deps.get("llm_fallback_check") or {}
        assert check.get("ok") is False
        assert "glm-4.7" in (check.get("missing") or [])
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb


def test_health_ok_when_fallback_present(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": True, "status_code": 200, "url": url},
    )
    monkeypatch.setattr(
        "backend.services.health_service.fetch_llm_model_ids",
        lambda base_url, api_key=None, timeout_s=1.0: {
            "ok": True,
            "url": "http://fake/v1/models",
            "model_ids": ["deepseek-v3.2", "glm-4.7"],
            "error": None,
        },
    )

    import backend.blueprints.web as web

    old_base = str(web.settings.llm.base_url or "")
    old_name = str(web.settings.llm.name or "")
    old_fb = str(web.settings.llm.fallback_models or "")
    try:
        web.settings.llm.base_url = "http://fake"
        web.settings.llm.name = "deepseek-v3.2"
        web.settings.llm.fallback_models = "glm-4.7"

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "ok"
        deps = data.get("deps") or {}
        assert (deps.get("llm_fallback_check") or {}).get("ok") is True
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb
