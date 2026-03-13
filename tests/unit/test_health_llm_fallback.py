from __future__ import annotations


def _setup_common_health_mocks(monkeypatch, *, pids: list[str]):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": pids},
    )
    # Avoid real dependency probes in health endpoints.
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": True, "status_code": 200, "url": url},
    )


def test_ready_fails_when_fallback_missing(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
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

        resp = client.get("/ready")
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


def test_ready_ok_when_fallback_present(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
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

        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "ok"
        deps = data.get("deps") or {}
        assert (deps.get("llm_fallback_check") or {}).get("ok") is True
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb


def test_health_degraded_when_fallback_missing(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
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
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "degraded"
        deps = data.get("deps") or {}
        check = deps.get("llm_fallback_check") or {}
        assert check.get("ok") is False
        assert "glm-4.7" in (check.get("missing") or [])
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb


def test_health_loading_when_no_papers(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=[])
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
    try:
        web.settings.llm.base_url = "http://fake"
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "loading"
    finally:
        web.settings.llm.base_url = old_base


def test_ready_loading_when_no_papers(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=[])
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
    try:
        web.settings.llm.base_url = "http://fake"
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.get_json() or {}
        assert data.get("status") == "loading"
    finally:
        web.settings.llm.base_url = old_base


def test_ready_fails_when_embedding_unreachable(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": False, "error": "connection refused"},
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

        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.get_json() or {}
        assert data.get("status") == "error"
        issues = data.get("issues") or []
        assert any("Embedding service is unreachable" in str(x) for x in issues)
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb


def test_ready_ok_when_embedding_unreachable_but_not_required(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": False, "error": "connection refused"},
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
    old_require_embedding = bool(getattr(web.settings.web, "ready_require_embedding", True))
    try:
        web.settings.llm.base_url = "http://fake"
        web.settings.llm.name = "deepseek-v3.2"
        web.settings.llm.fallback_models = "glm-4.7"
        web.settings.web.ready_require_embedding = False

        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "ok"
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb
        web.settings.web.ready_require_embedding = old_require_embedding


def test_ready_ok_when_mineru_unreachable_but_not_required(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )

    def _probe(url, timeout_s=1.0):
        if url.endswith("/health"):
            return {"reachable": False, "error": "mineru down"}
        return {"reachable": True, "status_code": 200}

    monkeypatch.setattr("backend.blueprints.web._http_probe", _probe)
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
    old_enabled = bool(web.settings.mineru.enabled)
    old_backend = str(web.settings.mineru.backend or "")
    old_require_mineru = bool(getattr(web.settings.web, "ready_require_mineru", True))
    try:
        web.settings.llm.base_url = "http://fake"
        web.settings.llm.name = "deepseek-v3.2"
        web.settings.llm.fallback_models = "glm-4.7"
        web.settings.mineru.enabled = True
        web.settings.mineru.backend = "vlm-http-client"
        web.settings.web.ready_require_mineru = False

        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "ok"
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb
        web.settings.mineru.enabled = old_enabled
        web.settings.mineru.backend = old_backend
        web.settings.web.ready_require_mineru = old_require_mineru


def test_ready_skips_local_mineru_probe_for_api_backend(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )

    def _probe(url, timeout_s=1.0):
        if url.endswith("/health"):
            return {"reachable": False, "error": "unexpected local probe"}
        return {"reachable": True, "status_code": 200}

    monkeypatch.setattr("backend.blueprints.web._http_probe", _probe)
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
    old_enabled = bool(web.settings.mineru.enabled)
    old_backend = str(web.settings.mineru.backend or "")
    old_api_key = str(web.settings.mineru.api_key or "")
    try:
        web.settings.llm.base_url = "http://fake"
        web.settings.llm.name = "deepseek-v3.2"
        web.settings.llm.fallback_models = "glm-4.7"
        web.settings.mineru.enabled = True
        web.settings.mineru.backend = "api"
        web.settings.mineru.api_key = "fake-key"

        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.get_json() or {}
        assert data.get("status") == "ok"
        deps = data.get("deps") or {}
        assert (deps.get("mineru") or {}).get("mode") == "api"
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb
        web.settings.mineru.enabled = old_enabled
        web.settings.mineru.backend = old_backend
        web.settings.mineru.api_key = old_api_key


def test_ready_fails_when_embedding_llm_api_base_missing(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": True, "status_code": 200},
    )

    import backend.blueprints.web as web

    old_llm_base = str(web.settings.llm.base_url or "")
    old_use_llm_api = bool(web.settings.embedding.use_llm_api)
    old_embed_base = str(web.settings.embedding.api_base or "")
    try:
        web.settings.llm.base_url = ""
        web.settings.embedding.use_llm_api = True
        web.settings.embedding.api_base = ""

        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.get_json() or {}
        assert data.get("status") == "error"
        issues = data.get("issues") or []
        assert any("Embedding service is unreachable" in str(x) for x in issues)
    finally:
        web.settings.llm.base_url = old_llm_base
        web.settings.embedding.use_llm_api = old_use_llm_api
        web.settings.embedding.api_base = old_embed_base


def test_ready_fails_when_mineru_api_key_missing(client, monkeypatch):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": ["2301.00001"]},
    )
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": True, "status_code": 200},
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
    old_enabled = bool(web.settings.mineru.enabled)
    old_backend = str(web.settings.mineru.backend or "")
    old_api_key = str(web.settings.mineru.api_key or "")
    try:
        web.settings.llm.base_url = "http://fake"
        web.settings.llm.name = "deepseek-v3.2"
        web.settings.llm.fallback_models = "glm-4.7"
        web.settings.mineru.enabled = True
        web.settings.mineru.backend = "api"
        web.settings.mineru.api_key = ""

        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.get_json() or {}
        assert data.get("status") == "error"
        issues = data.get("issues") or []
        assert any("MinerU service is unreachable" in str(x) for x in issues)
    finally:
        web.settings.llm.base_url = old_base
        web.settings.llm.name = old_name
        web.settings.llm.fallback_models = old_fb
        web.settings.mineru.enabled = old_enabled
        web.settings.mineru.backend = old_backend
        web.settings.mineru.api_key = old_api_key
