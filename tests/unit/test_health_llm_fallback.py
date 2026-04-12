from __future__ import annotations


def _setup_common_health_mocks(monkeypatch, *, pids: list[str]):
    monkeypatch.setattr(
        "backend.services.data_service.get_data_cached",
        lambda wait=False: {"pids": pids},
    )
    monkeypatch.setattr(
        "backend.blueprints.web._http_probe",
        lambda url, timeout_s=1.0: {"reachable": True, "status_code": 200, "url": url},
    )


def test_ready_fails_when_required_opencode_model_is_missing(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": False,
            "service": {"reachable": True, "healthy": True, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {
                "required": ["openai/gpt-5.4", "anthropic/claude-sonnet-4-6"],
                "missing": ["anthropic/claude-sonnet-4-6"],
            },
            "models": {"available_count": 1},
        },
    )

    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.get_json() or {}
    assert "Required OpenCode models are missing" in str(data.get("message") or "")


def test_health_degraded_when_required_opencode_model_is_missing(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": False,
            "service": {"reachable": True, "healthy": True, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {
                "required": ["openai/gpt-5.4", "anthropic/claude-sonnet-4-6"],
                "missing": ["anthropic/claude-sonnet-4-6"],
            },
            "models": {"available_count": 1},
        },
    )

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json() or {}
    assert data.get("status") == "degraded"
    assert any(
        "Required OpenCode models are missing" in str(item)
        for item in (data.get("warnings") or [])
    )


def test_ready_fails_when_opencode_authentication_fails(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": False,
            "service": {"reachable": True, "healthy": False, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {"required": ["openai/gpt-5.4"], "missing": []},
            "models": {"available_count": 10},
            "error_kind": "auth_failed",
            "error": "401 Unauthorized",
        },
    )

    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.get_json() or {}
    assert data.get("message") == "OpenCode authentication failed"


def test_ready_loading_when_no_papers(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=[])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": True,
            "service": {"reachable": True, "healthy": True, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {"required": ["openai/gpt-5.4"], "missing": []},
            "models": {"available_count": 10},
        },
    )

    resp = client.get("/ready")
    assert resp.status_code == 503
    assert (resp.get_json() or {}).get("status") == "loading"


def test_ready_probe_resolved_model_is_display_alias(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": True,
            "service": {"reachable": True, "healthy": True, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {"required": ["gpt-5.4"], "missing": []},
            "models": {"available_count": 1},
            "probe": {"ok": True, "resolved_model": "openai/gpt-5.4"},
        },
    )

    resp = client.get("/ready")

    assert resp.status_code == 200
    data = resp.get_json() or {}
    assert ((data.get("deps") or {}).get("llm_probe") or {}).get(
        "resolved_model"
    ) == "gpt-5.4"


def test_ready_fails_when_embedding_api_base_missing(client, monkeypatch):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": True,
            "service": {"reachable": True, "healthy": True, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {"required": ["openai/gpt-5.4"], "missing": []},
            "models": {"available_count": 10},
        },
    )

    import backend.blueprints.web as web

    old_use_llm_api = web.settings.embedding.use_llm_api
    old_api_base = str(web.settings.embedding.api_base or "")
    try:
        web.settings.embedding.use_llm_api = True
        web.settings.embedding.api_base = ""

        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.get_json() or {}
        issues = data.get("issues") or []
        assert any("Embedding service is unreachable" in str(item) for item in issues)
    finally:
        web.settings.embedding.use_llm_api = old_use_llm_api
        web.settings.embedding.api_base = old_api_base


def test_ready_reports_papers_db_file_when_path_is_string(
    client, monkeypatch, tmp_path
):
    _setup_common_health_mocks(monkeypatch, pids=["2301.00001"])
    monkeypatch.setattr(
        "backend.services.health_service.check_text_model_service",
        lambda *args, **kwargs: {
            "ok": True,
            "service": {"reachable": True, "healthy": True, "version": "1.4.3"},
            "base_url": "http://fake-opencode",
            "required": {"required": ["gpt-5.4"], "missing": []},
            "models": {"available_count": 10},
        },
    )

    db_path = tmp_path / "papers.db"
    db_path.write_text("", encoding="utf-8")
    monkeypatch.setattr("aslite.db.PAPERS_DB_FILE", str(db_path))

    resp = client.get("/ready")

    data = resp.get_json() or {}
    deps = data.get("deps") or {}
    assert (deps.get("papers_db_file") or {}).get("exists") is True
