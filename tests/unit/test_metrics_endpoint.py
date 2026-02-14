"""Unit tests for optional /metrics endpoint."""

from __future__ import annotations


def test_metrics_disabled_returns_404(client):
    resp = client.get("/metrics")
    assert resp.status_code == 404


def test_metrics_enabled_returns_200(monkeypatch):
    import config
    from backend import create_app

    old_enabled = bool(getattr(config.settings.web, "enable_metrics", False))
    old_key = str(getattr(config.settings.web, "metrics_key", "") or "")
    try:
        config.settings.web.enable_metrics = True
        config.settings.web.metrics_key = ""

        app = create_app()
        app.testing = True
        c = app.test_client()
        # First call generates the response before after_request hooks run, so custom
        # counters may only appear on subsequent scrapes.
        resp = c.get("/metrics")
        assert resp.status_code == 200
        resp2 = c.get("/metrics")
        assert resp2.status_code == 200
        body = resp2.get_data(as_text=True)
        assert "http_requests_total" in body
    finally:
        config.settings.web.enable_metrics = old_enabled
        config.settings.web.metrics_key = old_key


def test_metrics_key_enforced(monkeypatch):
    import config
    from backend import create_app

    old_enabled = bool(getattr(config.settings.web, "enable_metrics", False))
    old_key = str(getattr(config.settings.web, "metrics_key", "") or "")
    try:
        config.settings.web.enable_metrics = True
        config.settings.web.metrics_key = "secret"

        app = create_app()
        app.testing = True
        c = app.test_client()

        resp = c.get("/metrics")
        assert resp.status_code == 403

        resp2 = c.get("/metrics", headers={"X-ARXIV-SANITY-METRICS-KEY": "secret"})
        assert resp2.status_code == 200
        resp3 = c.get("/metrics", headers={"X-ARXIV-SANITY-METRICS-KEY": "secret"})
        assert resp3.status_code == 200
    finally:
        config.settings.web.enable_metrics = old_enabled
        config.settings.web.metrics_key = old_key
