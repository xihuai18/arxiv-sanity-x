"""Live tests for the OpenCode text-model service."""

from __future__ import annotations

import pytest

from tests.service_detection import (
    _opencode_auth,
    get_available_text_models,
    is_opencode_service_available,
    requires_opencode_service,
)


class TestOpenCodeServiceAvailability:
    def test_can_detect_opencode_service(self):
        result = is_opencode_service_available()
        assert isinstance(result, bool)
        if result:
            models = get_available_text_models()
            print(f"OpenCode service is available. Models: {models}")


@requires_opencode_service
class TestOpenCodeServiceLive:
    def test_opencode_health_endpoint(self):
        import requests

        from tests.service_detection import get_vars_config

        config = get_vars_config()
        base_url = str(config.get("OPENCODE_BASE_URL", "http://127.0.0.1:53000") or "").rstrip("/")

        resp = requests.get(f"{base_url}/global/health", timeout=5, auth=_opencode_auth())
        assert resp.status_code == 200
        assert (resp.json() or {}).get("healthy") is True

    def test_opencode_providers_endpoint(self):
        import requests

        from tests.service_detection import get_vars_config

        config = get_vars_config()
        base_url = str(config.get("OPENCODE_BASE_URL", "http://127.0.0.1:53000") or "").rstrip("/")

        resp = requests.get(f"{base_url}/config/providers", timeout=5, auth=_opencode_auth())
        assert resp.status_code == 200
        payload = resp.json()
        assert isinstance(payload, dict)
        assert isinstance(payload.get("providers"), list)

    def test_opencode_has_models(self):
        models = get_available_text_models()
        assert isinstance(models, list)

    def test_opencode_message_roundtrip(self):
        import requests

        from tests.service_detection import get_vars_config

        config = get_vars_config()
        base_url = str(config.get("OPENCODE_BASE_URL", "http://127.0.0.1:53000") or "").rstrip("/")
        models = get_available_text_models()
        if not models:
            pytest.skip("No OpenCode models available")

        provider_id, model_id = models[0].split("/", 1)
        session = requests.post(
            f"{base_url}/session",
            json={"title": "live test"},
            timeout=10,
            auth=_opencode_auth(),
        )
        session.raise_for_status()
        session_id = (session.json() or {}).get("id")
        assert session_id

        try:
            resp = requests.post(
                f"{base_url}/session/{session_id}/message",
                json={
                    "model": {"providerID": provider_id, "modelID": model_id},
                    "parts": [{"type": "text", "text": "Reply with the single word test."}],
                },
                timeout=30,
                auth=_opencode_auth(),
            )
            assert resp.status_code == 200
            payload = resp.json()
            assert isinstance(payload, dict)
            assert isinstance(payload.get("parts"), list)
        finally:
            requests.delete(f"{base_url}/session/{session_id}", timeout=10, auth=_opencode_auth())
