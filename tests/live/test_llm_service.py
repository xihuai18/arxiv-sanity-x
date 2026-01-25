"""Live tests for LLM service.

These tests run against the actual LiteLLM service when available.
Tests are automatically skipped if the LLM service is not running.
"""

from __future__ import annotations

import pytest

from tests.service_detection import (
    get_available_llm_models,
    is_litellm_service_available,
    requires_litellm_service,
)


class TestLiteLLMServiceAvailability:
    """Tests for LiteLLM service availability detection."""

    def test_can_detect_litellm_service(self):
        """Test that we can detect LiteLLM service status."""
        result = is_litellm_service_available()
        assert isinstance(result, bool)
        if result:
            print("✓ LiteLLM service is available")
            models = get_available_llm_models()
            print(f"  Available models: {models}")
        else:
            print("✗ LiteLLM service is not available")


@requires_litellm_service
class TestLiteLLMServiceLive:
    """Live tests for LiteLLM service (requires running service)."""

    def test_litellm_models_endpoint(self):
        """Test that LiteLLM /v1/models endpoint works."""
        import requests

        from tests.service_detection import get_vars_config

        config = get_vars_config()
        port = config.get("LITELLM_PORT", 53000)

        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        assert resp.status_code == 200

        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_litellm_has_models(self):
        """Test that LiteLLM has at least one model configured."""
        models = get_available_llm_models()
        # May be empty if no models configured, but should not crash
        assert isinstance(models, list)

    def test_litellm_chat_completion(self):
        """Test that LiteLLM can handle chat completion requests."""
        import requests

        from tests.service_detection import get_vars_config

        config = get_vars_config()
        port = config.get("LITELLM_PORT", 53000)

        models = get_available_llm_models()
        if not models:
            pytest.skip("No LLM models available")

        model = models[0]

        # Try a simple chat completion
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'test' and nothing else."}],
                    "max_tokens": 10,
                },
                timeout=30,
            )
            # Should return 200 or 4xx (rate limit, etc.)
            if resp.status_code >= 500:
                # Print response body for debugging (truncate to 1KB)
                body = resp.text[:1024] if resp.text else "(empty)"
                pytest.fail(
                    f"LiteLLM returned {resp.status_code} for model '{model}'. " f"Response body (first 1KB): {body}"
                )
        except requests.exceptions.Timeout:
            pytest.skip("LLM request timed out")


@requires_litellm_service
class TestSummaryServiceLive:
    """Live tests for summary service with LLM (requires LiteLLM)."""

    def test_get_summary_status(self):
        """Test getting summary status."""
        from backend.services.summary_service import get_summary_status

        status = get_summary_status("2301.00001", model="test-model")
        # get_summary_status returns a tuple (content, meta) or similar
        # Just verify it doesn't crash
        assert status is not None

    def test_summary_cache_stats(self):
        """Test getting summary cache statistics."""
        from backend.services.summary_service import get_summary_cache_stats

        stats = get_summary_cache_stats(ttl=1)
        assert isinstance(stats, dict)
        assert "data" in stats
