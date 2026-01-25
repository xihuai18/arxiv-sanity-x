"""Live tests for web server.

These tests run against the actual web server when available.
Tests are automatically skipped if the web server is not running.
"""

from __future__ import annotations

import re

import pytest

from tests.service_detection import (
    get_vars_config,
    is_web_server_available,
    requires_web_server,
)


class TestWebServerAvailability:
    """Tests for web server availability detection."""

    def test_can_detect_web_server(self):
        """Test that we can detect web server status."""
        result = is_web_server_available()
        assert isinstance(result, bool)
        if result:
            config = get_vars_config()
            port = config.get("SERVE_PORT", 55555)
            print(f"✓ Web server is available at port {port}")
        else:
            print("✗ Web server is not available")


@requires_web_server
class TestWebServerLive:
    """Live tests for web server (requires running server)."""

    @pytest.fixture
    def base_url(self):
        """Get base URL for the web server."""
        config = get_vars_config()
        port = config.get("SERVE_PORT", 55555)
        return f"http://localhost:{port}"

    @pytest.fixture
    def session(self):
        """Create a requests session."""
        import requests

        return requests.Session()

    def test_homepage_loads(self, base_url, session):
        """Test that homepage loads successfully."""
        resp = session.get(f"{base_url}/", timeout=10)
        assert resp.status_code == 200
        assert "arxiv" in resp.text.lower()

    def test_about_page_loads(self, base_url, session):
        """Test that about page loads successfully."""
        resp = session.get(f"{base_url}/about", timeout=10)
        assert resp.status_code == 200

    def test_stats_page_loads(self, base_url, session):
        """Test that stats page loads successfully."""
        resp = session.get(f"{base_url}/stats", timeout=10)
        assert resp.status_code == 200

    def test_csrf_token_present(self, base_url, session):
        """Test that CSRF token is present in pages."""
        resp = session.get(f"{base_url}/about", timeout=10)
        assert resp.status_code == 200
        assert re.search(r'csrf-token"\s+content="[^"]+"', resp.text)

    def test_api_llm_models(self, base_url, session):
        """Test LLM models API endpoint."""
        resp = session.get(f"{base_url}/api/llm_models", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_api_queue_stats(self, base_url, session):
        """Test queue stats API endpoint."""
        resp = session.get(f"{base_url}/api/queue_stats", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True

    def test_api_keyword_search(self, base_url, session):
        """Test keyword search API endpoint."""
        resp = session.post(
            f"{base_url}/api/keyword_search",
            json={"keyword": "transformer", "limit": 5},
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True
        assert "pids" in data


@requires_web_server
class TestWebServerAuthLive:
    """Live tests for web server authentication (requires running server)."""

    @pytest.fixture
    def base_url(self):
        """Get base URL for the web server."""
        config = get_vars_config()
        port = config.get("SERVE_PORT", 55555)
        return f"http://localhost:{port}"

    @pytest.fixture
    def session(self):
        """Create a requests session."""
        import requests

        return requests.Session()

    def _get_csrf_token(self, session, base_url) -> str:
        """Get CSRF token from a page."""
        resp = session.get(f"{base_url}/about")
        match = re.search(r'csrf-token"\s+content="([^"]+)"', resp.text)
        return match.group(1) if match else ""

    def test_login_flow(self, base_url, session):
        """Test login flow."""
        csrf = self._get_csrf_token(session, base_url)
        assert csrf, "CSRF token not found"

        # Login
        resp = session.post(
            f"{base_url}/login",
            data={"username": "live_test_user"},
            headers={"X-CSRF-Token": csrf},
            allow_redirects=False,
        )
        assert resp.status_code in [200, 302, 303]

        # Check user state
        resp = session.get(f"{base_url}/api/user_state")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True

    def test_authenticated_api_access(self, base_url, session):
        """Test authenticated API access."""
        csrf = self._get_csrf_token(session, base_url)

        # Login
        session.post(
            f"{base_url}/login",
            data={"username": "live_test_user"},
            headers={"X-CSRF-Token": csrf},
        )

        # Access reading list
        resp = session.get(f"{base_url}/api/readinglist/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True
        assert "items" in data
