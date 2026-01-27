"""Integration tests for Flask application setup."""

from __future__ import annotations

import pytest


class TestAppCreation:
    """Tests for Flask app creation."""

    def test_create_app_succeeds(self, app):
        """Test that create_app() succeeds."""
        assert app is not None

    def test_app_has_secret_key(self, app):
        """Test that app has a secret key."""
        assert app.secret_key is not None
        assert len(app.secret_key) > 0

    def test_app_is_in_testing_mode(self, app):
        """Test that app is in testing mode."""
        assert app.testing is True


class TestBlueprintRegistration:
    """Tests for blueprint registration."""

    def test_expected_blueprints_registered(self, app):
        """Test that all expected blueprints are registered."""
        expected_blueprints = {
            "web",
            "user",
            "sse",
            "summary",
            "search",
            "tags",
            "papers",
            "readinglist",
        }

        registered = set(app.blueprints.keys())
        missing = expected_blueprints - registered

        assert not missing, f"Missing blueprints: {sorted(missing)}"


class TestRouteRegistration:
    """Tests for route registration."""

    def test_minimum_route_count(self, app):
        """Test that minimum number of routes are registered."""
        routes = list(app.url_map.iter_rules())
        route_count = len(routes)

        # Expected minimum routes (excluding static)
        expected_min = 45
        assert route_count >= expected_min, f"Only {route_count} routes, expected >= {expected_min}"

    def test_critical_routes_exist(self, app):
        """Test that critical routes exist."""
        route_rules = {r.rule for r in app.url_map.iter_rules()}

        critical_routes = [
            "/",
            "/about",
            "/stats",
            "/profile",
            "/readinglist",
            "/summary",
            "/inspect",
            "/api/get_paper_summary",
            "/api/trigger_paper_summary",
            "/api/summary_status",
            "/api/readinglist/add",
            "/api/readinglist/remove",
            "/api/tag_feedback",
            "/api/keyword_search",
            "/api/user_stream",
        ]

        for route in critical_routes:
            assert route in route_rules, f"Missing route: {route}"


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_security_headers_present(self, client):
        """Test that security headers are present in responses."""
        resp = client.get("/")

        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Referrer-Policy",
        ]

        for header in expected_headers:
            assert header in resp.headers, f"Missing security header: {header}"

    def test_x_content_type_options_value(self, client):
        """Test X-Content-Type-Options header value."""
        resp = client.get("/about")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options_value(self, client):
        """Test X-Frame-Options header value."""
        resp = client.get("/about")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_cross_origin_resource_policy_value(self, client):
        """Test Cross-Origin-Resource-Policy header value."""
        resp = client.get("/about")
        assert resp.headers.get("Cross-Origin-Resource-Policy") == "same-origin"


class TestHtmlCacheHeaders:
    """Tests for HTML cache headers.

    HTML must not be cached to avoid stale CSRF tokens and stale hashed asset references.
    """

    @pytest.mark.parametrize("path", ["/", "/about", "/stats", "/profile", "/readinglist"])
    def test_html_pages_are_not_cached(self, client, path):
        resp = client.get(path)
        assert resp.mimetype == "text/html"
        cache_control = resp.headers.get("Cache-Control", "")
        assert "no-store" in cache_control


class TestPageRoutes:
    """Tests for page routes."""

    @pytest.mark.parametrize(
        "path,name",
        [
            ("/", "Home"),
            ("/about", "About"),
            ("/stats", "Stats"),
            ("/profile", "Profile"),
            ("/readinglist", "Reading List"),
        ],
    )
    def test_page_returns_200(self, client, path, name):
        """Test that page routes return 200."""
        resp = client.get(path)
        assert resp.status_code == 200, f"{name} page returned {resp.status_code}"

    def test_inspect_page_with_pid(self, client):
        """Test that inspect page with pid parameter returns 200."""
        resp = client.get("/inspect?pid=2301.00001")
        assert resp.status_code == 200

    def test_summary_page_with_pid(self, client):
        """Test that summary page with pid parameter returns 200."""
        resp = client.get("/summary?pid=2301.00001")
        assert resp.status_code == 200

    def test_summary_page_upload_pid_anonymous_no_leak(self, client):
        """Anonymous users should not learn whether an uploaded PID exists."""
        pid = "up_aaaaaaaaaaaa"
        resp = client.get(f"/summary?pid={pid}")
        assert resp.status_code == 404
        body = resp.get_data(as_text=True)
        assert pid not in body


class TestCsrfTokenInjection:
    """Tests for CSRF token injection."""

    def test_about_page_contains_csrf_token(self, client):
        """Test that about page contains CSRF token meta tag."""
        import re

        resp = client.get("/about")
        body = resp.get_data(as_text=True)

        assert re.search(r'<meta\s+name="csrf-token"\s+content="[^"]+"\s*>', body)

    def test_about_page_contains_hashed_assets(self, client):
        """Test that about page contains hashed static assets."""
        import re

        resp = client.get("/about")
        body = resp.get_data(as_text=True)

        # hashed_static may be unhashed (manifest absent) or hashed (manifest present)
        assert re.search(r"/static/dist/main(-[a-zA-Z0-9]{8,})?\.css", body)
        assert re.search(r"/static/dist/common_utils(-[a-zA-Z0-9]{8,})?\.js", body)


class TestCacheStatusEndpoint:
    """Tests for cache status endpoint."""

    def test_cache_status_disabled_by_default(self, client):
        """Test that cache_status is disabled by default."""
        resp = client.get("/cache_status")
        assert resp.status_code == 404
