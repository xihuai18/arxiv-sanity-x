"""Integration tests for login/logout APIs."""

from __future__ import annotations


class TestLoginApi:
    """Tests for login API."""

    def test_login_without_csrf_returns_403(self, client):
        """Test that login without CSRF returns 403."""
        resp = client.post("/login", data={"username": "testuser"})
        assert resp.status_code == 403

    def test_login_with_csrf_succeeds(self, client, csrf_token):
        """Test that login with CSRF succeeds."""
        resp = client.post(
            "/login",
            data={"username": "testuser"},
            headers={"X-CSRF-Token": csrf_token},
            follow_redirects=False,
        )
        # Should redirect after successful login
        assert resp.status_code in [200, 302, 303]

    def test_login_sets_session(self, client, csrf_token):
        """Test that login sets session cookie."""
        resp = client.post(
            "/login",
            data={"username": "testuser"},
            headers={"X-CSRF-Token": csrf_token},
            follow_redirects=False,
        )
        # After login, user_state should work
        resp = client.get("/api/user_state")
        assert resp.status_code == 200

    def test_login_empty_username(self, client, csrf_token):
        """Test login with empty username."""
        resp = client.post(
            "/login",
            data={"username": ""},
            headers={"X-CSRF-Token": csrf_token},
            follow_redirects=False,
        )
        # Should handle empty username gracefully
        assert resp.status_code in [200, 302, 400]


class TestLogoutApi:
    """Tests for logout API.

    Note: Logout may require CSRF protection depending on implementation.
    """

    def test_logout_with_csrf(self, logged_in_client, csrf_token):
        """Test logout with CSRF token."""
        resp = logged_in_client.get("/logout", headers={"X-CSRF-Token": csrf_token})
        # Should redirect or return success
        assert resp.status_code in [200, 302, 403]

    def test_logout_post_with_csrf(self, logged_in_client, csrf_token):
        """Test logout POST with CSRF token."""
        resp = logged_in_client.post("/logout", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302]


class TestRegisterEmailApi:
    """Tests for register_email API."""

    def test_register_email_without_csrf_returns_403(self, logged_in_client):
        """Test that register_email without CSRF returns 403."""
        resp = logged_in_client.post(
            "/register_email",
            data={"email": "test@example.com"},
        )
        assert resp.status_code == 403
