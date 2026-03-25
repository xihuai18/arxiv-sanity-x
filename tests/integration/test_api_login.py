"""Integration tests for login/logout APIs."""

from __future__ import annotations

import pytest


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

    def test_login_accepts_json_body(self, client, csrf_token):
        resp = client.post(
            "/login",
            json={"username": "jsonuser"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("user") == "jsonuser"

    @pytest.mark.parametrize("body", ["123", '"abc"', "true", "[1, 2]"])
    def test_login_non_object_json_returns_400(self, client, csrf_token, body):
        resp = client.post(
            "/login",
            data=body,
            content_type="application/json",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is False
        assert payload.get("error") == "Request body must be a JSON object"


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

    def test_register_email_with_csrf_but_no_login_returns_401_json(self, client, csrf_token):
        """Test that register_email returns JSON 401 when not logged in."""
        resp = client.post(
            "/register_email",
            data={"email": "test@example.com"},
            headers={"X-CSRF-Token": csrf_token},
            follow_redirects=False,
        )
        assert resp.status_code == 401

        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is False
        assert payload.get("error") == "Not logged in"

    def test_register_email_accepts_json_body(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/register_email",
            json={"email": "json@example.com"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert "json@example.com" in (payload.get("emails") or [])

    def test_register_email_invalid_json_with_existing_email_returns_400(self, logged_in_client, csrf_token):
        from aslite.repositories import UserRepository

        UserRepository.set_emails("test_user", ["old@example.com"])

        resp = logged_in_client.post(
            "/register_email",
            json={"email": "not-an-email"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is False
        assert UserRepository.get_emails("test_user") == ["old@example.com"]

    def test_register_email_json_missing_field_returns_400(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/register_email",
            json={},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        payload = resp.get_json(silent=True) or {}
        assert payload.get("error") == "email is required"

    def test_register_email_malformed_json_returns_400(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/register_email",
            data="{",
            content_type="application/json",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        payload = resp.get_json(silent=True) or {}
        assert payload.get("error") == "email is required"

    def test_register_email_json_rejects_non_string_value(self, logged_in_client, csrf_token):
        from aslite.repositories import UserRepository

        UserRepository.set_emails("test_user", ["old@example.com"])

        resp = logged_in_client.post(
            "/register_email",
            json={"email": ["bad@example.com"]},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        payload = resp.get_json(silent=True) or {}
        assert payload.get("error") == "email must be a string"
        assert UserRepository.get_emails("test_user") == ["old@example.com"]

    def test_register_email_json_null_returns_400(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/register_email",
            json={"email": None},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        payload = resp.get_json(silent=True) or {}
        assert payload.get("error") == "email must be a string"
