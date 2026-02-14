"""Unit tests for auth_service username validation boundary values."""

from __future__ import annotations

from unittest.mock import patch

from flask import g, session


class TestLoginUserUsernameValidation:
    """Tests for login_user username regex: ^[a-zA-Z0-9_]{3,64}$"""

    # --- Valid usernames ---

    def test_accepts_3_char_username(self, app):
        """Minimum length boundary: exactly 3 characters."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            login_user("abc")
            assert session.get("user") == "abc"

    def test_accepts_64_char_username(self, app):
        """Maximum length boundary: exactly 64 characters."""
        from backend.services.auth_service import login_user

        name = "a" * 64
        with app.test_request_context("/login", method="POST"):
            g.user = None
            login_user(name)
            assert session.get("user") == name

    def test_accepts_underscore_username(self, app):
        """Underscores are allowed."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            login_user("foo_bar")
            assert session.get("user") == "foo_bar"

    def test_accepts_numeric_start_username(self, app):
        """Digits at the start are allowed."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            login_user("123user")
            assert session.get("user") == "123user"

    # --- Invalid usernames ---

    def test_rejects_2_char_username(self, app):
        """Below minimum length: 2 characters."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            with patch("backend.services.auth_service.flash") as mock_flash:
                login_user("ab")
                mock_flash.assert_called_once()
            assert "user" not in session

    def test_rejects_65_char_username(self, app):
        """Above maximum length: 65 characters."""
        from backend.services.auth_service import login_user

        name = "a" * 65
        with app.test_request_context("/login", method="POST"):
            g.user = None
            with patch("backend.services.auth_service.flash") as mock_flash:
                login_user(name)
                mock_flash.assert_called_once()
            assert "user" not in session

    def test_rejects_script_tag(self, app):
        """XSS attempt: <script> in username."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            with patch("backend.services.auth_service.flash") as mock_flash:
                login_user("<script>")
                mock_flash.assert_called_once()
            assert "user" not in session

    def test_rejects_path_traversal(self, app):
        """Path traversal attempt: ../ in username."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            with patch("backend.services.auth_service.flash") as mock_flash:
                login_user("../etc")
                mock_flash.assert_called_once()
            assert "user" not in session

    def test_rejects_space_in_username(self, app):
        """Spaces are not allowed."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            with patch("backend.services.auth_service.flash") as mock_flash:
                login_user("foo bar")
                mock_flash.assert_called_once()
            assert "user" not in session

    def test_rejects_chinese_chars(self, app):
        """Non-ASCII characters are not allowed."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            with patch("backend.services.auth_service.flash") as mock_flash:
                login_user("用户名测试")
                mock_flash.assert_called_once()
            assert "user" not in session

    def test_rejects_empty_string(self, app):
        """Empty string should not set session."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            login_user("")
            assert "user" not in session

    def test_rejects_none(self, app):
        """None should not set session."""
        from backend.services.auth_service import login_user

        with app.test_request_context("/login", method="POST"):
            g.user = None
            login_user(None)
            assert "user" not in session
