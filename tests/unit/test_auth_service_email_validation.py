"""Unit tests for auth_service email validation."""

from __future__ import annotations

from unittest.mock import patch

from flask import g


def test_register_user_email_accepts_long_tld(app):
    from backend.services.auth_service import register_user_email

    with app.test_request_context("/profile", method="POST"):
        g.user = "test_user"
        with patch("backend.services.auth_service.csrf_protect"):
            with patch("backend.services.auth_service.UserRepository.set_emails") as mock_set_emails:
                register_user_email("foo@bar.engineering")
                mock_set_emails.assert_called_once_with("test_user", ["foo@bar.engineering"])


def test_register_user_email_rejects_missing_tld(app):
    from backend.services.auth_service import register_user_email

    with app.test_request_context("/profile", method="POST"):
        g.user = "test_user"
        with patch("backend.services.auth_service.csrf_protect"):
            with patch("backend.services.auth_service.UserRepository.set_emails") as mock_set_emails:
                register_user_email("foo@bar")
                mock_set_emails.assert_not_called()


def test_register_user_email_accepts_multiple_emails(app):
    from backend.services.auth_service import register_user_email

    with app.test_request_context("/profile", method="POST"):
        g.user = "test_user"
        with patch("backend.services.auth_service.csrf_protect"):
            with patch("backend.services.auth_service.UserRepository.set_emails") as mock_set_emails:
                register_user_email("Foo@Bar.com\nanother@example.com, third@example.org")
                mock_set_emails.assert_called_once_with(
                    "test_user",
                    ["foo@bar.com", "another@example.com", "third@example.org"],
                )
