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


def test_validate_user_email_input_rejects_invalid_value():
    from backend.services.auth_service import validate_user_email_input

    emails, is_valid = validate_user_email_input("not-an-email")

    assert emails == ["not-an-email"]
    assert is_valid is False


def test_validate_user_email_input_accepts_empty_string_for_clear():
    from backend.services.auth_service import validate_user_email_input

    emails, is_valid = validate_user_email_input("")

    assert emails == []
    assert is_valid is True


def test_validate_user_email_input_rejects_non_string_value():
    from backend.services.auth_service import validate_user_email_input

    emails, is_valid = validate_user_email_input(None)
    assert emails == []
    assert is_valid is False

    emails, is_valid = validate_user_email_input(123)  # type: ignore[arg-type]
    assert emails == []
    assert is_valid is False
