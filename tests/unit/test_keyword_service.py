"""Unit tests for keyword service functions.

Tests keyword service functions using mocks to avoid database dependencies.
"""

from __future__ import annotations

from unittest.mock import patch


class TestAddKeyword:
    """Tests for add_keyword function."""

    def test_add_keyword_not_logged_in(self, app):
        """Test that add_keyword returns error when not logged in."""
        from backend.services.keyword_service import add_keyword

        with app.app_context():
            from flask import g

            g.user = None
            result = add_keyword("test_keyword")
            assert "not logged in" in result

    def test_add_keyword_empty(self, app):
        """Test that add_keyword rejects empty keyword."""
        from backend.services.keyword_service import add_keyword

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_keyword("")
            assert "error" in result.lower() or "invalid" in result.lower()

    @patch("backend.services.keyword_service.KeywordRepository")
    @patch("backend.services.keyword_service.emit_user_event")
    def test_add_keyword_success(self, mock_emit, mock_repo, app):
        """Test successful keyword addition."""
        from backend.services.keyword_service import add_keyword

        mock_repo.get_user_keywords.return_value = {}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_keyword("machine learning")
            assert result == "ok"
            mock_repo.add_keyword.assert_called_once_with("test_user", "machine learning")
            mock_emit.assert_called_once()

    @patch("backend.services.keyword_service.KeywordRepository")
    def test_add_keyword_duplicate(self, mock_repo, app):
        """Test that add_keyword rejects duplicate keywords."""
        from backend.services.keyword_service import add_keyword

        mock_repo.get_user_keywords.return_value = {"machine learning": True}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_keyword("machine learning")
            assert "repeated" in result.lower()


class TestDeleteKeyword:
    """Tests for delete_keyword function."""

    def test_delete_keyword_not_logged_in(self, app):
        """Test that delete_keyword returns error when not logged in."""
        from backend.services.keyword_service import delete_keyword

        with app.app_context():
            from flask import g

            g.user = None
            result = delete_keyword("test_keyword")
            assert "not logged in" in result

    def test_delete_keyword_empty(self, app):
        """Test that delete_keyword rejects empty keyword."""
        from backend.services.keyword_service import delete_keyword

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = delete_keyword("")
            assert "error" in result.lower()

    @patch("backend.services.keyword_service.KeywordRepository")
    @patch("backend.services.keyword_service.emit_user_event")
    def test_delete_keyword_success(self, mock_emit, mock_repo, app):
        """Test successful keyword deletion."""
        from backend.services.keyword_service import delete_keyword

        mock_repo.get_user_keywords.return_value = {"machine learning": True}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = delete_keyword("machine learning")
            assert result == "ok"
            mock_repo.remove_keyword.assert_called_once_with("test_user", "machine learning")
            mock_emit.assert_called_once()

    @patch("backend.services.keyword_service.KeywordRepository")
    def test_delete_keyword_not_found(self, mock_repo, app):
        """Test that delete_keyword handles non-existent keyword."""
        from backend.services.keyword_service import delete_keyword

        mock_repo.get_user_keywords.return_value = {"other_keyword": True}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = delete_keyword("machine learning")
            assert "does not have" in result.lower()

    @patch("backend.services.keyword_service.KeywordRepository")
    def test_delete_keyword_no_library(self, mock_repo, app):
        """Test that delete_keyword handles user without library."""
        from backend.services.keyword_service import delete_keyword

        mock_repo.get_user_keywords.return_value = {}

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = delete_keyword("machine learning")
            assert "does not have" in result.lower()


class TestRenameKeyword:
    """Tests for rename_keyword function."""

    def test_rename_keyword_not_logged_in(self, app):
        """Test that rename_keyword returns error when not logged in."""
        from backend.services.keyword_service import rename_keyword

        with app.app_context():
            from flask import g

            g.user = None
            result = rename_keyword("old_keyword", "new_keyword")
            assert "not logged in" in result

    def test_rename_keyword_empty_params(self, app):
        """Test that rename_keyword rejects empty parameters."""
        from backend.services.keyword_service import rename_keyword

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = rename_keyword("", "new_keyword")
            assert "error" in result.lower()
            result = rename_keyword("old_keyword", "")
            assert "error" in result.lower()

    def test_rename_keyword_to_null(self, app):
        """Test that rename_keyword rejects 'null' as new name."""
        from backend.services.keyword_service import rename_keyword

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = rename_keyword("old_keyword", "null")
            assert "error" in result.lower() or "protected" in result.lower()

    def test_rename_keyword_same_name(self, app):
        """Test that rename_keyword handles same name gracefully."""
        from backend.services.keyword_service import rename_keyword

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = rename_keyword("keyword", "keyword")
            assert result == "ok"

    @patch("backend.services.keyword_service.KeywordRepository")
    @patch("backend.services.keyword_service.emit_user_event")
    def test_rename_keyword_success(self, mock_emit, mock_repo, app):
        """Test successful keyword rename."""
        from backend.services.keyword_service import rename_keyword

        mock_repo.rename_keyword.return_value = "ok"

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = rename_keyword("old_keyword", "new_keyword")
            assert result == "ok"
            mock_repo.rename_keyword.assert_called_once_with("test_user", "old_keyword", "new_keyword")
            mock_emit.assert_called_once()
