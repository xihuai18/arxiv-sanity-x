"""Unit tests for tag service functions.

Tests tag service functions using mocks to avoid database dependencies.
"""

from __future__ import annotations

from unittest.mock import patch


class TestCreateEmptyTag:
    """Tests for create_empty_tag function."""

    def test_create_empty_tag_not_logged_in(self, app):
        """Test that create_empty_tag returns error when not logged in."""
        from backend.services.tag_service import create_empty_tag

        with app.app_context():
            from flask import g

            g.user = None
            result = create_empty_tag("test_tag")
            assert "not logged in" in result

    def test_create_empty_tag_invalid_name(self, app):
        """Test that create_empty_tag rejects invalid tag names."""
        from backend.services.tag_service import create_empty_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            # Empty tag name
            result = create_empty_tag("")
            assert "error" in result.lower() or "invalid" in result.lower()

    def test_create_empty_tag_reserved_name(self, app):
        """Test that create_empty_tag rejects reserved tag names."""
        from backend.services.tag_service import create_empty_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = create_empty_tag("all")
            assert "error" in result.lower() or "reserved" in result.lower()

    @patch("backend.services.tag_service.TagRepository")
    @patch("backend.services.tag_service.emit_user_event")
    def test_create_empty_tag_success(self, mock_emit, mock_repo, app):
        """Test successful tag creation."""
        from backend.services.tag_service import create_empty_tag

        mock_repo.create_tag.return_value = "ok"

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = create_empty_tag("valid_tag")
            assert result == "ok"
            mock_repo.create_tag.assert_called_once_with("test_user", "valid_tag")
            mock_emit.assert_called_once()


class TestAddPaperToTag:
    """Tests for add_paper_to_tag function."""

    def test_add_paper_to_tag_not_logged_in(self, app):
        """Test that add_paper_to_tag returns error when not logged in."""
        from backend.services.tag_service import add_paper_to_tag

        with app.app_context():
            from flask import g

            g.user = None
            result = add_paper_to_tag("2301.00001", "test_tag")
            assert "not logged in" in result

    def test_add_paper_to_tag_empty_pid(self, app):
        """Test that add_paper_to_tag rejects empty pid."""
        from backend.services.tag_service import add_paper_to_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_paper_to_tag("", "test_tag")
            assert "error" in result.lower()

    @patch("backend.services.tag_service.TagRepository")
    @patch("backend.services.tag_service.emit_user_event")
    def test_add_paper_to_tag_success(self, mock_emit, mock_repo, app):
        """Test successful paper addition to tag."""
        from backend.services.tag_service import add_paper_to_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = add_paper_to_tag("2301.00001", "valid_tag")
            assert result == "ok"
            mock_repo.add_paper_to_tag_and_remove_neg.assert_called_once()


class TestRemovePaperFromTag:
    """Tests for remove_paper_from_tag function."""

    def test_remove_paper_from_tag_not_logged_in(self, app):
        """Test that remove_paper_from_tag returns error when not logged in."""
        from backend.services.tag_service import remove_paper_from_tag

        with app.app_context():
            from flask import g

            g.user = None
            result = remove_paper_from_tag("2301.00001", "test_tag")
            assert "not logged in" in result

    def test_remove_paper_from_tag_empty_params(self, app):
        """Test that remove_paper_from_tag rejects empty parameters."""
        from backend.services.tag_service import remove_paper_from_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = remove_paper_from_tag("", "test_tag")
            assert "error" in result.lower()
            result = remove_paper_from_tag("2301.00001", "")
            assert "error" in result.lower()


class TestDeleteTag:
    """Tests for delete_tag function."""

    def test_delete_tag_not_logged_in(self, app):
        """Test that delete_tag returns error when not logged in."""
        from backend.services.tag_service import delete_tag

        with app.app_context():
            from flask import g

            g.user = None
            result = delete_tag("test_tag")
            assert "not logged in" in result

    def test_delete_tag_empty_name(self, app):
        """Test that delete_tag rejects empty tag name."""
        from backend.services.tag_service import delete_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = delete_tag("")
            assert "error" in result.lower()

    @patch("backend.services.tag_service.TagRepository")
    @patch("backend.services.tag_service.emit_user_event")
    def test_delete_tag_success(self, mock_emit, mock_repo, app):
        """Test successful tag deletion."""
        from backend.services.tag_service import delete_tag

        mock_repo.delete_tag_full.return_value = "ok"

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = delete_tag("valid_tag")
            assert result == "ok"
            mock_repo.delete_tag_full.assert_called_once_with("test_user", "valid_tag")


class TestRenameTag:
    """Tests for rename_tag function."""

    def test_rename_tag_not_logged_in(self, app):
        """Test that rename_tag returns error when not logged in."""
        from backend.services.tag_service import rename_tag

        with app.app_context():
            from flask import g

            g.user = None
            result = rename_tag("old_tag", "new_tag")
            assert "not logged in" in result

    def test_rename_tag_empty_params(self, app):
        """Test that rename_tag rejects empty parameters."""
        from backend.services.tag_service import rename_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = rename_tag("", "new_tag")
            assert "error" in result.lower()
            result = rename_tag("old_tag", "")
            assert "error" in result.lower()

    @patch("backend.services.tag_service.TagRepository")
    @patch("backend.services.tag_service.emit_user_event")
    def test_rename_tag_success(self, mock_emit, mock_repo, app):
        """Test successful tag rename."""
        from backend.services.tag_service import rename_tag

        mock_repo.rename_tag_full.return_value = "ok"

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = rename_tag("old_tag", "new_tag")
            assert result == "ok"
            mock_repo.rename_tag_full.assert_called_once_with("test_user", "old_tag", "new_tag")


class TestCreateCombinedTag:
    """Tests for create_combined_tag function."""

    def test_create_combined_tag_not_logged_in(self, app):
        """Test that create_combined_tag returns error when not logged in."""
        from backend.services.tag_service import create_combined_tag

        with app.app_context():
            from flask import g

            g.user = None
            result = create_combined_tag("tag1,tag2")
            assert "not logged in" in result

    def test_create_combined_tag_empty(self, app):
        """Test that create_combined_tag rejects empty ctag."""
        from backend.services.tag_service import create_combined_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = create_combined_tag("")
            assert "error" in result.lower()

    def test_create_combined_tag_null(self, app):
        """Test that create_combined_tag rejects 'null' ctag."""
        from backend.services.tag_service import create_combined_tag

        with app.app_context():
            from flask import g

            g.user = "test_user"
            result = create_combined_tag("null")
            assert "error" in result.lower() or "cannot" in result.lower()
