"""Unit tests for validation functions."""

from __future__ import annotations


class TestValidateTagName:
    """Tests for validate_tag_name function."""

    def test_validate_tag_name_rejects_empty(self):
        """Test that empty tag is rejected."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("")
        assert error is not None

    def test_validate_tag_name_rejects_reserved_all(self):
        """Test that 'all' is rejected as reserved."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("all")
        assert error is not None

    def test_validate_tag_name_accepts_valid(self):
        """Test that valid tag names are accepted."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("valid_tag")
        assert error is None

    def test_validate_tag_name_accepts_with_numbers(self):
        """Test that tag names with numbers are accepted."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("tag123")
        assert error is None

    def test_validate_tag_name_rejects_forward_slash(self):
        """Tag names with forward slash are rejected."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("tag/")
        assert error is not None
        assert "slash" in error.lower()

    def test_validate_tag_name_rejects_backslash(self):
        """Tag names with backslash are rejected."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("tag\\name")
        assert error is not None
        assert "slash" in error.lower()

    def test_validate_tag_name_rejects_slash_in_middle(self):
        """Tag names with slash in the middle are rejected."""
        from backend.utils.validation import validate_tag_name

        error = validate_tag_name("tag/name")
        assert error is not None
        assert "slash" in error.lower()


class TestValidateKeywordName:
    """Tests for validate_keyword_name function."""

    def test_validate_keyword_name_rejects_empty(self):
        """Test that empty keyword is rejected."""
        from backend.utils.validation import validate_keyword_name

        error = validate_keyword_name("")
        assert error is not None

    def test_validate_keyword_name_rejects_null(self):
        """Test that 'null' is rejected."""
        from backend.utils.validation import validate_keyword_name

        error = validate_keyword_name("null")
        assert error is not None

    def test_validate_keyword_name_accepts_valid(self):
        """Test that valid keyword names are accepted."""
        from backend.utils.validation import validate_keyword_name

        error = validate_keyword_name("valid_keyword")
        assert error is None

    def test_validate_keyword_name_rejects_slash(self):
        """Keyword names with slashes are rejected."""
        from backend.utils.validation import validate_keyword_name

        error = validate_keyword_name("key/word")
        assert error is not None
        assert "slash" in error.lower()

    def test_validate_keyword_name_rejects_backslash(self):
        """Keyword names with backslash are rejected."""
        from backend.utils.validation import validate_keyword_name

        error = validate_keyword_name("key\\word")
        assert error is not None
        assert "slash" in error.lower()


class TestApiHelpers:
    """Tests for API helper functions.

    Note: api_error and api_success use Flask's jsonify which requires
    an application context. These tests verify the functions exist and
    are callable. Full integration tests cover actual behavior.
    """

    def test_api_error_exists(self):
        """Test that api_error function exists."""
        from backend.services.api_helpers import api_error

        assert callable(api_error)

    def test_api_success_exists(self):
        """Test that api_success function exists."""
        from backend.services.api_helpers import api_success

        assert callable(api_success)
