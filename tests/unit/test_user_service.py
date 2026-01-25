"""Unit tests for user service functions."""

from __future__ import annotations


class TestGetTags:
    """Tests for get_tags function."""

    def test_get_tags_exists(self):
        """Test that get_tags function exists."""
        from backend.services.user_service import get_tags

        assert callable(get_tags)


class TestGetNegTags:
    """Tests for get_neg_tags function."""

    def test_get_neg_tags_exists(self):
        """Test that get_neg_tags function exists."""
        from backend.services.user_service import get_neg_tags

        assert callable(get_neg_tags)


class TestGetCombinedTags:
    """Tests for get_combined_tags function."""

    def test_get_combined_tags_exists(self):
        """Test that get_combined_tags function exists."""
        from backend.services.user_service import get_combined_tags

        assert callable(get_combined_tags)


class TestGetKeys:
    """Tests for get_keys function."""

    def test_get_keys_exists(self):
        """Test that get_keys function exists."""
        from backend.services.user_service import get_keys

        assert callable(get_keys)


class TestBuildUserTagList:
    """Tests for build_user_tag_list function."""

    def test_build_user_tag_list_exists(self):
        """Test that build_user_tag_list function exists."""
        from backend.services.user_service import build_user_tag_list

        assert callable(build_user_tag_list)


class TestBuildUserKeyList:
    """Tests for build_user_key_list function."""

    def test_build_user_key_list_exists(self):
        """Test that build_user_key_list function exists."""
        from backend.services.user_service import build_user_key_list

        assert callable(build_user_key_list)


class TestBuildUserCombinedTagList:
    """Tests for build_user_combined_tag_list function."""

    def test_build_user_combined_tag_list_exists(self):
        """Test that build_user_combined_tag_list function exists."""
        from backend.services.user_service import build_user_combined_tag_list

        assert callable(build_user_combined_tag_list)


class TestBeforeRequest:
    """Tests for before_request function."""

    def test_before_request_exists(self):
        """Test that before_request function exists."""
        from backend.services.user_service import before_request

        assert callable(before_request)


class TestCloseConnection:
    """Tests for close_connection function."""

    def test_close_connection_exists(self):
        """Test that close_connection function exists."""
        from backend.services.user_service import close_connection

        assert callable(close_connection)


class TestTemporaryUserContext:
    """Tests for temporary_user_context function."""

    def test_temporary_user_context_exists(self):
        """Test that temporary_user_context function exists."""
        from backend.services.user_service import temporary_user_context

        assert callable(temporary_user_context)
