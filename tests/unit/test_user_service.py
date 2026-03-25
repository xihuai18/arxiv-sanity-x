"""Unit tests for user service functions."""

from __future__ import annotations


class TestGetTags:
    """Tests for get_tags function."""

    def test_get_tags_exists(self):
        """Test that get_tags function exists."""
        from backend.services.user_service import get_tags

        assert callable(get_tags)

    def test_get_tags_accepts_explicit_user(self, monkeypatch):
        from backend.services.user_service import get_tags

        monkeypatch.setattr(
            "backend.services.user_service.TagRepository.get_user_tags",
            lambda user: {"ml": {user}},
        )

        assert get_tags(user="alice") == {"ml": {"alice"}}


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

    def test_get_keys_accepts_explicit_user(self, monkeypatch):
        from backend.services.user_service import get_keys

        monkeypatch.setattr(
            "backend.services.user_service.KeywordRepository.get_user_keywords",
            lambda user: {"graph": {user}},
        )

        assert get_keys(user="alice") == {"graph": {"alice"}}


class TestBuildUserTagList:
    """Tests for build_user_tag_list function."""

    def test_build_user_tag_list_exists(self):
        """Test that build_user_tag_list function exists."""
        from backend.services.user_service import build_user_tag_list

        assert callable(build_user_tag_list)

    def test_build_pid_tag_reverse_index_filters_candidates(self):
        from backend.services.user_service import build_pid_tag_reverse_index

        pid_to_tags = build_pid_tag_reverse_index(
            {
                "alpha": {"p1", "p2"},
                "beta": ["p2", "p3"],
                "": {"p1"},
            },
            candidate_pids={"p2", "p3", "p4"},
        )

        assert pid_to_tags == {
            "p2": ["alpha", "beta"],
            "p3": ["beta"],
            "p4": [],
        }

    def test_build_pid_tag_reverse_index_respects_empty_candidate_set(self):
        from backend.services.user_service import build_pid_tag_reverse_index

        pid_to_tags = build_pid_tag_reverse_index(
            {
                "alpha": {"p1", "p2"},
                "beta": ["p2", "p3"],
            },
            candidate_pids=set(),
        )

        assert pid_to_tags == {}


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

    def test_temporary_user_context_swaps_and_restores_all_cached_user_fields(self, app, monkeypatch):
        from backend.services.user_service import temporary_user_context

        monkeypatch.setattr(
            "backend.services.user_service.TagRepository.get_user_tags",
            lambda user: {"tag": {user}},
        )
        monkeypatch.setattr(
            "backend.services.user_service.NegativeTagRepository.get_user_neg_tags",
            lambda user: {"neg": {user}},
        )
        monkeypatch.setattr(
            "backend.services.user_service.KeywordRepository.get_user_keywords",
            lambda user: {"kw": {user}},
        )
        monkeypatch.setattr(
            "backend.services.user_service.TagRepository.get_user_combined_tags",
            lambda user: {f"combo:{user}"},
        )

        with app.app_context():
            from flask import g

            g.user = "original"
            g._tags = {"orig": {"original"}}
            g._neg_tags = {"orig_neg": {"original"}}
            g._keys = {"orig_kw": {"original"}}
            g._combined_tags = {"combo:original"}

            with temporary_user_context("alice") as user_tags:
                assert g.user == "alice"
                assert user_tags == {"tag": {"alice"}}
                assert g._tags == {"tag": {"alice"}}
                assert g._neg_tags == {"neg": {"alice"}}
                assert g._keys == {"kw": {"alice"}}
                assert g._combined_tags == {"combo:alice"}

            assert g.user == "original"
            assert g._tags == {"orig": {"original"}}
            assert g._neg_tags == {"orig_neg": {"original"}}
            assert g._keys == {"orig_kw": {"original"}}
            assert g._combined_tags == {"combo:original"}
