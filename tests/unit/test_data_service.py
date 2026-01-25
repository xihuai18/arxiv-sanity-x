"""Unit tests for data service functions."""

from __future__ import annotations

import pytest


class TestGetPids:
    """Tests for get_pids function."""

    def test_get_pids_returns_list(self):
        """Test that get_pids returns a list."""
        from backend.services.data_service import get_pids

        try:
            pids = get_pids()
            assert isinstance(pids, list)
        except FileNotFoundError:
            # Expected in test environment without data files
            pytest.skip("No data file available")


class TestGetMetas:
    """Tests for get_metas function."""

    def test_get_metas_returns_dict(self):
        """Test that get_metas returns a dict."""
        from backend.services.data_service import get_metas

        try:
            metas = get_metas()
            assert isinstance(metas, dict)
        except FileNotFoundError:
            pytest.skip("No data file available")


class TestPaperExists:
    """Tests for paper_exists function."""

    def test_paper_exists_returns_false_for_empty_pid(self):
        """Test that paper_exists returns False for empty pid."""
        from backend.services.data_service import paper_exists

        try:
            assert paper_exists("") is False
        except FileNotFoundError:
            pytest.skip("No data file available")

    def test_paper_exists_returns_false_for_invalid_pid(self):
        """Test that paper_exists returns False for invalid pid."""
        from backend.services.data_service import paper_exists

        try:
            assert paper_exists("nonexistent_pid_12345") is False
        except FileNotFoundError:
            pytest.skip("No data file available")


class TestGetPaper:
    """Tests for get_paper function."""

    def test_get_paper_returns_none_for_invalid_pid(self):
        """Test that get_paper returns None for invalid pid."""
        from backend.services.data_service import get_paper

        try:
            result = get_paper("nonexistent_pid_12345")
            assert result is None
        except FileNotFoundError:
            pytest.skip("No data file available")


class TestGetPapersBulk:
    """Tests for get_papers_bulk function."""

    def test_get_papers_bulk_returns_dict(self):
        """Test that get_papers_bulk returns a dict."""
        from backend.services.data_service import get_papers_bulk

        try:
            result = get_papers_bulk(["nonexistent1", "nonexistent2"])
            assert isinstance(result, dict)
        except FileNotFoundError:
            pytest.skip("No data file available")

    def test_get_papers_bulk_empty_list(self):
        """Test get_papers_bulk with empty list."""
        from backend.services.data_service import get_papers_bulk

        try:
            result = get_papers_bulk([])
            assert isinstance(result, dict)
            assert len(result) == 0
        except FileNotFoundError:
            pytest.skip("No data file available")


class TestGetDataCached:
    """Tests for get_data_cached function."""

    def test_get_data_cached_returns_expected_structure(self):
        """Test that get_data_cached returns expected structure."""
        from backend.services.data_service import get_data_cached

        try:
            data = get_data_cached()
            assert isinstance(data, dict)
            assert "pids" in data
            assert "metas" in data
        except FileNotFoundError:
            pytest.skip("No data file available")


class TestGetFeaturesCached:
    """Tests for get_features_cached function."""

    def test_get_features_cached_returns_dict(self):
        """Test that get_features_cached returns a dict."""
        from backend.services.data_service import get_features_cached

        try:
            features = get_features_cached()
            assert isinstance(features, dict)
        except FileNotFoundError:
            pytest.skip("No features file available")
