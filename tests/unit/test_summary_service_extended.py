"""Extended unit tests for summary service functions.

Tests summary service functions using mocks to avoid file system dependencies.
"""

from __future__ import annotations

import time
from unittest.mock import patch


class TestTLDRCache:
    """Tests for _TLDRCache class."""

    def test_tldr_cache_get_missing(self):
        """Test getting missing key from TLDR cache."""
        from backend.services.summary_service import TLDR_CACHE

        result = TLDR_CACHE.get("nonexistent_key_12345")
        assert result is None

    def test_tldr_cache_set_and_get(self):
        """Test setting and getting from TLDR cache."""
        from backend.services.summary_service import TLDR_CACHE

        test_key = f"test_key_{time.time()}"
        test_value = "This is a test TLDR"

        TLDR_CACHE.set(test_key, test_value)
        result = TLDR_CACHE.get(test_key)
        assert result == test_value


class TestWriteSummaryMeta:
    """Tests for write_summary_meta function."""

    @patch("backend.services.summary_service.atomic_write_json")
    def test_write_summary_meta_success(self, mock_write):
        """Test successful summary meta write."""
        from pathlib import Path

        from backend.services.summary_service import write_summary_meta

        meta_path = Path("/tmp/test_meta.json")
        data = {"generated_at": time.time(), "source": "test"}

        write_summary_meta(meta_path, data)
        mock_write.assert_called_once_with(meta_path, data)

    @patch("backend.services.summary_service.atomic_write_json")
    @patch("backend.services.summary_service.logger")
    def test_write_summary_meta_handles_exception(self, mock_logger, mock_write):
        """Test that write_summary_meta handles exceptions."""
        from pathlib import Path

        from backend.services.summary_service import write_summary_meta

        mock_write.side_effect = Exception("Write error")

        meta_path = Path("/tmp/test_meta.json")
        data = {"generated_at": time.time()}

        # Should not raise
        write_summary_meta(meta_path, data)
        mock_logger.warning.assert_called()


class TestPublicSummaryMeta:
    """Tests for public_summary_meta function."""

    def test_public_summary_meta_filters_fields(self):
        """Test that public_summary_meta filters internal fields."""
        from backend.services.summary_service import public_summary_meta

        meta = {
            "generated_at": 123456,
            "source": "mineru",
            "llm": "gpt-4",
            "llm_model": "gpt-4-turbo",
            "prompt": "internal prompt",  # Should be filtered
            "quality": 0.9,  # Should be filtered
        }

        result = public_summary_meta(meta)

        assert "generated_at" in result
        assert "source" in result
        assert "llm" in result
        assert "prompt" not in result
        assert "quality" not in result

    def test_public_summary_meta_empty_input(self):
        """Test public_summary_meta with empty input."""
        from backend.services.summary_service import public_summary_meta

        assert public_summary_meta({}) == {}
        assert public_summary_meta(None) == {}

    def test_public_summary_meta_non_dict(self):
        """Test public_summary_meta with non-dict input."""
        from backend.services.summary_service import public_summary_meta

        assert public_summary_meta("string") == {}
        assert public_summary_meta([1, 2, 3]) == {}


class TestSanitizeSummaryMeta:
    """Tests for sanitize_summary_meta function."""

    def test_sanitize_summary_meta_removes_internal(self):
        """Test that sanitize_summary_meta removes internal fields."""
        from backend.services.summary_service import sanitize_summary_meta

        meta = {
            "generated_at": 123456,
            "source": "mineru",
            "prompt": "internal prompt",
            "updated_at": 789,
            "quality": 0.9,
            "chinese_ratio": 0.1,
            "model": "gpt-4",
        }

        result = sanitize_summary_meta(meta)

        assert "generated_at" in result
        assert "source" in result
        assert "prompt" not in result
        assert "updated_at" not in result
        assert "quality" not in result
        assert "chinese_ratio" not in result
        assert "model" not in result

    def test_sanitize_summary_meta_empty_input(self):
        """Test sanitize_summary_meta with empty input."""
        from backend.services.summary_service import sanitize_summary_meta

        assert sanitize_summary_meta({}) == {}
        assert sanitize_summary_meta(None) == {}


class TestGetSummaryStatus:
    """Tests for get_summary_status function."""

    @patch("backend.services.summary_service.LLM_NAME", "")
    def test_get_summary_status_no_model(self):
        """Test get_summary_status with no model."""
        from backend.services.summary_service import get_summary_status

        status, error = get_summary_status("2301.00001", None)
        assert status == ""
        assert error is None

    @patch("backend.services.summary_service.LLM_NAME", "gpt-4")
    @patch("backend.services.summary_service.summary_cache_paths")
    @patch("backend.services.summary_service.normalize_summary_source")
    def test_get_summary_status_cache_exists(self, mock_normalize, mock_paths):
        """Test get_summary_status when cache exists."""
        from unittest.mock import MagicMock

        from backend.services.summary_service import get_summary_status

        mock_normalize.return_value = "mineru"

        # Create mock paths
        cache_file = MagicMock()
        cache_file.exists.return_value = True
        meta_file = MagicMock()
        meta_file.exists.return_value = False
        lock_file = MagicMock()
        lock_file.exists.return_value = False
        legacy_cache = MagicMock()
        legacy_cache.exists.return_value = False
        legacy_meta = MagicMock()
        legacy_lock = MagicMock()
        legacy_lock.exists.return_value = False

        mock_paths.return_value = (cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock)

        with patch("backend.services.summary_service.read_summary_meta") as mock_read:
            mock_read.return_value = {"source": "mineru"}
            with patch("backend.services.summary_service.summary_source_matches") as mock_match:
                mock_match.return_value = True

                status, error = get_summary_status("2301.00001", "gpt-4")
                assert status == "ok"
                assert error is None


class TestExtractTldrFromSummary:
    """Tests for extract_tldr_from_summary function."""

    def test_extract_tldr_from_summary_exists(self):
        """Test that extract_tldr_from_summary function exists."""
        from backend.services.summary_service import extract_tldr_from_summary

        assert callable(extract_tldr_from_summary)

    @patch("backend.services.summary_service.TLDR_CACHE")
    def test_extract_tldr_from_summary_uses_cache(self, mock_cache):
        """Test that extract_tldr_from_summary uses cache."""
        from backend.services.summary_service import extract_tldr_from_summary

        mock_cache.get.return_value = "Cached TLDR"

        result = extract_tldr_from_summary("2301.00001")
        assert result == "Cached TLDR"
        mock_cache.get.assert_called_once_with("2301.00001")


class TestBuildSnapshotData:
    """Tests for _build_snapshot_data function."""

    def test_build_snapshot_data(self):
        """Test _build_snapshot_data returns expected structure."""
        from backend.services.summary_service import _build_snapshot_data

        result = _build_snapshot_data(
            total=100,
            paper_count=50,
            model_counts={"gpt-4": 30, "gpt-3.5": 20},
        )

        assert result["summary_cache_total"] == 100
        assert result["summary_cache_paper_count"] == 50
        assert isinstance(result["summary_cache_model_counts"], list)


class TestComputeSummaryCacheStats:
    """Tests for compute_summary_cache_stats function."""

    def test_compute_summary_cache_stats_exists(self):
        """Test that compute_summary_cache_stats function exists."""
        from backend.services.summary_service import compute_summary_cache_stats

        assert callable(compute_summary_cache_stats)


class TestGetSummaryCacheStats:
    """Tests for get_summary_cache_stats function."""

    def test_get_summary_cache_stats_exists(self):
        """Test that get_summary_cache_stats function exists."""
        from backend.services.summary_service import get_summary_cache_stats

        assert callable(get_summary_cache_stats)

    def test_get_summary_cache_stats_returns_dict(self):
        """Test that get_summary_cache_stats returns a dict."""
        from backend.services.summary_service import get_summary_cache_stats

        result = get_summary_cache_stats()
        assert isinstance(result, dict)
