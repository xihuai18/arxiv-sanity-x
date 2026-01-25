"""Unit tests for summary service functions."""

from __future__ import annotations


class TestPublicSummaryMeta:
    """Tests for public_summary_meta function."""

    def test_public_summary_meta_filters_internal_fields(self):
        """Test that internal fields are filtered out."""
        from backend.services.summary_service import public_summary_meta

        meta = {
            "generated_at": "2024-01-01",
            "source": "llm",
            "llm_model": "gpt-4",
            "prompt": "secret prompt",  # Should be filtered
        }
        public = public_summary_meta(meta)

        assert "generated_at" in public
        assert "prompt" not in public

    def test_public_summary_meta_preserves_public_fields(self):
        """Test that public fields are preserved."""
        from backend.services.summary_service import public_summary_meta

        meta = {
            "generated_at": "2024-01-01",
            "source": "llm",
            "llm_model": "gpt-4",
        }
        public = public_summary_meta(meta)

        assert public.get("generated_at") == "2024-01-01"
        assert public.get("source") == "llm"
        assert public.get("llm_model") == "gpt-4"


class TestSanitizeSummaryMeta:
    """Tests for sanitize_summary_meta function."""

    def test_sanitize_summary_meta_removes_internal_fields(self):
        """Test that internal fields are removed."""
        from backend.services.summary_service import sanitize_summary_meta

        meta = {
            "generated_at": "2024-01-01",
            "prompt": "secret prompt",
        }
        sanitized = sanitize_summary_meta(meta)

        assert "prompt" not in sanitized


class TestTldrCache:
    """Tests for TLDR cache."""

    def test_tldr_cache_exists(self):
        """Test that TLDR_CACHE exists."""
        from backend.services.summary_service import TLDR_CACHE

        assert TLDR_CACHE is not None


class TestSummaryCacheStats:
    """Tests for summary cache statistics functions."""

    def test_compute_summary_cache_stats_returns_expected_structure(self):
        """Test that compute_summary_cache_stats returns expected structure."""
        from backend.services.summary_service import compute_summary_cache_stats

        stats = compute_summary_cache_stats()
        expected_keys = ["summary_cache_total", "summary_cache_paper_count", "summary_cache_model_counts"]

        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_get_summary_cache_stats_returns_expected_structure(self):
        """Test that get_summary_cache_stats returns expected structure."""
        from backend.services.summary_service import get_summary_cache_stats

        result = get_summary_cache_stats(ttl=1)

        assert isinstance(result, dict)
        assert "data" in result


class TestGetSummaryStatus:
    """Tests for get_summary_status function."""

    def test_get_summary_status_exists(self):
        """Test that get_summary_status function exists."""
        from backend.services.summary_service import get_summary_status

        assert callable(get_summary_status)


class TestExtractTldrFromSummary:
    """Tests for extract_tldr_from_summary function."""

    def test_extract_tldr_from_summary_exists(self):
        """Test that extract_tldr_from_summary function exists."""
        from backend.services.summary_service import extract_tldr_from_summary

        assert callable(extract_tldr_from_summary)
