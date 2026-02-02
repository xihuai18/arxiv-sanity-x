"""Unit tests for summary service functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


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


class TestClearModelSummary:
    """Tests for clear_model_summary behavior."""

    def test_clear_model_summary_removes_legacy_mismatch_cache(self):
        """Clear should remove caches where meta declares the target model.

        Some legacy caches may be saved under a different filename, but the meta contains
        `model/llm_model` (actual model). We should still delete these files when clearing
        by the declared model, even if the filename doesn't match.
        """
        from backend.services import summary_service as ss

        # Safety guard: never mutate the real repo data directory in unit tests.
        repo_root = Path(__file__).resolve().parents[2]
        real_data_dir = (repo_root / "data").resolve()
        if Path(ss.DATA_DIR).resolve() == real_data_dir:
            pytest.skip("Refusing to modify real data/ directory; set ARXIV_SANITY_DATA_DIR to a temp dir for tests.")

        clear_model_summary = ss.clear_model_summary
        compute_summary_cache_stats = ss.compute_summary_cache_stats

        pid = "9999.99999"
        target_model = "mimo-v2-flash-test-legacy"
        cache_dir = Path(ss.SUMMARY_DIR) / pid
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a mismatch cache: filename looks like another model, but meta declares target_model.
        filename_model_key = "requested-model"
        (cache_dir / f"{filename_model_key}.md").write_text("# ok", encoding="utf-8")
        (cache_dir / f"{filename_model_key}.meta.json").write_text(
            json.dumps({"llm_model": target_model, "source": "html"}, ensure_ascii=False),
            encoding="utf-8",
        )

        before = compute_summary_cache_stats()
        before_count = 0
        for row in before.get("summary_cache_model_counts") or []:
            if row.get("model") == filename_model_key:
                before_count = int(row.get("count") or 0)
                break
        assert before_count == 1

        clear_model_summary(pid, target_model, metas_getter=lambda: {})

        assert not (cache_dir / f"{filename_model_key}.meta.json").exists()
        assert not (cache_dir / f"{filename_model_key}.md").exists()

        after = compute_summary_cache_stats()
        after_count = 0
        for row in after.get("summary_cache_model_counts") or []:
            if row.get("model") == filename_model_key:
                after_count = int(row.get("count") or 0)
                break
        assert after_count == 0


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
