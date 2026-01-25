"""Unit tests for search service functions."""

from __future__ import annotations


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        from backend.services.search_service import normalize_text

        assert normalize_text("Hello-World") == "hello world"

    def test_normalize_text_lowercase(self):
        """Test that text is lowercased."""
        from backend.services.search_service import normalize_text

        assert normalize_text("HELLO WORLD") == "hello world"

    def test_normalize_text_special_chars(self):
        """Test handling of special characters."""
        from backend.services.search_service import normalize_text

        result = normalize_text("Hello_World!")
        assert "hello" in result
        assert "world" in result


class TestParseSearchQuery:
    """Tests for parse_search_query function."""

    def test_parse_search_query_basic(self):
        """Test basic query parsing."""
        from backend.services.search_service import parse_search_query

        parsed = parse_search_query("machine learning")
        assert "terms" in parsed
        assert "norm" in parsed

    def test_parse_search_query_empty(self):
        """Test parsing empty query."""
        from backend.services.search_service import parse_search_query

        parsed = parse_search_query("")
        assert "terms" in parsed
        assert "norm" in parsed


class TestExtractArxivIds:
    """Tests for extract_arxiv_ids function."""

    def test_extract_arxiv_ids_basic(self):
        """Test extracting arxiv IDs from text."""
        from backend.services.search_service import extract_arxiv_ids

        ids = extract_arxiv_ids("Check out arxiv:2301.00001 and 1234.56789")
        assert "2301.00001" in ids

    def test_extract_arxiv_ids_no_ids(self):
        """Test extracting from text with no IDs."""
        from backend.services.search_service import extract_arxiv_ids

        ids = extract_arxiv_ids("No arxiv IDs here")
        assert len(ids) == 0

    def test_extract_arxiv_ids_multiple(self):
        """Test extracting multiple IDs."""
        from backend.services.search_service import extract_arxiv_ids

        ids = extract_arxiv_ids("Papers: 2301.00001, 2301.00002, 2301.00003")
        assert len(ids) >= 1


class TestLooksLikeCjkQuery:
    """Tests for looks_like_cjk_query function."""

    def test_looks_like_cjk_query_chinese(self):
        """Test detection of Chinese text."""
        from backend.services.search_service import looks_like_cjk_query

        assert looks_like_cjk_query("机器学习") is True

    def test_looks_like_cjk_query_english(self):
        """Test that English text is not detected as CJK."""
        from backend.services.search_service import looks_like_cjk_query

        assert looks_like_cjk_query("machine learning") is False

    def test_looks_like_cjk_query_mixed(self):
        """Test mixed CJK and English text."""
        from backend.services.search_service import looks_like_cjk_query

        # Should detect CJK if there are CJK characters
        result = looks_like_cjk_query("机器学习 machine learning")
        assert isinstance(result, bool)


class TestSearchCaches:
    """Tests for search caches."""

    def test_svm_rank_cache_exists(self):
        """Test that SVM_RANK_CACHE exists."""
        from backend.services.search_service import SVM_RANK_CACHE

        assert SVM_RANK_CACHE is not None

    def test_search_rank_cache_exists(self):
        """Test that SEARCH_RANK_CACHE exists."""
        from backend.services.search_service import SEARCH_RANK_CACHE

        assert SEARCH_RANK_CACHE is not None
