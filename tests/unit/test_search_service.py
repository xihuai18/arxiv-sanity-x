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


class TestSvmRankWithUploads:
    """Tests for svm_rank behavior with uploaded paper training samples."""

    def test_svm_rank_allows_upload_positive_only(self):
        """Upload-tagged positives should enable ranking even without in-slice positives."""
        import scipy.sparse as sp

        from backend.services.search_service import svm_rank

        pids = ["p1", "p2", "p3"]
        x = sp.csr_matrix(
            [
                [1.0, 0.0],  # p1
                [0.0, 1.0],  # p2 (closest to upload positive)
                [0.2, 0.2],  # p3
            ],
            dtype=float,
        )

        def get_features():
            return {"x": x, "pids": pids, "vocab": {}}

        def get_tags():
            return {"t": {"up_abcdefghijkl"}}

        def get_neg_tags():
            return {}

        def get_metas():
            return {}

        def compute_upload_features(_pid: str):
            return {"x": sp.csr_matrix([[0.0, 1.0]], dtype=float)}

        out_pids, scores, words = svm_rank(
            tags="t",
            limit=3,
            get_features_fn=get_features,
            get_tags_fn=get_tags,
            get_neg_tags_fn=get_neg_tags,
            get_metas_fn=get_metas,
            compute_upload_features_fn=compute_upload_features,
            user="u",
        )

        assert out_pids
        assert out_pids[0] == "p2"
        assert len(out_pids) == len(scores)
        assert isinstance(words, list)

    def test_svm_rank_skips_missing_upload_features(self):
        """If upload features cannot be computed, svm_rank should safely degrade."""
        import scipy.sparse as sp

        from backend.services.search_service import svm_rank

        pids = ["p1", "p2"]
        x = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]], dtype=float)

        def get_features():
            return {"x": x, "pids": pids, "vocab": {}}

        def get_tags():
            return {"t": {"up_abcdefghijkl"}}

        def get_neg_tags():
            return {}

        def get_metas():
            return {}

        def compute_upload_features(_pid: str):
            return None

        out_pids, scores, words = svm_rank(
            tags="t",
            limit=10,
            get_features_fn=get_features,
            get_tags_fn=get_tags,
            get_neg_tags_fn=get_neg_tags,
            get_metas_fn=get_metas,
            compute_upload_features_fn=compute_upload_features,
            user="u",
        )

        assert out_pids == []
        assert scores == []
        assert words == []

    def test_svm_rank_does_not_bypass_when_upload_adds_negative(self):
        """When all in-slice are positive, upload negatives should still affect ranking."""
        import scipy.sparse as sp

        from backend.services.search_service import svm_rank

        # p2 is closer to upload negative; training should push p1 above p2.
        pids = ["p2", "p1"]
        x = sp.csr_matrix(
            [
                [0.0, 1.0],  # p2
                [1.0, 0.0],  # p1
            ],
            dtype=float,
        )

        def get_features():
            return {"x": x, "pids": pids, "vocab": {}}

        def get_tags():
            return {"t": {"p1", "p2"}}

        def get_neg_tags():
            return {"t": {"up_abcdefghijkl"}}

        def get_metas():
            return {}

        def compute_upload_features(_pid: str):
            return {"x": sp.csr_matrix([[0.0, 1.0]], dtype=float)}

        out_pids, _scores, _words = svm_rank(
            tags="t",
            limit=2,
            get_features_fn=get_features,
            get_tags_fn=get_tags,
            get_neg_tags_fn=get_neg_tags,
            get_metas_fn=get_metas,
            compute_upload_features_fn=compute_upload_features,
            user="u",
        )

        assert out_pids[0] == "p1"


class TestSearchDowngradeSwitches:
    """Tests for search downgrade switches."""

    def test_semantic_disabled_downgrades_semantic_and_hybrid(self, monkeypatch):
        """When semantic is disabled, semantic/hybrid should downgrade to keyword-only."""
        import importlib
        import sys
        import types

        import config

        monkeypatch.setenv("ARXIV_SANITY_SEARCH_SEMANTIC_DISABLED", "true")
        config.reload_settings()

        import backend.services.search_service as search_service

        search_service = importlib.reload(search_service)

        def _kw_stub(_q: str, _limit=None):
            return ["p1", "p2"], [10.0, 9.0]

        monkeypatch.setattr(search_service, "search_rank", _kw_stub)

        # Guardrail: if code tries to import semantic_service, it should crash.
        sentinel = types.ModuleType("backend.services.semantic_service")

        def _missing_attr(_name: str):
            raise AssertionError("semantic_service should not be used when semantic_disabled=true")

        sentinel.__getattr__ = _missing_attr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "backend.services.semantic_service", sentinel)

        pids, scores, details = search_service.enhanced_search_rank(q="test query", limit=10, search_mode="semantic")
        assert pids == ["p1", "p2"]
        assert scores == [10.0, 9.0]
        assert details == {}

        pids, scores, details = search_service.enhanced_search_rank(q="test query", limit=10, search_mode="hybrid")
        assert pids == ["p1", "p2"]
        assert scores == [10.0, 9.0]
        assert details == {}

    def test_disable_fullscan_skips_fullscan_fallback(self, monkeypatch):
        """When fullscan fallback is disabled, keyword search should not call lexical_rank_fullscan."""
        import importlib

        import config

        monkeypatch.setenv("ARXIV_SANITY_SEARCH_DISABLE_FULLSCAN", "true")
        config.reload_settings()

        import backend.services.search_service as search_service

        search_service = importlib.reload(search_service)

        def _should_not_run(*_args, **_kwargs):
            raise AssertionError("lexical_rank_fullscan should be skipped when disable_fullscan=true")

        monkeypatch.setattr(search_service, "lexical_rank_fullscan", _should_not_run)

        pids, scores = search_service.search_rank(
            "foo",
            limit=10,
            get_features_fn=lambda: {},  # falsy -> forces fallback path
            get_pids_fn=lambda: [],
            get_papers_bulk_fn=lambda _pids: {},
            get_metas_fn=lambda: {},
            paper_exists_fn=lambda _pid: False,
            paper_text_fields_fn=lambda _paper: {},
        )

        assert pids == []
        assert scores == []

    def test_disable_fullscan_uses_bounded_title_scan_fallback(self, monkeypatch):
        """When fullscan is disabled and TF-IDF misses, do a bounded title scan instead of returning empty."""
        import importlib

        import config

        monkeypatch.setenv("ARXIV_SANITY_SEARCH_DISABLE_FULLSCAN", "true")
        config.reload_settings()

        import backend.services.search_service as search_service

        search_service = importlib.reload(search_service)

        def _should_not_run(*_args, **_kwargs):
            raise AssertionError("lexical_rank_fullscan should be skipped when disable_fullscan=true")

        monkeypatch.setattr(search_service, "lexical_rank_fullscan", _should_not_run)

        papers = {
            "p1": {"title": "Graph Neural Networks"},
            "p2": {"title": "Unrelated Paper"},
        }

        def get_pids():
            return list(papers.keys())

        def get_papers_bulk(pids):
            return {pid: papers.get(pid) for pid in pids}

        def paper_text_fields(paper):
            title = (paper or {}).get("title") or ""
            return {
                "title_norm_loose": search_service.normalize_text_loose(title),
                "title_norm": search_service.normalize_text(title),
            }

        pids, _scores = search_service.search_rank(
            "Graph Neural Networks",
            limit=10,
            get_features_fn=lambda: {},  # falsy -> forces TF-IDF miss path
            get_pids_fn=get_pids,
            get_papers_bulk_fn=get_papers_bulk,
            get_metas_fn=lambda: {},
            paper_exists_fn=lambda _pid: False,
            paper_text_fields_fn=paper_text_fields,
        )

        assert pids[:1] == ["p1"]
