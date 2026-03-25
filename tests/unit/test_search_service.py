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

    def test_parse_search_query_populates_precomputed_fields(self):
        from backend.services.search_service import parse_search_query

        parsed = parse_search_query('ti:"Graph Networks" 2301.00001 graph')

        assert parsed["raw_loose_norm"] == "ti graph networks 2301 00001 graph"
        assert "2301.00001" in parsed["mentioned_ids"]
        assert "2301.00001" in parsed["exact_id_terms"]
        assert parsed["has_any_field_filters"] is True
        assert parsed["has_text_field_filters"] is True
        assert parsed["title_terms"] == ["graph", "networks"]
        assert parsed["general_phrase"] == "2301.00001 graph"


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

    def test_filter_by_time_zero_days_keeps_all_candidates(self):
        from backend.services.search_service import filter_by_time

        pids = ["p1", "p2"]
        metas = {"p1": {"_time": 1.0}, "p2": {"_time": 2.0}}

        kept_pids, kept_indices = filter_by_time(pids, metas, "0.0")

        assert kept_pids == pids
        assert kept_indices == [0, 1]

    def test_search_rank_cache_invalidates_when_papers_db_mtime_changes(self, monkeypatch):
        import backend.services.search_service as ss
        from backend.utils.cache import LRUCacheTTL

        monkeypatch.setattr(ss, "SEARCH_RANK_CACHE", LRUCacheTTL(maxsize=8, ttl_s=60.0))

        call_count = {"count": 0}
        papers_mtimes = iter([1.0, 2.0])

        monkeypatch.setattr("backend.services.data_service.get_features_file_mtime", lambda: 10.0)
        monkeypatch.setattr(
            "backend.services.data_service._sqlite_effective_mtime",
            lambda _path: next(papers_mtimes),
        )

        def _fake_fullscan(*_args, **_kwargs):
            call_count["count"] += 1
            return ["p1"], [1.0]

        monkeypatch.setattr(ss, "lexical_rank_fullscan", _fake_fullscan)

        ss.search_rank("title:test", limit=5, get_features_fn=lambda: None)
        ss.search_rank("title:test", limit=5, get_features_fn=lambda: None)

        assert call_count["count"] == 2

    def test_svm_rank_cache_invalidates_when_dict_wal_mtime_changes(self, monkeypatch):
        import os

        import backend.services.search_service as ss
        from backend.utils.cache import LRUCacheTTL

        monkeypatch.setattr(ss, "SVM_RANK_CACHE", LRUCacheTTL(maxsize=8, ttl_s=60.0))
        monkeypatch.setattr("backend.services.data_service._sqlite_effective_mtime", lambda _path: 1.0)

        call_count = {"count": 0}

        def _fake_get_features():
            import scipy.sparse as sp

            call_count["count"] += 1
            return {
                "x": sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]], dtype=float),
                "pids": ["p1", "p2"],
                "vocab": {"a": 0, "b": 1},
            }

        monkeypatch.setattr(os.path, "exists", lambda path: path.endswith("features.p"))
        feature_mtimes = iter([10.0, 10.0])
        monkeypatch.setattr(os.path, "getmtime", lambda _path: next(feature_mtimes))

        tags_db = {"t": {"p1"}}

        ss.svm_rank(
            tags="t",
            limit=2,
            get_features_fn=_fake_get_features,
            get_tags_fn=lambda: tags_db,
            get_neg_tags_fn=lambda: {},
            get_metas_fn=lambda: {},
            user="u",
        )

        monkeypatch.setattr("backend.services.data_service._sqlite_effective_mtime", lambda _path: 2.0)

        ss.svm_rank(
            tags="t",
            limit=2,
            get_features_fn=_fake_get_features,
            get_tags_fn=lambda: tags_db,
            get_neg_tags_fn=lambda: {},
            get_metas_fn=lambda: {},
            user="u",
        )

        assert call_count["count"] == 2

    def test_lexical_rank_over_pids_reads_metas_once(self):
        from backend.services.search_service import (
            lexical_rank_over_pids,
            parse_search_query,
        )

        calls = {"count": 0}

        def _get_metas():
            calls["count"] += 1
            return {"p1": {"_time": 1.0}, "p2": {"_time": 2.0}}

        parsed = parse_search_query("graph")
        pids, scores = lexical_rank_over_pids(
            ["p1", "p2"],
            parsed,
            get_papers_bulk_fn=lambda ids: {
                pid: {
                    "title": f"{pid} graph paper",
                    "authors": [{"name": "Alice"}],
                    "summary": "graph abstract",
                    "tags": [{"term": "ml"}],
                }
                for pid in ids
            },
            paper_text_fields_fn=lambda paper: {
                "title_norm": paper["title"],
                "title_lower": paper["title"],
                "title_norm_loose": paper["title"],
                "authors_norm": "alice",
                "summary_norm": "graph abstract",
                "tags_norm": "ml",
            },
            get_metas_fn=_get_metas,
            apply_limit_fn=lambda out_pids, out_scores, _limit: (out_pids, out_scores),
            limit=None,
        )

        assert pids == ["p2", "p1"] or pids == ["p1", "p2"]
        assert len(scores) == 2
        assert calls["count"] == 1

    def test_lexical_rank_fullscan_reads_metas_once(self):
        from backend.services.search_service import (
            lexical_rank_fullscan,
            parse_search_query,
        )

        calls = {"count": 0}

        def _get_metas():
            calls["count"] += 1
            return {"p1": {"_time": 1.0}, "p2": {"_time": 2.0}}

        parsed = parse_search_query("graph")
        pids, scores = lexical_rank_fullscan(
            parsed,
            get_pids_fn=lambda: ["p1", "p2"],
            get_papers_fn=lambda: {
                "p1": {
                    "title": "p1 graph paper",
                    "authors": [{"name": "Alice"}],
                    "summary": "graph abstract",
                    "tags": [{"term": "ml"}],
                },
                "p2": {
                    "title": "p2 graph paper",
                    "authors": [{"name": "Bob"}],
                    "summary": "graph abstract",
                    "tags": [{"term": "ml"}],
                },
            },
            get_papers_bulk_fn=lambda _ids: {},
            paper_text_fields_fn=lambda paper: {
                "title_norm": paper["title"],
                "title_lower": paper["title"],
                "title_norm_loose": paper["title"],
                "authors_norm": "authors",
                "summary_norm": "graph abstract",
                "tags_norm": "ml",
            },
            get_metas_fn=_get_metas,
            max_results=10,
            limit=10,
        )

        assert len(pids) == 2
        assert len(scores) == 2
        assert calls["count"] == 1


def test_filter_public_results_drops_missing_pids():
    from backend.services.search_service import filter_public_results

    pids, scores = filter_public_results(
        ["p1", "p2", "p3"],
        [1.0, 2.0, 3.0],
        get_papers_bulk_fn=lambda _pids: {
            "p1": {"title": "one"},
            "p3": {"title": "three"},
        },
    )

    assert pids == ["p1", "p3"]
    assert scores == [1.0, 3.0]


def test_search_rank_explicit_id_uses_data_service_visibility(monkeypatch):
    import backend.services.search_service as ss

    monkeypatch.setattr("backend.services.data_service.paper_exists", lambda _pid: False)

    pids, scores = ss.search_rank(
        "2301.00001",
        limit=5,
        get_features_fn=lambda: None,
        get_metas_fn=lambda: {"2301.00001": {"_time": 1.0}},
        get_pids_fn=lambda: [],
        get_papers_bulk_fn=lambda _pids: {},
    )

    assert pids == []
    assert scores == []


def test_search_rank_explicit_id_still_filters_public_results(monkeypatch):
    import backend.services.search_service as ss

    monkeypatch.setattr("backend.services.data_service.paper_exists", lambda _pid: True)
    monkeypatch.setattr(
        "aslite.repositories.PaperTombstoneRepository.get_by_ids",
        lambda _pids: {"2301.00001": {"reason": "withdrawn_only"}},
    )

    pids, scores = ss.search_rank(
        "2301.00001",
        limit=5,
        get_features_fn=lambda: None,
        get_pids_fn=lambda: [],
        get_papers_bulk_fn=lambda _pids: {},
    )

    assert pids == []
    assert scores == []


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

    def test_svm_rank_logic_and_uses_strict_tag_intersection(self, monkeypatch):
        import numpy as np
        import scipy.sparse as sp

        import backend.services.search_service as ss

        pids = ["p1", "p2", "p3"]
        x = sp.csr_matrix([[1.0], [0.8], [0.0]], dtype=float)
        captured = {}

        class _FakeLinearSVC:
            def __init__(self, **_kwargs):
                self.coef_ = np.array([0.0])

            def fit(self, x_train, y_train, sample_weight=None):
                captured["y_train"] = np.asarray(y_train)
                self.coef_ = np.zeros(x_train.shape[1], dtype=float)
                return self

            def decision_function(self, x_data):
                return np.asarray(x_data.toarray()).reshape(-1)

        monkeypatch.setattr("sklearn.svm.LinearSVC", _FakeLinearSVC)

        out_pids, _scores, _words = ss.svm_rank(
            tags="tag_a,tag_b",
            logic="and",
            limit=3,
            get_features_fn=lambda: {"x": x, "pids": pids, "vocab": {}},
            get_tags_fn=lambda: {"tag_a": {"p1", "p2"}, "tag_b": {"p2"}},
            get_neg_tags_fn=lambda: {"tag_a": {"p3"}},
            get_metas_fn=lambda: {},
            user="alice",
        )

        assert out_pids
        assert int((captured["y_train"] == 1).sum()) == 1

    def test_svm_rank_keeps_positive_label_when_pid_is_also_negative(self, monkeypatch):
        import numpy as np
        import scipy.sparse as sp

        import backend.services.search_service as ss

        x = sp.csr_matrix([[1.0], [0.0]], dtype=float)
        captured = {}

        class _FakeLinearSVC:
            def __init__(self, **_kwargs):
                self.coef_ = np.array([0.0])

            def fit(self, x_train, y_train, sample_weight=None):
                captured["y_train"] = np.asarray(y_train)
                captured["sample_weight"] = np.asarray(sample_weight)
                self.coef_ = np.zeros(x_train.shape[1], dtype=float)
                return self

            def decision_function(self, x_data):
                return np.asarray(x_data.toarray()).reshape(-1)

        monkeypatch.setattr("sklearn.svm.LinearSVC", _FakeLinearSVC)

        out_pids, _scores, _words = ss.svm_rank(
            tags="tag_a",
            limit=2,
            get_features_fn=lambda: {"x": x, "pids": ["p1", "p2"], "vocab": {}},
            get_tags_fn=lambda: {"tag_a": {"p1"}},
            get_neg_tags_fn=lambda: {"tag_a": {"p1"}},
            get_metas_fn=lambda: {},
            user="alice",
        )

        assert out_pids
        assert captured["y_train"].tolist() == [1, 0]
        assert captured["sample_weight"][0] == 1.0

    def test_svm_rank_time_filter_keeps_tagged_and_seed_papers(self, monkeypatch):
        import numpy as np
        import scipy.sparse as sp

        import backend.services.search_service as ss

        x = sp.csr_matrix([[1.0], [0.5], [0.25]], dtype=float)
        captured = {}

        class _FakeLinearSVC:
            def __init__(self, **_kwargs):
                self.coef_ = np.array([0.0])

            def fit(self, x_train, y_train, sample_weight=None):
                captured["x_train_shape"] = x_train.shape
                captured["y_train"] = np.asarray(y_train)
                self.coef_ = np.zeros(x_train.shape[1], dtype=float)
                return self

            def decision_function(self, x_data):
                return np.asarray(x_data.toarray()).reshape(-1)

        monkeypatch.setattr("sklearn.svm.LinearSVC", _FakeLinearSVC)
        monkeypatch.setattr(ss.time, "time", lambda: 1000.0)

        out_pids, _scores, _words = ss.svm_rank(
            tags="tag_a",
            s_pids="p3",
            time_filter="1",
            limit=3,
            get_features_fn=lambda: {"x": x, "pids": ["p1", "p2", "p3"], "vocab": {}},
            get_tags_fn=lambda: {"tag_a": {"p1"}},
            get_neg_tags_fn=lambda: {"tag_a": {"p2"}},
            get_metas_fn=lambda: {
                "p1": {"_time": 0.0},
                "p2": {"_time": 0.0},
                "p3": {"_time": 0.0},
            },
            user="alice",
        )

        assert out_pids == ["p1", "p2", "p3"]
        assert captured["x_train_shape"][0] == 3
        assert captured["y_train"].tolist() == [1, 0, 1]


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
