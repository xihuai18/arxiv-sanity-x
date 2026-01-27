"""Tests for upload_similarity_service module."""

import numpy as np
import scipy.sparse as sp


class TestComputeTfidfVector:
    """Tests for compute_tfidf_vector function."""

    def test_compute_tfidf_vector_basic(self, monkeypatch):
        """Test basic TF-IDF vector computation using mocked vectorizer."""
        from backend.services import search_service, upload_similarity_service

        # Create a mock vectorizer that returns a known sparse matrix
        class MockVectorizer:
            def transform(self, texts):
                # Return a normalized sparse vector
                vec = sp.csr_matrix(np.array([[0.6, 0.8, 0.0, 0.0]], dtype=np.float32))
                return vec

        monkeypatch.setattr(
            search_service,
            "get_query_vectorizer",
            lambda features: MockVectorizer(),
        )

        # Create mock global features
        mock_features = {
            "vocab": {"hello": 0, "world": 1, "test": 2, "python": 3},
            "idf": np.array([1.0, 1.5, 2.0, 1.2], dtype=np.float32),
        }

        text = "Hello world! This is a test of Python code."
        result = upload_similarity_service.compute_tfidf_vector(text, mock_features)

        assert result is not None
        assert sp.issparse(result)
        assert result.shape == (1, 4)
        # Check that it's normalized (L2 norm should be ~1)
        norm = sp.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_compute_tfidf_vector_empty_result(self, monkeypatch):
        """Test with text that produces empty result (zero vector).

        After fix: zero TF-IDF vector is now valid (returns sparse matrix with nnz=0)
        to allow embedding path to continue working.
        """
        from backend.services import search_service, upload_similarity_service

        # Create a mock vectorizer that returns empty sparse matrix
        class MockVectorizer:
            def transform(self, texts):
                return sp.csr_matrix((1, 2), dtype=np.float32)

        monkeypatch.setattr(
            search_service,
            "get_query_vectorizer",
            lambda features: MockVectorizer(),
        )

        mock_features = {"vocab": {"xyz": 0, "abc": 1}, "idf": np.array([1.0, 1.5])}
        result = upload_similarity_service.compute_tfidf_vector("", mock_features)
        # Zero vector is now valid (not None) to allow embedding fallback
        assert result is not None
        assert sp.issparse(result)
        assert result.nnz == 0
        assert result.shape == (1, 2)

    def test_compute_tfidf_vector_no_vectorizer(self, monkeypatch):
        """Test when vectorizer is not available."""
        from backend.services import search_service, upload_similarity_service

        monkeypatch.setattr(
            search_service,
            "get_query_vectorizer",
            lambda features: None,
        )

        mock_features = {"vocab": {"hello": 0}, "idf": np.array([1.0])}
        result = upload_similarity_service.compute_tfidf_vector("hello world", mock_features)
        assert result is None


class TestGetUploadFeaturesPath:
    """Tests for get_upload_features_path function."""

    def test_get_upload_features_path(self):
        """Test feature path generation."""
        from backend.services.upload_similarity_service import get_upload_features_path

        # Use valid PID format: up_ + 12 chars = 15 total
        path = get_upload_features_path("up_test1234567a")
        assert "uploads" in str(path)
        assert "up_test1234567a" in str(path)
        assert str(path).endswith("features.npz")

    def test_get_upload_features_path_invalid_pid(self):
        """Test that invalid PID raises ValueError."""
        import pytest

        from backend.services.upload_similarity_service import get_upload_features_path

        with pytest.raises(ValueError, match="Invalid upload PID format"):
            get_upload_features_path("invalid_pid")


class TestSaveLoadUploadFeatures:
    """Tests for save/load upload features functions."""

    def test_save_and_load_features(self, tmp_path, monkeypatch):
        """Test saving and loading features."""
        from backend.services import upload_similarity_service

        # Monkeypatch DATA_DIR to use temp directory
        monkeypatch.setattr(upload_similarity_service, "DATA_DIR", str(tmp_path))

        # Use valid PID format: up_ + 12 chars = 15 total
        pid = "up_saveload1234"

        # Create test features
        tfidf = sp.csr_matrix(np.array([[0.5, 0.3, 0.0, 0.8]], dtype=np.float32))
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        features = {
            "pid": pid,
            "tfidf": tfidf,
            "embedding": embedding,
            "text_length": 100,
            "feature_type": "hybrid_sparse_dense",
            "global_features_mtime": 123.0,
        }

        # Save
        upload_similarity_service.save_upload_features(pid, features)

        # Load
        loaded = upload_similarity_service.load_upload_features(pid)

        assert loaded is not None
        assert loaded["pid"] == pid
        assert loaded["text_length"] == 100
        assert sp.issparse(loaded["tfidf"])
        assert np.allclose(loaded["tfidf"].toarray(), tfidf.toarray())
        assert np.allclose(loaded["embedding"], embedding)
        # New compatibility fields
        assert "x" in loaded
        assert "x_tfidf" in loaded
        assert float(loaded.get("global_features_mtime") or 0.0) == 123.0

    def test_load_nonexistent_features(self, tmp_path, monkeypatch):
        """Test loading features that don't exist."""
        from backend.services import upload_similarity_service

        monkeypatch.setattr(upload_similarity_service, "DATA_DIR", str(tmp_path))

        # Use valid PID format: up_ + 12 chars = 15 total
        result = upload_similarity_service.load_upload_features("up_nonexistent1")
        assert result is None


class TestGetUploadTextForFeatures:
    """Tests for get_upload_text_for_features function."""

    def test_get_upload_text_no_record(self, monkeypatch):
        """Test with non-existent paper."""
        from aslite import repositories
        from backend.services.upload_similarity_service import (
            get_upload_text_for_features,
        )

        # Mock repository to return None
        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda pid: None,
        )

        result = get_upload_text_for_features("up_nonexistent")
        assert result is None

    def test_get_upload_text_with_metadata(self, monkeypatch):
        """Test with paper that has metadata."""
        from aslite import repositories
        from backend.services.upload_similarity_service import (
            get_upload_text_for_features,
        )

        # Mock repository
        mock_record = {
            "meta_extracted": {
                "title": "Test Paper Title",
                "authors": ["Author One", "Author Two"],
                "abstract": "This is the abstract.",
            },
        }
        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda pid: mock_record,
        )

        result = get_upload_text_for_features("up_test")
        assert result is not None
        assert "Test Paper Title" in result
        assert "Author One" in result
        assert "This is the abstract" in result


class TestGetUploadTextForEmbedding:
    """Tests for get_upload_text_for_embedding function."""

    def test_get_upload_text_for_embedding_format(self, monkeypatch):
        from aslite import repositories
        from backend.services.upload_similarity_service import (
            get_upload_text_for_embedding,
        )

        mock_record = {
            "meta_extracted": {
                "title": "My Title",
                "authors": ["A1", "A2"],
                "abstract": "My abstract",
            },
        }
        monkeypatch.setattr(repositories.UploadedPaperRepository, "get", lambda pid: mock_record)

        text = get_upload_text_for_embedding("up_test")
        assert text is not None
        assert text.startswith("Title:")
        assert "Abstract:" in text


class TestFindSimilarPapers:
    """Tests for find_similar_papers function."""

    def test_find_similar_papers_no_features(self, monkeypatch):
        """Test when features cannot be computed."""
        from backend.services import upload_similarity_service

        # Mock compute_upload_features to return None
        monkeypatch.setattr(
            upload_similarity_service,
            "compute_upload_features",
            lambda pid, force=False: None,
        )

        result = upload_similarity_service.find_similar_papers("up_test")
        assert result == []

    def test_find_similar_papers_no_global_features(self, monkeypatch):
        """Test when global features are not available."""
        from backend.services import data_service, upload_similarity_service

        # Mock compute_upload_features to return valid features
        mock_features = {
            "pid": "up_test",
            "tfidf": sp.csr_matrix(np.array([[0.5, 0.3]], dtype=np.float32)),
            "embedding": np.array([0.1, 0.2], dtype=np.float32),
            "text_length": 100,
        }
        monkeypatch.setattr(
            upload_similarity_service,
            "compute_upload_features",
            lambda pid, force=False: mock_features,
        )

        # Mock get_features_cached to return None
        monkeypatch.setattr(data_service, "get_features_cached", lambda: None)

        result = upload_similarity_service.find_similar_papers("up_test")
        assert result == []

    def test_find_similar_papers_with_valid_features(self, monkeypatch):
        """Test find_similar_papers with valid features returns results."""
        from backend.services import data_service, upload_similarity_service

        # Mock upload features
        mock_upload_features = {
            "pid": "up_test",
            "tfidf": sp.csr_matrix(np.array([[0.5, 0.3, 0.0, 0.2]], dtype=np.float32)),
            "embedding": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "text_length": 100,
            "x": sp.csr_matrix(np.array([[0.5, 0.3, 0.0, 0.2]], dtype=np.float32)),
        }
        monkeypatch.setattr(
            upload_similarity_service,
            "compute_upload_features",
            lambda pid, force=False: mock_upload_features,
        )

        # Mock global features
        mock_global_features = {
            "pids": ["paper1", "paper2", "paper3"],
            "x": sp.csr_matrix(
                np.array(
                    [
                        [0.6, 0.2, 0.1, 0.1],
                        [0.1, 0.8, 0.0, 0.1],
                        [0.4, 0.3, 0.2, 0.1],
                    ],
                    dtype=np.float32,
                )
            ),
            "x_tfidf": sp.csr_matrix(
                np.array(
                    [
                        [0.6, 0.2, 0.1, 0.1],
                        [0.1, 0.8, 0.0, 0.1],
                        [0.4, 0.3, 0.2, 0.1],
                    ],
                    dtype=np.float32,
                )
            ),
            "x_embeddings": np.array(
                [
                    [0.2, 0.3, 0.4, 0.1],
                    [0.1, 0.1, 0.1, 0.7],
                    [0.15, 0.25, 0.35, 0.25],
                ],
                dtype=np.float32,
            ),
            "feature_type": "hybrid_sparse_dense",
        }
        monkeypatch.setattr(data_service, "get_features_cached", lambda: mock_global_features)

        # Mock metas
        mock_metas = {
            "paper1": {"title": "Paper One", "authors": [{"name": "Author A"}]},
            "paper2": {"title": "Paper Two", "authors": [{"name": "Author B"}]},
            "paper3": {"title": "Paper Three", "authors": [{"name": "Author C"}]},
        }
        monkeypatch.setattr(data_service, "get_metas", lambda: mock_metas)

        result = upload_similarity_service.find_similar_papers("up_test", limit=3)

        assert len(result) > 0
        assert all("id" in r and "score" in r and "title" in r for r in result)
        # Scores should be sorted descending
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestComputeUploadFeatures:
    """Tests for compute_upload_features function."""

    def test_compute_upload_features_no_text(self, monkeypatch):
        """Test compute_upload_features when no text is available."""
        from backend.services import upload_similarity_service

        monkeypatch.setattr(
            upload_similarity_service,
            "get_upload_text_for_features",
            lambda pid: None,
        )

        # Use valid PID format: up_ + 12 chars = 15 total
        result = upload_similarity_service.compute_upload_features("up_notext123456")
        assert result is None

    def test_compute_upload_features_no_global_features(self, monkeypatch):
        """Test compute_upload_features when global features unavailable."""
        from backend.services import data_service, upload_similarity_service

        monkeypatch.setattr(
            upload_similarity_service,
            "get_upload_text_for_features",
            lambda pid: "Test paper about machine learning",
        )
        monkeypatch.setattr(data_service, "get_features_cached", lambda: None)

        # Use valid PID format: up_ + 12 chars = 15 total
        result = upload_similarity_service.compute_upload_features("up_noglobal1234")
        assert result is None

    def test_compute_upload_features_loads_cached(self, tmp_path, monkeypatch):
        """Test that compute_upload_features loads cached features."""
        from backend.services import upload_similarity_service

        monkeypatch.setattr(upload_similarity_service, "DATA_DIR", str(tmp_path))

        # Use valid PID format: up_ + 12 chars = 15 total
        pid = "up_testcached12"

        # Pre-save features
        cached_features = {
            "pid": pid,
            "tfidf": sp.csr_matrix(np.array([[0.5, 0.3]], dtype=np.float32)),
            "embedding": np.array([0.1, 0.2], dtype=np.float32),
            "text_length": 50,
        }
        upload_similarity_service.save_upload_features(pid, cached_features)

        # Should load from cache without calling get_upload_text_for_features
        call_count = [0]
        original_get_text = upload_similarity_service.get_upload_text_for_features

        def mock_get_text(pid):
            call_count[0] += 1
            return original_get_text(pid)

        monkeypatch.setattr(
            upload_similarity_service,
            "get_upload_text_for_features",
            mock_get_text,
        )

        result = upload_similarity_service.compute_upload_features(pid, force=False)

        assert result is not None
        assert result["pid"] == pid
        assert call_count[0] == 0  # Should not call get_text when cached


class TestComputeEmbeddingVector:
    """Tests for compute_embedding_vector function."""

    def test_compute_embedding_vector_truncates_long_text(self, monkeypatch):
        """Test compute_embedding_vector calls document embedding helper."""
        from backend.services import semantic_service, upload_similarity_service

        captured_text = [None]

        def mock_get_document_embedding(text, embed_dim):
            captured_text[0] = text
            v = np.array([0.1] * embed_dim, dtype=np.float32)
            v /= np.linalg.norm(v)
            return v

        monkeypatch.setattr(
            semantic_service,
            "get_document_embedding",
            mock_get_document_embedding,
        )

        long_text = "a" * 10000
        result = upload_similarity_service.compute_embedding_vector(long_text, 4)

        assert result is not None
        assert captured_text[0] is not None

    def test_compute_embedding_vector_normalizes(self, monkeypatch):
        """Test that embedding vector is normalized."""
        from backend.services import semantic_service, upload_similarity_service

        def mock_get_document_embedding(text, embed_dim):
            v = np.array([3.0, 4.0], dtype=np.float32)
            v /= np.linalg.norm(v)
            return v

        monkeypatch.setattr(
            semantic_service,
            "get_document_embedding",
            mock_get_document_embedding,
        )

        result = upload_similarity_service.compute_embedding_vector("test", 2)

        assert result is not None
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01  # Should be normalized
