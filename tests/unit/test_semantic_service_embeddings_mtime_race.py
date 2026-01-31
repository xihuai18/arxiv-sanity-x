"""Unit tests for semantic_service embeddings mtime behavior."""

from __future__ import annotations

import numpy as np


def test_get_paper_embeddings_does_not_advance_cache_mtime_past_loaded_data(monkeypatch):
    import backend.services.semantic_service as semantic_service

    # Reset module-level caches to ensure test isolation.
    monkeypatch.setattr(semantic_service, "_cached_embeddings", None)
    monkeypatch.setattr(semantic_service, "_cached_embeddings_mtime", 0.0)
    monkeypatch.setattr(semantic_service, "_EMBEDDINGS_LOADING", False)

    embeddings = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

    state = {"features_calls": 0, "updated": False}

    def fake_getmtime(_path: str) -> float:
        # Simulate: features.p is updated *during* the load window.
        return 2.0 if state["updated"] else 1.0

    def fake_get_features_cached():
        state["features_calls"] += 1
        # The file changes while we're "loading" the features, so post-load mtime is newer.
        state["updated"] = True
        return {
            "feature_type": "hybrid_sparse_dense",
            "x_embeddings": embeddings,
            "pids": ["p1", "p2"],
        }

    monkeypatch.setattr(semantic_service.os.path, "getmtime", fake_getmtime)
    monkeypatch.setattr(semantic_service, "get_features_cached", fake_get_features_cached)

    # First load: cache must be keyed by the pre-load mtime (1.0), even though the file
    # is updated during the load window.
    cache1 = semantic_service.get_paper_embeddings()
    assert cache1 is not None
    assert semantic_service.get_cached_embeddings_mtime() == 1.0

    # Second call: since the real file mtime is now 2.0, we must reload (not keep serving
    # stale embeddings marked as fresh).
    cache2 = semantic_service.get_paper_embeddings()
    assert cache2 is not None
    assert state["features_calls"] == 2
    assert semantic_service.get_cached_embeddings_mtime() == 2.0
