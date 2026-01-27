"""Unit tests for semantic_service defensive behavior."""

from __future__ import annotations

import numpy as np


def test_semantic_search_rank_returns_empty_when_pid_list_missing(monkeypatch):
    import backend.services.semantic_service as semantic_service

    embeddings = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        semantic_service,
        "get_paper_embeddings",
        lambda: {"embeddings": embeddings, "pids": []},
    )
    monkeypatch.setattr(
        semantic_service,
        "get_query_embedding",
        lambda q, embed_dim: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )

    pids, scores = semantic_service.semantic_search_rank("hello", limit=5)
    assert pids == []
    assert scores == []
