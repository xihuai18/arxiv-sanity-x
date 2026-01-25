"""Semantic embedding and search services."""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import numpy as np
from loguru import logger

from aslite.db import FEATURES_FILE
from config import settings
from tools.compute import Qwen3EmbeddingVllm

EMBED_PORT = settings.embedding.port
EMBED_USE_LLM_API = settings.embedding.use_llm_api
EMBED_MODEL_NAME = settings.embedding.model_name
EMBED_API_BASE = settings.embedding.api_base
EMBED_API_KEY = settings.embedding.api_key
LLM_BASE_URL = settings.llm.base_url
LLM_API_KEY = settings.llm.api_key

from .data_service import get_features_cached
from .search_service import QUERY_EMBED_CACHE, SEARCH_RANK_CACHE

_MODEL_LOCK = Lock()
_EMBEDDINGS_LOCK = Lock()

_semantic_model: Qwen3EmbeddingVllm | None = None
_cached_embeddings: dict[str, Any] | None = None
_cached_embeddings_mtime: float = 0.0


def get_semantic_model() -> Qwen3EmbeddingVllm | None:
    """Get semantic model client for query encoding."""
    global _semantic_model

    if _semantic_model is not None:
        return _semantic_model

    with _MODEL_LOCK:
        if _semantic_model is not None:
            return _semantic_model

        try:
            if EMBED_USE_LLM_API:
                api_base = (EMBED_API_BASE or LLM_BASE_URL or "").rstrip("/")
                api_key = (EMBED_API_KEY or LLM_API_KEY or "").strip() or None
            else:
                api_base = f"http://localhost:{EMBED_PORT}"
                api_key = None

            api_type = "OpenAI-compatible" if EMBED_USE_LLM_API else "Ollama"
            logger.debug(f"Initializing semantic model {api_type} API client for query encoding...")
            model = Qwen3EmbeddingVllm(
                model_name_or_path=EMBED_MODEL_NAME,
                instruction="Extract key concepts from this query to search computer science and AI paper",
                api_base=api_base,
                api_key=api_key,
                use_openai_api=EMBED_USE_LLM_API,
            )
            if not model.initialize():
                logger.error("Failed to initialize semantic model API client")
                _semantic_model = None
                return None
            _semantic_model = model
            return _semantic_model
        except Exception as e:
            logger.error(f"Error initializing semantic model API client: {e}")
            _semantic_model = None
            return None


def get_cached_embeddings_mtime() -> float:
    return float(_cached_embeddings_mtime or 0.0)


def get_paper_embeddings() -> dict[str, Any] | None:
    """Load and cache paper embeddings from features file."""
    global _cached_embeddings, _cached_embeddings_mtime

    try:
        current_mtime = os.path.getmtime(FEATURES_FILE)
    except Exception:
        current_mtime = 0.0

    if _cached_embeddings is not None and _cached_embeddings_mtime == current_mtime:
        return _cached_embeddings

    with _EMBEDDINGS_LOCK:
        try:
            current_mtime = os.path.getmtime(FEATURES_FILE)
        except Exception:
            current_mtime = 0.0

        if _cached_embeddings is not None and _cached_embeddings_mtime == current_mtime:
            return _cached_embeddings

        try:
            features = get_features_cached()
            if features.get("feature_type") == "hybrid_sparse_dense" and "x_embeddings" in features:
                logger.trace("Using pre-computed embeddings from features file")
                embeddings = features["x_embeddings"]

                # Normalize once for fast cosine similarity via dot product
                try:
                    if getattr(embeddings, "dtype", None) != np.float32:
                        embeddings = embeddings.astype(np.float32, copy=False)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32, copy=False)
                    norms[norms == 0] = 1.0
                    embeddings /= norms
                except Exception as e:
                    logger.warning(f"Failed to normalize embeddings, fallback to raw vectors: {e}")

                _cached_embeddings = {"embeddings": embeddings, "pids": features.get("pids") or []}
                _cached_embeddings_mtime = current_mtime
                return _cached_embeddings

            logger.warning("No embeddings found in features file")
            return None
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None


def get_query_embedding(q: str, embed_dim: int):
    """Get cached query embedding to avoid repeated API calls."""
    key = (q.lower(), int(embed_dim), get_cached_embeddings_mtime())
    cached = QUERY_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    model = get_semantic_model()
    if model is None:
        return None

    query_embedding = model.encode([q], dim=embed_dim)
    if query_embedding is None:
        logger.error("Failed to encode query")
        return None

    try:
        if hasattr(query_embedding, "cpu"):
            query_embedding = query_embedding.cpu().numpy()
        else:
            query_embedding = np.asarray(query_embedding)
    except Exception as e:
        logger.error(f"Failed to convert query embedding: {e}")
        return None

    query_vec = query_embedding.reshape(-1).astype(np.float32, copy=False)
    qn = float(np.linalg.norm(query_vec))
    if qn > 0:
        query_vec /= qn

    QUERY_EMBED_CACHE.set(key, query_vec)
    return query_vec


def semantic_search_rank(q: str = "", limit=None) -> tuple[list[str], list[float]]:
    """Execute pure semantic search."""
    q = (q or "").strip()
    if not q:
        return [], []

    paper_data = get_paper_embeddings()
    if paper_data is None:
        logger.error("No paper embeddings available")
        return [], []

    cache_key = None
    try:
        cache_key = ("sem", q.lower(), int(limit) if limit is not None else None, get_cached_embeddings_mtime())
        cached = SEARCH_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    embeddings = paper_data.get("embeddings")
    pids_all = paper_data.get("pids") or []
    if embeddings is None:
        return [], []

    try:
        embed_dim = int(getattr(embeddings, "shape", [0, 0])[1])
    except Exception:
        embed_dim = 0
    if embed_dim <= 0:
        return [], []

    query_vec = get_query_embedding(q, embed_dim)
    if query_vec is None:
        logger.error("Semantic query embedding unavailable")
        return [], []

    try:
        similarities = embeddings @ query_vec
    except Exception as e:
        logger.error(f"Failed to compute semantic similarities: {e}")
        return [], []

    n = int(len(similarities))
    if n <= 0:
        return [], []

    if limit is not None:
        try:
            k = min(int(limit), n)
        except Exception:
            k = n
        if k <= 0:
            return [], []
        top_indices = np.argpartition(-similarities, min(k, n - 1))[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
    else:
        top_indices = np.argsort(-similarities)

    if pids_all and len(pids_all) != n:
        # Be defensive if embeddings/pids length mismatch.
        if len(pids_all) > n:
            pids_all = pids_all[:n]
        else:
            top_indices = [i for i in top_indices if int(i) < len(pids_all)]

    out_pids = [pids_all[int(i)] for i in top_indices]
    out_scores = [float(similarities[int(i)]) * 100.0 for i in top_indices]

    if cache_key is not None:
        try:
            SEARCH_RANK_CACHE.set(cache_key, (out_pids, out_scores))
        except Exception:
            pass

    return out_pids, out_scores


def compute_top_tags_for_paper(
    pid: str,
    user_tags: dict,
    max_tags: int = 3,
    threshold: float = 0.3,
) -> list[str]:
    """Return the top-N most relevant tags for a paper based on embedding similarity.

    Args:
        pid: Paper id (unversioned).
        user_tags: Mapping tag -> iterable[pids]. Usually positive tags.
        max_tags: Max tags to return.
        threshold: Similarity threshold (cosine). Only tags above this are considered.
    """
    if not pid or not user_tags:
        return []

    try:
        paper_data = get_paper_embeddings()
        if not paper_data:
            return []
        embeddings = paper_data.get("embeddings")
        pids_list = paper_data.get("pids") or []
        if embeddings is None or not pids_list:
            return []

        features = get_features_cached()
        pid_to_index = features.get("pid_to_index") if isinstance(features, dict) else None
        if not isinstance(pid_to_index, dict):
            pid_to_index = {}

        # Find target paper embedding
        if pid in pid_to_index:
            paper_idx = pid_to_index[pid]
        elif pid in pids_list:
            paper_idx = pids_list.index(pid)
        else:
            return []

        paper_emb = embeddings[int(paper_idx)]
        try:
            paper_emb = paper_emb / (np.linalg.norm(paper_emb) + 1e-9)
        except Exception:
            pass

        tag_scores = []
        for tag_name, tag_pids in (user_tags or {}).items():
            if not tag_name or not tag_pids:
                continue

            tag_embs = []
            for tpid in tag_pids:
                if not tpid:
                    continue
                if tpid in pid_to_index:
                    tidx = pid_to_index[tpid]
                    tag_embs.append(embeddings[int(tidx)])
                elif tpid in pids_list:
                    tidx = pids_list.index(tpid)
                    tag_embs.append(embeddings[int(tidx)])

            if not tag_embs:
                continue

            tag_mean_emb = np.mean(tag_embs, axis=0)
            tag_mean_emb = tag_mean_emb / (np.linalg.norm(tag_mean_emb) + 1e-9)
            similarity = float(np.dot(paper_emb, tag_mean_emb))

            if similarity >= float(threshold):
                tag_scores.append((tag_name, similarity))

        tag_scores.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in tag_scores[: int(max_tags)]]

    except Exception as e:
        logger.warning(f"Failed to compute top tags for {pid}: {e}")
        return []
