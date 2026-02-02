"""Service for computing similarity between uploaded papers and arXiv papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from loguru import logger

from backend.utils.upload_utils import validate_upload_pid
from config import settings

DATA_DIR = settings.data_dir


def get_upload_features_path(pid: str) -> Path:
    """Get the path to store computed features for an uploaded paper.

    Args:
        pid: Upload PID (must be validated format)

    Returns:
        Path to the features file

    Raises:
        ValueError: If pid format is invalid
    """
    if not validate_upload_pid(pid):
        raise ValueError(f"Invalid upload PID format: {pid}")
    return Path(DATA_DIR) / "uploads" / pid / "features.npz"


def _get_upload_meta_fields(pid: str) -> dict[str, Any] | None:
    """Return normalized meta fields for an uploaded paper.

    Keep this centralized so TF-IDF and embedding inputs can reuse it.
    """
    from aslite.repositories import UploadedPaperRepository

    record = UploadedPaperRepository.get(pid)
    if not record:
        return None

    meta = record.get("meta_extracted", {})
    override = record.get("meta_override", {})

    title = (override.get("title") or meta.get("title") or "").strip()
    authors_list = override.get("authors") or meta.get("authors") or []
    authors_list = [str(a).strip() for a in authors_list if str(a).strip()]
    abstract = (override.get("abstract") or meta.get("abstract") or "").strip()

    return {
        "title": title,
        "authors_list": authors_list,
        "abstract": abstract,
        "record": record,
    }


def get_upload_text_for_features(pid: str) -> str | None:
    """Build TF-IDF text for an uploaded paper.

    To minimize drift, this mirrors the global TF-IDF corpus construction in
    tools/compute.py (title + summary/abstract + authors).

    Note:
    - We intentionally avoid TL;DR here because global TF-IDF training does not
      include summary model outputs.
    - For uploads, we use abstract as the closest analogue of arXiv "summary".
    """
    fields = _get_upload_meta_fields(pid)
    if not fields:
        return None

    title = fields.get("title") or ""
    abstract = fields.get("abstract") or ""
    authors_list = fields.get("authors_list") or []
    author_str = " ".join(authors_list)

    parts = [p for p in (title, abstract, author_str) if p]
    if not parts:
        return None
    return " ".join(parts)


def get_upload_text_for_embedding(pid: str, max_chars: int = 8000) -> str | None:
    """Build document-embedding text for an uploaded paper.

    To minimize drift, this mirrors tools/compute.py embedding corpus format:
    "Title: <title>\nAbstract: <summary/abstract>"
    """
    fields = _get_upload_meta_fields(pid)
    if not fields:
        return None

    title = (fields.get("title") or "").strip()
    abstract = (fields.get("abstract") or "").strip()
    if not title and not abstract:
        return None

    text = f"Title: {title}\nAbstract: {abstract}".strip()
    if max_chars and len(text) > int(max_chars):
        text = text[: int(max_chars)]
    return text


def build_hybrid_feature_vector(
    tfidf_vec: sp.csr_matrix | None,
    embedding_vec: np.ndarray | None,
    global_features: dict,
) -> sp.csr_matrix | None:
    """Build a hybrid feature vector matching global feature space.

    Handles cases where either TF-IDF or embedding is missing by padding with zeros.

    Args:
        tfidf_vec: Sparse TF-IDF vector (1 x vocab_size) or None
        embedding_vec: Dense embedding vector (embed_dim,) or None
        global_features: Global features dict for dimension info

    Returns:
        Hybrid sparse matrix matching global_features['x'] dimensions, or None on failure
    """
    from tools.compute import sparse_dense_concatenation

    try:
        if global_features.get("feature_type") != "hybrid_sparse_dense":
            return tfidf_vec

        # Get dimensions from global features
        x_embeddings = global_features.get("x_embeddings")
        vocab = global_features.get("vocab")
        embed_dim = x_embeddings.shape[1] if x_embeddings is not None else 0
        vocab_size = len(vocab) if vocab is not None else 0

        if tfidf_vec is not None and embedding_vec is not None:
            return sparse_dense_concatenation(tfidf_vec, embedding_vec.reshape(1, -1))
        elif tfidf_vec is not None and embed_dim > 0:
            # Pad with zero embedding
            zero_emb = np.zeros((1, embed_dim), dtype=np.float32)
            return sparse_dense_concatenation(tfidf_vec, zero_emb)
        elif embedding_vec is not None and vocab_size > 0:
            # Pad with zero TF-IDF
            zero_tfidf = sp.csr_matrix((1, vocab_size), dtype=np.float32)
            return sparse_dense_concatenation(zero_tfidf, embedding_vec.reshape(1, -1))
        elif embedding_vec is not None:
            # Can't build hybrid, convert embedding to sparse
            return sp.csr_matrix(embedding_vec.reshape(1, -1))
        else:
            return tfidf_vec
    except Exception as e:
        logger.warning(f"Failed to build hybrid feature vector: {e}")
        return None


def compute_upload_features(pid: str, force: bool = False) -> dict | None:
    """Compute TF-IDF and embedding features for an uploaded paper.

    Args:
        pid: Upload paper ID
        force: If True, recompute even if features exist

    Returns:
        Dict with computed features, or None if failed
    """
    import time as time_module

    t_start = time_module.time()
    logger.trace(f"[BLOCKING] compute_upload_features: starting for pid={pid}, force={force}")

    from backend.services.data_service import (
        get_features_cached,
        get_features_file_mtime,
    )

    features_path = get_upload_features_path(pid)

    global_features = None
    global_feat_mtime = 0.0

    # Check if features already exist and are compatible with current global features
    if not force and features_path.exists():
        try:
            cached = load_upload_features(pid)
            if cached is not None:
                # Best-effort mtime check: if global features are unavailable, keep cached.
                try:
                    logger.trace("[BLOCKING] compute_upload_features: loading global features for cache validation...")
                    global_features = get_features_cached()
                    global_feat_mtime = float(get_features_file_mtime() or 0.0)
                except Exception:
                    global_features = None
                    global_feat_mtime = 0.0

                cached_mtime = float(cached.get("global_features_mtime") or 0.0)
                # If we cannot validate (missing mtimes), keep old behavior (best effort).
                if cached_mtime <= 0 or global_feat_mtime <= 0:
                    logger.trace(
                        f"[BLOCKING] compute_upload_features: using cached features (mtime unavailable) for {pid} "
                        f"in {time_module.time() - t_start:.2f}s"
                    )
                    return cached

                if cached_mtime == global_feat_mtime:
                    logger.trace(
                        f"[BLOCKING] compute_upload_features: using cached features (mtime ok) for {pid} "
                        f"in {time_module.time() - t_start:.2f}s"
                    )
                    return cached
        except Exception as e:
            logger.warning(f"Failed to load existing features for {pid}: {e}")

    # Get TF-IDF text content
    text = get_upload_text_for_features(pid)
    if not text:
        logger.error(f"No text content available for {pid}")
        return None

    # Load global features (used for TF-IDF params + feature type).
    if global_features is None:
        try:
            logger.trace("[BLOCKING] compute_upload_features: loading global features...")
            global_features = get_features_cached()
        except FileNotFoundError:
            global_features = None
    if not global_features:
        logger.error("Global features not available")
        return None

    # Check required fields in global features
    vocab = global_features.get("vocab")
    idf = global_features.get("idf")

    if vocab is None or idf is None:
        logger.error("Vocab or IDF not found in global features")
        return None

    # Compute TF-IDF vector using the same logic as global features
    t_tfidf = time_module.time()
    tfidf_vec = compute_tfidf_vector(text, global_features)
    logger.trace(f"[BLOCKING] compute_upload_features: TF-IDF computed in {time_module.time() - t_tfidf:.2f}s")
    if tfidf_vec is None:
        logger.warning(f"Failed to compute TF-IDF for {pid}, will try embedding only")

    # Compute embedding vector if available
    embedding_vec = None
    if global_features.get("feature_type") == "hybrid_sparse_dense":
        x_embeddings = global_features.get("x_embeddings")
        if x_embeddings is not None:
            embed_dim = x_embeddings.shape[1]
            embed_text = get_upload_text_for_embedding(pid)
            if not embed_text:
                logger.warning(f"No embedding text available for {pid}")
            else:
                t_emb = time_module.time()
                embedding_vec = compute_embedding_vector(embed_text, embed_dim)
                logger.trace(
                    f"[BLOCKING] compute_upload_features: embedding computed in {time_module.time() - t_emb:.2f}s "
                    f"(dim={embed_dim}, text_len={len(embed_text)})"
                )

    # Check if we have at least one feature type available
    if tfidf_vec is None and embedding_vec is None:
        logger.error(f"No features available for {pid} (both TF-IDF and embedding failed)")
        return None

    # Build upload feature vector in the *same feature space* as global_features['x'].
    upload_x = build_hybrid_feature_vector(tfidf_vec, embedding_vec, global_features)

    # Save features
    result = {
        # Keep legacy keys for backward compatibility
        "pid": pid,
        "tfidf": tfidf_vec,
        "embedding": embedding_vec,
        "text_length": len(text),
        # New keys aligned with global features structure
        "x": upload_x,
        "x_tfidf": tfidf_vec,
        "x_embeddings": embedding_vec.reshape(1, -1) if embedding_vec is not None else None,
        "feature_type": global_features.get("feature_type") or "unknown",
        "global_features_mtime": global_feat_mtime,
    }

    try:
        save_upload_features(pid, result)
    except Exception as e:
        logger.warning(f"Failed to save features for {pid}: {e}")

    logger.trace(
        f"[BLOCKING] compute_upload_features: completed for {pid} in {time_module.time() - t_start:.2f}s "
        f"(tfidf={'ok' if tfidf_vec is not None else 'none'}, emb={'ok' if embedding_vec is not None else 'none'})"
    )
    return result


def compute_tfidf_vector(text: str, global_features: dict) -> sp.csr_matrix | None:
    """Compute TF-IDF vector for a single document using the same logic as global features.

    This function reuses the query vectorizer from search_service to ensure consistency
    with global feature computation (including n-gram support).

    Args:
        text: Document text
        global_features: Global features dict containing vocab, idf, tfidf_params

    Returns:
        Sparse TF-IDF vector (1 x vocab_size)
    """
    from backend.services.search_service import get_query_vectorizer

    try:
        import time as time_module

        t_start = time_module.time()
        # Get the query vectorizer (reuses cached instance)
        vectorizer = get_query_vectorizer(global_features)
        if vectorizer is None:
            logger.error("Failed to get query vectorizer")
            return None

        # Transform text to TF-IDF vector
        vec = vectorizer.transform([text])
        logger.trace(f"[BLOCKING] compute_tfidf_vector: transform completed in {time_module.time() - t_start:.2f}s")

        if vec.nnz == 0:
            # Zero matches is valid (e.g., non-English text, short text).
            # Return the zero vector so embedding path can still work.
            logger.warning("No vocabulary matches found in text (zero TF-IDF vector)")

        return vec

    except Exception as e:
        logger.error(f"Failed to compute TF-IDF vector: {e}")
        return None


def compute_embedding_vector(text: str, embed_dim: int) -> np.ndarray | None:
    """Compute document embedding vector for a single paper.

    This MUST match tools/compute.py document-embedding behavior, not query
    embedding behavior.
    """
    from backend.services.semantic_service import get_document_embedding

    try:
        return get_document_embedding(text, embed_dim)
    except Exception as e:
        logger.error(f"Failed to compute embedding vector: {e}")
        return None


def save_upload_features(pid: str, features: dict) -> None:
    """Save computed features to disk.

    Only saves numeric arrays to allow loading with allow_pickle=False for security.
    """
    features_path = get_upload_features_path(pid)
    features_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving - only numeric arrays (no pickle required)
    save_data = {
        # Store text_length as array instead of scalar for allow_pickle=False compatibility
        "text_length": np.array([features.get("text_length", 0)], dtype=np.int64),
    }

    # Cache invalidation metadata
    if features.get("global_features_mtime") is not None:
        try:
            save_data["global_features_mtime"] = np.array(
                [float(features.get("global_features_mtime") or 0.0)], dtype=np.float64
            )
        except Exception:
            save_data["global_features_mtime"] = np.array([0.0], dtype=np.float64)

    # Save TF-IDF as sparse matrix components (all numeric arrays)
    if features.get("tfidf") is not None:
        tfidf = features["tfidf"]
        save_data["tfidf_data"] = tfidf.data
        save_data["tfidf_indices"] = tfidf.indices
        save_data["tfidf_indptr"] = tfidf.indptr
        save_data["tfidf_shape"] = np.array(tfidf.shape)

    # Save embedding as dense array
    if features.get("embedding") is not None:
        save_data["embedding"] = features["embedding"]

    # Save feature_type as a numpy string array (no pickle needed)
    if features.get("feature_type") is not None:
        save_data["feature_type"] = np.array([str(features.get("feature_type"))], dtype="U64")

    np.savez_compressed(features_path, **save_data)
    logger.debug(f"Saved features for {pid} to {features_path}")


def load_upload_features(pid: str) -> dict | None:
    """Load computed features from disk."""
    features_path = get_upload_features_path(pid)

    if not features_path.exists():
        return None

    try:
        # Use allow_pickle=False for security - we only store numeric arrays
        data = np.load(features_path, allow_pickle=False)

        result = {
            "pid": pid,  # pid is not stored in npz with allow_pickle=False
            "text_length": int(data["text_length"][0]) if "text_length" in data and data["text_length"].size > 0 else 0,
        }

        # Cache invalidation metadata
        if "global_features_mtime" in data:
            try:
                m = data.get("global_features_mtime")
                # Stored as 1-element array
                if isinstance(m, np.ndarray) and m.size:
                    result["global_features_mtime"] = float(m.reshape(-1)[0])
                else:
                    result["global_features_mtime"] = float(m)
            except Exception:
                pass

        # feature_type is stored as a string array
        if "feature_type" in data:
            try:
                ft = data.get("feature_type")
                if isinstance(ft, np.ndarray):
                    result["feature_type"] = str(ft.item()) if ft.size == 1 else str(ft)
                else:
                    result["feature_type"] = str(ft)
            except Exception:
                pass

        # Reconstruct TF-IDF sparse matrix
        if "tfidf_data" in data:
            tfidf = sp.csr_matrix(
                (data["tfidf_data"], data["tfidf_indices"], data["tfidf_indptr"]), shape=tuple(data["tfidf_shape"])
            )
            result["tfidf"] = tfidf

        # Load embedding
        if "embedding" in data:
            result["embedding"] = data["embedding"]

        # Build aligned fields for callers that expect global-like keys.
        if "tfidf" in result:
            result.setdefault("x_tfidf", result.get("tfidf"))
        if "embedding" in result and result.get("embedding") is not None:
            emb = np.asarray(result.get("embedding")).reshape(1, -1)
            result.setdefault("x_embeddings", emb)

        # Attempt to reconstruct x for compatibility with new scorer.
        try:
            if result.get("x_tfidf") is not None and result.get("x_embeddings") is not None:
                from tools.compute import sparse_dense_concatenation

                result.setdefault(
                    "x",
                    sparse_dense_concatenation(result["x_tfidf"], result["x_embeddings"]),
                )
            else:
                result.setdefault("x", result.get("x_tfidf"))
        except Exception:
            result.setdefault("x", result.get("x_tfidf"))

        return result

    except Exception as e:
        logger.error(f"Failed to load features for {pid}: {e}")
        return None


def find_similar_papers(
    pid: str,
    limit: int = 20,
    use_tfidf: bool = True,
    use_embedding: bool = True,
    tfidf_weight: float = 0.3,
) -> list[dict]:
    """Find similar arXiv papers for an uploaded paper.

    Args:
        pid: Upload paper ID
        limit: Maximum number of results
        use_tfidf: Whether to use TF-IDF similarity
        use_embedding: Whether to use embedding similarity
        tfidf_weight: Weight for TF-IDF (embedding weight = 1 - tfidf_weight)

    Returns:
        List of similar papers with scores
    """
    import time as time_module

    t_start = time_module.time()
    logger.trace(f"[BLOCKING] find_similar_papers: starting for pid={pid}, limit={limit}")

    from backend.services.data_service import get_features_cached

    try:
        # Compute or load features for uploaded paper
        logger.trace(f"[BLOCKING] find_similar_papers: computing upload features...")
        upload_features = compute_upload_features(pid)
        logger.trace(f"[BLOCKING] find_similar_papers: upload features computed in {time_module.time() - t_start:.2f}s")
        if not upload_features:
            logger.error(f"Failed to get features for {pid}")
            return []

        # Load global features
        logger.trace(f"[BLOCKING] find_similar_papers: loading global features...")
        t_global = time_module.time()
        global_features = get_features_cached()
        logger.trace(f"[BLOCKING] find_similar_papers: global features loaded in {time_module.time() - t_global:.2f}s")
        if not global_features:
            logger.error("Global features not available")
            return []

        pids_all = global_features.get("pids", [])
        if not pids_all:
            return []

        scores = np.zeros(len(pids_all), dtype=np.float32)
        scores_computed = False

        # Prefer computing similarity in the unified global feature space.
        # This minimizes drift by reusing the exact same feature vector representation.
        upload_x = upload_features.get("x")
        x_global = global_features.get("x")
        if upload_x is not None and x_global is not None:
            try:
                t_sim = time_module.time()
                logger.trace(
                    f"[BLOCKING] find_similar_papers: unified similarity (x_global={getattr(x_global, 'shape', None)}, "
                    f"upload_x={getattr(upload_x, 'shape', None)})"
                )
                scores = (x_global @ upload_x.T).toarray().flatten()
                logger.trace(
                    f"[BLOCKING] find_similar_papers: unified similarity computed in {time_module.time() - t_sim:.2f}s"
                )
                scores_computed = np.any(scores != 0)
            except Exception as e:
                logger.warning(f"Unified feature-space similarity failed, fallback to component scoring: {e}")

        if not scores_computed:
            # Fallback path (kept for robustness / partial feature availability)
            scores = np.zeros(len(pids_all), dtype=np.float32)

            # TF-IDF similarity
            upload_tfidf = upload_features.get("tfidf")
            if use_tfidf and upload_tfidf is not None:
                x_tfidf = global_features.get("x_tfidf")
                if x_tfidf is not None:
                    t_sim = time_module.time()
                    tfidf_scores = (x_tfidf @ upload_tfidf.T).toarray().flatten()
                    logger.trace(
                        f"[BLOCKING] find_similar_papers: TF-IDF similarity computed in {time_module.time() - t_sim:.2f}s"
                    )
                    scores += tfidf_weight * tfidf_scores

            # Embedding similarity (use normalized embeddings from semantic_service to avoid drift)
            upload_emb = upload_features.get("embedding")
            if use_embedding and upload_emb is not None:
                from backend.services.semantic_service import get_paper_embeddings

                paper_emb_data = get_paper_embeddings()
                x_embeddings = None
                if paper_emb_data is not None:
                    x_embeddings = paper_emb_data.get("embeddings")
                if x_embeddings is None:
                    x_embeddings = global_features.get("x_embeddings")
                if x_embeddings is not None:
                    t_sim = time_module.time()
                    emb_scores = (x_embeddings @ upload_emb).flatten()
                    logger.trace(
                        f"[BLOCKING] find_similar_papers: embedding similarity computed in {time_module.time() - t_sim:.2f}s"
                    )
                    scores += (1 - tfidf_weight) * emb_scores

        # Get top results
        top_indices = np.argsort(scores)[::-1][:limit]

        # Build result list
        results = []

        # Get paper PIDs for top results
        top_pids = [pids_all[idx] for idx in top_indices if scores[idx] > 0]

        # Fetch full paper data (title, authors, summary, etc.)
        from backend.services import data_service

        papers_data: dict[str, Any] = {}
        metas_data: dict[str, Any] = {}
        try:
            papers_data = data_service.get_papers_bulk(top_pids)
        except FileNotFoundError:
            papers_data = {}
        except Exception:
            papers_data = {}
        try:
            metas_data = data_service.get_metas()
        except Exception:
            metas_data = {}

        # Import TL;DR extraction for enriched results
        from backend.services.summary_service import extract_tldr_from_summary

        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            paper_pid = pids_all[idx]
            paper = papers_data.get(paper_pid, {}) or {}
            meta = metas_data.get(paper_pid, {}) if isinstance(metas_data, dict) else {}

            # Get TL;DR if available
            tldr = ""
            try:
                tldr = extract_tldr_from_summary(paper_pid) or ""
            except Exception:
                pass

            # Get abstract from paper data
            abstract = paper.get("summary") or meta.get("summary") or ""

            # Format authors (limit to reasonable length)
            authors_list = paper.get("authors") or meta.get("authors") or []
            if authors_list:
                authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
                if len(authors_list) > 5:
                    authors_str += f" et al. ({len(authors_list)} authors)"
            else:
                authors_str = ""

            # Get time string
            time_str = paper.get("_time_str") or meta.get("_time_str") or ""

            results.append(
                {
                    "id": paper_pid,
                    "score": float(scores[idx]),
                    "title": paper.get("title") or meta.get("title") or "",
                    "authors": authors_str,
                    "tldr": tldr,
                    "abstract": abstract,
                    "time": time_str,
                }
            )

        logger.info(f"Found {len(results)} similar papers for {pid}")
        logger.trace(f"[BLOCKING] find_similar_papers: completed in {time_module.time() - t_start:.2f}s")
        return results
    except Exception as e:
        logger.error(f"Error in find_similar_papers for {pid}: {e}")
        return []
