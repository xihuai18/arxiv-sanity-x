"""
Extracts tfidf features from all paper abstracts and saves them to disk.
Now also supports generating embedding features and concatenating with TF-IDF.
"""

# Multi-core optimization configuration - Ubuntu system
import os
from multiprocessing import cpu_count

from loguru import logger

from vars import VLLM_EMBED_PORT

# Set multi-threading environment variables
num_threads = min(cpu_count(), 192)  # Reasonable thread limit
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

# Try to use Intel extensions (if available)
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    logger.info(f"Intel scikit-learn extension enabled with {num_threads} threads")
    USE_INTEL_EXT = True
except ImportError:
    logger.info(f"Using standard sklearn with {num_threads} threads")
    USE_INTEL_EXT = False

import argparse
import shutil
import time
from random import shuffle

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from aslite.db import (
    FEATURES_FILE,
    FEATURES_FILE_NEW,
    get_papers_db,
    load_features,
    save_features,
)

# -----------------------------------------------------------------------------


def sparse_dense_concatenation(tfidf_sparse, embedding_dense):
    """
    Concatenation scheme that preserves TF-IDF sparsity

    Args:
        tfidf_sparse: Sparse TF-IDF matrix (n_samples, n_tfidf_features) - already L2 normalized
        embedding_dense: Dense embedding matrix (n_samples, n_embedding_features)

    Returns:
        concatenated_sparse: Concatenated sparse matrix (n_samples, n_tfidf_features + n_embedding_features)
    """
    # 1. Normalize dense embeddings (to maintain same scale as TF-IDF)
    embedding_norm = normalize(embedding_dense, norm="l2", axis=1)

    # 2. TF-IDF is already normalized in TfidfVectorizer, use directly

    # 3. Convert dense matrix to sparse format
    embedding_sparse = sp.csr_matrix(embedding_norm)

    # 4. Use scipy sparse matrix concatenation
    concatenated_sparse = sp.hstack([tfidf_sparse, embedding_sparse], format="csr")

    logger.info(
        f"Sparse-dense concatenation complete: TF-IDF {tfidf_sparse.shape} + Embedding {embedding_norm.shape} = {concatenated_sparse.shape}"
    )
    logger.info(
        f"Sparsity after concatenation: {1 - concatenated_sparse.nnz / (concatenated_sparse.shape[0] * concatenated_sparse.shape[1]):.4f}"
    )

    return concatenated_sparse


class Qwen3EmbeddingVllm:
    """Qwen3 embedding model API client implementation"""

    def __init__(self, model_name_or_path, instruction=None, api_base=None):
        if api_base is None:
            api_base = f"http://localhost:{VLLM_EMBED_PORT}/v1"
        if instruction is None:
            instruction = "Extract key concepts from this computer science and AI paper: algorithmic contributions, theoretical insights, implementation techniques, empirical validations, and potential research impacts"
        self.instruction = instruction
        self.client = None
        self.model_path = model_name_or_path
        self.api_base = api_base
        self.model_name = None

    def initialize(self):
        """Initialize API client"""
        try:
            from openai import OpenAI

            logger.info(f"Connecting to vLLM API server: {self.api_base}")
            self.client = OpenAI(api_key="EMPTY", base_url=self.api_base)  # vLLM doesn't need real API key

            # Get available model list
            try:
                models = self.client.models.list()
                if models.data:
                    self.model_name = models.data[0].id
                    logger.info(f"Using model: {self.model_name}")
                else:
                    logger.error("No available model found")
                    return False
            except Exception as e:
                logger.error(f"Failed to get model list: {e}")
                return False

            logger.info("Embedding API client initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"Failed to import OpenAI library: {e}")
            logger.error("Please make sure openai is installed: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Error occurred while initializing API client: {e}")
            logger.error(f"API address: {self.api_base}")
            return False

    def get_detailed_instruct(self, query: str) -> str:
        """Add instruction prefix to document"""
        return f"Instruct: {self.instruction}\nQuery: {query}"

    def encode(self, sentences, dim=1024):
        """Encode text to embedding vectors via API"""
        try:
            if self.client is None:
                logger.error("API client not initialized, cannot encode")
                return None

            if not sentences:
                logger.error("Input sentence list is empty")
                return None

            if not self.model_name:
                logger.error("Model name not set")
                return None

            # Add instruction prefix
            try:
                instructed_sentences = [self.get_detailed_instruct(sent) for sent in sentences]
            except Exception as e:
                logger.error(f"Error adding instruction prefix: {e}")
                return None

            # Call API to generate embeddings
            try:
                response = self.client.embeddings.create(
                    input=instructed_sentences, model=self.model_name, dimensions=dim
                )

                # Extract embedding vectors
                embeddings = np.array([data.embedding for data in response.data], dtype=np.float32)

                # Convert to torch tensor to maintain compatibility with original interface
                import torch

                return torch.from_numpy(embeddings)

            except Exception as e:
                logger.error(f"API call failed: {e}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import required library: {e}")
            return None
        except Exception as e:
            logger.error(f"Unknown error occurred during encoding: {e}")
            logger.error(f"Input sentence count: {len(sentences) if sentences else 0}")
            logger.error(f"Target dimension: {dim}")
            return None

    def stop(self):
        """Clean up client resources"""
        try:
            # API client doesn't need special cleanup
            self.client = None
            logger.debug("API client resources cleaned up")
        except Exception as e:
            logger.warning(f"Error occurred while cleaning up API client: {e}")


def load_existing_embeddings(embed_dim=512):
    """
    Load existing embedding features

    Returns:
        existing_embeddings: dict with 'pids', 'embeddings', 'params'
    """
    try:
        # Try to load existing feature file (with numpy pickle compatibility)
        features = load_features()

        if features.get("feature_type") == "hybrid_sparse_dense":
            # Check if embedding parameters match
            embed_params = features.get("embedding_params", {})
            if embed_params.get("embed_dim") == embed_dim:
                logger.info(f"Loaded existing embedding features: {features['x_embeddings'].shape}")
                return {"pids": features["pids"], "embeddings": features["x_embeddings"], "params": embed_params}
            else:
                logger.warning(
                    f"Embedding dimension mismatch: existing {embed_params.get('embed_dim')} vs target {embed_dim}"
                )
                logger.warning("Will regenerate all embeddings")

    except FileNotFoundError:
        logger.info("No existing feature file found")
    except Exception as e:
        logger.error(f"Error occurred while loading existing embeddings: {e}")
        logger.error(f"File path: {FEATURES_FILE}")
        import traceback

        logger.error(f"Detailed error info: {traceback.format_exc()}")

    return None


def generate_embeddings_incremental(
    all_pids,
    pdb,
    model_path="./qwen3-embed-0.6B",
    embed_dim=512,
    batch_size=512,
    api_base=None,
):
    if api_base is None:
        api_base = f"http://localhost:{VLLM_EMBED_PORT}/v1"
    """
    Generate embedding vectors incrementally, optimize corpus preparation order

    Args:
        all_pids: List of all paper IDs
        pdb: Paper database object
        model_path: Embedding model path
        embed_dim: Embedding dimension
        batch_size: Batch size
        api_base: vLLM server API address

    Returns:
        embeddings: numpy array (n_samples, embed_dim)
    """
    logger.info("Checking for existing embeddings...")
    existing = load_existing_embeddings(embed_dim)

    if existing is not None:
        existing_pids_set = set(existing["pids"])
        logger.info(f"Found {len(existing_pids_set)} existing embeddings")

        # Find paper IDs that need new generation
        new_pids = []
        new_indices = []
        for i, pid in enumerate(all_pids):
            if pid not in existing_pids_set:
                new_pids.append(pid)
                new_indices.append(i)

        logger.info(f"Need to generate {len(new_pids)} new embeddings")

        if len(new_pids) == 0:
            logger.info("All papers already have embeddings, returning existing embeddings")
            # Reorder to match current pids order
            ordered_embeddings = np.zeros((len(all_pids), embed_dim), dtype=np.float32)
            pid_to_idx = {pid: i for i, pid in enumerate(existing["pids"])}

            for i, pid in enumerate(all_pids):
                if pid in pid_to_idx:
                    ordered_embeddings[i] = existing["embeddings"][pid_to_idx[pid]]
                else:
                    # This should not happen in theory, fill with random vectors
                    ordered_embeddings[i] = np.random.randn(embed_dim).astype(np.float32)

            return ordered_embeddings

        # Prepare corpus only for papers that need updates
        logger.info(f"Preparing embedding corpus for {len(new_pids)} new papers...")
        new_texts = []
        for pid in tqdm(new_pids, desc="Preparing new paper embedding corpus"):
            d = pdb[pid]
            # Build text for embedding
            text = f"Title: {d['title']}\n"
            text += f"Abstract: {d['summary']}"
            new_texts.append(text)
    else:
        logger.info("No existing embeddings found, will generate all embeddings")
        existing_pids_set = set()
        new_pids = all_pids
        new_indices = list(range(len(all_pids)))

        # Prepare corpus for all papers
        logger.info(f"Preparing embedding corpus for all {len(all_pids)} papers...")
        new_texts = []
        for pid in tqdm(all_pids, desc="Preparing embedding corpus"):
            d = pdb[pid]
            # Build text for embedding
            text = f"Title: {d['title']}\n"
            text += f"Abstract: {d['summary']}"
            new_texts.append(text)

    # Generate new embeddings
    new_embeddings = None
    if len(new_texts) > 0:
        logger.info(f"Initializing embedding API client: {api_base}")
        model = Qwen3EmbeddingVllm(model_name_or_path=model_path, api_base=api_base)

        if not model.initialize():
            logger.error("API client initialization failed, using random embeddings")
            new_embeddings = np.random.randn(len(new_texts), embed_dim).astype(np.float32)
        else:
            # Batch generate new embeddings
            new_embeddings_list = []

            for i in tqdm(range(0, len(new_texts), batch_size), desc="Generating new embedding vectors"):
                batch_texts = new_texts[i : i + batch_size]

                try:
                    batch_output = model.encode(batch_texts, dim=embed_dim)

                    if batch_output is not None:
                        new_embeddings_list.append(batch_output.cpu().numpy())
                    else:
                        logger.warning(f"Batch {i//batch_size + 1} encoding returned None, using random embeddings")
                        new_embeddings_list.append(np.random.randn(len(batch_texts), embed_dim).astype(np.float32))
                except Exception as e:
                    logger.error(f"Batch {i//batch_size + 1} failed to generate embeddings: {e}")
                    new_embeddings_list.append(np.random.randn(len(batch_texts), embed_dim).astype(np.float32))

            # Clean up model
            model.stop()

            # Merge new embeddings
            new_embeddings = np.vstack(new_embeddings_list).astype(np.float32)
            logger.info(f"New embeddings generated: {new_embeddings.shape}")

    # Assemble final embedding matrix
    final_embeddings = np.zeros((len(all_pids), embed_dim), dtype=np.float32)

    if existing is not None:
        # Fill existing embeddings first
        pid_to_idx = {pid: i for i, pid in enumerate(existing["pids"])}
        for i, pid in enumerate(all_pids):
            if pid in pid_to_idx:
                final_embeddings[i] = existing["embeddings"][pid_to_idx[pid]]

    # Fill newly generated embeddings
    if new_embeddings is not None:
        new_embed_idx = 0
        for i in new_indices:
            final_embeddings[i] = new_embeddings[new_embed_idx]
            new_embed_idx += 1

    logger.info(f"Incremental embedding generation complete: total {final_embeddings.shape}")
    return final_embeddings


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arxiv Computor - Optimized for 400k papers with hybrid features")
    parser.add_argument(
        "-n", "--num", type=int, default=50000, help="number of tfidf features (optimized for 400k papers)"
    )
    parser.add_argument("--min_df", type=int, default=20, help="min document frequency (for 400k papers)")
    parser.add_argument("--max_df", type=float, default=0.10, help="max document frequency (for 400k papers)")

    parser.add_argument(
        "--ngram_max", type=int, default=1, help="maximum n-gram size (unigram only for large datasets)"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="maximum number of documents to use when training tfidf, or -1 to disable",
    )

    # New embedding related parameters
    parser.add_argument(
        "--use_embeddings",
        action="store_true",
        default=True,
        help="Enable embedding vectors (default: on; use --no-embeddings to disable)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_false",
        dest="use_embeddings",
        help="Disable embedding vectors",
    )
    parser.add_argument("--embed_model", type=str, default="./qwen3-embed-0.6B", help="Embedding model path")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding vector dimension")
    parser.add_argument("--embed_batch_size", type=int, default=2048, help="Embedding generation batch size")
    parser.add_argument(
        "--embed_api_base",
        type=str,
        default=f"http://localhost:{VLLM_EMBED_PORT}/v1",
        help="vLLM embedding server API address",
    )

    args = parser.parse_args()
    print(args)

    start_time = time.time()

    # Optimized TF-IDF parameters for academic papers
    v = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        # Improved token pattern: capture more academic vocabulary (e.g. COVID-19, state-of-the-art)
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b|\b[a-zA-Z]+\-[a-zA-Z]+\b",
        ngram_range=(1, args.ngram_max),
        max_features=args.num,
        norm="l2",  # L2 normalization, important for similarity calculation
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,  # Use logarithmic scale term frequency
        max_df=args.max_df,
        min_df=args.min_df,
        dtype=np.float32,  # Use float32 to save space
    )

    pdb = get_papers_db(flag="r")

    def make_corpus(training: bool):
        assert isinstance(training, bool)

        # determine which papers we will use to build tfidf
        if training and args.max_docs > 0 and args.max_docs < len(pdb):
            # crop to a random subset of papers
            keys = list(pdb.keys())
            shuffle(keys)
            keys = keys[: args.max_docs]
        else:
            keys = pdb.keys()

        # yield the abstracts of the papers
        for p in tqdm(keys, desc="loading db"):
            d = pdb[p]
            author_str = " ".join([a["name"] for a in d["authors"]])
            yield " ".join([d["title"], d["summary"], author_str])

    logger.info("Training TF-IDF vectors...")
    t0 = time.time()
    v.fit(make_corpus(training=True))
    logger.info(f"TF-IDF training completed in {time.time() - t0:.1f}s")
    logger.info(f"Vocabulary size: {len(v.vocabulary_)}")

    logger.info("Transforming all documents...")
    t0 = time.time()
    x = v.transform(make_corpus(training=False)).astype(np.float32)
    logger.info(f"Transform completed in {time.time() - t0:.1f}s")
    logger.info(f"Original TF-IDF feature matrix shape: {x.shape}")
    logger.info(f"TF-IDF sparsity: {1 - x.nnz / (x.shape[0] * x.shape[1]):.4f}")

    # Get all paper IDs
    pids = list(pdb.keys())

    # Decide final features to use
    if args.use_embeddings:
        logger.info("Generating embedding features...")

        # Generate embedding vectors incrementally (corpus preparation done inside function)
        embeddings = generate_embeddings_incremental(
            pids,
            pdb,
            model_path=args.embed_model,
            embed_dim=args.embed_dim,
            batch_size=args.embed_batch_size,
            api_base=args.embed_api_base,
        )

        if embeddings is not None:
            # Perform sparse-dense concatenation
            logger.info("Performing sparse-dense matrix concatenation...")
            x_final = sparse_dense_concatenation(x, embeddings)

            # Save features
            features = {
                "pids": pids,
                "x": x_final,  # Concatenated hybrid features
                "x_tfidf": x,  # Original TF-IDF features
                "x_embeddings": embeddings,  # Original embedding features
                "vocab": v.vocabulary_,
                "idf": v._tfidf.idf_,
                "feature_type": "hybrid_sparse_dense",
                "feature_config": {
                    "tfidf_features": args.num,
                    "tfidf_shape": x.shape,
                    "embedding_dim": args.embed_dim,
                    "embedding_shape": embeddings.shape,
                    "hybrid_shape": x_final.shape,
                    "hybrid_sparsity": 1 - x_final.nnz / (x_final.shape[0] * x_final.shape[1]),
                },
                "tfidf_params": {
                    "num_features": args.num,
                    "min_df": args.min_df,
                    "max_df": args.max_df,
                    "ngram_max": args.ngram_max,
                },
                "embedding_params": {
                    "model_path": args.embed_model,
                    "embed_dim": args.embed_dim,
                },
            }
        else:
            logger.warning("Embedding generation failed, falling back to TF-IDF features only")
            x_final = x
            features = {
                "pids": pids,
                "x": x_final,
                "x_tfidf": x,
                "vocab": v.vocabulary_,
                "idf": v._tfidf.idf_,
                "feature_type": "tfidf_only",
                "tfidf_params": {
                    "num_features": args.num,
                    "min_df": args.min_df,
                    "max_df": args.max_df,
                    "ngram_max": args.ngram_max,
                },
            }
    else:
        # Use only TF-IDF features
        logger.info("Using only TF-IDF features (embeddings not enabled)")
        x_final = x
        features = {
            "pids": pids,
            "x": x_final,
            "x_tfidf": x,
            "vocab": v.vocabulary_,
            "idf": v._tfidf.idf_,
            "feature_type": "tfidf_only",
            "tfidf_params": {
                "num_features": args.num,
                "min_df": args.min_df,
                "max_df": args.max_df,
                "ngram_max": args.ngram_max,
            },
        }

    logger.info("Saving features to disk...")
    save_features(features)

    logger.info("Copying to production file...")
    shutil.copyfile(FEATURES_FILE_NEW, FEATURES_FILE)

    total_time = time.time() - start_time
    logger.info(f"Feature extraction complete, total time: {total_time:.1f}s")
    logger.info(f"Final feature shape: {x_final.shape}")
    logger.info(f"Final feature type: {features.get('feature_type', 'unknown')}")

    if "feature_config" in features:
        logger.info("Feature configuration details:")
        for key, value in features["feature_config"].items():
            logger.info(f"  {key}: {value}")

    # Ensure distributed resources are cleaned up
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed resources cleaned up before program exit")
    except Exception as e:
        logger.warning(f"Failed to clean up distributed resources before program exit: {e}")
