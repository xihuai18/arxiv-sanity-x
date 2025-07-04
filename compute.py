"""
Extracts tfidf features from all paper abstracts and saves them to disk.
Now also supports generating embedding features and concatenating with TF-IDF.
"""

# Multi-core optimization configuration - Ubuntu system
import os
from multiprocessing import cpu_count

from loguru import logger

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

from aslite.db import FEATURES_FILE, FEATURES_FILE_NEW, get_papers_db, save_features

# -----------------------------------------------------------------------------


def sparse_dense_concatenation(tfidf_sparse, embedding_dense):
    """
    保持TF-IDF稀疏性的拼接方案

    Args:
        tfidf_sparse: 稀疏TF-IDF矩阵 (n_samples, n_tfidf_features) - 已经L2归一化
        embedding_dense: 稠密嵌入矩阵 (n_samples, n_embedding_features)

    Returns:
        concatenated_sparse: 拼接后的稀疏矩阵 (n_samples, n_tfidf_features + n_embedding_features)
    """
    # 1. 对稠密嵌入进行归一化（与TF-IDF保持相同尺度）
    embedding_norm = normalize(embedding_dense, norm="l2", axis=1)

    # 2. TF-IDF已经在TfidfVectorizer中归一化，直接使用

    # 3. 将稠密矩阵转换为稀疏格式
    embedding_sparse = sp.csr_matrix(embedding_norm)

    # 4. 使用scipy的稀疏矩阵拼接
    concatenated_sparse = sp.hstack([tfidf_sparse, embedding_sparse], format="csr")

    logger.info(
        f"Sparse-dense concatenation complete: TF-IDF {tfidf_sparse.shape} + Embedding {embedding_norm.shape} = {concatenated_sparse.shape}"
    )
    logger.info(
        f"Sparsity after concatenation: {1 - concatenated_sparse.nnz / (concatenated_sparse.shape[0] * concatenated_sparse.shape[1]):.4f}"
    )

    return concatenated_sparse


class Qwen3EmbeddingVllm:
    """Qwen3 嵌入模型的 API 客户端实现"""

    def __init__(self, model_name_or_path, instruction=None, api_base="http://localhost:51000/v1"):
        if instruction is None:
            instruction = "Extract key concepts from this computer science and AI paper: algorithmic contributions, theoretical insights, implementation techniques, empirical validations, and potential research impacts"
        self.instruction = instruction
        self.client = None
        self.model_path = model_name_or_path
        self.api_base = api_base
        self.model_name = None

    def initialize(self):
        """初始化 API 客户端"""
        try:
            from openai import OpenAI

            logger.info(f"Connecting to vLLM API server: {self.api_base}")
            self.client = OpenAI(api_key="EMPTY", base_url=self.api_base)  # vLLM 不需要真实的 API key

            # 获取可用模型列表
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
        """为文档添加指令前缀"""
        return f"Instruct: {self.instruction}\nQuery: {query}"

    def encode(self, sentences, dim=1024):
        """通过 API 编码文本为嵌入向量"""
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

            # 添加指令前缀
            try:
                instructed_sentences = [self.get_detailed_instruct(sent) for sent in sentences]
            except Exception as e:
                logger.error(f"Error adding instruction prefix: {e}")
                return None

            # 调用 API 生成嵌入
            try:
                response = self.client.embeddings.create(
                    input=instructed_sentences, model=self.model_name, dimensions=dim
                )

                # 提取嵌入向量
                embeddings = np.array([data.embedding for data in response.data], dtype=np.float32)

                # 转换为 torch tensor 以保持与原有接口兼容
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
        """清理客户端资源"""
        try:
            # API 客户端不需要特殊清理
            self.client = None
            logger.debug("API client resources cleaned up")
        except Exception as e:
            logger.warning(f"Error occurred while cleaning up API client: {e}")


def load_existing_embeddings(embed_dim=512):
    """
    加载现有的嵌入特征

    Returns:
        existing_embeddings: dict with 'pids', 'embeddings', 'params'
    """
    try:
        # 尝试加载现有特征文件
        with open(FEATURES_FILE, "rb") as f:
            import pickle

            features = pickle.load(f)

        if features.get("feature_type") == "hybrid_sparse_dense":
            # 检查嵌入参数是否匹配
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
    api_base="http://localhost:51000/v1",
):
    """
    增量生成嵌入向量，优化语料准备顺序

    Args:
        all_pids: 所有论文ID列表
        pdb: 论文数据库对象
        model_path: 嵌入模型路径
        embed_dim: 嵌入维度
        batch_size: 批处理大小
        api_base: vLLM 服务器 API 地址

    Returns:
        embeddings: numpy数组 (n_samples, embed_dim)
    """
    logger.info("Checking for existing embeddings...")
    existing = load_existing_embeddings(embed_dim)

    if existing is not None:
        existing_pids_set = set(existing["pids"])
        logger.info(f"Found {len(existing_pids_set)} existing embeddings")

        # 找出需要新生成的论文ID
        new_pids = []
        new_indices = []
        for i, pid in enumerate(all_pids):
            if pid not in existing_pids_set:
                new_pids.append(pid)
                new_indices.append(i)

        logger.info(f"Need to generate {len(new_pids)} new embeddings")

        if len(new_pids) == 0:
            logger.info("All papers already have embeddings, returning existing embeddings")
            # 重新排序以匹配当前 pids 顺序
            ordered_embeddings = np.zeros((len(all_pids), embed_dim), dtype=np.float32)
            pid_to_idx = {pid: i for i, pid in enumerate(existing["pids"])}

            for i, pid in enumerate(all_pids):
                if pid in pid_to_idx:
                    ordered_embeddings[i] = existing["embeddings"][pid_to_idx[pid]]
                else:
                    # 这种情况理论上不应该发生，用随机向量填充
                    ordered_embeddings[i] = np.random.randn(embed_dim).astype(np.float32)

            return ordered_embeddings

        # 只为需要更新的论文准备语料
        logger.info(f"Preparing embedding corpus for {len(new_pids)} new papers...")
        new_texts = []
        for pid in tqdm(new_pids, desc="准备新论文嵌入语料"):
            d = pdb[pid]
            # 构建用于嵌入的文本
            text = f"Title: {d['title']}\n"
            text += f"Abstract: {d['summary']}"
            new_texts.append(text)
    else:
        logger.info("No existing embeddings found, will generate all embeddings")
        existing_pids_set = set()
        new_pids = all_pids
        new_indices = list(range(len(all_pids)))

        # 为所有论文准备语料
        logger.info(f"Preparing embedding corpus for all {len(all_pids)} papers...")
        new_texts = []
        for pid in tqdm(all_pids, desc="准备嵌入语料"):
            d = pdb[pid]
            # 构建用于嵌入的文本
            text = f"Title: {d['title']}\n"
            text += f"Abstract: {d['summary']}"
            new_texts.append(text)

    # 生成新嵌入
    new_embeddings = None
    if len(new_texts) > 0:
        logger.info(f"Initializing embedding API client: {api_base}")
        model = Qwen3EmbeddingVllm(model_name_or_path=model_path, api_base=api_base)

        if not model.initialize():
            logger.error("API client initialization failed, using random embeddings")
            new_embeddings = np.random.randn(len(new_texts), embed_dim).astype(np.float32)
        else:
            # 批量生成新嵌入
            new_embeddings_list = []

            for i in tqdm(range(0, len(new_texts), batch_size), desc="生成新嵌入向量"):
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

            # 清理模型
            model.stop()

            # 合并新嵌入
            new_embeddings = np.vstack(new_embeddings_list).astype(np.float32)
            logger.info(f"New embeddings generated: {new_embeddings.shape}")

    # 组装最终的嵌入矩阵
    final_embeddings = np.zeros((len(all_pids), embed_dim), dtype=np.float32)

    if existing is not None:
        # 先填充现有嵌入
        pid_to_idx = {pid: i for i, pid in enumerate(existing["pids"])}
        for i, pid in enumerate(all_pids):
            if pid in pid_to_idx:
                final_embeddings[i] = existing["embeddings"][pid_to_idx[pid]]

    # 填充新生成的嵌入
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

    # 新增嵌入相关参数
    parser.add_argument("--use_embeddings", action="store_false", help="是否生成和使用嵌入向量")
    parser.add_argument("--embed_model", type=str, default="./qwen3-embed-0.6B", help="嵌入模型路径")
    parser.add_argument("--embed_dim", type=int, default=512, help="嵌入向量维度")
    parser.add_argument("--embed_batch_size", type=int, default=2048, help="嵌入生成批大小")
    parser.add_argument(
        "--embed_api_base", type=str, default="http://localhost:51000/v1", help="vLLM 嵌入服务器 API 地址"
    )

    args = parser.parse_args()
    print(args)

    start_time = time.time()

    # 优化的 TF-IDF 参数，针对学术论文特点
    v = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        # 改进的 token pattern：捕获更多学术词汇（如 COVID-19, state-of-the-art）
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b|\b[a-zA-Z]+\-[a-zA-Z]+\b",
        ngram_range=(1, args.ngram_max),
        max_features=args.num,
        norm="l2",  # L2 归一化，对相似度计算很重要
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,  # 使用对数尺度的词频
        max_df=args.max_df,
        min_df=args.min_df,
        dtype=np.float32,  # 使用 float32 节省空间
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

    # 获取所有论文ID
    pids = list(pdb.keys())

    # 决定最终使用的特征
    if args.use_embeddings:
        logger.info("Generating embedding features...")

        # 增量生成嵌入向量（语料准备在函数内部完成）
        embeddings = generate_embeddings_incremental(
            pids,
            pdb,
            model_path=args.embed_model,
            embed_dim=args.embed_dim,
            batch_size=args.embed_batch_size,
            api_base=args.embed_api_base,
        )

        if embeddings is not None:
            # 执行稀疏-稠密拼接
            logger.info("Performing sparse-dense matrix concatenation...")
            x_final = sparse_dense_concatenation(x, embeddings)

            # 保存特征
            features = {
                "pids": pids,
                "x": x_final,  # 拼接后的混合特征
                "x_tfidf": x,  # 原始TF-IDF特征
                "x_embeddings": embeddings,  # 原始嵌入特征
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
        # 仅使用TF-IDF特征
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

    # 确保分布式资源被清理
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed resources cleaned up before program exit")
    except Exception as e:
        logger.warning(f"Failed to clean up distributed resources before program exit: {e}")
