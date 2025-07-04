"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

# Multi-core optimization configuration - Ubuntu system
import os
from multiprocessing import Pool, cpu_count

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

import re
import sys
import time
from pathlib import Path
from random import shuffle

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from flask import g  # global session-level object
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from sklearn import svm
from tqdm import tqdm

from aslite.db import (
    FEATURES_FILE,
    get_combined_tags_db,
    get_email_db,
    get_keywords_db,
    get_last_active_db,
    get_metas_db,
    get_papers_db,
    get_tags_db,
    load_features,
)
from compute import Qwen3EmbeddingVllm
from paper_summarizer import (
    generate_paper_summary as generate_paper_summary_from_module,
)

# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = 100  # number of papers to return per page
MAX_RESULTS = RET_NUM * 10  # Process at most 10 pages of results, avoid processing all data

# Feature cache related global variables
FEATURES_CACHE = None
FEATURES_FILE_MTIME = 0  # Feature file modification time
FEATURES_CACHE_TIME = 0  # Cache creation time

# Papers and metas cache related global variables (存储在同一个数据库文件中)
PAPERS_CACHE = None
METAS_CACHE = None
PIDS_CACHE = None
PAPERS_DB_FILE_MTIME = 0  # Papers数据库文件修改时间
PAPERS_DB_CACHE_TIME = 0  # 数据库缓存创建时间

# Database file path
from aslite.db import PAPERS_DB_FILE as PAPERS_DB_PATH

app = Flask(__name__)

# set the secret key so we can cryptographically sign cookies and maintain sessions
if os.path.isfile("secret_key.txt"):
    # example of generating a good key on your system is:
    # import secrets; secrets.token_urlsafe(16)
    sk = open("secret_key.txt").read().strip()
else:
    logger.warning("No secret key found, using default `devkey`")
    sk = "devkey"
app.secret_key = sk


# -----------------------------------------------------------------------------
# Helper function for Chinese text detection
def calculate_chinese_ratio(text: str) -> float:
    """
    计算文本中中文字符的占比

    Args:
        text: 输入文本

    Returns:
        float: 中文字符占比 (0.0 到 1.0)
    """
    if not text or not text.strip():
        return 0.0

    # 移除空白字符后的文本
    clean_text = re.sub(r"\s+", "", text)
    if not clean_text:
        return 0.0

    # 统计中文字符数量 (包括中文标点符号)
    chinese_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", clean_text)
    chinese_count = len(chinese_chars)

    # 计算占比
    total_chars = len(clean_text)
    ratio = chinese_count / total_chars if total_chars > 0 else 0.0

    return ratio


# -----------------------------------------------------------------------------
# Helper function to reduce code duplication
def _get_user_data(db_func, attr_name):
    """Generic function to get user data from database"""
    if g.user is None:
        return {}
    if not hasattr(g, attr_name):
        with db_func() as db:
            data = db[g.user] if g.user in db else {}
        setattr(g, attr_name, data)
    return getattr(g, attr_name)


# globals that manage the (lazy) loading of various state for a request
def get_tags():
    return _get_user_data(get_tags_db, "_tags")


def get_combined_tags():
    return _get_user_data(get_combined_tags_db, "_combined_tags")


def get_keys():
    return _get_user_data(get_keywords_db, "_keys")


# -----------------------------------------------------------------------------
# Intelligent unified data caching functionality


def get_data_cached():
    """
    智能统一数据缓存加载函数
    - 统一管理papers和metas数据的加载
    - 检测数据库文件更新并自动重新加载
    - 记录详细的缓存状态日志
    """
    global PAPERS_CACHE, METAS_CACHE, PIDS_CACHE
    global PAPERS_DB_FILE_MTIME, PAPERS_DB_CACHE_TIME

    current_time = time.time()

    # 检查数据库文件是否存在
    if not os.path.exists(PAPERS_DB_PATH):
        logger.error(f"Papers database file not found: {PAPERS_DB_PATH}")
        raise FileNotFoundError(f"Papers database file not found: {PAPERS_DB_PATH}")

    # 获取当前数据库文件修改时间
    current_file_mtime = os.path.getmtime(PAPERS_DB_PATH)

    # 检查是否需要重新加载（首次加载或文件更新）
    need_reload = (
        PAPERS_CACHE is None  # 首次加载
        or METAS_CACHE is None  # 首次加载
        or current_file_mtime > PAPERS_DB_FILE_MTIME  # 文件更新
    )

    if need_reload:
        logger.info("Loading papers and metas from database...")
        if PAPERS_CACHE is not None:
            logger.trace(f"Database file updated (old mtime: {PAPERS_DB_FILE_MTIME}, new mtime: {current_file_mtime})")

        start_time = time.time()

        try:
            # 同时加载papers和metas数据
            with get_papers_db() as papers_db:
                PAPERS_CACHE = {k: v for k, v in tqdm(papers_db.items(), desc="loading papers db")}

            with get_metas_db() as metas_db:
                METAS_CACHE = {k: v for k, v in tqdm(metas_db.items(), desc="loading metas db")}
                PIDS_CACHE = list(METAS_CACHE.keys())

            # 更新缓存时间戳
            PAPERS_DB_FILE_MTIME = current_file_mtime
            PAPERS_DB_CACHE_TIME = current_time

            load_time = time.time() - start_time
            logger.info(f"Data loaded successfully in {load_time:.3f}s")
            logger.trace(f"Number of papers: {len(PAPERS_CACHE)}")
            logger.trace(f"Number of metas: {len(METAS_CACHE)}")
            logger.trace(f"Number of pids: {len(PIDS_CACHE)}")

        except Exception as e:
            logger.error(f"Failed to load papers and metas: {e}")
            raise
    else:
        # 使用缓存
        pass

    return PAPERS_CACHE, METAS_CACHE, PIDS_CACHE


# -----------------------------------------------------------------------------
# Cached data access functions


def get_pids():
    """获取所有论文ID列表"""
    get_data_cached()  # 确保缓存已加载
    return PIDS_CACHE


def get_papers():
    """获取论文数据库"""
    get_data_cached()  # 确保缓存已加载
    return PAPERS_CACHE


def get_metas():
    """获取元数据数据库"""
    get_data_cached()  # 确保缓存已加载
    return METAS_CACHE


# 初始化缓存（在应用启动时预加载）
logger.info("Initializing unified data caches on startup...")
get_data_cached()

# 设置定时刷新缓存的调度器
scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
scheduler.add_job(get_data_cached, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
scheduler.start()


@app.before_request
def before_request():
    g.user = session.get("user", None)

    # record activity on this user so we can reserve periodic
    # recommendations heavy compute only for active users
    if g.user:
        with get_last_active_db(flag="c") as last_active_db:
            last_active_db[g.user] = int(time.time())


@app.teardown_request
def close_connection(error=None):
    # papers和metas现在使用全局缓存，无需清理
    # 所有数据访问都直接使用全局变量
    pass


# -----------------------------------------------------------------------------
# Intelligent feature caching functionality


def get_features_cached():
    """
    Intelligent feature cache loading function
    - Detect if feature file is updated
    - Automatically reload updated features
    - Record detailed cache status logs
    """
    global FEATURES_CACHE, FEATURES_FILE_MTIME, FEATURES_CACHE_TIME

    current_time = time.time()

    # Check if feature file exists
    if not os.path.exists(FEATURES_FILE):
        logger.error(f"Features file not found: {FEATURES_FILE}")
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    # Get current modification time of feature file
    current_file_mtime = os.path.getmtime(FEATURES_FILE)

    # Determine if features need to be reloaded
    need_reload = FEATURES_CACHE is None or current_file_mtime > FEATURES_FILE_MTIME  # First load  # File updated

    if need_reload:
        logger.info(f"Loading features from disk...")
        if FEATURES_CACHE is not None:
            logger.trace(f"Features file updated (old mtime: {FEATURES_FILE_MTIME}, new mtime: {current_file_mtime})")

        start_time = time.time()

        try:
            # Load features
            FEATURES_CACHE = load_features()
            FEATURES_FILE_MTIME = current_file_mtime
            FEATURES_CACHE_TIME = current_time

            load_time = time.time() - start_time
            logger.info(f"Features loaded successfully in {load_time:.3f}s")
            logger.trace(f"Feature matrix shape: {FEATURES_CACHE['x'].shape}")
            logger.trace(f"Number of papers: {len(FEATURES_CACHE['pids'])}")
            logger.trace(f"Vocabulary size: {len(FEATURES_CACHE['vocab'])}")

        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            raise
    else:
        # Use cache
        pass
        # cache_age = current_time - FEATURES_CACHE_TIME
        # logger.trace(f"Using cached features (age: {cache_age:.1f}s)")

    return FEATURES_CACHE


# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests


def render_pid(pid):
    # render a single paper with just the information we need for the UI
    pdb = get_papers()
    tags = get_tags()
    thumb_path = "static/thumb/" + pid + ".jpg"
    thumb_url = thumb_path if os.path.isfile(thumb_path) else ""
    d = pdb[pid]
    return dict(
        weight=0.0,
        id=d["_id"],
        title=d["title"],
        time=d["_time_str"],
        authors=", ".join(a["name"] for a in d["authors"]),
        tags=", ".join(t["term"] for t in d["tags"]),
        utags=[t for t, pids in tags.items() if pid in pids],
        summary=d["summary"],
        thumb_url=thumb_url,
    )


def _apply_limit(pids, scores, limit):
    """Apply limit to results if specified"""
    if limit is not None and len(pids) > limit:
        return pids[:limit], scores[:limit]
    return pids, scores


def random_rank(limit=None):
    mdb = get_metas()
    pids = list(mdb.keys())
    shuffle(pids)
    scores = [0 for _ in pids]
    return _apply_limit(pids, scores, limit)


def time_rank(limit=None):
    mdb = get_metas()
    ms = sorted(mdb.items(), key=lambda kv: kv[1]["_time"], reverse=True)

    if limit is not None:
        ms = ms[:limit]

    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v["_time"]) / 60 / 60 / 24 for k, v in ms]  # time delta in days
    return pids, scores


def _filter_by_time_with_tags(pids, time_filter, user_tagged_pids=None):
    """
    智能时间过滤：保留有标签的论文（即使超出时间窗口）和时间窗口内的论文
    Filter papers by time but keep tagged papers even if outside time window
    """
    if not time_filter:
        return pids, list(range(len(pids)))

    mdb = get_metas()
    kv = {k: v for k, v in mdb.items()}
    tnow = time.time()
    deltat = float(time_filter) * 60 * 60 * 24

    # 获取用户标记的所有论文ID集合
    tagged_set = set()
    if user_tagged_pids:
        tagged_set = user_tagged_pids

    # Find indices of papers that meet criteria:
    # 1. 在时间窗口内的论文，或
    # 2. 用户已标记的论文（即使超出时间窗口）
    valid_indices = []
    valid_pids = []
    for i, pid in enumerate(pids):
        if pid in kv:
            # 检查是否在时间窗口内
            in_time_window = (tnow - kv[pid]["_time"]) < deltat
            # 检查是否被用户标记
            is_tagged = pid in tagged_set

            # 满足任一条件即保留
            if in_time_window or is_tagged:
                valid_indices.append(i)
                valid_pids.append(pid)

    return valid_pids, valid_indices


def _filter_by_time(pids, time_filter):
    """原始时间过滤函数，保持向后兼容"""
    return _filter_by_time_with_tags(pids, time_filter, None)


def svm_rank(tags: str = "", s_pids: str = "", C: float = 0.02, logic: str = "and", time_filter: str = ""):
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or s_pids):
        return [], [], []

    # Use intelligent cache to load features
    s_time = time.time()
    features = get_features_cached()  # Replace: features = load_features()
    x, pids = features["x"], features["pids"]

    # 收集用户标记的所有论文ID，用于智能时间过滤
    user_tagged_pids = set()
    if tags:
        tags_db = get_tags()
        tags_filter_to = tags_db.keys() if tags == "all" else set(map(str.strip, tags.split(",")))
        for tag in tags_filter_to:
            if tag in tags_db:
                user_tagged_pids.update(tags_db[tag])

    if s_pids:
        user_tagged_pids.update(map(str.strip, s_pids.split(",")))

    # 应用智能时间过滤：保留有标签的论文和时间窗口内的论文
    if time_filter:
        pids, time_valid_indices = _filter_by_time_with_tags(pids, time_filter, user_tagged_pids)
        if not pids:
            return [], [], []
        x = x[time_valid_indices]
        logger.trace(
            f"After intelligent time filtering, kept {len(pids)} papers (including tagged papers and those within {time_filter} days)"
        )

    n, d = x.shape
    ptoi, itop = {}, {}
    for i, p in enumerate(pids):
        ptoi[p] = i
        itop[i] = p

    # construct the positive set
    y = np.zeros(n, dtype=np.float32)
    weight_offset = 0.0

    # Process PIDs
    if s_pids:
        s_pids = set(map(str.strip, s_pids.split(",")))
        for p_i, pid in enumerate(s_pids):
            if pid in ptoi:  # Ensure PID exists
                if logic == "and":
                    y[ptoi[pid]] = 1.0 + p_i + weight_offset
                else:
                    y[ptoi[pid]] = max(y[ptoi[pid]], 1.0)
        weight_offset += len(s_pids)  # Reserve space for tags weight

    # Process Tags (can coexist with PIDs)
    if tags:
        tags_db = get_tags()
        logger.trace(f"Available tags in tags_db: {list(tags_db.keys())}")
        tags_filter_to = tags_db.keys() if tags == "all" else set(map(str.strip, tags.split(",")))
        logger.trace(f"Tags to filter: {tags_filter_to}")
        for t_i, tag in enumerate(tags_filter_to):
            if tag in tags_db:  # Ensure tag exists
                t_pids = tags_db[tag]
                logger.trace(f"Tag '{tag}' has {len(t_pids)} papers")
                found_count = 0
                for p_i, pid in enumerate(t_pids):
                    if pid in ptoi:  # Ensure PID exists
                        found_count += 1
                        if logic == "and":
                            y[ptoi[pid]] = max(y[ptoi[pid]], 1.0 + t_i + weight_offset)
                        else:
                            y[ptoi[pid]] = max(y[ptoi[pid]], 1.0)
                logger.trace(f"Found {found_count} papers from tag '{tag}' in current features")
            else:
                logger.trace(f"Tag '{tag}' not found in tags_db")
    e_time = time.time()

    logger.trace(f"feature loading/caching for {e_time - s_time:.5f}s")

    if y.sum() == 0:
        return [], [], []  # there are no positives?

    s_time = time.time()

    # classify - 优化参数加速训练，保持准确率
    clf = svm.LinearSVC(
        class_weight="balanced",
        verbose=0,
        max_iter=5000,  # 适当减少迭代次数
        tol=1e-3,
        C=C,
        dual=False,  # 对高维特征更快
        random_state=42,  # 确保结果可重现
        fit_intercept=True,  # 通常提高准确率
        multi_class="ovr",  # 一对其余，对二分类更快
    )
    # feature_map_nystroem = Nystroem(
    #     random_state=0, n_components=100, n_jobs=-1
    # )
    # x = feature_map_nystroem.fit_transform(x)
    # rbf_feature = RBFSampler(gamma=1, random_state=0, n_components=200)
    # x = rbf_feature.fit_transform(x)
    # e_time = time.time()
    # logger.trace(f"Dimension reduction for {e_time - s_time:.5f}s")

    clf.fit(x, y)
    e_time = time.time()
    logger.trace(f"SVM fitting data for {e_time - s_time:.5f}s")

    if logic == "and":
        s = clf.decision_function(x)
        # logger.trace(f"svm_rank: {s.shape}")
        if len(s.shape) > 1:
            s = s[:, 1:].mean(axis=-1)
    else:
        s = clf.decision_function(x)
    e_time = time.time()
    logger.trace(f"SVM decsion function for {e_time - s_time:.5f}s")
    sortix = np.argsort(-s)
    pids = [itop[ix] for ix in sortix]
    scores = [100 * float(s[ix]) for ix in sortix]

    # get the words that score most positively and most negatively for the svm
    ivocab = {v: k for k, v in features["vocab"].items()}  # index to word mapping
    weights = clf.coef_[0]  # (n_features,) weights of the trained svm

    # Only analyze TF-IDF part weights (vocab size), ignore embedding dimensions
    vocab_size = len(ivocab)
    tfidf_weights = weights[:vocab_size]  # Only TF-IDF weights

    sortix = np.argsort(-tfidf_weights)
    e_time = time.time()
    logger.trace(f"rank calculation for {e_time - s_time:.5f}s")
    # logger.trace(f"Total features: {len(weights)}, TF-IDF features/: {vocab_size}")

    words = []
    for ix in list(sortix[:40]) + list(sortix[-20:]):
        words.append(
            {
                "word": ivocab[ix],
                "weight": float(tfidf_weights[ix]),
            }
        )

    return pids, scores, words


def count_match(q, pid_start, n_pids):
    q = q.lower().strip()
    qs = q.split()  # split query by spaces and lowercase
    sub_pairs = []
    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs) / len(qs)
    # matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs) / len(qs)
    match_t = lambda s: int(q in s.lower())
    pdb = get_papers()
    for pid in get_pids()[pid_start : pid_start + n_pids]:
        p = pdb[pid]
        score = 0.0
        score += 20.0 * match_t(" ".join([a["name"] for a in p["authors"]]))
        score += 20.0 * match(p["title"])
        score += 10.0 * match_t(p["title"])
        score += 5.0 * match(p["summary"])
        if score > 0:
            sub_pairs.append((score, pid))
    return sub_pairs


def search_rank(q: str = "", limit=None):
    if not q:
        return [], []  # no query? no results
    n_pids = len(get_pids())
    chunk_size = 20000
    n_process = min(cpu_count() // 2, n_pids // chunk_size)
    with Pool(n_process) as pool:
        sub_pairs_list = pool.starmap(
            count_match,
            [(q, pid_start, chunk_size) for pid_start in range(0, n_pids, chunk_size)],
        )
        pairs = sum(sub_pairs_list, [])

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return _apply_limit(pids, scores, limit)


# 全局语义搜索相关变量
_semantic_model = None
_cached_embeddings = None


def get_semantic_model():
    """获取语义模型实例（通过 API 调用）"""
    global _semantic_model
    if _semantic_model is None:
        try:
            logger.info("Initializing semantic model API client for query encoding...")
            _semantic_model = Qwen3EmbeddingVllm(
                model_name_or_path="./qwen3-embed-0.6B",
                instruction="Extract key concepts from this query to search computer science and AI paper",
                api_base="http://localhost:51000/v1",
            )
            if not _semantic_model.initialize():
                logger.error("Failed to initialize semantic model API client")
                _semantic_model = None
        except Exception as e:
            logger.error(f"Error initializing semantic model API client: {e}")
            _semantic_model = None
    return _semantic_model


def get_paper_embeddings():
    """获取论文嵌入向量（从特征文件加载）"""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    try:
        features = get_features_cached()
        if features.get("feature_type") == "hybrid_sparse_dense" and "x_embeddings" in features:
            logger.trace("Using pre-computed embeddings from features file")
            _cached_embeddings = {"embeddings": features["x_embeddings"], "pids": features["pids"]}
            return _cached_embeddings
        else:
            logger.warning("No embeddings found in features file")
            return None
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None


get_semantic_model()
get_paper_embeddings()


def semantic_search_rank(q: str = "", limit=None):
    """执行纯语义搜索"""
    if not q:
        return [], []

    # 获取论文嵌入
    paper_data = get_paper_embeddings()
    if paper_data is None:
        logger.error("No paper embeddings available")
        return [], []

    # 获取模型
    model = get_semantic_model()
    if model is None:
        logger.error("Semantic model not available")
        return [], []

    try:
        # 编码查询
        # logger.trace(f"Encoding query: {q}")
        query_embedding = model.encode(
            [model.get_detailed_instruct(q)],
            dim=512,
        )
        if query_embedding is None:
            logger.error("Failed to encode query")
            return [], []
        query_embedding = query_embedding.cpu().numpy()

        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(query_embedding, paper_data["embeddings"])[0]

        # 获取top-k结果
        if limit:
            top_indices = np.argpartition(-similarities, min(limit, len(similarities) - 1))[:limit]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)

        pids = [paper_data["pids"][i] for i in top_indices]
        scores = [float(similarities[i]) * 100 for i in top_indices]

        return pids, scores

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return [], []


def hybrid_search_rank(q: str = "", limit=None, semantic_weight=0.5):
    """混合搜索：融合关键词和语义搜索结果"""
    if not q:
        return [], [], {}

    # 并行执行两种搜索
    keyword_pids, keyword_scores = search_rank(q, limit=limit * 2 if limit else None)
    semantic_pids, semantic_scores = semantic_search_rank(q, limit=limit * 2 if limit else None)

    # 创建分数映射
    keyword_score_map = {pid: score for pid, score in zip(keyword_pids, keyword_scores)} if keyword_pids else {}
    semantic_score_map = {pid: score for pid, score in zip(semantic_pids, semantic_scores)} if semantic_pids else {}

    # 融合结果
    combined_scores = {}
    score_details = {}  # 保存每个paper的详细分数信息

    # 归一化参数
    max_keyword = max(keyword_scores) if keyword_scores else 1.0
    max_semantic = max(semantic_scores) if semantic_scores else 1.0

    # 获取所有涉及的PIDs
    all_pids = set(keyword_pids + semantic_pids)

    for pid in all_pids:
        keyword_score = keyword_score_map.get(pid, 0.0)
        semantic_score = semantic_score_map.get(pid, 0.0)

        # 归一化分数
        normalized_keyword = keyword_score / max_keyword if max_keyword > 0 else 0.0
        normalized_semantic = semantic_score / max_semantic if max_semantic > 0 else 0.0

        # 计算加权分数
        weighted_keyword = (1 - semantic_weight) * normalized_keyword
        weighted_semantic = semantic_weight * normalized_semantic
        final_score = weighted_keyword + weighted_semantic

        combined_scores[pid] = final_score

        # 保存详细信息，用于显示
        score_details[pid] = {
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
            "keyword_weight": 1 - semantic_weight,  # 转换为百分制
            "semantic_weight": semantic_weight,  # 转换为百分制
            "final_score": final_score * 100,
        }

    # 排序并限制结果数量
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    if limit:
        sorted_results = sorted_results[:limit]

    # 分离PIDs和分数
    pids = [pid for pid, _ in sorted_results]
    scores = [score * 100 for _, score in sorted_results]

    return pids, scores, score_details


def enhanced_search_rank(q: str = "", limit=None, search_mode="keyword", semantic_weight=0.5):
    """
    增强的搜索函数，支持多种搜索模式

    Args:
        q: 搜索查询
        limit: 结果数量限制
        search_mode: 搜索模式 ('keyword', 'semantic', 'hybrid')
        semantic_weight: 语义搜索权重 (0-1)，仅在 hybrid 模式下使用

    Returns:
        (paper_ids, scores) 元组
    """
    if not q:
        return [], []

    if search_mode == "keyword":
        # 使用现有的关键词搜索
        return search_rank(q, limit)

    elif search_mode == "semantic":
        # 纯语义搜索
        return semantic_search_rank(q, limit)

    elif search_mode == "hybrid":
        # 混合搜索
        return hybrid_search_rank(q, limit, semantic_weight)

    else:
        raise ValueError(f"Unknown search mode: {search_mode}")


# -----------------------------------------------------------------------------
# primary application endpoints


def default_context():
    # any global context across all pages, e.g. related to the current user
    context = {}
    context["user"] = g.user if g.user is not None else ""
    return context


@app.route("/", methods=["GET"])
def main():
    # default settings
    default_rank = "time"
    default_tags = ""
    default_time_filter = ""
    default_skip_have = "no"
    default_logic = "and"
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # override variables with any provided options via the interface
    opt_rank = request.args.get("rank", default_rank)  # rank type. search|tags|pid|time|random
    opt_q = request.args.get("q", "")  # search request in the text box
    opt_tags = request.args.get("tags", default_tags)  # tags to rank by if opt_rank == 'tag'
    opt_pid = request.args.get("pid", "")  # pid to find nearest neighbors to
    opt_time_filter = request.args.get("time_filter", default_time_filter)  # number of days to filter by
    opt_skip_have = request.args.get("skip_have", default_skip_have)  # hide papers we already have?
    opt_logic = request.args.get("logic", default_logic)  # tags logic?
    opt_svm_c = request.args.get("svm_c", "")  # svm C parameter
    opt_page_number = request.args.get("page_number", "1")  # page number for pagination
    opt_search_mode = request.args.get("search_mode", "hybrid")  # search mode: keyword|semantic|hybrid
    opt_semantic_weight = request.args.get("semantic_weight", "0.5")  # semantic weight for hybrid search

    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if opt_q:
        opt_rank = "search"

    # if using svm_rank (tags or pid) and no time filter is specified, default to 365 days
    if opt_rank in ["tags", "pid"] and not opt_time_filter:
        opt_time_filter = "365"
        # logger.trace(f"Applied default 365-day time filter for SVM ranking ({opt_rank})")

    # try to parse opt_svm_c into something sensible (a float)
    try:
        C = float(opt_svm_c)
    except ValueError:
        C = 0.02  # sensible default, i think

    # Calculate required number of results (considering pagination and possible need for more results after filtering)
    try:
        page_number = max(1, int(opt_page_number))
    except ValueError:
        page_number = 1

    # Calculate how many results to get to support current page display
    # Basic requirement: results needed for current page = RET_NUM * page_number
    base_needed = RET_NUM * page_number

    # Consider that filtering conditions will reduce available results, so need to get more
    buffer_multiplier = 1
    if opt_time_filter:
        buffer_multiplier += 2  # Time filtering may remove many results, take 2x more
    if opt_skip_have == "yes":
        buffer_multiplier += 1  # skip_have will also remove some, take 1x more

    # Final limit = basic requirement * buffer multiplier, but not exceeding maximum
    dynamic_limit = min(MAX_RESULTS, base_needed * buffer_multiplier)

    # logger.trace(
    #     f"Page {page_number}, base_needed={base_needed}, buffer_multiplier={buffer_multiplier}, dynamic_limit={dynamic_limit}"
    # )

    # rank papers: by tags, by time, by random
    words = []  # only populated in the case of svm rank
    score_details = {}  # 保存详细分数信息，仅在 hybrid 模式下使用

    if opt_rank == "search":
        t_s = time.time()
        # 使用增强的搜索函数
        try:
            semantic_weight = float(opt_semantic_weight)
        except ValueError:
            semantic_weight = 0.5

        if opt_search_mode == "hybrid":
            pids, scores, score_details = enhanced_search_rank(
                q=opt_q, limit=dynamic_limit, search_mode=opt_search_mode, semantic_weight=semantic_weight
            )
        else:
            pids, scores = enhanced_search_rank(
                q=opt_q, limit=dynamic_limit, search_mode=opt_search_mode, semantic_weight=semantic_weight
            )
        logger.info(
            f"User {g.user} {opt_search_mode} search '{opt_q}' weight={semantic_weight}, time {time.time() - t_s:.3f}s"
        )
    elif opt_rank == "tags":
        t_s = time.time()
        pids, scores, words = svm_rank(tags=opt_tags, C=C, logic=opt_logic, time_filter=opt_time_filter)
        logger.info(
            f"User {g.user} tags {opt_tags} C {C} logic {opt_logic} time_filter {opt_time_filter}, time {time.time() - t_s:.3f}s"
        )
        # SVM results are usually already sorted, trim early
        if len(pids) > dynamic_limit:
            pids = pids[:dynamic_limit]
            scores = scores[:dynamic_limit]
    elif opt_rank == "pid":
        t_s = time.time()
        pids, scores, words = svm_rank(s_pids=opt_pid, C=C, logic=opt_logic, time_filter=opt_time_filter)
        logger.info(
            f"User {g.user} pid {opt_pid} C {C} logic {opt_logic} time_filter {opt_time_filter}, time {time.time() - t_s:.3f}s"
        )
        # SVM results are usually already sorted, trim early
        if len(pids) > dynamic_limit:
            pids = pids[:dynamic_limit]
            scores = scores[:dynamic_limit]
    elif opt_rank == "time":
        t_s = time.time()
        pids, scores = time_rank(limit=dynamic_limit)
        logger.info(f"User {g.user} time rank, time {time.time() - t_s:.3f}s")
    elif opt_rank == "random":
        t_s = time.time()
        pids, scores = random_rank(limit=dynamic_limit)
        logger.info(f"User {g.user} random rank, time {time.time() - t_s:.3f}s")
    else:
        raise ValueError(f"opt_rank {opt_rank} is not a thing")

    # filter by time (now handled within svm_rank for SVM-based rankings)
    if opt_time_filter and opt_rank not in ["tags", "pid"]:
        # 收集用户标记的论文，用于智能时间过滤
        user_tagged_pids = set()
        tags = get_tags()
        for tag_pids in tags.values():
            user_tagged_pids.update(tag_pids)

        # 使用智能时间过滤
        pids, time_valid_indices = _filter_by_time_with_tags(pids, opt_time_filter, user_tagged_pids)
        scores = [scores[i] for i in time_valid_indices]

    # optionally hide papers we already have
    if opt_skip_have == "yes":
        tags = get_tags()
        have = set().union(*tags.values())
        keep = [i for i, pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # Pagination processing (page_number already calculated above)
    start_index = (page_number - 1) * RET_NUM  # desired starting index
    end_index = min(start_index + RET_NUM, len(pids))  # desired ending index
    pids = pids[start_index:end_index]
    scores = scores[start_index:end_index]

    # logger.trace(f"Final pagination: start={start_index}, end={end_index}, got {len(pids)} papers")

    # render all papers to just the information we need for the UI
    papers = [render_pid(pid) for pid in pids]
    for i, p in enumerate(papers):
        p["weight"] = float(scores[i])

        # 如果是 hybrid 搜索模式，添加详细分数信息
        if opt_rank == "search" and opt_search_mode == "hybrid" and score_details:
            pid = p["id"]
            if pid in score_details:
                details = score_details[pid]
                # 格式化显示：A: {(1 - semantic_weight)} Keyword {Score} + {semantic_weight} Semantic {Score}
                try:
                    semantic_weight = float(opt_semantic_weight)
                except ValueError:
                    semantic_weight = 0.5

                keyword_weight = 1 - semantic_weight
                p["score_breakdown"] = (
                    f"{keyword_weight:.1f} Keyword {details['keyword_score']:.2f} + "
                    f"{semantic_weight:.1f} Semantic {details['semantic_score']:.2f}"
                )

    # build the current tags for the user, and append the special 'all' tag
    tags = get_tags()
    rtags = [{"name": t, "n": len(pids)} for t, pids in tags.items()]
    if rtags:
        rtags.append({"name": "all"})

    keys = get_keys()
    rkeys = [{"name": k} for k, pids in keys.items()]
    rkeys.append({"name": "Artificial general intelligence"})

    combined_tags = get_combined_tags()
    rctags = [{"name": k} for k in combined_tags]

    # build the page context information and render
    context = default_context()
    context["papers"] = papers
    context["words"] = words

    context["tags"] = sorted(rtags, key=lambda item: item["name"])
    context["keys"] = sorted(rkeys, key=lambda item: item["name"])
    context["combined_tags"] = sorted(rctags, key=lambda item: item["name"])

    # test keys
    # context["keys"] = [
    #     {"name": "Reinforcement Learning", "pids": []},
    #     {"name": "Zero-shot Coordination", "pids": []},
    # ]

    context["words_desc"] = (
        "Here are the top 40 most positive and bottom 20 most negative weights of the SVM. If they don't look great then try tuning the regularization strength hyperparameter of the SVM, svm_c, above. Lower C is higher regularization."
    )
    context["gvars"] = {}
    context["gvars"]["rank"] = opt_rank
    context["gvars"]["tags"] = opt_tags
    context["gvars"]["pid"] = opt_pid
    context["gvars"]["time_filter"] = opt_time_filter
    context["gvars"]["skip_have"] = opt_skip_have
    context["gvars"]["logic"] = opt_logic
    context["gvars"]["search_query"] = opt_q
    context["gvars"]["svm_c"] = str(C)
    context["gvars"]["page_number"] = str(page_number)
    context["gvars"]["search_mode"] = opt_search_mode
    context["gvars"]["semantic_weight"] = opt_semantic_weight
    context["show_score_breakdown"] = opt_rank == "search" and opt_search_mode == "hybrid"
    logger.trace(
        f'User: {context["user"]}\ntags {context["tags"]}\nkeys {context["keys"]}\nctags {context["combined_tags"]}'
    )
    return render_template("index.html", **context)


@app.route("/inspect", methods=["GET"])
def inspect():
    # fetch the paper of interest based on the pid
    pid = request.args.get("pid", "")
    pdb = get_papers()
    if pid not in pdb:
        return "error, malformed pid"  # todo: better error handling

    # Use intelligent cache to load features
    features = get_features_cached()  # Replace: features = load_features()

    # 使用原始 TF-IDF 矩阵进行 inspect（如果存在）
    x = features.get("x_tfidf", features["x"])
    idf = features["idf"]
    ivocab = {v: k for k, v in features["vocab"].items()}
    pix = features["pids"].index(pid)
    wixs = np.flatnonzero(np.asarray(x[pix].todense()))
    words = []
    for ix in wixs:
        words.append(
            {
                "word": ivocab[ix],
                "weight": float(x[pix, ix]),
                "idf": float(idf[ix]),
            }
        )
    words.sort(key=lambda w: w["weight"], reverse=True)

    # package everything up and render
    paper = render_pid(pid)
    context = default_context()
    context["paper"] = paper
    context["words"] = words
    context["words_desc"] = (
        "The following are the tokens and their (tfidf) weight in the paper vector. This is the actual summary that feeds into the SVM to power recommendations, so hopefully it is good and representative!"
    )
    return render_template("inspect.html", **context)


@app.route("/summary", methods=["GET"])
def summary():
    """
    显示论文的 AI 生成的 markdown 格式总结
    """
    # 获取论文 ID
    pid = request.args.get("pid", "")
    logger.info(f"show paper summary page for paper {pid}")
    pdb = get_papers()
    if pid not in pdb:
        return f"<h1>Error</h1><p>Paper with ID '{pid}' not found in database.</p>", 404

    # 获取论文基本信息
    paper = render_pid(pid)

    # 构建页面上下文，不在这里调用 get_paper_summary
    context = default_context()
    context["paper"] = paper
    context["pid"] = pid  # 传递 pid 给前端，用于异步获取 summary

    return render_template("summary.html", **context)


@app.route("/api/get_paper_summary", methods=["POST"])
def api_get_paper_summary():
    """
    API endpoint: Get paper summary asynchronously
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        pid = data.get("pid", "").strip()
        if not pid:
            return jsonify({"success": False, "error": "Paper ID is required"}), 400

        logger.info(f"Getting paper summary for: {pid}")

        # Check if paper exists
        pdb = get_papers()
        if pid not in pdb:
            return jsonify({"success": False, "error": "Paper not found"}), 404

        # Generate paper summary - currently returns placeholder
        summary_content = generate_paper_summary(pid)

        return jsonify({"success": True, "pid": pid, "summary_content": summary_content})

    except Exception as e:
        logger.error(f"Paper summary API error: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


def generate_paper_summary(pid: str) -> str:
    """
    Generate paper summary with intelligent caching mechanism
    1. Check if data/summary/{pid}.md exists
    2. If exists, return cached summary
    3. If not exists, call paper_summarizer to generate and cache the summary
    """
    try:
        pdb = get_papers()
        if pid not in pdb:
            return "# 錯誤\n\n論文未找到。"

        # Define cache file path
        cache_dir = Path("data/summary")
        cache_file = cache_dir / f"{pid}.md"

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if cached summary exists
        if cache_file.exists():
            logger.info(f"Using cached paper summary: {pid}")
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cached_summary = f.read()
                if cached_summary.strip():
                    return cached_summary
                else:
                    logger.warning(f"Cache file is empty, regenerating summary: {pid}")
            except Exception as e:
                logger.error(f"Failed to read cached summary: {e}")

        # Generate new summary using paper_summarizer module
        logger.info(f"Generating new paper summary: {pid}")
        summary_content = generate_paper_summary_from_module(pid)

        # Only cache successful summaries (not error messages)
        if not summary_content.startswith("# 錯誤") and not summary_content.startswith("# 错误"):
            # Check Chinese ratio before caching
            chinese_ratio = calculate_chinese_ratio(summary_content)
            logger.trace(f"Paper {pid} summary Chinese ratio: {chinese_ratio:.2%}")

            if chinese_ratio >= 0.25:
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(summary_content)
                    logger.info(f"Paper summary cached to: {cache_file} (Chinese ratio: {chinese_ratio:.2%})")
                except Exception as e:
                    logger.error(f"Failed to cache paper summary: {e}")
            else:
                logger.warning(f"Summary Chinese ratio too low ({chinese_ratio:.2%} < 50%), not caching: {pid}")
        else:
            logger.warning(f"Summary generation failed, not caching: {pid}")

        return summary_content

    except Exception as e:
        logger.error(f"Error occurred while generating paper summary: {e}")
        return f"# Error\n\nFailed to generate summary: {str(e)}"


@app.route("/profile")
def profile():
    context = default_context()
    with get_email_db() as edb:
        email = edb.get(g.user, "")
        context["email"] = email
    return render_template("profile.html", **context)


@app.route("/stats")
def stats():
    context = default_context()
    mdb = get_metas()
    kv = {k: v for k, v in mdb.items()}  # read all of metas to memory at once, for efficiency
    times = [v["_time"] for v in kv.values()]
    tstr = lambda t: time.strftime("%b %d %Y", time.localtime(t))

    context["num_papers"] = len(kv)
    if len(kv) > 0:
        context["earliest_paper"] = tstr(min(times))
        context["latest_paper"] = tstr(max(times))
    else:
        context["earliest_paper"] = "N/A"
        context["latest_paper"] = "N/A"

    # count number of papers from various time deltas to now
    tnow = time.time()
    for thr in [1, 6, 12, 24, 48, 72, 96]:
        context["thr_%d" % thr] = len([t for t in times if t > tnow - thr * 60 * 60])

    context["thr_week"] = len([t for t in times if t > tnow - 7 * 24 * 60 * 60])
    context["thr_month"] = len([t for t in times if t > tnow - 30.25 * 24 * 60 * 60])
    context["thr_quarter"] = len([t for t in times if t > tnow - 91.5 * 24 * 60 * 60])
    context["thr_semiannual"] = len([t for t in times if t > tnow - 182.5 * 24 * 60 * 60])
    context["thr_year"] = len([t for t in times if t > tnow - 365 * 24 * 60 * 60])

    return render_template("stats.html", **context)


@app.route("/about")
def about():
    context = default_context()
    return render_template("about.html", **context)


# -----------------------------------------------------------------------------
# Helper for API endpoints
# -----------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _temporary_user_context(user):
    """Context manager to temporarily set g.user and g._tags for API calls"""
    original_user = getattr(g, "user", None)
    original_tags = getattr(g, "_tags", None)

    try:
        # Get user tags
        with get_tags_db() as tags_db:
            user_tags = tags_db.get(user, {})

        # Set temporary context
        g.user = user
        g._tags = user_tags

        yield user_tags

    finally:
        # Restore original context
        if original_user is not None:
            g.user = original_user
        else:
            if hasattr(g, "user"):
                delattr(g, "user")
        if original_tags is not None:
            g._tags = original_tags
        else:
            if hasattr(g, "_tags"):
                delattr(g, "_tags")


# -----------------------------------------------------------------------------
# API endpoints for email recommendation system
# -----------------------------------------------------------------------------


@app.route("/api/keyword_search", methods=["POST"])
def api_keyword_search():
    """API接口：单个关键词搜索"""
    try:
        data = request.get_json()
        # logger.info(f"API keyword search data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        keyword = data.get("keyword", "")
        time_delta = data.get("time_delta", 3)  # 天数
        limit = data.get("limit", 50)

        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400

        # 使用增强搜索
        pids, scores = enhanced_search_rank(
            q=keyword, limit=limit * 5, search_mode="keyword"  # 多取一些，因为需要时间过滤
        )

        # 应用时间过滤
        if time_delta:
            mdb = get_metas()
            kv = {k: v for k, v in mdb.items()}
            tnow = time.time()
            deltat = float(time_delta) * 60 * 60 * 24
            keep = [i for i, pid in enumerate(pids) if pid in kv and (tnow - kv[pid]["_time"]) < deltat]
            pids = [pids[i] for i in keep]
            scores = [scores[i] for i in keep]

        # 限制最终结果数量
        if len(pids) > limit:
            pids = pids[:limit]
            scores = scores[:limit]
        # logger.trace(f"API keyword search results: {len(pids)} papers found")
        return jsonify({"success": True, "pids": pids, "scores": scores, "total_count": len(pids)})

    except Exception as e:
        logger.error(f"API keyword search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tag_search", methods=["POST"])
def api_tag_search():
    """API接口：单个tag推荐"""
    try:
        data = request.get_json()
        logger.trace(f"API tag search data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        tag_name = data.get("tag_name", "")  # tag名称
        user = data.get("user", "")  # 用户名
        time_delta = data.get("time_delta", 3)  # 天数
        limit = data.get("limit", 50)
        C = data.get("C", 0.1)

        if not tag_name:
            return jsonify({"error": "tag_name is required"}), 400
        if not user:
            return jsonify({"error": "user is required"}), 400

        # 获取该用户的标签数据
        with get_tags_db() as tags_db:
            user_tags = tags_db.get(user, {})

        # 检查用户是否有该标签
        if tag_name not in user_tags or len(user_tags[tag_name]) == 0:
            logger.warning(f"User {user} has no papers tagged with '{tag_name}'")
            return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})

        logger.trace(f"User {user} has {len(user_tags[tag_name])} papers tagged with '{tag_name}'")

        # 临时设置g.user和g._tags，让svm_rank能正确工作
        original_user = getattr(g, "user", None)
        original_tags = getattr(g, "_tags", None)

        g.user = user
        g._tags = user_tags

        try:
            # 使用tag名称进行推荐
            rec_pids, rec_scores, words = svm_rank(
                tags=tag_name, s_pids="", C=C, logic="and", time_filter=str(time_delta)
            )

            logger.trace(f"svm_rank returned {len(rec_pids)} results before filtering")

            # 获取该用户该tag已标记的论文，用于排除
            if tag_name in user_tags:
                tagged_pids = user_tags[tag_name]
                tagged_set = set(tagged_pids)
                logger.trace(f"User has {len(tagged_set)} papers tagged with '{tag_name}'")

                # 计算推荐结果和已标记论文的交集
                intersection = [pid for pid in rec_pids if pid in tagged_set]
                logger.trace(f"Found {len(intersection)} recommended papers that are already tagged")

                keep = [i for i, pid in enumerate(rec_pids) if pid not in tagged_set]
                rec_pids = [rec_pids[i] for i in keep]
                rec_scores = [rec_scores[i] for i in keep]

                logger.trace(f"After filtering out tagged papers: {len(rec_pids)} papers remain")
        finally:
            # 恢复原始的g.user和g._tags
            if original_user is not None:
                g.user = original_user
            else:
                if hasattr(g, "user"):
                    delattr(g, "user")
            if original_tags is not None:
                g._tags = original_tags
            else:
                if hasattr(g, "_tags"):
                    delattr(g, "_tags")

        # 限制结果数量
        if len(rec_pids) > limit:
            rec_pids = rec_pids[:limit]
            rec_scores = rec_scores[:limit]
        logger.trace(f"API tag search results: {len(rec_pids)} papers found")
        return jsonify({"success": True, "pids": rec_pids, "scores": rec_scores, "total_count": len(rec_pids)})

    except Exception as e:
        logger.error(f"API tag search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tags_search", methods=["POST"])
def api_tags_search():
    """API接口：联合tag推荐"""
    try:
        data = request.get_json()
        logger.trace(f"API combined tags search data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        tags_list = data.get("tags", [])  # List[tag_name]
        user = data.get("user", "")  # 用户名
        logic = data.get("logic", "and")  # and|or
        time_delta = data.get("time_delta", 3)  # 天数
        limit = data.get("limit", 50)
        C = data.get("C", 0.1)

        if not tags_list:
            return jsonify({"error": "Tags list is required"}), 400
        if not user:
            return jsonify({"error": "user is required"}), 400

        # 获取该用户的标签数据
        with get_tags_db() as tags_db:
            user_tags = tags_db.get(user, {})

        # 检查用户是否有任一标签
        valid_tags = [tag for tag in tags_list if tag in user_tags and len(user_tags[tag]) > 0]
        if not valid_tags:
            logger.warning(f"User {user} has no papers tagged with any of {tags_list}")
            return jsonify({"success": True, "pids": [], "scores": [], "total_count": 0})

        # 将tag列表转换为逗号分隔的字符串
        tags_str = ",".join(valid_tags)

        # 临时设置g.user和g._tags，让svm_rank能正确工作
        original_user = getattr(g, "user", None)
        original_tags = getattr(g, "_tags", None)

        g.user = user
        g._tags = user_tags

        try:
            # 使用SVM推荐
            rec_pids, rec_scores, words = svm_rank(
                tags=tags_str, s_pids="", C=C, logic=logic, time_filter=str(time_delta)
            )

            # 获取该用户所有相关tags的已标记论文，用于排除
            all_tagged = set()
            for tag in valid_tags:
                if tag in user_tags:
                    all_tagged.update(user_tags[tag])

            keep = [i for i, pid in enumerate(rec_pids) if pid not in all_tagged]
            rec_pids = [rec_pids[i] for i in keep]
            rec_scores = [rec_scores[i] for i in keep]
        finally:
            # 恢复原始的g.user和g._tags
            if original_user is not None:
                g.user = original_user
            else:
                if hasattr(g, "user"):
                    delattr(g, "user")
            if original_tags is not None:
                g._tags = original_tags
            else:
                if hasattr(g, "_tags"):
                    delattr(g, "_tags")

        # 限制结果数量
        if len(rec_pids) > limit:
            rec_pids = rec_pids[:limit]
            rec_scores = rec_scores[:limit]

        logger.trace(f"API combined tags search results: {len(rec_pids)} papers found")

        return jsonify({"success": True, "pids": rec_pids, "scores": rec_scores, "total_count": len(rec_pids)})

    except Exception as e:
        logger.error(f"API tags search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/cache_status")
def cache_status():
    """Debug endpoint to display cache status"""
    if not g.user:
        return "Access denied"

    global FEATURES_CACHE, FEATURES_FILE_MTIME, FEATURES_CACHE_TIME
    global PAPERS_CACHE, METAS_CACHE, PIDS_CACHE, PAPERS_DB_FILE_MTIME, PAPERS_DB_CACHE_TIME

    status = {
        "current_time": time.time(),
        "features": {
            "cached": FEATURES_CACHE is not None,
            "cache_time": FEATURES_CACHE_TIME,
            "file_mtime": FEATURES_FILE_MTIME,
        },
        "papers_and_metas": {
            "papers_cached": PAPERS_CACHE is not None,
            "metas_cached": METAS_CACHE is not None,
            "cache_time": PAPERS_DB_CACHE_TIME,
            "db_file_mtime": PAPERS_DB_FILE_MTIME,
        },
    }

    # Features details
    if FEATURES_CACHE:
        status["features"].update(
            {
                "cache_age_seconds": time.time() - FEATURES_CACHE_TIME,
                "feature_shape": str(FEATURES_CACHE["x"].shape),
                "num_papers": len(FEATURES_CACHE["pids"]),
                "vocab_size": len(FEATURES_CACHE["vocab"]),
            }
        )

    if os.path.exists(FEATURES_FILE):
        current_mtime = os.path.getmtime(FEATURES_FILE)
        status["features"].update(
            {
                "file_exists": True,
                "current_file_mtime": current_mtime,
                "file_is_newer": current_mtime > FEATURES_FILE_MTIME,
            }
        )
    else:
        status["features"]["file_exists"] = False

    # Papers and metas details (unified since they share the same db file)
    if PAPERS_CACHE and METAS_CACHE:
        status["papers_and_metas"].update(
            {
                "cache_age_seconds": time.time() - PAPERS_DB_CACHE_TIME,
                "num_papers": len(PAPERS_CACHE),
                "num_metas": len(METAS_CACHE),
                "num_pids": len(PIDS_CACHE) if PIDS_CACHE else 0,
            }
        )

    # Database file details
    if os.path.exists(PAPERS_DB_PATH):
        current_db_mtime = os.path.getmtime(PAPERS_DB_PATH)
        status["papers_and_metas"].update(
            {
                "db_file_exists": True,
                "current_db_file_mtime": current_db_mtime,
                "db_file_is_newer": current_db_mtime > PAPERS_DB_FILE_MTIME,
            }
        )
    else:
        status["papers_and_metas"]["db_file_exists"] = False

    return status


# -----------------------------------------------------------------------------
# tag related endpoints: add, delete tags for any paper


@app.route("/add/<pid>/<tag>")
def add(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if tag == "all":
        return "error, cannot add the protected tag 'all'"
    elif tag == "null":
        return "error, cannot add the protected tag 'null'"

    with get_tags_db(flag="c") as tags_db:
        # create the user if we don't know about them yet with an empty library
        if not g.user in tags_db:
            tags_db[g.user] = {}

        # fetch the user library object
        d = tags_db[g.user]

        # add the paper to the tag
        if tag not in d:
            d[tag] = set()
        d[tag].add(pid)

        # write back to database
        tags_db[g.user] = d

    logger.info(f"added paper {pid} to tag {tag} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route("/sub/<pid>/<tag>")
def sub(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"

    with get_tags_db(flag="c") as tags_db:
        # if the user doesn't have any tags, there is nothing to do
        if not g.user in tags_db:
            return r"user has no library of tags ¯\_(ツ)_/¯"

        # fetch the user library object
        d = tags_db[g.user]

        # add the paper to the tag
        if tag not in d:
            return f"user doesn't have the tag {tag}"
        else:
            if pid in d[tag]:
                # remove this pid from the tag
                d[tag].remove(pid)

                # if this was the last paper in this tag, also delete the tag
                if len(d[tag]) == 0:
                    del d[tag]

                # write back the resulting dict to database
                tags_db[g.user] = d
                return f"ok removed pid {pid} from tag {tag}"
            else:
                return f"user doesn't have paper {pid} in tag {tag}"


@app.route("/del/<tag>")
def delete_tag(tag=None):
    if g.user is None:
        return "error, not logged in"

    with get_tags_db(flag="c") as tags_db:
        with get_combined_tags_db(flag="c") as ctags_db:
            if g.user not in tags_db:
                return "user does not have a library"

            d = tags_db[g.user]

            if tag not in d:
                return "user does not have this tag"

            # delete the tag
            del d[tag]

            # write back to database
            tags_db[g.user] = d
            # remove ctag
            d = ctags_db[g.user]
            for ctag in d:
                if tag in ctag.split(","):
                    d.remove(ctag)
            ctags_db[g.user] = d

    logger.info(f"deleted tag {tag} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route("/rename/<otag>/<ntag>")
def rename_tag(otag=None, ntag=None):
    if g.user is None:
        return "error, not logged in"

    with get_tags_db(flag="c") as tags_db:
        with get_combined_tags_db(flag="c") as ctags_db:
            if g.user not in tags_db:
                return "user does not have a library"

            d = tags_db[g.user]

            if otag not in d:
                return "user does not have this tag"

            # logger.trace(d)
            o_pids = d[otag]
            del d[otag]
            if ntag not in d:
                d[ntag] = o_pids
            else:
                d[ntag] = d[ntag].union(o_pids)
            # logger.trace(d)

            # write back to database
            tags_db[g.user] = d

            # rename ctag
            d = ctags_db[g.user]
            # logger.error(d)
            for ctag in d:
                # logger.error(f"{ctag}, {otag}, {ctag.split(',')}")
                if otag in (ctag_split := ctag.split(",")):
                    ctag_split = [ct_s if ct_s != otag else ntag for ct_s in ctag_split]
                    n_ctag = ",".join(ctag_split)
                    # logger.error(f"{ctag}, {n_ctag}")
                    d.remove(ctag)
                    d.add(n_ctag)
            ctags_db[g.user] = d

    logger.info(f"renamed tag {otag} to {ntag} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route("/add_ctag/<ctag>")
def add_ctag(ctag=None):
    if g.user is None:
        return "error, not logged in"
    elif ctag == "null":
        return "error, cannot add the ctag 'null'"

    tags = get_tags()
    for tag in map(str.strip, ctag.split(",")):
        if tag not in tags:
            return "invalid ctag"

    with get_combined_tags_db(flag="c") as ctags_db:
        # logger.trace(f"{ctags_db}")
        # create the user if we don't know about them yet with an empty library
        if g.user not in ctags_db:
            ctags_db[g.user] = set()

        # fetch the user library object
        d = ctags_db[g.user]

        # if isinstance(d, dict):
        #     d = set(d.keys())

        # add the paper to the key
        if ctag in d:
            return "user has repeated ctag"

        d.add(ctag)

        # write back to database
        ctags_db[g.user] = d

    logger.info(f"added ctag {ctag} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route("/del_ctag/<ctag>")
def delete_ctag(ctag=None):
    if g.user is None:
        return "error, not logged in"

    with get_combined_tags_db(flag="c") as ctags_db:
        if g.user not in ctags_db:
            return "user does not have a library"

        d = ctags_db[g.user]

        if ctag not in d:
            return "user does not have this ctag"

        # delete the tag
        d.remove(ctag)

        # write back to database
        ctags_db[g.user] = d

    logger.info(f"deleted ctag {ctag} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route("/add_key/<keyword>")
def add_key(keyword=None):
    if g.user is None:
        return "error, not logged in"
    elif keyword == "null":
        return "error, cannot add the protected keyword 'null'"

    with get_keywords_db(flag="c") as keys_db:
        # create the user if we don't know about them yet with an empty library
        if g.user not in keys_db:
            keys_db[g.user] = {}

        # fetch the user library object
        d = keys_db[g.user]

        # add the paper to the key
        if keyword in d:
            return "user has repeated keywords"

        d[keyword] = set()

        # write back to database
        keys_db[g.user] = d

    logger.info(f"added keyword {keyword} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route("/del_key/<keyword>")
def delete_key(keyword=None):
    if g.user is None:
        return "error, not logged in"

    with get_keywords_db(flag="c") as keys_db:
        if g.user not in keys_db:
            return "user does not have a library"

        d = keys_db[g.user]

        if keyword not in d:
            return "user does not have this keyword"

        # delete the keyword
        del d[keyword]

        # write back to database
        keys_db[g.user] = d

    logger.info(f"deleted keyword {keyword} for user {g.user}")
    return "ok: " + str(d)  # return back the user library for debugging atm


# -----------------------------------------------------------------------------
# endpoints to log in and out


@app.route("/login", methods=["POST"])
def login():
    # the user is logged out but wants to log in, ok
    if g.user is None and request.form["username"]:
        username = request.form["username"]
        if len(username) > 0:  # one more paranoid check
            session["user"] = username

    return redirect(url_for("profile"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("profile"))


# -----------------------------------------------------------------------------
# user settings and configurations


@app.route("/register_email", methods=["POST"])
def register_email():
    email = request.form["email"]

    if g.user:
        # do some basic input validation
        proper_email = re.match(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$", email, re.IGNORECASE)
        if email == "" or proper_email:  # allow empty email, meaning no email
            # everything checks out, write to the database
            with get_email_db(flag="c") as edb:
                edb[g.user] = email

    return redirect(url_for("profile"))
