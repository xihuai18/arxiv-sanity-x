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
from random import shuffle

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from flask import g  # global session-level object
from flask import Flask, redirect, render_template, request, session, url_for
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

# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = 100  # number of papers to return per page
MAX_RESULTS = RET_NUM * 10  # Process at most 10 pages of results, avoid processing all data

# Feature cache related global variables
FEATURES_CACHE = None
FEATURES_FILE_MTIME = 0  # Feature file modification time
FEATURES_CACHE_TIME = 0  # Cache creation time

app = Flask(__name__)

# set the secret key so we can cryptographically sign cookies and maintain sessions
if os.path.isfile("secret_key.txt"):
    # example of generating a good key on your system is:
    # import secrets; secrets.token_urlsafe(16)
    sk = open("secret_key.txt").read().strip()
else:
    logger.warning("no secret key found, using default `devkey`")
    sk = "devkey"
app.secret_key = sk


# -----------------------------------------------------------------------------
# globals that manage the (lazy) loading of various state for a request
def get_tags():
    if g.user is None:
        return {}
    if not hasattr(g, "_tags"):
        with get_tags_db() as tags_db:
            tags_dict = tags_db[g.user] if g.user in tags_db else {}
        g._tags = tags_dict
    return g._tags


def get_combined_tags():
    if g.user is None:
        return {}
    if not hasattr(g, "_combined_tags"):
        with get_combined_tags_db() as combined_tags_db:
            combined_tags_dict = combined_tags_db[g.user] if g.user in combined_tags_db else {}
        g._combined_tags = combined_tags_dict
    return g._combined_tags


def get_keys():
    if g.user is None:
        return {}
    if not hasattr(g, "_keys"):
        with get_keywords_db() as keys_db:
            keys_dict = keys_db[g.user] if g.user in keys_db else {}
        g._keys = keys_dict
    return g._keys


# -----------------------------------------------------------------------------
# init papers and meta db
def get_all_papers():
    if "db_papers" not in globals():
        global db_papers
    with get_papers_db() as papers_db:
        db_papers = {k: v for k, v in tqdm(papers_db.items(), desc="initing papers db")}


def get_all_metas():
    if "db_metas" not in globals():
        global db_metas
    if "db_pids" not in globals():
        global db_pids
    with get_metas_db() as metas_db:
        db_metas = {k: v for k, v in tqdm(metas_db.items(), desc="initing metas db")}
        db_pids = list(db_metas.keys())


get_all_papers()
get_all_metas()

scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
scheduler.add_job(get_all_papers, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
scheduler.add_job(get_all_metas, "cron", day_of_week="tue,wed,thu,fri,mon", hour=18)
scheduler.start()


def get_pids():
    if not hasattr(g, "_pids"):
        if "db_pids" in globals():
            g._pids = db_pids
        else:
            g._pids = list(get_metas().keys())
    return g._pids


def get_papers():
    if not hasattr(g, "_pdb"):
        if "db_papers" in globals():
            g._pdb = db_papers
        else:
            # logger.warning("reading papers db from disk")
            g._pdb = get_papers_db()
        # g._pdb = {k: v for k, v in g._pdb.items()}
    return g._pdb


def get_metas():
    if not hasattr(g, "_mdb"):
        if "db_metas" in globals():
            g._mdb = db_metas
        else:
            # logger.warning("reading metas db from disk")
            g._mdb = get_metas_db()
        # g._mdb = {k: v for k, v in g._mdb.items()}
    return g._mdb


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
    # close any opened database connections
    if hasattr(g, "_pdb") and not isinstance(g._pdb, dict):
        g._pdb.close()
    if hasattr(g, "_mdb") and not isinstance(g._mdb, dict):
        g._mdb.close()


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
            logger.info(f"Features file updated (old mtime: {FEATURES_FILE_MTIME}, new mtime: {current_file_mtime})")

        start_time = time.time()

        try:
            # Load features
            FEATURES_CACHE = load_features()
            FEATURES_FILE_MTIME = current_file_mtime
            FEATURES_CACHE_TIME = current_time

            load_time = time.time() - start_time
            logger.info(f"Features loaded successfully in {load_time:.3f}s")
            logger.info(f"Feature matrix shape: {FEATURES_CACHE['x'].shape}")
            logger.info(f"Number of papers: {len(FEATURES_CACHE['pids'])}")
            logger.info(f"Vocabulary size: {len(FEATURES_CACHE['vocab'])}")

        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            raise
    else:
        # Use cache
        cache_age = current_time - FEATURES_CACHE_TIME
        logger.debug(f"Using cached features (age: {cache_age:.1f}s)")

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


def random_rank(limit=None):
    mdb = get_metas()
    pids = list(mdb.keys())
    shuffle(pids)

    # If limit is specified, only take first limit results
    if limit is not None:
        pids = pids[:limit]
        logger.debug(f"Random rank limited to {limit} results")

    scores = [0 for _ in pids]
    return pids, scores


def time_rank(limit=None):
    mdb = get_metas()
    ms = sorted(mdb.items(), key=lambda kv: kv[1]["_time"], reverse=True)

    # If limit is specified, only take first limit results
    if limit is not None:
        ms = ms[:limit]
        logger.debug(f"Time rank limited to {limit} results")

    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v["_time"]) / 60 / 60 / 24 for k, v in ms]  # time delta in days
    return pids, scores


def svm_rank(tags: str = "", s_pids: str = "", C: float = 0.02, logic: str = "and", time_filter: str = ""):
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or s_pids):
        return [], [], []

    # Use intelligent cache to load features
    s_time = time.time()
    features = get_features_cached()  # Replace: features = load_features()
    x, pids = features["x"], features["pids"]

    # Apply time filtering first to reduce computation
    if time_filter:
        mdb = get_metas()
        kv = {k: v for k, v in mdb.items()}  # read all of metas to memory at once, for efficiency
        tnow = time.time()
        deltat = float(time_filter) * 60 * 60 * 24  # allowed time delta in seconds

        # Find indices of papers that meet time criteria
        time_valid_indices = []
        time_valid_pids = []
        for i, pid in enumerate(pids):
            if pid in kv and (tnow - kv[pid]["_time"]) < deltat:
                time_valid_indices.append(i)
                time_valid_pids.append(pid)

        # Filter features and pids based on time
        if time_valid_indices:
            x = x[time_valid_indices]
            pids = time_valid_pids
            logger.debug(f"Time filter reduced dataset from {len(features['pids'])} to {len(pids)} papers")
        else:
            logger.debug("No papers match time filter criteria")
            return [], [], []

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
        tags_filter_to = tags_db.keys() if tags == "all" else set(map(str.strip, tags.split(",")))
        for t_i, tag in enumerate(tags_filter_to):
            if tag in tags_db:  # Ensure tag exists
                t_pids = tags_db[tag]
                for p_i, pid in enumerate(t_pids):
                    if pid in ptoi:  # Ensure PID exists
                        if logic == "and":
                            y[ptoi[pid]] = max(y[ptoi[pid]], 1.0 + t_i + weight_offset)
                        else:
                            y[ptoi[pid]] = max(y[ptoi[pid]], 1.0)
    e_time = time.time()

    logger.debug(f"feature loading/caching for {e_time - s_time:.5f}s")

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
    # logger.debug(f"Dimension reduction for {e_time - s_time:.5f}s")

    clf.fit(x, y)
    e_time = time.time()
    logger.debug(f"SVM fitting data for {e_time - s_time:.5f}s")

    if logic == "and":
        s = clf.decision_function(x)
        # logger.debug(f"svm_rank: {s.shape}")
        if len(s.shape) > 1:
            s = s[:, 1:].mean(axis=-1)
    else:
        s = clf.decision_function(x)
    e_time = time.time()
    logger.debug(f"SVM decsion function for {e_time - s_time:.5f}s")
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
    logger.debug(f"rank calculation for {e_time - s_time:.5f}s")
    logger.debug(f"Total features: {len(weights)}, TF-IDF features: {vocab_size}")

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

    # If limit is specified, only take first limit results
    if limit is not None:
        pairs = pairs[:limit]
        logger.debug(f"Search rank limited to {limit} results")

    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores


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

    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if opt_q:
        opt_rank = "search"

    # if using svm_rank (tags or pid) and no time filter is specified, default to 365 days
    if opt_rank in ["tags", "pid"] and not opt_time_filter:
        opt_time_filter = "365"
        logger.debug(f"Applied default 365-day time filter for SVM ranking ({opt_rank})")

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

    logger.debug(
        f"Page {page_number}, base_needed={base_needed}, buffer_multiplier={buffer_multiplier}, dynamic_limit={dynamic_limit}"
    )

    # rank papers: by tags, by time, by random
    words = []  # only populated in the case of svm rank
    if opt_rank == "search":
        t_s = time.time()
        pids, scores = search_rank(q=opt_q, limit=dynamic_limit)
        logger.info(f"User {g.user} search {opt_q}, time {time.time() - t_s:.3f}s")
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
        mdb = get_metas()
        kv = {k: v for k, v in mdb.items()}  # read all of metas to memory at once, for efficiency
        tnow = time.time()
        deltat = float(opt_time_filter) * 60 * 60 * 24  # allowed time delta in seconds
        keep = [i for i, pid in enumerate(pids) if (tnow - kv[pid]["_time"]) < deltat]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

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

    logger.debug(f"Final pagination: start={start_index}, end={end_index}, got {len(pids)} papers")

    # render all papers to just the information we need for the UI
    papers = [render_pid(pid) for pid in pids]
    for i, p in enumerate(papers):
        p["weight"] = float(scores[i])

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
    logger.info(
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


@app.route("/cache_status")
def cache_status():
    """Debug endpoint to display cache status"""
    if not g.user:
        return "Access denied"

    global FEATURES_CACHE, FEATURES_FILE_MTIME, FEATURES_CACHE_TIME

    status = {
        "features_cached": FEATURES_CACHE is not None,
        "cache_time": FEATURES_CACHE_TIME,
        "file_mtime": FEATURES_FILE_MTIME,
        "current_time": time.time(),
    }

    if FEATURES_CACHE:
        status.update(
            {
                "cache_age_seconds": time.time() - FEATURES_CACHE_TIME,
                "feature_shape": str(FEATURES_CACHE["x"].shape),
                "num_papers": len(FEATURES_CACHE["pids"]),
                "vocab_size": len(FEATURES_CACHE["vocab"]),
            }
        )

    if os.path.exists(FEATURES_FILE):
        current_mtime = os.path.getmtime(FEATURES_FILE)
        status.update(
            {
                "file_exists": True,
                "current_file_mtime": current_mtime,
                "file_is_newer": current_mtime > FEATURES_FILE_MTIME,
            }
        )
    else:
        status["file_exists"] = False

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

            # logger.debug(d)
            o_pids = d[otag]
            del d[otag]
            if ntag not in d:
                d[ntag] = o_pids
            else:
                d[ntag] = d[ntag].union(o_pids)
            # logger.debug(d)

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
        # logger.debug(f"{ctags_db}")
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
