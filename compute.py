"""
Extracts tfidf features from all paper abstracts and saves them to disk.
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
from random import shuffle

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from aslite.db import FEATURES_FILE, FEATURES_FILE_NEW, get_papers_db, save_features

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arxiv Computor")
    parser.add_argument("-n", "--num", type=int, default=20000, help="number of tfidf features")
    parser.add_argument("--min_df", type=int, default=1, help="min df")
    parser.add_argument("--max_df", type=float, default=0.05, help="max df")
    parser.add_argument(
        "--max_docs",
        type=int,
        default=-1,
        help="maximum number of documents to use when training tfidf, or -1 to disable",
    )
    args = parser.parse_args()
    print(args)

    v = TfidfVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
        ngram_range=(1, 3),
        max_features=args.num,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        max_df=args.max_df,
        min_df=args.min_df,
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

    logger.info("training tfidf vectors...")
    v.fit(make_corpus(training=True))

    logger.info("running inference...")
    x = v.transform(make_corpus(training=False)).astype(np.float32)
    logger.info(x.shape)

    logger.info("saving to features to disk...")
    features = {
        "pids": list(pdb.keys()),
        "x": x,
        "vocab": v.vocabulary_,
        "idf": v._tfidf.idf_,
    }
    save_features(features)
    logger.info("copy...")
    shutil.copyfile(FEATURES_FILE_NEW, FEATURES_FILE)
    logger.info("feature updated")
