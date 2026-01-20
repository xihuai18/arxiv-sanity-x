"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.
"""

import os
import pickle
import sqlite3
import tempfile
import zlib
from contextlib import contextmanager

from sqlitedict import SqliteDict

from vars import DATA_DIR

# -----------------------------------------------------------------------------
# utilities for safe writing of a pickle file


# Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
@contextmanager
def _tempfile(*args, **kws):
    """Context for temporary file.
    Will find a free temporary filename upon entering
    and will try to delete the file on leaving
    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """Open temporary file object that atomically moves to destination upon
    exiting.
    Allows reading and writing to and from the same filename.
    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop("fsync", False)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)


def safe_pickle_dump(obj, fname):
    """
    prevents a case where one process could be writing a pickle file
    while another process is reading it, causing a crash. the solution
    is to write the pickle file to a temporary file and then move it.
    """
    with open_atomic(fname, "wb") as f:
        pickle.dump(obj, f, -1)  # -1 specifies highest binary protocol


# -----------------------------------------------------------------------------


class CompressedSqliteDict(SqliteDict):
    """overrides the encode/decode methods to use zlib, so we get compressed storage"""

    def __init__(self, *args, **kwargs):
        def encode(obj):
            return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

        def decode(obj):
            return pickle.loads(zlib.decompress(bytes(obj)))

        super().__init__(*args, **kwargs, encode=encode, decode=decode)


# -----------------------------------------------------------------------------
"""
some docs to self:
flag='c': default mode, open for read/write, and creating the db/table if necessary
flag='r': open for read-only
"""

# stores info about papers, and also their lighter-weight metadata
PAPERS_DB_FILE = os.path.join(DATA_DIR, "papers.db")
# stores account-relevant info, like which tags exist for which papers
DICT_DB_FILE = os.path.join(DATA_DIR, "dict.db")


def get_papers_db(flag="r", autocommit=True):
    assert flag in ["r", "c"]
    pdb = CompressedSqliteDict(PAPERS_DB_FILE, tablename="papers", flag=flag, autocommit=autocommit)
    return pdb


def get_metas_db(flag="r", autocommit=True):
    assert flag in ["r", "c"]
    mdb = SqliteDict(PAPERS_DB_FILE, tablename="metas", flag=flag, autocommit=autocommit)
    return mdb


def _safe_open_db(db_class, db_file, tablename, flag="r", autocommit=True):
    """
    Safely open a database table. If flag='r' and table doesn't exist,
    automatically create it first with flag='c', then reopen with flag='r'.
    """
    assert flag in ["r", "c"]
    try:
        return db_class(db_file, tablename=tablename, flag=flag, autocommit=autocommit)
    except RuntimeError as e:
        if "read-only" in str(e) and flag == "r":
            # Table doesn't exist, create it first
            db_class(db_file, tablename=tablename, flag="c", autocommit=True).close()
            return db_class(db_file, tablename=tablename, flag=flag, autocommit=autocommit)
        raise


def get_tags_db(flag="r", autocommit=True):
    return _safe_open_db(CompressedSqliteDict, DICT_DB_FILE, "tags", flag, autocommit)


def get_neg_tags_db(flag="r", autocommit=True):
    return _safe_open_db(CompressedSqliteDict, DICT_DB_FILE, "neg_tags", flag, autocommit)


def get_combined_tags_db(flag="r", autocommit=True):
    return _safe_open_db(CompressedSqliteDict, DICT_DB_FILE, "combined_tags", flag, autocommit)


def get_keywords_db(flag="r", autocommit=True):
    return _safe_open_db(CompressedSqliteDict, DICT_DB_FILE, "keywords", flag, autocommit)


def get_last_active_db(flag="r", autocommit=True):
    return _safe_open_db(SqliteDict, DICT_DB_FILE, "last_active", flag, autocommit)


def get_email_db(flag="r", autocommit=True):
    return _safe_open_db(SqliteDict, DICT_DB_FILE, "email", flag, autocommit)


def get_readinglist_db(flag="r", autocommit=True):
    """
    Reading list database with atomic per-entry storage.
    Key format: "user::pid" (e.g., "alice::2301.00001")
    Value: {added_time, top_tags, summary_triggered}

    This design avoids lost updates under concurrent writes by making each
    (user, pid) pair an independent key rather than nesting all pids under user.
    """
    return _safe_open_db(CompressedSqliteDict, DICT_DB_FILE, "readinglist", flag, autocommit)


def get_readinglist_index_db(flag="r", autocommit=True):
    """Per-user reading list index to avoid full scans.

    Key: "user::pid"
    Value: 1
    """
    return _safe_open_db(CompressedSqliteDict, DICT_DB_FILE, "readinglist_index", flag, autocommit)


def get_summary_status_db(flag="r", autocommit=True):
    """Summary status store.

    Key: "pid::model"
    Value: {status, last_error, updated_time}
    """
    return _safe_open_db(SqliteDict, DICT_DB_FILE, "summary_status", flag, autocommit)


def readinglist_key(user: str, pid: str) -> str:
    """Generate a reading list key from user and pid."""
    return f"{user}::{pid}"


def parse_readinglist_key(key: str) -> tuple:
    """Parse a reading list key into (user, pid). Returns (None, None) if invalid."""
    if "::" not in key:
        return None, None
    parts = key.split("::", 1)
    return parts[0], parts[1]


# -----------------------------------------------------------------------------
"""
our "feature store" is currently just a pickle file, may want to consider hdf5 in the future
"""

# stores tfidf features a bunch of other metadata
FEATURES_FILE_NEW = os.path.join(DATA_DIR, "features_new.p")
FEATURES_FILE = os.path.join(DATA_DIR, "features.p")


def save_features(features):
    """takes the features dict and save it to disk in a simple pickle file"""
    safe_pickle_dump(features, FEATURES_FILE_NEW)


def load_features():
    """loads the features dict from disk"""
    try:
        with open(FEATURES_FILE, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if str(e).startswith("No module named 'numpy._core'"):
            raise ModuleNotFoundError(
                "Failed to load features.p due to NumPy version mismatch: this file was likely created under NumPy 2.x, "
                "but the current environment is NumPy 1.x. Fix by upgrading NumPy to 2.x or regenerating features via "
                "`python3 compute.py` under the current environment."
            ) from e
        raise
