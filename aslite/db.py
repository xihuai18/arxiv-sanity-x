"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.

This version uses native sqlite3 instead of sqlitedict to avoid lock conflicts.
"""

import os
import pickle
import re
import sqlite3
import tempfile
import time
import zlib
from contextlib import contextmanager

from config import settings

DATA_DIR = str(settings.data_dir)

# -----------------------------------------------------------------------------
# SQLite configuration (from centralized settings)
DB_TIMEOUT = settings.db.timeout
DB_MAX_RETRIES = settings.db.max_retries
DB_RETRY_BASE_SLEEP = settings.db.retry_base_sleep


def _init_connection(conn: sqlite3.Connection, enable_wal: bool = False):
    """Initialize connection with optimal settings for concurrency."""
    # Apply settings in an order that minimizes startup lock contention.
    # - busy_timeout: how long SQLite waits when encountering a locked DB
    # - synchronous: durability/performance tradeoff (WAL + NORMAL is common)
    # - journal_mode=WAL: enables concurrent readers + single writer
    conn.execute(f"PRAGMA busy_timeout={DB_TIMEOUT * 1000}")  # milliseconds
    conn.execute("PRAGMA synchronous=NORMAL")
    if enable_wal:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # Another process may be changing journal mode or holding a lock.
            # We'll still benefit from busy_timeout + retry logic.
            pass


# -----------------------------------------------------------------------------
# Native SQLite dict-like wrapper


# Valid table name pattern to prevent SQL injection
_VALID_TABLENAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class SqliteKV:
    """
    A dict-like interface over a SQLite table with (key TEXT, value BLOB).
    Compatible with the legacy sqlitedict table schema for seamless migration.
    Uses WAL mode and busy_timeout for better concurrency.
    """

    def __init__(
        self,
        db_path: str,
        tablename: str,
        flag: str = "r",
        autocommit: bool = True,
        compressed: bool = False,
    ):
        """
        Args:
            db_path: Path to SQLite database file
            tablename: Table name to use
            flag: 'r' for read-only, 'c' for read-write (create if needed)
            autocommit: If True, commit after each write operation
            compressed: If True, use zlib compression for values
        """
        # Validate flag parameter
        if flag not in ("r", "c"):
            raise ValueError(f"Invalid flag '{flag}': must be 'r' (read-only) or 'c' (read-write/create)")

        # Validate tablename to prevent SQL injection
        if not _VALID_TABLENAME_RE.match(tablename):
            raise ValueError(
                f"Invalid table name '{tablename}': must be alphanumeric with underscores, starting with letter or underscore"
            )

        self.db_path = db_path
        self.tablename = tablename
        self.flag = flag
        self.autocommit = autocommit
        self.compressed = compressed
        self._conn = None
        self._closed = False

        # Open connection
        if flag == "r":
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found: {db_path}")
            self._conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                timeout=DB_TIMEOUT,
                check_same_thread=False,
            )
        else:
            # Create directory if needed (handle empty dirname case)
            db_dir = os.path.dirname(db_path)
            if db_dir:  # Only create if there's actually a directory component
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT, check_same_thread=False)

        _init_connection(self._conn, enable_wal=(flag == "c"))

        # Create table if needed (only for write mode)
        if flag == "c":
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {tablename} (key TEXT PRIMARY KEY, value BLOB)")
            if self.autocommit:
                self._conn.commit()
        elif flag == "r":
            # Verify table exists for read-only mode
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (tablename,),
            )
            if cursor.fetchone() is None:
                self._conn.close()
                raise sqlite3.OperationalError(f"no such table: {tablename}")

    def _encode(self, obj):
        """Encode object to bytes for storage."""
        data = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        if self.compressed:
            data = zlib.compress(data)
        return sqlite3.Binary(data)

    def _decode(self, data):
        """Decode bytes from storage to object."""
        if data is None:
            return None
        data = bytes(data)
        if self.compressed:
            data = zlib.decompress(data)
        return pickle.loads(data)

    def _execute_with_retry(self, sql: str, params=()):
        for attempt in range(DB_MAX_RETRIES):
            try:
                return self._conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    if attempt >= DB_MAX_RETRIES - 1:
                        raise
                    time.sleep(DB_RETRY_BASE_SLEEP * (2**attempt))
                    continue
                raise

    def _commit_with_retry(self):
        if not self._conn:
            return
        for attempt in range(DB_MAX_RETRIES):
            try:
                self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    if attempt >= DB_MAX_RETRIES - 1:
                        raise
                    time.sleep(DB_RETRY_BASE_SLEEP * (2**attempt))
                    continue
                raise

    def __getitem__(self, key: str):
        cursor = self._execute_with_retry(f"SELECT value FROM {self.tablename} WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        return self._decode(row[0])

    def __setitem__(self, key: str, value):
        if self.flag == "r":
            raise RuntimeError("Cannot write to read-only database")
        self._execute_with_retry(
            f"INSERT OR REPLACE INTO {self.tablename} (key, value) VALUES (?, ?)",
            (key, self._encode(value)),
        )
        if self.autocommit:
            self._commit_with_retry()

    def __delitem__(self, key: str):
        if self.flag == "r":
            raise RuntimeError("Cannot write to read-only database")
        cursor = self._execute_with_retry(f"DELETE FROM {self.tablename} WHERE key = ?", (key,))
        if cursor.rowcount == 0:
            raise KeyError(key)
        if self.autocommit:
            self._commit_with_retry()

    def __contains__(self, key: str) -> bool:
        cursor = self._execute_with_retry(f"SELECT 1 FROM {self.tablename} WHERE key = ? LIMIT 1", (key,))
        return cursor.fetchone() is not None

    def __len__(self) -> int:
        cursor = self._execute_with_retry(f"SELECT COUNT(*) FROM {self.tablename}")
        return cursor.fetchone()[0]

    def __iter__(self):
        return self.keys()

    def get(self, key: str, default=None):
        """Get value by key, return default if not found."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Iterate over all keys."""
        cursor = self._execute_with_retry(f"SELECT key FROM {self.tablename}")
        for row in cursor:
            yield row[0]

    def iterkeys(self):
        return self.keys()

    def values(self):
        """Iterate over all values."""
        cursor = self._execute_with_retry(f"SELECT value FROM {self.tablename}")
        for row in cursor:
            yield self._decode(row[0])

    def itervalues(self):
        return self.values()

    def items(self):
        """Iterate over all (key, value) pairs."""
        cursor = self._execute_with_retry(f"SELECT key, value FROM {self.tablename}")
        for row in cursor:
            yield row[0], self._decode(row[1])

    def iteritems(self):
        return self.items()

    def update(self, mapping):
        for key, value in mapping.items():
            self[key] = value

    def get_many(self, keys: list) -> dict:
        """Batch get multiple keys in a single query. Returns {key: value} for found keys."""
        if not keys:
            return {}
        # SQLite has a limit on the number of variables, chunk if needed
        result = {}
        chunk_size = 500
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursor = self._execute_with_retry(
                f"SELECT key, value FROM {self.tablename} WHERE key IN ({placeholders})",
                tuple(chunk),
            )
            for row in cursor:
                result[row[0]] = self._decode(row[1])
        return result

    def set_many(self, mapping: dict):
        """Batch set multiple key-value pairs in a single transaction."""
        if self.flag == "r":
            raise RuntimeError("Cannot write to read-only database")
        if not mapping:
            return
        for key, value in mapping.items():
            self._execute_with_retry(
                f"INSERT OR REPLACE INTO {self.tablename} (key, value) VALUES (?, ?)",
                (key, self._encode(value)),
            )
        if self.autocommit:
            self._commit_with_retry()

    def keys_with_prefix(self, prefix: str):
        """Iterate over keys that start with the given prefix."""
        cursor = self._execute_with_retry(
            f"SELECT key FROM {self.tablename} WHERE key LIKE ?",
            (prefix + "%",),
        )
        for row in cursor:
            yield row[0]

    def items_with_prefix(self, prefix: str):
        """Iterate over (key, value) pairs where key starts with the given prefix."""
        cursor = self._execute_with_retry(
            f"SELECT key, value FROM {self.tablename} WHERE key LIKE ?",
            (prefix + "%",),
        )
        for row in cursor:
            yield row[0], self._decode(row[1])

    def commit(self):
        """Commit pending changes."""
        self._commit_with_retry()

    def close(self):
        """Close the database connection."""
        # Use getattr to handle case where __init__ failed before setting attributes
        conn = getattr(self, "_conn", None)
        closed = getattr(self, "_closed", True)
        if conn and not closed:
            try:
                if self.flag == "c" and not self.autocommit:
                    try:
                        self._commit_with_retry()
                    except Exception:
                        pass
                conn.close()
            finally:
                self._closed = True
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def transaction(self, mode: str = "IMMEDIATE"):
        """Execute multiple operations in a single SQLite transaction.

        This is useful for read-modify-write updates on a single key where we
        want to avoid lost updates under concurrency.
        """
        if self.flag == "r":
            raise RuntimeError("Cannot start a transaction on read-only database")

        mode_u = (mode or "IMMEDIATE").upper()
        if mode_u not in ("DEFERRED", "IMMEDIATE", "EXCLUSIVE"):
            raise ValueError(f"Invalid transaction mode: {mode}")

        self._execute_with_retry(f"BEGIN {mode_u}")
        try:
            yield self
        except Exception:
            try:
                self._execute_with_retry("ROLLBACK")
            except Exception:
                pass
            raise
        else:
            self._commit_with_retry()

    def __del__(self):
        """Ensure connection is closed when object is garbage collected."""
        self.close()


# -----------------------------------------------------------------------------
# utilities for safe writing of a pickle file


@contextmanager
def _tempfile(*args, **kws):
    """Context for temporary file."""
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
    """Open temporary file object that atomically moves to destination upon exiting."""
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
    while another process is reading it, causing a crash.
    """
    with open_atomic(fname, "wb") as f:
        pickle.dump(obj, f, -1)


# -----------------------------------------------------------------------------
# Database file paths

PAPERS_DB_FILE = os.path.join(DATA_DIR, "papers.db")
DICT_DB_FILE = os.path.join(DATA_DIR, "dict.db")


# -----------------------------------------------------------------------------
# Database accessor functions (compatible with old sqlitedict interface)


def _safe_open_db(
    db_path: str,
    tablename: str,
    flag: str = "r",
    autocommit: bool = True,
    compressed: bool = False,
    allow_create_on_read: bool = False,
):
    """
    Safely open a database table. If flag='r' and table doesn't exist,
    automatically create it first with flag='c', then reopen with flag='r'.
    """
    try:
        return SqliteKV(db_path, tablename, flag=flag, autocommit=autocommit, compressed=compressed)
    except (FileNotFoundError, sqlite3.OperationalError):
        if flag == "r" and allow_create_on_read:
            # Table or DB doesn't exist, create it first
            with SqliteKV(db_path, tablename, flag="c", autocommit=True, compressed=compressed):
                pass  # Just create and close
            return SqliteKV(
                db_path,
                tablename,
                flag=flag,
                autocommit=autocommit,
                compressed=compressed,
            )
        raise


def get_papers_db(flag="r", autocommit=True):
    """Get papers database (compressed)."""
    return _safe_open_db(
        PAPERS_DB_FILE,
        "papers",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=False,
    )


def get_metas_db(flag="r", autocommit=True):
    """Get metas database (not compressed, for faster access)."""
    return _safe_open_db(
        PAPERS_DB_FILE,
        "metas",
        flag,
        autocommit,
        compressed=False,
        allow_create_on_read=False,
    )


def get_tags_db(flag="r", autocommit=True):
    return _safe_open_db(
        DICT_DB_FILE,
        "tags",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_neg_tags_db(flag="r", autocommit=True):
    return _safe_open_db(
        DICT_DB_FILE,
        "neg_tags",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_combined_tags_db(flag="r", autocommit=True):
    return _safe_open_db(
        DICT_DB_FILE,
        "combined_tags",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_keywords_db(flag="r", autocommit=True):
    return _safe_open_db(
        DICT_DB_FILE,
        "keywords",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_email_db(flag="r", autocommit=True):
    return _safe_open_db(
        DICT_DB_FILE,
        "email",
        flag,
        autocommit,
        compressed=False,
        allow_create_on_read=True,
    )


def get_readinglist_db(flag="r", autocommit=True):
    """
    Reading list database with atomic per-entry storage.
    Key format: "user::pid" (e.g., "alice::2301.00001")
    Value: {added_time, top_tags, summary_triggered}
    """
    return _safe_open_db(
        DICT_DB_FILE,
        "readinglist",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_readinglist_index_db(flag="r", autocommit=True):
    """Per-user reading list index to avoid full scans."""
    return _safe_open_db(
        DICT_DB_FILE,
        "readinglist_index",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_uploaded_papers_db(flag="r", autocommit=True):
    """
    Uploaded papers database.
    Key format: pid (e.g., "up_V1StGXR8_Z5j")
    Value: UploadedPaper dict with metadata, parse status, etc.
    """
    return _safe_open_db(
        DICT_DB_FILE,
        "uploaded_papers",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_uploaded_papers_index_db(flag="r", autocommit=True):
    """
    Per-user uploaded papers index to avoid full scans.
    Key format: user
    Value: list of pids
    """
    return _safe_open_db(
        DICT_DB_FILE,
        "uploaded_papers_index",
        flag,
        autocommit,
        compressed=True,
        allow_create_on_read=True,
    )


def get_summary_status_db(flag="r", autocommit=True):
    """Summary status store."""
    return _safe_open_db(
        DICT_DB_FILE,
        "summary_status",
        flag,
        autocommit,
        compressed=False,
        allow_create_on_read=True,
    )


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
# Feature store (pickle files)

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
