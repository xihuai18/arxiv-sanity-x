"""Tests for SqliteKV retry behavior under lock contention."""

from __future__ import annotations

import sqlite3
import threading
import time


def test_sqlitekv_retries_when_db_locked(tmp_path):
    import aslite.db as db

    # Speed up retry loop for tests.
    old_timeout = db.DB_TIMEOUT
    old_max_retries = db.DB_MAX_RETRIES
    old_sleep = db.DB_RETRY_BASE_SLEEP

    try:
        db.DB_TIMEOUT = 0
        db.DB_MAX_RETRIES = 8
        db.DB_RETRY_BASE_SLEEP = 0.01

        # Create a dedicated db file for this test to avoid interference.
        db_path = tmp_path / "dict.db"

        # Ensure table exists before taking the lock (avoids CREATE TABLE lock errors).
        kv = db.SqliteKV(str(db_path), "tags", flag="c", autocommit=True, compressed=True)
        kv.close()

        # Take an exclusive lock in a separate connection.
        locker = sqlite3.connect(str(db_path), timeout=0, check_same_thread=False)
        locker.execute("PRAGMA busy_timeout=0")
        locker.execute("BEGIN EXCLUSIVE")

        # Release the lock shortly after the writer starts.
        def release_lock():
            time.sleep(0.05)
            try:
                locker.rollback()
            finally:
                locker.close()

        threading.Thread(target=release_lock, daemon=True).start()

        result = {"ok": False, "error": ""}

        def writer():
            try:
                w = db.SqliteKV(str(db_path), "tags", flag="c", autocommit=True, compressed=True)
                w["u"] = {"t": {"p1"}}
                w.close()
                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        t.join(timeout=3.0)

        assert result["ok"] is True, result["error"]

        r = db.SqliteKV(str(db_path), "tags", flag="r", autocommit=True, compressed=True)
        assert r.get("u") == {"t": {"p1"}}
        r.close()
    finally:
        db.DB_TIMEOUT = old_timeout
        db.DB_MAX_RETRIES = old_max_retries
        db.DB_RETRY_BASE_SLEEP = old_sleep
