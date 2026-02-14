"""Concurrency tests that cover SqliteKV retry and transaction handling."""

from __future__ import annotations

import sqlite3
import threading
import time

from aslite import db as aslite_db
from aslite.db import SqliteKV


class _FailingConnectionProxy:
    """Proxy connection that fails the first few execute/commit calls."""

    def __init__(self, conn: sqlite3.Connection, fail_execute: int = 0, fail_commit: int = 0):
        self._conn = conn
        self.execute_failures = fail_execute
        self.commit_failures = fail_commit

    def execute(self, *args, **kwargs):
        if self.execute_failures > 0:
            self.execute_failures -= 1
            raise sqlite3.OperationalError("database is locked")
        return self._conn.execute(*args, **kwargs)

    def commit(self, *args, **kwargs):
        if self.commit_failures > 0:
            self.commit_failures -= 1
            raise sqlite3.OperationalError("database is locked")
        return self._conn.commit(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._conn, name)


def _prepare_db(path: str) -> SqliteKV:
    """Create a writable SqliteKV instance for the test database."""
    return SqliteKV(path, "tokens", flag="c")


def test_execute_with_retry_retries_locked(tmp_path, monkeypatch):
    """Ensure execute retry handles busy errors before succeeding."""
    db_path = str(tmp_path / "sqlitekv_execute.db")
    kv = _prepare_db(db_path)
    try:
        proxy = _FailingConnectionProxy(kv.conn, fail_execute=2)
        kv._conn = proxy

        monkeypatch.setattr(aslite_db, "DB_MAX_RETRIES", 5)
        monkeypatch.setattr(aslite_db.time, "sleep", lambda *_: None)

        kv["key"] = "value"
        assert kv["key"] == "value"
        assert proxy.execute_failures == 0
    finally:
        kv.close()


def test_commit_with_retry_retries_locked(tmp_path, monkeypatch):
    """Ensure commit retry handles transient lock errors."""
    db_path = str(tmp_path / "sqlitekv_commit.db")
    kv = _prepare_db(db_path)
    try:
        proxy = _FailingConnectionProxy(kv.conn, fail_commit=2)
        kv._conn = proxy

        monkeypatch.setattr(aslite_db, "DB_MAX_RETRIES", 5)
        monkeypatch.setattr(aslite_db.time, "sleep", lambda *_: None)

        kv["persist"] = {"a": 1}
        assert kv["persist"]["a"] == 1
        assert proxy.commit_failures == 0
    finally:
        kv.close()


def test_transaction_handles_concurrent_lock(tmp_path, monkeypatch):
    """Validate lock contention when another writer holds a transaction."""
    db_path = str(tmp_path / "sqlitekv_transaction.db")

    monkeypatch.setattr(aslite_db, "DB_MAX_RETRIES", 12)
    monkeypatch.setattr(aslite_db, "DB_TIMEOUT", 0.05)
    monkeypatch.setattr(aslite_db, "DB_RETRY_BASE_SLEEP", 0.001)
    monkeypatch.setattr(aslite_db.time, "sleep", lambda *_: None)

    writer = SqliteKV(db_path, "tokens", flag="c")
    try:
        errors: list[Exception] = []
        ready = threading.Event()

        def worker():
            ready.set()
            try:
                writer["busy"] = "ok"
            except Exception as exc:
                errors.append(exc)

        locker = SqliteKV(db_path, "tokens", flag="c", autocommit=False)
        try:
            thread = threading.Thread(target=worker)
            thread.daemon = True
            with locker.transaction(mode="IMMEDIATE"):
                thread.start()
                assert ready.wait(timeout=1), "worker never started"
                time.sleep(0.05)
            thread.join(timeout=1)
            assert not thread.is_alive()
            assert not errors
            assert writer["busy"] == "ok"
        finally:
            locker.close()
    finally:
        writer.close()
