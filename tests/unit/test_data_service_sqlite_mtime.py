"""Unit tests for SQLite mtime helper used by data_service cache invalidation."""

from __future__ import annotations

import os


def test_sqlite_effective_mtime_includes_wal_and_shm(tmp_path):
    from backend.services.data_service import _sqlite_effective_mtime

    db_path = tmp_path / "papers.db"
    wal_path = tmp_path / "papers.db-wal"
    shm_path = tmp_path / "papers.db-shm"

    db_path.write_text("db")
    wal_path.write_text("wal")
    shm_path.write_text("shm")

    os.utime(db_path, (100, 100))
    os.utime(shm_path, (150, 150))
    os.utime(wal_path, (200, 200))

    assert _sqlite_effective_mtime(str(db_path)) == 200.0
