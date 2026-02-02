import time


def test_sse_sqlite_ipc_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("ARXIV_SANITY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ARXIV_SANITY_SSE_ENABLED", "true")
    monkeypatch.setenv("ARXIV_SANITY_SSE_DB_PATH", str(tmp_path / "sse_events.db"))
    monkeypatch.setenv("ARXIV_SANITY_SSE_POLL_INTERVAL", "0.05")
    monkeypatch.setenv("ARXIV_SANITY_SSE_CLEANUP_INTERVAL", "9999")

    from config import reload_settings

    reload_settings()

    from backend.utils.sse import emit_user_event, register_user_stream
    from backend.utils.sse_bus import get_sse_bus

    q = register_user_stream("alice")

    emit_user_event("alice", {"type": "hello"})

    bus = get_sse_bus()
    assert bus is not None
    # Publishing may be async; allow a short window for the writer thread.
    t0 = time.time()
    while True:
        rows = bus.fetch_after(0, limit=50)
        if any(r.user == "alice" and r.payload.get("type") == "hello" for r in rows):
            break
        if time.time() - t0 > 3.0:
            raise AssertionError("Timed out waiting for SSE event to be persisted to the bus")
        time.sleep(0.05)

    t0 = time.time()
    while True:
        try:
            payload = q.get(timeout=0.5)
        except Exception:
            payload = None
        if payload and payload.get("type") == "hello":
            break
        if time.time() - t0 > 3.0:
            raise AssertionError("Timed out waiting for SSE IPC event")


def test_sse_bus_publish_locked_is_retried_async(tmp_path):
    import sqlite3

    from backend.utils.sse_bus import SQLiteSSEBus

    db_path = tmp_path / "sse_events.db"
    bus = SQLiteSSEBus(str(db_path), busy_timeout_ms=1)
    bus.initialize()

    lock_conn = sqlite3.connect(str(db_path), timeout=0.1, check_same_thread=False)
    lock_conn.execute("BEGIN IMMEDIATE")
    try:
        # Under lock contention, publish should enqueue for async retry instead of dropping.
        assert bus.publish(None, {"type": "locked"}) is True
    finally:
        try:
            lock_conn.rollback()
        except Exception:
            pass
        lock_conn.close()

    t0 = time.time()
    while True:
        rows = bus.fetch_after(0, limit=50)
        if any(r.payload.get("type") == "locked" for r in rows):
            break
        if time.time() - t0 > 3.0:
            raise AssertionError("Timed out waiting for async SSE publish retry")
        time.sleep(0.05)


def test_sse_bus_publish_with_origin_pid_column(tmp_path):
    import os
    import sqlite3

    from backend.utils.sse_bus import SQLiteSSEBus

    db_path = tmp_path / "sse_events.db"
    conn = sqlite3.connect(str(db_path), timeout=0.1, check_same_thread=False)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sse_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NULL,
                payload TEXT NOT NULL,
                ts REAL NOT NULL,
                origin_pid INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    bus = SQLiteSSEBus(str(db_path), busy_timeout_ms=50)
    bus.initialize()
    assert bus.publish_async(None, {"type": "origin_pid"}) is True

    t0 = time.time()
    while True:
        rows = bus.fetch_after(0, limit=50)
        if any(r.payload.get("type") == "origin_pid" for r in rows):
            break
        if time.time() - t0 > 3.0:
            raise AssertionError("Timed out waiting for SSE publish into origin_pid schema")
        time.sleep(0.05)

    conn = sqlite3.connect(str(db_path), timeout=0.1, check_same_thread=False)
    try:
        r = conn.execute("SELECT origin_pid FROM sse_events ORDER BY id DESC LIMIT 1").fetchone()
        assert r is not None
        assert int(r[0]) == int(os.getpid())
    finally:
        conn.close()
