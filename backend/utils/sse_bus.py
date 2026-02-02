"""SQLite-backed cross-process event bus for SSE.

This module provides a lightweight IPC mechanism using SQLite as the shared medium.
It is designed for single-host deployments and avoids external services (e.g., Redis).
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

from loguru import logger

import config


def _get_native_thread_class():
    """Return a real OS Thread class even under gevent monkey-patching."""
    try:
        import gevent.monkey

        if gevent.monkey.is_module_patched("threading"):
            return gevent.monkey.get_original("threading", "Thread")
    except Exception:
        return threading.Thread
    return threading.Thread


def _start_daemon_thread(*, target, name: str) -> threading.Thread:
    Thread = _get_native_thread_class()
    t = Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def _is_locked_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


@dataclass(frozen=True)
class EventRow:
    id: int
    user: str | None
    payload: dict
    ts: float


class SQLiteSSEBus:
    """Append-only event bus using a SQLite table.

    Concurrency model:
    - Many writers (web worker threads + Huey workers) insert rows.
    - Many readers (each gunicorn worker) tail the table and fan-out locally.
    """

    def __init__(
        self,
        db_path: str,
        *,
        busy_timeout_ms: int = 2000,
        wal_autocheckpoint: int = 1000,
        retry_queue_maxsize: int = 2000,
        retry_backoff_max_s: float = 1.0,
    ) -> None:
        self.db_path = str(db_path or "")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.wal_autocheckpoint = int(wal_autocheckpoint)
        self.retry_queue_maxsize = int(max(1, retry_queue_maxsize))
        self.retry_backoff_max_s = float(max(0.05, retry_backoff_max_s))
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        # Serialize writes within a single process to avoid N connections/threads
        # contending on the same SQLite write lock.
        self._write_lock = threading.Lock()
        self._retry_q: queue.Queue[tuple[str | None, str, float]] = queue.Queue(maxsize=self.retry_queue_maxsize)
        self._writer_lock = threading.Lock()
        self._writer_started = False
        self._writer_stop = threading.Event()
        self._sse_events_has_origin_pid: bool | None = None
        self._sse_events_has_origin_pid_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "publish_immediate_ok": 0,
            "publish_queued": 0,
            "publish_dropped": 0,
            "publish_failed": 0,
            "cleanup_runs": 0,
            "cleanup_deleted": 0,
        }

    def _connect(self) -> sqlite3.Connection:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=max(0.1, self.busy_timeout_ms / 1000), check_same_thread=False)
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Important: avoid setting journal_mode on every connection creation.
        # Switching journal mode needs an exclusive lock and can contend with concurrent
        # writers/readers (especially when the poller thread and a request thread start together).
        conn.row_factory = sqlite3.Row
        return conn

    def _table_has_column(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except Exception:
            return False
        for r in rows or []:
            try:
                name = r["name"]  # sqlite3.Row
            except Exception:
                try:
                    name = r[1]  # (cid, name, type, notnull, dflt_value, pk)
                except Exception:
                    name = None
            if str(name or "") == str(column):
                return True
        return False

    def _has_origin_pid_column(self, conn: sqlite3.Connection) -> bool:
        cached = self._sse_events_has_origin_pid
        if cached is not None:
            return bool(cached)
        with self._sse_events_has_origin_pid_lock:
            cached = self._sse_events_has_origin_pid
            if cached is not None:
                return bool(cached)
            has = self._table_has_column(conn, "sse_events", "origin_pid")
            self._sse_events_has_origin_pid = bool(has)
            return bool(has)

    def _insert_event(self, conn: sqlite3.Connection, user: str | None, msg: str, ts: float) -> sqlite3.Cursor:
        if self._has_origin_pid_column(conn):
            return conn.execute(
                "INSERT INTO sse_events(user, payload, ts, origin_pid) VALUES (?, ?, ?, ?)",
                (user, msg, ts, int(os.getpid())),
            )
        return conn.execute("INSERT INTO sse_events(user, payload, ts) VALUES (?, ?, ?)", (user, msg, ts))

    def _ensure_writer_started(self) -> None:
        if self._writer_started:
            return
        with self._writer_lock:
            if self._writer_started:
                return

            def _run() -> None:
                # Dedicated writer connection for async retries.
                conn = self._connect()
                while not self._writer_stop.is_set():
                    try:
                        user, msg, ts = self._retry_q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    backoff = 0.05
                    while not self._writer_stop.is_set():
                        try:
                            with self._write_lock:
                                self._insert_event(conn, user, msg, ts)
                                conn.commit()
                            break
                        except sqlite3.OperationalError as exc:
                            if _is_locked_error(exc):
                                try:
                                    conn.rollback()
                                except Exception:
                                    pass
                                time.sleep(backoff)
                                backoff = min(backoff * 2, self.retry_backoff_max_s)
                                continue
                            logger.warning(f"SSE bus async publish failed: {exc}")
                            break
                        except Exception as exc:
                            logger.warning(f"SSE bus async publish failed: {exc}")
                            break

                    self._retry_q.task_done()

                try:
                    conn.close()
                except Exception:
                    pass

            _start_daemon_thread(target=_run, name="sse-sqlite-writer")
            self._writer_started = True

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._get_conn()
            # Ensure WAL mode deterministically (avoids readers blocking writers).
            for attempt in range(3):
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if _is_locked_error(exc) and attempt < 2:
                        time.sleep(0.05 * (2**attempt))
                        continue
                    break
            try:
                conn.execute(f"PRAGMA wal_autocheckpoint={max(1, self.wal_autocheckpoint)}")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sse_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sse_leases (
                    name TEXT PRIMARY KEY,
                    holder TEXT NOT NULL,
                    expires_ts REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sse_events_ts ON sse_events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sse_events_user_id ON sse_events(user, id)")
            conn.commit()
            self._initialized = True

    def initialize(self) -> None:
        """Initialize schema and WAL mode (safe to call multiple times)."""
        self._ensure_schema()
        # Start the async writer thread lazily; it will only do work if lock retries happen.
        # Keeping it running reduces tail latency when the first lock contention occurs.
        self._ensure_writer_started()

    def publish(self, user: str | None, payload: dict) -> bool:
        ok, _bus_id = self.publish_with_id(user, payload)
        return ok

    def publish_async(self, user: str | None, payload: dict) -> bool:
        """Enqueue an event for async publishing (never blocks on SQLite locks)."""
        if not self.db_path:
            return False
        self._ensure_schema()
        try:
            msg = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            logger.warning(f"SSE bus payload not JSON-serializable: {exc}")
            return False

        # Use server insert time for retention/cleanup to avoid callers accidentally
        # providing stale timestamps that get cleaned up before being delivered.
        ts = time.time()
        self._ensure_writer_started()
        try:
            self._retry_q.put_nowait((user, msg, ts))
            with self._stats_lock:
                self._stats["publish_queued"] += 1
            return True
        except queue.Full:
            with self._stats_lock:
                self._stats["publish_dropped"] += 1
            return False

    def publish_with_id(self, user: str | None, payload: dict) -> tuple[bool, int | None]:
        """Publish an event and (best-effort) return the bus row id.

        - user=None means broadcast to all users.
        - payload must be JSON-serializable.
        - On lock contention, the event may be enqueued for async retry; in that case id is None.
        """
        if not self.db_path:
            return False, None
        self._ensure_schema()
        try:
            msg = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            logger.warning(f"SSE bus payload not JSON-serializable: {exc}")
            return False, None

        # Use server insert time for retention/cleanup to avoid callers accidentally
        # providing stale timestamps that get cleaned up before being delivered.
        ts = time.time()
        conn = self._get_conn()
        # Retry briefly in-line; if still locked, enqueue for async retry.
        for attempt in range(2):
            try:
                with self._write_lock:
                    cur = self._insert_event(conn, user, msg, ts)
                    bus_id = int(cur.lastrowid or 0) or None
                    conn.commit()
                with self._stats_lock:
                    self._stats["publish_immediate_ok"] += 1
                return True, bus_id
            except sqlite3.OperationalError as exc:
                if _is_locked_error(exc):
                    if attempt < 1:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        time.sleep(0.03)
                        continue
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    break
                logger.warning(f"Failed to publish SSE event: {exc}")
                with self._stats_lock:
                    self._stats["publish_failed"] += 1
                return False, None
            except Exception as exc:
                logger.warning(f"Failed to publish SSE event: {exc}")
                with self._stats_lock:
                    self._stats["publish_failed"] += 1
                return False, None
        # Locked: enqueue for async retry (best-effort).
        self._ensure_writer_started()
        try:
            self._retry_q.put_nowait((user, msg, ts))
            with self._stats_lock:
                self._stats["publish_queued"] += 1
            return True, None
        except queue.Full:
            with self._stats_lock:
                self._stats["publish_dropped"] += 1
            return False, None

    def get_latest_id(self) -> int:
        """Return the current max(id) in the table (0 if empty)."""
        if not self.db_path:
            return 0
        self._ensure_schema()
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT MAX(id) AS max_id FROM sse_events").fetchone()
            if row is None:
                return 0
            return int(row["max_id"] or 0)
        except Exception:
            return 0

    def fetch_after(self, last_id: int, *, limit: int) -> list[EventRow]:
        """Fetch events with id > last_id in ascending order."""
        if not self.db_path:
            return []
        self._ensure_schema()
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "SELECT id, user, payload, ts FROM sse_events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (int(last_id), int(limit)),
            )
            rows = cur.fetchall()
        except Exception as exc:
            logger.warning(f"Failed to fetch SSE events: {exc}")
            return []

        out: list[EventRow] = []
        for r in rows:
            try:
                payload = json.loads(r["payload"])
                if not isinstance(payload, dict):
                    continue
                out.append(EventRow(id=int(r["id"]), user=r["user"], payload=payload, ts=float(r["ts"])))
            except Exception:
                continue
        return out

    def cleanup_older_than(self, cutoff_ts: float, *, chunk: int = 2000) -> int:
        """Best-effort deletion of old events in small chunks to reduce lock contention."""
        if not self.db_path:
            return 0
        self._ensure_schema()
        conn = self._get_conn()
        deleted = 0
        try:
            # Chunked delete avoids holding a long write lock.
            with self._write_lock:
                cur = conn.execute(
                    """
                    DELETE FROM sse_events
                    WHERE id IN (
                        SELECT id FROM sse_events
                        WHERE ts < ?
                        ORDER BY id ASC
                        LIMIT ?
                    )
                    """,
                    (float(cutoff_ts), int(chunk)),
                )
                deleted = int(cur.rowcount or 0)
                if deleted:
                    conn.commit()
        except sqlite3.OperationalError as exc:
            if _is_locked_error(exc):
                try:
                    conn.rollback()
                except Exception:
                    pass
                return 0
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning(f"Failed to cleanup SSE events: {exc}")
            return 0
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning(f"Failed to cleanup SSE events: {exc}")
            return 0
        return deleted

    def try_acquire_lease(self, name: str, *, ttl_s: float) -> bool:
        """Best-effort cross-process lease using SQLite.

        Used to ensure only one process performs heavy housekeeping (e.g., cleanup) at a time.
        """
        if not self.db_path:
            return False
        self._ensure_schema()
        conn = self._get_conn()
        holder = str(os.getpid())
        now = time.time()
        expires = now + float(max(1.0, ttl_s))
        try:
            with self._write_lock:
                # Portable lease implementation (avoid SQLite version dependence on UPSERT ... WHERE).
                # 1) If expired, delete the old lease.
                conn.execute(
                    "DELETE FROM sse_leases WHERE name = ? AND expires_ts < ?",
                    (str(name), float(now)),
                )
                # 2) Try to acquire by inserting a new lease.
                cur = conn.execute(
                    "INSERT OR IGNORE INTO sse_leases(name, holder, expires_ts) VALUES (?, ?, ?)",
                    (str(name), holder, float(expires)),
                )
                acquired = int(cur.rowcount or 0) > 0
                # 3) If already held by us, renew.
                if not acquired:
                    cur2 = conn.execute(
                        "UPDATE sse_leases SET expires_ts = ? WHERE name = ? AND holder = ?",
                        (float(expires), str(name), holder),
                    )
                    acquired = int(cur2.rowcount or 0) > 0
                conn.commit()
            return bool(acquired)
        except sqlite3.OperationalError:
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return False

    def get_stats(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["retry_queue_size"] = int(getattr(self._retry_q, "qsize", lambda: 0)() or 0)
        stats["retry_queue_maxsize"] = int(self.retry_queue_maxsize)
        stats["busy_timeout_ms"] = int(self.busy_timeout_ms)
        # Avoid leaking filesystem layout via APIs; return only filename.
        stats["db_file"] = os.path.basename(str(self.db_path))
        return stats


_BUS_LOCK = threading.Lock()
_BUS: SQLiteSSEBus | None = None
_BUS_PATH: str | None = None
_BUS_PID: int | None = None


def get_sse_bus() -> SQLiteSSEBus | None:
    """Return a process-local bus handle bound to settings.sse.db_path."""
    global _BUS, _BUS_PATH, _BUS_PID
    pid = os.getpid()
    if _BUS_PID is not None and _BUS_PID != pid:
        # Fork safety: reset inherited handles/state.
        with _BUS_LOCK:
            _BUS = None
            _BUS_PATH = None
            _BUS_PID = pid
    if _BUS_PID is None:
        _BUS_PID = pid
    settings = config.settings
    if not getattr(settings, "sse", None):
        return None
    if not settings.sse.enabled:
        return None
    db_path = (settings.sse.db_path or "").strip()
    if not db_path:
        return None
    role = (os.environ.get("ARXIV_SANITY_PROCESS_ROLE") or "").strip().lower()
    timeout_s = 2.0
    try:
        if role == "worker" and hasattr(settings.db, "timeout_worker"):
            timeout_s = float(settings.db.timeout_worker)
        elif hasattr(settings.db, "timeout_web"):
            timeout_s = float(settings.db.timeout_web)
        else:
            timeout_s = float(getattr(settings.db, "timeout", 2) or 2)
    except Exception:
        timeout_s = 2.0
    busy_timeout_ms = int(max(100, timeout_s * 1000))
    with _BUS_LOCK:
        if _BUS is not None and _BUS_PATH == db_path:
            return _BUS
        # If config reloaded and db_path changed, bind to the new path for subsequent calls.
        _BUS = SQLiteSSEBus(
            db_path,
            busy_timeout_ms=busy_timeout_ms,
            wal_autocheckpoint=1000,
            retry_queue_maxsize=int(getattr(settings.sse, "publish_retry_queue_maxsize", 2000) or 2000),
            retry_backoff_max_s=float(getattr(settings.sse, "publish_retry_backoff_max_s", 1.0) or 1.0),
        )
        _BUS_PATH = db_path
        _BUS_PID = pid
        return _BUS


_POLL_LOCK = threading.Lock()
_POLL_STARTED = False
_POLL_THREAD: threading.Thread | None = None
_POLL_STOP = threading.Event()
_POLL_BUS_PATH: str | None = None
_POLL_PID: int | None = None


def ensure_poller_started(
    *,
    dispatch_user: Callable[[str, dict], None],
    dispatch_all: Callable[[dict], None],
) -> None:
    """Start a background poller thread (once per process).

    The poller tails the SQLite event table and dispatches events into in-process queues.
    """
    global _POLL_STARTED, _POLL_THREAD, _POLL_STOP, _POLL_BUS_PATH, _POLL_PID

    bus = get_sse_bus()
    if bus is None:
        return
    pid = os.getpid()
    if _POLL_PID is not None and _POLL_PID != pid:
        # Fork safety: never inherit a "started" flag across processes.
        with _POLL_LOCK:
            _POLL_STARTED = False
            _POLL_THREAD = None
            _POLL_STOP = threading.Event()
            _POLL_BUS_PATH = None
            _POLL_PID = pid
    if _POLL_PID is None:
        _POLL_PID = pid
    if _POLL_STARTED and _POLL_BUS_PATH == bus.db_path and _POLL_THREAD is not None and _POLL_THREAD.is_alive():
        return
    with _POLL_LOCK:
        if _POLL_STARTED and _POLL_BUS_PATH == bus.db_path and _POLL_THREAD is not None and _POLL_THREAD.is_alive():
            return
        # If a previous thread died, allow restart.
        if _POLL_THREAD is not None and not _POLL_THREAD.is_alive():
            _POLL_STARTED = False
            _POLL_THREAD = None
            _POLL_STOP = threading.Event()

        # Initialize bus before starting the poller thread to avoid first-use races
        # where multiple threads open connections before WAL is enabled.
        try:
            bus.initialize()
        except Exception:
            pass

        settings = config.settings
        interval = float(getattr(settings.sse, "poll_interval", 0.05) or 0.05)
        batch = int(getattr(settings.sse, "batch_size", 500) or 500)
        retention = int(getattr(settings.sse, "retention_seconds", 86400) or 86400)
        cleanup_every = float(getattr(settings.sse, "cleanup_interval", 60.0) or 60.0)
        # Start from the current tail to avoid replaying historical events on process start.
        initial_last_id = 0
        try:
            initial_last_id = int(bus.get_latest_id() or 0)
        except Exception:
            initial_last_id = 0

        def _run():
            last_id = initial_last_id
            last_cleanup = 0.0
            drain_loops = 0
            # Use a dedicated thread-local connection for polling.
            while not _POLL_STOP.is_set():
                try:
                    rows = bus.fetch_after(last_id, limit=batch)
                    if rows:
                        drain_loops += 1
                        for ev in rows:
                            payload = dict(ev.payload or {})
                            payload.setdefault("bus_id", ev.id)
                            if ev.user:
                                dispatch_user(str(ev.user), payload)
                            else:
                                dispatch_all(payload)
                        last_id = rows[-1].id
                        # Avoid starving the scheduler under sustained event floods.
                        # Yield periodically without adding latency when the backlog is small.
                        if drain_loops >= 20:
                            _POLL_STOP.wait(timeout=0)
                            drain_loops = 0
                        continue  # Drain backlog quickly without sleeping.
                    drain_loops = 0

                    now = time.time()
                    if retention > 0 and (now - last_cleanup) >= cleanup_every:
                        # Only one process should perform housekeeping at a time.
                        if not bus.try_acquire_lease("cleanup", ttl_s=max(5.0, cleanup_every * 2)):
                            last_cleanup = now
                        else:
                            cutoff = now - float(retention)
                            # Try a few chunks; stop early if no more deletions.
                            for _ in range(3):
                                n = bus.cleanup_older_than(cutoff, chunk=2000)
                                if n > 0:
                                    with bus._stats_lock:
                                        bus._stats["cleanup_deleted"] += int(n)
                                if n <= 0:
                                    break
                            with bus._stats_lock:
                                bus._stats["cleanup_runs"] += 1
                            last_cleanup = now
                except Exception as exc:
                    logger.warning(f"SSE poller loop error: {exc}")

                _POLL_STOP.wait(timeout=max(0.01, interval))

        _POLL_THREAD = _start_daemon_thread(target=_run, name="sse-sqlite-poller")
        _POLL_STARTED = True
        _POLL_BUS_PATH = bus.db_path
