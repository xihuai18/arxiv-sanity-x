"""Server-sent events helpers."""

from __future__ import annotations

import os
import queue
import threading
import time
import weakref
from hashlib import sha256
from secrets import token_hex

from loguru import logger

import config

from .sse_bus import ensure_poller_started, get_sse_bus

USER_EVENT_SUBS: dict[str, weakref.WeakSet[queue.Queue]] = {}
USER_EVENT_LOCK = threading.Lock()

_ORIGIN_FIELD = "__sse_origin"
_ORIGIN_LOCK = threading.Lock()
_ORIGIN_PID: int | None = None
_ORIGIN_ID: str | None = None

_WARN_LOCK = threading.Lock()
_WARN_LAST: dict[str, float] = {}

_SSE_STATS_LOCK = threading.Lock()
_SSE_STATS = {
    "queue_full_drops": 0,
    "enqueue_fail_drops": 0,
}

_JANITOR_LOCK = threading.Lock()
_JANITOR_STARTED = False

_CONN_LOCK = threading.Lock()
_CONN_LOCAL: dict[str, set[str]] = {}


def _get_native_thread_class():
    """Return a real OS Thread class even under gevent monkey-patching."""
    try:
        import gevent.monkey

        if gevent.monkey.is_module_patched("threading"):
            return gevent.monkey.get_original("threading", "Thread")
    except Exception:
        return threading.Thread
    return threading.Thread


def _start_daemon_thread(*, target, name: str) -> None:
    Thread = _get_native_thread_class()
    t = Thread(target=target, name=name, daemon=True)
    t.start()


def _snapshot_weakset(subs: weakref.WeakSet[queue.Queue] | None) -> list[queue.Queue]:
    """Best-effort snapshot for WeakSet (may mutate during GC)."""
    if not subs:
        return []
    for _ in range(3):
        try:
            return list(subs)
        except RuntimeError:
            continue
        except Exception:
            break
    return []


def _warn_rate_limited(key: str, interval_s: float, message: str) -> None:
    """Emit warning at most once per interval (per key)."""
    now = time.time()
    with _WARN_LOCK:
        last = _WARN_LAST.get(key, 0.0)
        if now - last < float(interval_s):
            return
        _WARN_LAST[key] = now
    logger.warning(message)


def _get_origin_id() -> str:
    """Return a per-process origin id (regenerated after fork)."""
    global _ORIGIN_ID, _ORIGIN_PID
    pid = os.getpid()
    if _ORIGIN_ID is not None and _ORIGIN_PID == pid:
        return _ORIGIN_ID
    with _ORIGIN_LOCK:
        pid = os.getpid()
        if _ORIGIN_ID is not None and _ORIGIN_PID == pid:
            return _ORIGIN_ID
        _ORIGIN_PID = pid
        _ORIGIN_ID = f"{pid}-{token_hex(8)}"
        return _ORIGIN_ID


def _strip_origin(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    if _ORIGIN_FIELD not in payload:
        return payload
    clean = dict(payload)
    clean.pop(_ORIGIN_FIELD, None)
    return clean


def _enqueue_with_drop(q: queue.Queue, payload: dict, *, max_drop_attempts: int = 3) -> None:
    """Best-effort enqueue. When full, drop oldest entries first."""
    for _ in range(max(1, int(max_drop_attempts))):
        try:
            q.put_nowait(payload)
            return
        except queue.Full:
            with _SSE_STATS_LOCK:
                _SSE_STATS["queue_full_drops"] += 1
            try:
                q.get_nowait()
            except Exception:
                pass
            continue
        except Exception:
            with _SSE_STATS_LOCK:
                _SSE_STATS["enqueue_fail_drops"] += 1
            return
    with _SSE_STATS_LOCK:
        _SSE_STATS["enqueue_fail_drops"] += 1


def _ensure_janitor_started() -> None:
    global _JANITOR_STARTED
    if _JANITOR_STARTED:
        return
    with _JANITOR_LOCK:
        if _JANITOR_STARTED:
            return

        def _run() -> None:
            while True:
                time.sleep(60.0)
                with USER_EVENT_LOCK:
                    for u, subs in list(USER_EVENT_SUBS.items()):
                        try:
                            if not subs or len(subs) == 0:
                                USER_EVENT_SUBS.pop(u, None)
                        except Exception:
                            continue

        _start_daemon_thread(target=_run, name="sse-janitor")
        _JANITOR_STARTED = True


class SSEUserConnectionLease:
    """A best-effort per-user connection slot.

    - Cross-process: backed by SQLite leases (sse_leases).
    - Fallback: in-process reservation (for development/testing).
    """

    def __init__(
        self,
        *,
        user_key: str,
        lease_name: str,
        holder: str,
        ttl_s: float,
        release_fn=None,
    ) -> None:
        self.user_key = str(user_key or "")
        self.lease_name = str(lease_name or "")
        self.holder = str(holder or "")
        self.ttl_s = float(max(1.0, ttl_s))
        self._released = False
        self._release_fn = release_fn
        self._last_renew_ts = 0.0
        self._last_renew_attempt_ts = 0.0
        self._lock = threading.Lock()

    def renew(self) -> None:
        with self._lock:
            if self._released:
                return
            now = time.time()
            # Avoid excessive SQLite writes: renew at most every ~ttl/3 (capped at 30s),
            # but retry failed renewals with a small backoff.
            min_success_interval = min(30.0, max(1.0, self.ttl_s / 3.0))
            min_attempt_interval = 1.0
            if (now - self._last_renew_attempt_ts) < min_attempt_interval:
                return
            if (now - self._last_renew_ts) < min_success_interval:
                return
            self._last_renew_attempt_ts = now
            lease_name = self.lease_name
            holder = self.holder
            ttl_s = self.ttl_s
        bus = get_sse_bus()
        if bus is None:
            return
        try:
            ok = bus.try_acquire_lease(lease_name, ttl_s=ttl_s, holder=holder)
            if ok:
                with self._lock:
                    self._last_renew_ts = now
            # If the lease was released while renew was in-flight, best-effort cleanup.
            if ok:
                with self._lock:
                    released = self._released
                if released:
                    try:
                        bus.release_lease(lease_name, holder=holder)
                    except Exception:
                        pass
        except Exception:
            return

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        if self._release_fn is not None:
            try:
                self._release_fn()
            except Exception:
                pass
            return
        bus = get_sse_bus()
        if bus is None:
            return
        try:
            bus.release_lease(self.lease_name, holder=self.holder)
        except Exception:
            return


def _user_conn_key(user: str) -> str:
    """Stable, non-PII key for lease names."""
    u = str(user or "")
    if not u:
        return "anon"
    return sha256(u.encode("utf-8")).hexdigest()[:24]


def _user_stream_key(user: str) -> str:
    """Stable, non-PII key for SSE routing and SQLite bus storage."""
    return _user_conn_key(user)


def get_max_connections_per_user() -> int:
    settings = config.settings
    try:
        limit = int(getattr(settings.sse, "max_connections_per_user", 2) or 2)
    except Exception:
        limit = 2
    return int(limit)


def try_acquire_user_connection_lease(user: str) -> tuple[bool, SSEUserConnectionLease | None, int]:
    """Try to reserve a per-user SSE connection slot.

    Returns (ok, lease, limit). If ok is False, the caller should return HTTP 429.
    """
    limit = get_max_connections_per_user()
    if limit <= 0:
        return True, None, limit

    settings = config.settings
    try:
        ttl_s = float(getattr(settings.sse, "connection_lease_ttl_s", 90.0) or 90.0)
    except Exception:
        ttl_s = 90.0

    bus = get_sse_bus()
    user_key = _user_conn_key(user)
    holder = f"{os.getpid()}-{token_hex(8)}"

    if bus is None:
        with _CONN_LOCK:
            holders = _CONN_LOCAL.setdefault(user_key, set())
            if len(holders) >= limit:
                return False, None, limit
            holders.add(holder)

        def _local_release() -> None:
            with _CONN_LOCK:
                hs = _CONN_LOCAL.get(user_key)
                if not hs:
                    return
                hs.discard(holder)
                if len(hs) == 0:
                    _CONN_LOCAL.pop(user_key, None)

        lease = SSEUserConnectionLease(
            user_key=user_key,
            lease_name=f"local:{user_key}",
            holder=holder,
            ttl_s=ttl_s,
            release_fn=_local_release,
        )
        return True, lease, limit

    # Cross-process lease slots: sse_conn:<user_hash>:<slot>
    for slot in range(int(limit)):
        lease_name = f"sse_conn:{user_key}:{slot}"
        try:
            if bus.try_acquire_lease(lease_name, ttl_s=ttl_s, holder=holder):
                return (
                    True,
                    SSEUserConnectionLease(user_key=user_key, lease_name=lease_name, holder=holder, ttl_s=ttl_s),
                    limit,
                )
        except Exception:
            continue

    return False, None, limit


def register_user_stream(user: str) -> queue.Queue:
    # Starting the poller here makes SSE robust even if the app forgot to call ensure_sse_runtime_started().
    try:
        ensure_sse_runtime_started()
    except Exception:
        pass
    _ensure_janitor_started()
    settings = config.settings
    maxsize = int(getattr(settings.sse, "queue_maxsize", 200) or 200)
    # queue.Queue(maxsize=0) means "infinite" (never Full).
    qmax = 0 if maxsize <= 0 else max(1, maxsize)
    q = queue.Queue(maxsize=qmax)
    user_key = _user_stream_key(user)
    with USER_EVENT_LOCK:
        USER_EVENT_SUBS.setdefault(user_key, weakref.WeakSet()).add(q)
    return q


def unregister_user_stream(user: str, q: queue.Queue) -> None:
    user_key = _user_stream_key(user)
    with USER_EVENT_LOCK:
        subs = USER_EVENT_SUBS.get(user_key)
        if not subs:
            return
        try:
            subs.discard(q)
        except Exception:
            pass
        if len(subs) == 0:
            USER_EVENT_SUBS.pop(user_key, None)


def _enqueue_user_event(user_key: str, payload: dict) -> None:
    payload = _strip_origin(payload)
    with USER_EVENT_LOCK:
        subs = USER_EVENT_SUBS.get(user_key)
        queues = _snapshot_weakset(subs)
        if subs is not None and len(subs) == 0:
            USER_EVENT_SUBS.pop(user_key, None)
    for q in queues:
        _enqueue_with_drop(q, payload, max_drop_attempts=3)


def _enqueue_all_event(payload: dict) -> None:
    payload = _strip_origin(payload)
    with USER_EVENT_LOCK:
        all_queues = []
        for u, subs in list(USER_EVENT_SUBS.items()):
            try:
                qs = _snapshot_weakset(subs)
                if not qs:
                    USER_EVENT_SUBS.pop(u, None)
                    continue
                all_queues.extend(qs)
            except Exception:
                continue
    for q in all_queues:
        _enqueue_with_drop(q, payload, max_drop_attempts=3)


def ensure_sse_runtime_started() -> None:
    """Ensure the SQLite SSE poller is running in this process."""
    bus = get_sse_bus()
    if bus is None:
        return

    origin = _get_origin_id()

    def _dispatch_user_from_bus(user: str, payload: dict) -> None:
        if isinstance(payload, dict) and payload.get(_ORIGIN_FIELD) == origin:
            return
        _enqueue_user_event(str(user), payload)

    def _dispatch_all_from_bus(payload: dict) -> None:
        if isinstance(payload, dict) and payload.get(_ORIGIN_FIELD) == origin:
            return
        _enqueue_all_event(payload)

    ensure_poller_started(dispatch_user=_dispatch_user_from_bus, dispatch_all=_dispatch_all_from_bus)


def emit_user_event(user: str | None, payload: dict) -> None:
    if not user:
        return
    user_key = _user_stream_key(str(user))
    payload = dict(payload or {})
    payload.setdefault("ts", time.time())
    bus = get_sse_bus()
    if bus is not None:
        payload.setdefault(_ORIGIN_FIELD, _get_origin_id())
        settings = config.settings
        publish_async = bool(getattr(settings.sse, "publish_async", True))
        if publish_async:
            ok = bus.publish_async(user_key, payload)
        else:
            ok, bus_id = bus.publish_with_id(user_key, payload)
            if bus_id is not None:
                payload.setdefault("bus_id", bus_id)
        if not ok:
            _warn_rate_limited(
                f"sse_sqlite_publish:{bus.db_path}",
                5.0,
                f"Failed to publish SSE event to SQLite bus (db busy/locked): {bus.db_path}; cross-process delivery may be lost",
            )
        # Immediate in-process delivery (poller will skip by origin id).
        _enqueue_user_event(user_key, payload)
        return
    _enqueue_user_event(user_key, payload)


def emit_all_event(payload: dict) -> None:
    payload = dict(payload or {})
    payload.setdefault("ts", time.time())
    bus = get_sse_bus()
    if bus is not None:
        payload.setdefault(_ORIGIN_FIELD, _get_origin_id())
        settings = config.settings
        publish_async = bool(getattr(settings.sse, "publish_async", True))
        if publish_async:
            ok = bus.publish_async(None, payload)
        else:
            ok, bus_id = bus.publish_with_id(None, payload)
            if bus_id is not None:
                payload.setdefault("bus_id", bus_id)
        if not ok:
            _warn_rate_limited(
                f"sse_sqlite_publish:{bus.db_path}",
                5.0,
                f"Failed to publish SSE broadcast event to SQLite bus (db busy/locked): {bus.db_path}; cross-process delivery may be lost",
            )
        # Immediate in-process delivery (poller will skip by origin id).
        _enqueue_all_event(payload)
        return
    _enqueue_all_event(payload)


def get_sse_stats() -> dict:
    """Return process-local SSE stats (best-effort)."""
    with USER_EVENT_LOCK:
        users = len(USER_EVENT_SUBS)
        queues = 0
        for subs in USER_EVENT_SUBS.values():
            try:
                queues += len(subs)
            except Exception:
                continue
    settings = config.settings
    maxsize = int(getattr(settings.sse, "queue_maxsize", 200) or 200)
    maxsize = 0 if maxsize <= 0 else int(maxsize)
    with _SSE_STATS_LOCK:
        extra = dict(_SSE_STATS)
    extra.update(
        {
            "subscribed_users": int(users),
            "subscriber_queues": int(queues),
            "queue_maxsize": int(maxsize),
        }
    )
    return extra
