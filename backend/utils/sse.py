"""Server-sent events helpers."""

from __future__ import annotations

import os
import queue
import threading
import time
import weakref
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
    with USER_EVENT_LOCK:
        USER_EVENT_SUBS.setdefault(user, weakref.WeakSet()).add(q)
    return q


def unregister_user_stream(user: str, q: queue.Queue) -> None:
    with USER_EVENT_LOCK:
        subs = USER_EVENT_SUBS.get(user)
        if not subs:
            return
        try:
            subs.discard(q)
        except Exception:
            pass
        if len(subs) == 0:
            USER_EVENT_SUBS.pop(user, None)


def _enqueue_user_event(user: str, payload: dict) -> None:
    payload = _strip_origin(payload)
    with USER_EVENT_LOCK:
        subs = USER_EVENT_SUBS.get(user)
        queues = _snapshot_weakset(subs)
        if subs is not None and len(subs) == 0:
            USER_EVENT_SUBS.pop(user, None)
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
        _enqueue_user_event(user, payload)

    def _dispatch_all_from_bus(payload: dict) -> None:
        if isinstance(payload, dict) and payload.get(_ORIGIN_FIELD) == origin:
            return
        _enqueue_all_event(payload)

    ensure_poller_started(dispatch_user=_dispatch_user_from_bus, dispatch_all=_dispatch_all_from_bus)


def emit_user_event(user: str | None, payload: dict) -> None:
    if not user:
        return
    payload = dict(payload or {})
    payload.setdefault("ts", time.time())
    bus = get_sse_bus()
    if bus is not None:
        payload.setdefault(_ORIGIN_FIELD, _get_origin_id())
        settings = config.settings
        publish_async = bool(getattr(settings.sse, "publish_async", True))
        if publish_async:
            ok = bus.publish_async(user, payload)
        else:
            ok, bus_id = bus.publish_with_id(user, payload)
            if bus_id is not None:
                payload.setdefault("bus_id", bus_id)
        if not ok:
            _warn_rate_limited(
                f"sse_sqlite_publish:{bus.db_path}",
                5.0,
                f"Failed to publish SSE event to SQLite bus (db busy/locked): {bus.db_path}; cross-process delivery may be lost",
            )
        # Immediate in-process delivery (poller will skip by origin id).
        _enqueue_user_event(user, payload)
        return
    _enqueue_user_event(user, payload)


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
