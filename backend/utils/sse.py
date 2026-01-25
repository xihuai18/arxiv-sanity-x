"""Server-sent events helpers."""

from __future__ import annotations

import queue
import threading
import time

USER_EVENT_SUBS = {}
USER_EVENT_LOCK = threading.Lock()


def register_user_stream(user: str) -> queue.Queue:
    q = queue.Queue(maxsize=200)
    with USER_EVENT_LOCK:
        USER_EVENT_SUBS.setdefault(user, set()).add(q)
    return q


def unregister_user_stream(user: str, q: queue.Queue) -> None:
    with USER_EVENT_LOCK:
        subs = USER_EVENT_SUBS.get(user)
        if not subs:
            return
        subs.discard(q)
        if not subs:
            USER_EVENT_SUBS.pop(user, None)


def emit_user_event(user: str | None, payload: dict) -> None:
    if not user:
        return
    payload = dict(payload or {})
    payload.setdefault("ts", time.time())
    with USER_EVENT_LOCK:
        queues = list(USER_EVENT_SUBS.get(user, []))
    for q in queues:
        try:
            q.put_nowait(payload)
        except queue.Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(payload)
            except Exception:
                pass


def emit_all_event(payload: dict) -> None:
    payload = dict(payload or {})
    payload.setdefault("ts", time.time())
    with USER_EVENT_LOCK:
        all_queues = [q for qs in USER_EVENT_SUBS.values() for q in qs]
    for q in all_queues:
        try:
            q.put_nowait(payload)
        except queue.Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(payload)
            except Exception:
                pass
