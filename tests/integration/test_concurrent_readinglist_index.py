from __future__ import annotations

import threading

import aslite.repositories as repos
from aslite.repositories import ReadingListRepository, readinglist_key


def test_add_to_readinglist_holds_write_lock_until_index_updated(monkeypatch):
    """Ensure add_to_reading_list serializes writers before reading index.

    The correctness property we need is: while one writer is in the middle of
    adding an item (including index update), another writer must not read and
    write back a stale index snapshot (lost update).
    """

    user = "lock_user"
    pid_a = "2301.00001"
    pid_b = "2301.00002"

    a_in_index_set = threading.Event()
    allow_a_continue = threading.Event()
    b_reached_index_get = threading.Event()
    errors: list[BaseException] = []

    orig_kv_get = repos._kv_get
    orig_kv_set = repos._kv_set

    def patched_kv_get(conn, tablename: str, key: str, decode_fn):
        if tablename == "readinglist_index" and key == user and threading.current_thread().name == "B":
            b_reached_index_get.set()
        return orig_kv_get(conn, tablename, key, decode_fn)

    def patched_kv_set(conn, tablename: str, key: str, value, encode_fn) -> None:
        if tablename == "readinglist_index" and key == user and threading.current_thread().name == "A":
            a_in_index_set.set()
            # Hold the transaction open: B should not be able to even read index yet.
            allow_a_continue.wait(timeout=5)
        return orig_kv_set(conn, tablename, key, value, encode_fn)

    monkeypatch.setattr(repos, "_kv_get", patched_kv_get)
    monkeypatch.setattr(repos, "_kv_set", patched_kv_set)

    def worker_a():
        try:
            ReadingListRepository.add_to_reading_list(user, pid_a, {"pid": pid_a})
        except BaseException as exc:
            errors.append(exc)

    def worker_b():
        try:
            ReadingListRepository.add_to_reading_list(user, pid_b, {"pid": pid_b})
        except BaseException as exc:
            errors.append(exc)

    ta = threading.Thread(name="A", target=worker_a)
    tb = threading.Thread(name="B", target=worker_b)

    ta.start()
    assert a_in_index_set.wait(timeout=5), "worker A did not reach index update in time"

    tb.start()
    # While A holds the IMMEDIATE transaction, B should not reach index read.
    assert not b_reached_index_get.wait(timeout=0.15)

    allow_a_continue.set()
    ta.join(timeout=5)
    tb.join(timeout=5)

    assert not errors, f"unexpected errors: {errors!r}"

    items = ReadingListRepository.get_user_reading_list(user)
    assert pid_a in items
    assert pid_b in items

