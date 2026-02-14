from __future__ import annotations

import threading
import time

from aslite.repositories import ReadingListRepository, TagRepository, readinglist_key


def test_concurrent_readinglist_updates_no_lost_update(monkeypatch):
    """Ensure reading list updates don't lose fields under concurrency.

    This test forces an interleaving where one worker reads the old value and
    is paused before write, while the other worker completes a different field
    update. Without a surrounding transaction, the last writer would overwrite
    the entire item (lost update).
    """

    user = "concurrent_user"
    pid = "2301.00001"
    rl_key = readinglist_key(user, pid)

    ReadingListRepository.add_to_reading_list(
        user,
        pid,
        {
            "pid": pid,
            "summary_status": "",
            "top_tags": [],
        },
    )

    import aslite.db as db_mod

    original_get = db_mod.SqliteKV.get
    a_has_read = threading.Event()
    allow_a_continue = threading.Event()

    def patched_get(self, key: str, default=None):
        value = original_get(self, key, default)
        if key == rl_key and threading.current_thread().name == "A" and not a_has_read.is_set():
            a_has_read.set()
            # Pause worker A after read, before write.
            allow_a_continue.wait(timeout=5)
        return value

    monkeypatch.setattr(db_mod.SqliteKV, "get", patched_get)

    errors: list[BaseException] = []

    def worker_a():
        try:
            ReadingListRepository.update_reading_list_item(user, pid, {"summary_status": "ok"})
        except BaseException as exc:
            errors.append(exc)

    def worker_b():
        try:
            # Cross-table update: tags + readinglist.
            TagRepository.set_tag_label(user, pid, "alpha", 1)
            ReadingListRepository.update_reading_list_item(user, pid, {"top_tags": ["alpha"]})
        except BaseException as exc:
            errors.append(exc)

    ta = threading.Thread(name="A", target=worker_a)
    tb = threading.Thread(name="B", target=worker_b)

    ta.start()
    assert a_has_read.wait(timeout=5), "worker A did not reach the read pause point in time"

    tb.start()
    # In the buggy implementation, worker B finishes while A is paused, then A overwrites.
    time.sleep(0.05)
    allow_a_continue.set()

    ta.join(timeout=5)
    tb.join(timeout=5)

    assert not errors, f"unexpected errors: {errors!r}"

    item = ReadingListRepository.get_reading_list_item(user, pid) or {}
    assert item.get("summary_status") == "ok"
    assert item.get("top_tags") == ["alpha"]

    tags = TagRepository.get_user_tags(user)
    assert pid in (tags.get("alpha") or set())

