"""Regression tests for read-modify-write transaction guards.

Covers:
- Fix #1: ReadingListRepository.update_reading_list_item must not resurrect deleted entries
- Fix #2: KeywordRepository methods use IMMEDIATE transactions
- Fix #3: CombinedTagRepository methods use IMMEDIATE transactions
- Fix #4: SummaryStatusRepository.bump_generation_epoch uses IMMEDIATE transaction
"""

from __future__ import annotations

import threading

from aslite.db import (
    get_combined_tags_db,
    get_keywords_db,
    get_metas_db,
    get_neg_tags_db,
    get_readinglist_db,
    get_readinglist_index_db,
    get_summary_status_db,
    get_tags_db,
)
from aslite.repositories import (
    CombinedTagRepository,
    KeywordRepository,
    NegativeTagRepository,
    PaperTombstoneRepository,
    ReadingListRepository,
    SummaryStatusRepository,
    TagRepository,
)

# ---------------------------------------------------------------------------
# Fix #1 — update_reading_list_item must not resurrect deleted entries
# ---------------------------------------------------------------------------


class TestReadingListNoResurrection:
    """Verify that updating a non-existent reading list item is a no-op."""

    def test_update_on_absent_key_is_noop(self):
        """update_reading_list_item should NOT create a new entry for a key
        that was never added (or was already deleted)."""
        user = "ghost_user_noop"
        pid = "9999.99999"

        # Ensure the key does not exist.
        from aslite.repositories import readinglist_key

        rl_key = readinglist_key(user, pid)
        with get_readinglist_db(flag="c") as rldb:
            if rl_key in rldb:
                del rldb[rl_key]

        # Attempt to update — should be silently ignored.
        ReadingListRepository.update_reading_list_item(user, pid, {"summary_status": "done"})

        with get_readinglist_db() as rldb:
            assert rldb.get(rl_key) is None, "Deleted entry was resurrected!"

    def test_update_after_delete_does_not_recreate(self):
        """Simulate the delete-then-late-update race: add, delete, update.
        The final state must remain deleted."""
        user = "ghost_user_race"
        pid = "8888.88888"

        from aslite.repositories import readinglist_key

        rl_key = readinglist_key(user, pid)

        # Step 1: add the item.
        with get_readinglist_db(flag="c") as rldb:
            rldb[rl_key] = {"pid": pid, "user": user}

        # Step 2: delete it.
        with get_readinglist_db(flag="c") as rldb:
            del rldb[rl_key]

        # Step 3: late background update arrives.
        ReadingListRepository.update_reading_list_item(user, pid, {"summary_status": "done"})

        with get_readinglist_db() as rldb:
            assert rldb.get(rl_key) is None, "Deleted entry was resurrected by late update!"

    def test_update_existing_item_still_works(self):
        """Normal path: updating an existing item should still apply."""
        user = "normal_user_update"
        pid = "7777.77777"

        from aslite.repositories import readinglist_key

        rl_key = readinglist_key(user, pid)

        with get_readinglist_db(flag="c") as rldb:
            rldb[rl_key] = {"pid": pid, "user": user}

        ReadingListRepository.update_reading_list_item(user, pid, {"summary_status": "done"})

        with get_readinglist_db() as rldb:
            item = rldb.get(rl_key)
        assert item is not None
        assert item["summary_status"] == "done"
        assert item["pid"] == pid

    def test_real_remove_path_blocks_late_update_and_cleans_index(self):
        """Deleting through the repository should stay deleted after a late update attempt."""
        user = "ghost_user_real_delete"
        pid = "6666.66666"

        from aslite.repositories import readinglist_key

        rl_key = readinglist_key(user, pid)
        ReadingListRepository.add_to_reading_list(user, pid, {"pid": pid, "user": user})

        removed = ReadingListRepository.remove_from_reading_list(user, pid)
        assert removed is True

        ReadingListRepository.update_reading_list_item(user, pid, {"summary_status": "done"})

        with get_readinglist_db() as rldb:
            assert rldb.get(rl_key) is None
        with get_readinglist_index_db() as idx_db:
            assert pid not in (idx_db.get(user, []) or [])


# ---------------------------------------------------------------------------
# Fix #2 — KeywordRepository transaction coverage
# ---------------------------------------------------------------------------


class TestKeywordRepositoryTransactions:
    """Verify KeywordRepository write methods use transactions."""

    def test_add_keyword_concurrent_no_lost_update(self):
        """Two threads adding different keywords should not lose either one."""
        user = "kw_txn_user"
        # Ensure clean state.
        with get_keywords_db(flag="c") as kdb:
            if user in kdb:
                del kdb[user]

        barrier = threading.Barrier(2)
        errors = []

        def add_kw(keyword):
            try:
                barrier.wait(timeout=5)
                KeywordRepository.add_keyword(user, keyword)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=add_kw, args=("kw_alpha",))
        t2 = threading.Thread(target=add_kw, args=("kw_beta",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with get_keywords_db() as kdb:
            keywords = kdb.get(user, {}) or {}
        assert "kw_alpha" in keywords, "kw_alpha lost!"
        assert "kw_beta" in keywords, "kw_beta lost!"

    def test_remove_keyword_uses_transaction(self):
        """remove_keyword should work correctly and not corrupt state."""
        user = "kw_rm_user"
        with get_keywords_db(flag="c") as kdb:
            kdb[user] = {"a": set(), "b": {"p1"}}

        result = KeywordRepository.remove_keyword(user, "a")
        assert result is True

        with get_keywords_db() as kdb:
            keywords = kdb.get(user, {}) or {}
        assert "a" not in keywords
        assert "b" in keywords

    def test_rename_keyword_uses_transaction(self):
        """rename_keyword should atomically move the keyword."""
        user = "kw_rename_user"
        with get_keywords_db(flag="c") as kdb:
            kdb[user] = {"old_kw": {"p1", "p2"}}

        result = KeywordRepository.rename_keyword(user, "old_kw", "new_kw")
        assert result == "ok"

        with get_keywords_db() as kdb:
            keywords = kdb.get(user, {}) or {}
        assert "old_kw" not in keywords
        assert "new_kw" in keywords
        assert keywords["new_kw"] == {"p1", "p2"}

    def test_add_paper_to_keyword_concurrent_same_keyword_keeps_both_pids(self, monkeypatch):
        """Forced interleaving should not lose one pid when both threads update the same keyword."""
        import aslite.db as db_mod

        user = "kw_pid_txn_user"
        keyword = "shared_kw"
        with get_keywords_db(flag="c") as kdb:
            kdb[user] = {keyword: set()}

        original_get = db_mod.SqliteKV.get
        a_has_read = threading.Event()
        allow_a_continue = threading.Event()

        def patched_get(self, key: str, default=None):
            value = original_get(self, key, default)
            if key == user and threading.current_thread().name == "A" and not a_has_read.is_set():
                a_has_read.set()
                allow_a_continue.wait(timeout=5)
            return value

        monkeypatch.setattr(db_mod.SqliteKV, "get", patched_get)

        errors = []

        def add_pid(pid):
            try:
                KeywordRepository.add_paper_to_keyword(user, keyword, pid)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(name="A", target=add_pid, args=("p_alpha",))
        t2 = threading.Thread(name="B", target=add_pid, args=("p_beta",))
        t1.start()
        assert a_has_read.wait(timeout=5), "worker A did not reach the read pause point in time"
        t2.start()
        allow_a_continue.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with get_keywords_db() as kdb:
            keywords = kdb.get(user, {}) or {}
        assert keywords.get(keyword) == {"p_alpha", "p_beta"}


# ---------------------------------------------------------------------------
# Fix #3 — CombinedTagRepository transaction coverage
# ---------------------------------------------------------------------------


class TestCombinedTagRepositoryTransactions:
    """Verify CombinedTagRepository write methods use transactions."""

    def test_add_combined_tag_concurrent_no_lost_update(self):
        """Two threads adding different combined tags should not lose either."""
        user = "ct_txn_user"
        with get_combined_tags_db(flag="c") as ctdb:
            if user in ctdb:
                del ctdb[user]

        barrier = threading.Barrier(2)
        errors = []

        def add_ct(tag):
            try:
                barrier.wait(timeout=5)
                CombinedTagRepository.add_combined_tag(user, tag)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=add_ct, args=("RL,NLP",))
        t2 = threading.Thread(target=add_ct, args=("CV,GAN",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with get_combined_tags_db() as ctdb:
            ctags = ctdb.get(user, set()) or set()
        assert "RL,NLP" in ctags, "RL,NLP lost!"
        assert "CV,GAN" in ctags, "CV,GAN lost!"

    def test_remove_combined_tag_uses_transaction(self):
        """remove_combined_tag should work correctly."""
        user = "ct_rm_user"
        with get_combined_tags_db(flag="c") as ctdb:
            ctdb[user] = {"RL,NLP", "CV,GAN"}

        result = CombinedTagRepository.remove_combined_tag(user, "RL,NLP")
        assert result is True

        with get_combined_tags_db() as ctdb:
            ctags = ctdb.get(user, set()) or set()
        assert "RL,NLP" not in ctags
        assert "CV,GAN" in ctags

    def test_rename_combined_tag_uses_transaction(self):
        """rename_combined_tag should atomically replace the tag."""
        user = "ct_rename_user"
        with get_combined_tags_db(flag="c") as ctdb:
            ctdb[user] = {"old,tag"}

        result = CombinedTagRepository.rename_combined_tag(user, "old,tag", "new,tag")
        assert result is True

        with get_combined_tags_db() as ctdb:
            ctags = ctdb.get(user, set()) or set()
        assert "old,tag" not in ctags
        assert "new,tag" in ctags


class TestTagRepositoryVerboseTransactions:
    """Verify legacy verbose tag removal keeps transaction safety."""

    def test_remove_paper_from_tag_verbose_concurrent_same_tag_keeps_empty(self, monkeypatch):
        """Two concurrent removals from the same tag should not resurrect either pid."""
        import aslite.db as db_mod

        user = "tag_verbose_txn_user"
        tag = "shared-tag"
        with get_tags_db(flag="c") as tdb:
            tdb[user] = {tag: {"p_alpha", "p_beta"}}

        original_getitem = db_mod.SqliteKV.__getitem__
        a_has_read = threading.Event()
        allow_a_continue = threading.Event()

        def patched_getitem(self, key):
            value = original_getitem(self, key)
            if key == user and threading.current_thread().name == "A" and not a_has_read.is_set():
                a_has_read.set()
                allow_a_continue.wait(timeout=5)
            return value

        monkeypatch.setattr(db_mod.SqliteKV, "__getitem__", patched_getitem)

        errors = []

        def remove_pid(pid):
            try:
                result = TagRepository.remove_paper_from_tag_verbose(user, pid, tag)
                assert result == "ok"
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(name="A", target=remove_pid, args=("p_alpha",))
        t2 = threading.Thread(name="B", target=remove_pid, args=("p_beta",))
        t1.start()
        assert a_has_read.wait(timeout=5), "worker A did not reach the read pause point in time"
        t2.start()
        allow_a_continue.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with get_tags_db() as tdb:
            tags = tdb.get(user, {}) or {}
        assert tag not in tags


# ---------------------------------------------------------------------------
# Fix #3b — NegativeTagRepository transaction coverage
# ---------------------------------------------------------------------------


class TestNegativeTagRepositoryTransactions:
    """Verify NegativeTagRepository write methods use transactions."""

    def test_add_paper_to_neg_tag_concurrent_same_tag_keeps_both_pids(self, monkeypatch):
        """Forced interleaving should not lose one pid when both threads update the same neg tag."""
        import aslite.db as db_mod

        user = "neg_txn_user"
        tag = "hard-negative"
        with get_neg_tags_db(flag="c") as ntdb:
            ntdb[user] = {tag: set()}

        original_get = db_mod.SqliteKV.get
        a_has_read = threading.Event()
        allow_a_continue = threading.Event()

        def patched_get(self, key: str, default=None):
            value = original_get(self, key, default)
            if key == user and threading.current_thread().name == "A" and not a_has_read.is_set():
                a_has_read.set()
                allow_a_continue.wait(timeout=5)
            return value

        monkeypatch.setattr(db_mod.SqliteKV, "get", patched_get)

        errors = []

        def add_pid(pid):
            try:
                NegativeTagRepository.add_paper_to_neg_tag(user, pid, tag)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(name="A", target=add_pid, args=("p_alpha",))
        t2 = threading.Thread(name="B", target=add_pid, args=("p_beta",))
        t1.start()
        assert a_has_read.wait(timeout=5), "worker A did not reach the read pause point in time"
        t2.start()
        allow_a_continue.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with get_neg_tags_db() as ntdb:
            neg_tags = ntdb.get(user, {}) or {}
        assert neg_tags.get(tag) == {"p_alpha", "p_beta"}

    def test_remove_paper_from_neg_tag_uses_transaction(self):
        """remove_paper_from_neg_tag should remove only the requested pid."""
        user = "neg_rm_user"
        tag = "hard-negative"
        with get_neg_tags_db(flag="c") as ntdb:
            ntdb[user] = {tag: {"p1", "p2"}}

        result = NegativeTagRepository.remove_paper_from_neg_tag(user, "p1", tag)
        assert result is True

        with get_neg_tags_db() as ntdb:
            neg_tags = ntdb.get(user, {}) or {}
        assert neg_tags.get(tag) == {"p2"}


class TestUserPidCleanup:
    def test_get_user_tags_cleans_tombstoned_pids_but_keeps_empty_tag(self):
        user = "tag_cleanup_user"
        live_pid = "3401.00001"
        dead_pid = "3401.00002"
        upload_pid = "up_keepme001"

        with get_metas_db(flag="c") as mdb:
            mdb[live_pid] = {"_time": 1.0}
        PaperTombstoneRepository.save(dead_pid, {"pid": dead_pid, "reason": "withdrawn_only"})
        with get_tags_db(flag="c") as tdb:
            tdb[user] = {
                "topic": {live_pid, dead_pid, upload_pid},
                "keep_empty": {dead_pid},
            }

        cleaned = TagRepository.get_user_tags(user)

        assert cleaned["topic"] == {live_pid, upload_pid}
        assert cleaned["keep_empty"] == set()

        with get_tags_db() as tdb:
            persisted = tdb.get(user, {}) or {}
        assert persisted["topic"] == {live_pid, upload_pid}
        assert persisted["keep_empty"] == set()

    def test_get_user_neg_tags_cleans_tombstoned_pids_and_drops_empty_bucket(self):
        user = "neg_cleanup_user"
        live_pid = "3501.00001"
        dead_pid = "3501.00002"

        with get_metas_db(flag="c") as mdb:
            mdb[live_pid] = {"_time": 1.0}
        PaperTombstoneRepository.save(dead_pid, {"pid": dead_pid, "reason": "withdrawn_only"})
        with get_neg_tags_db(flag="c") as ntdb:
            ntdb[user] = {"topic": {live_pid, dead_pid}, "drop_me": {dead_pid}}

        cleaned = NegativeTagRepository.get_user_neg_tags(user)

        assert cleaned["topic"] == {live_pid}
        assert "drop_me" not in cleaned

        with get_neg_tags_db() as ntdb:
            persisted = ntdb.get(user, {}) or {}
        assert persisted["topic"] == {live_pid}
        assert "drop_me" not in persisted

    def test_get_user_keywords_cleans_tombstoned_pids_but_keeps_keyword(self):
        user = "kw_cleanup_user"
        live_pid = "3301.00001"
        dead_pid = "3301.00002"

        with get_metas_db(flag="c") as mdb:
            mdb[live_pid] = {"_time": 1.0}
        PaperTombstoneRepository.save(dead_pid, {"pid": dead_pid, "reason": "withdrawn_only"})
        with get_keywords_db(flag="c") as kdb:
            kdb[user] = {"topic": {live_pid, dead_pid}, "empty_ok": {dead_pid}}

        cleaned = KeywordRepository.get_user_keywords(user)

        assert cleaned["topic"] == {live_pid}
        assert cleaned["empty_ok"] == set()

        with get_keywords_db() as kdb:
            persisted = kdb.get(user, {}) or {}
        assert persisted["topic"] == {live_pid}
        assert persisted["empty_ok"] == set()


# ---------------------------------------------------------------------------
# Fix #4 — bump_generation_epoch transaction coverage
# ---------------------------------------------------------------------------


class TestBumpGenerationEpochTransaction:
    """Verify bump_generation_epoch uses a transaction to avoid lost increments."""

    def test_sequential_bumps(self):
        """Sequential bumps should monotonically increase."""
        pid, model = "epoch_seq_pid", "gpt-4"
        # Reset.
        from aslite.repositories import summary_generation_epoch_key

        key = summary_generation_epoch_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            if key in sdb:
                del sdb[key]

        v1 = SummaryStatusRepository.bump_generation_epoch(pid, model)
        v2 = SummaryStatusRepository.bump_generation_epoch(pid, model)
        v3 = SummaryStatusRepository.bump_generation_epoch(pid, model)

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_concurrent_bumps_no_lost_increment(self):
        """Concurrent bumps should not lose increments."""
        pid, model = "epoch_conc_pid", "gpt-4"
        from aslite.repositories import summary_generation_epoch_key

        key = summary_generation_epoch_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            if key in sdb:
                del sdb[key]

        n_threads = 10
        barrier = threading.Barrier(n_threads)
        errors = []

        def bump():
            try:
                barrier.wait(timeout=5)
                SummaryStatusRepository.bump_generation_epoch(pid, model)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=bump) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Unexpected errors: {errors}"

        final = SummaryStatusRepository.get_generation_epoch(pid, model)
        assert final == n_threads, f"Expected epoch={n_threads} after {n_threads} concurrent bumps, got {final}"


class TestSummaryStatusUpdateTransactions:
    """Verify summary status partial updates do not lose fields under concurrency."""

    def test_update_status_concurrent_field_merges(self, monkeypatch):
        import aslite.db as db_mod

        pid, model = "summary_update_pid", "gpt-4"
        from aslite.repositories import summary_status_key

        key = summary_status_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            sdb[key] = {"status": "queued", "task_id": ""}

        original_get = db_mod.SqliteKV.get
        a_has_read = threading.Event()
        allow_a_continue = threading.Event()

        def patched_get(self, lookup_key: str, default=None):
            value = original_get(self, lookup_key, default)
            if lookup_key == key and threading.current_thread().name == "A" and not a_has_read.is_set():
                a_has_read.set()
                allow_a_continue.wait(timeout=5)
            return value

        monkeypatch.setattr(db_mod.SqliteKV, "get", patched_get)

        errors = []

        def update_fields(updates):
            try:
                SummaryStatusRepository.update_status(pid, model, updates)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(name="A", target=update_fields, args=({"status": "running"},))
        t2 = threading.Thread(name="B", target=update_fields, args=({"task_id": "task-1"},))
        t1.start()
        assert a_has_read.wait(timeout=5), "worker A did not reach the read pause point in time"
        t2.start()
        allow_a_continue.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with get_summary_status_db() as sdb:
            status = sdb.get(key, {}) or {}
        assert status.get("status") == "running"
        assert status.get("task_id") == "task-1"
