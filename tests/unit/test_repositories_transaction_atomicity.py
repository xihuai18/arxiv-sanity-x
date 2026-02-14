"""Repository transaction atomicity tests."""

from __future__ import annotations

import pytest


class TestTagRepositoryTransactions:
    def test_delete_tag_full_is_atomic_across_tables(self, monkeypatch):
        import aslite.repositories as repos
        from aslite.db import get_combined_tags_db, get_neg_tags_db, get_tags_db
        from aslite.repositories import TagRepository

        user = "u"

        with get_tags_db(flag="c") as tdb:
            tdb[user] = {"t1": {"p1", "p2"}, "t2": {"p3"}}
        with get_neg_tags_db(flag="c") as ntdb:
            ntdb[user] = {"t1": {"p9"}}
        with get_combined_tags_db(flag="c") as cdb:
            cdb[user] = {"t1,t2"}

        original_kv_set = repos._kv_set

        def boom(conn, tablename: str, key: str, value, encode_fn):
            if tablename == "combined_tags":
                raise RuntimeError("boom")
            return original_kv_set(conn, tablename, key, value, encode_fn)

        monkeypatch.setattr(repos, "_kv_set", boom)

        with pytest.raises(RuntimeError):
            TagRepository.delete_tag_full(user, "t1")

        # Verify rollback: nothing should change.
        with get_tags_db() as tdb:
            tags = tdb.get(user, {})
        with get_neg_tags_db() as ntdb:
            neg_tags = ntdb.get(user, {})
        with get_combined_tags_db() as cdb:
            combined = cdb.get(user, set())

        assert "t1" in tags
        assert tags["t1"] == {"p1", "p2"}
        assert neg_tags.get("t1") == {"p9"}
        assert "t1,t2" in combined

    def test_rename_tag_full_is_atomic_across_tables(self, monkeypatch):
        import aslite.repositories as repos
        from aslite.db import get_combined_tags_db, get_neg_tags_db, get_tags_db
        from aslite.repositories import TagRepository

        user = "u2"

        with get_tags_db(flag="c") as tdb:
            tdb[user] = {"old": {"p1"}}
        with get_neg_tags_db(flag="c") as ntdb:
            ntdb[user] = {"old": {"p2"}}
        with get_combined_tags_db(flag="c") as cdb:
            cdb[user] = {"old,x"}

        original_kv_set = repos._kv_set

        def boom(conn, tablename: str, key: str, value, encode_fn):
            if tablename == "neg_tags":
                raise RuntimeError("boom")
            return original_kv_set(conn, tablename, key, value, encode_fn)

        monkeypatch.setattr(repos, "_kv_set", boom)

        with pytest.raises(RuntimeError):
            TagRepository.rename_tag_full(user, "old", "new")

        with get_tags_db() as tdb:
            tags = tdb.get(user, {})
        with get_neg_tags_db() as ntdb:
            neg_tags = ntdb.get(user, {})
        with get_combined_tags_db() as cdb:
            combined = cdb.get(user, set())

        assert "old" in tags
        assert "new" not in tags
        assert neg_tags.get("old") == {"p2"}
        assert "old,x" in combined
