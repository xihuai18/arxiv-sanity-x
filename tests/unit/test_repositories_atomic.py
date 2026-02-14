"""Ensure cross-table repository transactions remain atomic."""

from __future__ import annotations

import pytest

from aslite import repositories
from aslite.db import get_tags_db


def test_tag_label_transaction_rolls_back_on_failure(monkeypatch):
    """`set_tag_label` should not partially persist writes when a nested write fails."""
    user = "atomic_user"
    tag = "atomic_tag"
    pid = "2301.00001"

    with get_tags_db(flag="c") as tags_db:
        tags_db[user] = {tag: {pid}}

    with get_tags_db(flag="c") as tags_db:
        repositories._ensure_kv_table(tags_db.conn, "neg_tags")
        initial_tags = repositories._kv_get(tags_db.conn, "tags", user, tags_db._decode)
        initial_neg = repositories._kv_get(tags_db.conn, "neg_tags", user, tags_db._decode) or {}

    call_state = {"count": 0}
    original_kv_set = repositories._kv_set

    def failing_kv_set(conn, tablename, key, value, encode_fn):
        original_kv_set(conn, tablename, key, value, encode_fn)
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(repositories, "_kv_set", failing_kv_set)

    with pytest.raises(RuntimeError):
        repositories.TagRepository.set_tag_label(user, pid, tag, label=-1)

    with get_tags_db(flag="c") as tags_db:
        repositories._ensure_kv_table(tags_db.conn, "neg_tags")
        after_tags = repositories._kv_get(tags_db.conn, "tags", user, tags_db._decode)
        after_neg = repositories._kv_get(tags_db.conn, "neg_tags", user, tags_db._decode) or {}
        if user in tags_db:
            del tags_db[user]

    assert after_tags == initial_tags
    assert after_neg == initial_neg
