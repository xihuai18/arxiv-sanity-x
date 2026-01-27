"""Unit tests for SqliteKV prefix iteration.

This ensures prefix scanning treats SQL LIKE wildcards (% and _) as literals.
"""

from __future__ import annotations

import os
import tempfile


def test_items_with_prefix_escapes_like_wildcards():
    from aslite.db import SqliteKV

    fd, db_path = tempfile.mkstemp(prefix="arxiv_sanity_sqlitekv_", suffix=".db")
    os.close(fd)
    try:
        with SqliteKV(db_path, "t", flag="c") as kv:
            kv["up_A_B::m"] = 1
            kv["up_AxB::m"] = 2  # would match with '_' wildcard if not escaped
            kv["up_A_B::n"] = 3

        with SqliteKV(db_path, "t", flag="r") as kv:
            got_keys = [k for k, _v in kv.items_with_prefix("up_A_B::")]
            assert sorted(got_keys) == ["up_A_B::m", "up_A_B::n"]
    finally:
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_keys_with_prefix_escapes_percent():
    from aslite.db import SqliteKV

    fd, db_path = tempfile.mkstemp(prefix="arxiv_sanity_sqlitekv_", suffix=".db")
    os.close(fd)
    try:
        with SqliteKV(db_path, "t", flag="c") as kv:
            kv["p%id::m"] = 1
            kv["pXid::m"] = 2  # would match with '%' wildcard if not escaped
            kv["p%id::n"] = 3

        with SqliteKV(db_path, "t", flag="r") as kv:
            got_keys = sorted(list(kv.keys_with_prefix("p%id::")))
            assert got_keys == ["p%id::m", "p%id::n"]
    finally:
        try:
            os.remove(db_path)
        except Exception:
            pass
