"""Unit tests for SqliteKV table name whitelist validation."""

from __future__ import annotations

import tempfile

import pytest


class TestSqliteKVTableNameValidation:
    """Tests for _VALID_TABLENAME_RE enforcement in SqliteKV.__init__."""

    def test_sqlitekv_rejects_sql_injection_tablename(self):
        """SQL injection attempt in table name raises ValueError."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            with pytest.raises(ValueError, match="Invalid table name"):
                SqliteKV(tmp.name, "users; DROP TABLE users", flag="c")

    def test_sqlitekv_rejects_hyphen_tablename(self):
        """Hyphenated table name raises ValueError."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            with pytest.raises(ValueError, match="Invalid table name"):
                SqliteKV(tmp.name, "table-name", flag="c")

    def test_sqlitekv_rejects_dot_tablename(self):
        """Dotted table name raises ValueError."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            with pytest.raises(ValueError, match="Invalid table name"):
                SqliteKV(tmp.name, "table.name", flag="c")

    def test_sqlitekv_rejects_space_tablename(self):
        """Table name with spaces raises ValueError."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            with pytest.raises(ValueError, match="Invalid table name"):
                SqliteKV(tmp.name, "my table", flag="c")

    def test_sqlitekv_rejects_empty_tablename(self):
        """Empty table name raises ValueError."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            with pytest.raises(ValueError, match="Invalid table name"):
                SqliteKV(tmp.name, "", flag="c")

    def test_sqlitekv_accepts_simple_name(self):
        """Simple alphanumeric table name is accepted."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            kv = SqliteKV(tmp.name, "users", flag="c")
            kv.close()

    def test_sqlitekv_accepts_underscore_prefix(self):
        """Table name starting with underscore is accepted."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            kv = SqliteKV(tmp.name, "_private", flag="c")
            kv.close()

    def test_sqlitekv_accepts_name_with_digits(self):
        """Table name with trailing digits is accepted."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            kv = SqliteKV(tmp.name, "table_123", flag="c")
            kv.close()

    def test_sqlitekv_rejects_digit_start(self):
        """Table name starting with a digit raises ValueError."""
        from aslite.db import SqliteKV

        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            with pytest.raises(ValueError, match="Invalid table name"):
                SqliteKV(tmp.name, "123table", flag="c")
