"""Unit tests for cache_only summary fetch behavior (non-blocking)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


def test_cache_only_does_not_try_to_acquire_lock(monkeypatch, tmp_path):
    """When cache_only=True, we should not call acquire_summary_lock (which can block)."""
    import backend.services.summary_service as ss

    lock_file = tmp_path / ".p1.lock"
    lock_file.write_text(f"{os.getpid()}\n{time.time()}\n", encoding="utf-8")

    cache_file = tmp_path / "p1.md"
    meta_file = tmp_path / "p1.meta.json"

    monkeypatch.setattr(
        ss,
        "summary_cache_paths",
        lambda _pid, _model: (cache_file, meta_file, lock_file, cache_file, meta_file, lock_file),
    )

    def _should_not_be_called(*_a, **_kw):
        raise AssertionError("acquire_summary_lock should not be called for cache_only requests")

    monkeypatch.setattr(ss, "acquire_summary_lock", _should_not_be_called)

    content, meta = ss.generate_paper_summary(
        "p1",
        model="m",
        cache_only=True,
        metas_getter=lambda: {},
        paper_exists_fn=lambda _pid: True,
    )
    assert "Summary is being generated" in content
    assert meta == {}


def test_cache_only_stale_lock_is_ignored(monkeypatch, tmp_path):
    """A stale lock should not block cache-only requests or force 'generating'."""
    import backend.services.summary_service as ss

    lock_file = tmp_path / ".p1.lock"
    lock_file.write_text("999999\n0\n", encoding="utf-8")

    # Make it stale.
    os.utime(lock_file, (0, 0))

    cache_file = tmp_path / "p1.md"
    meta_file = tmp_path / "p1.meta.json"

    monkeypatch.setattr(
        ss,
        "summary_cache_paths",
        lambda _pid, _model: (cache_file, meta_file, lock_file, cache_file, meta_file, lock_file),
    )

    def _should_not_be_called(*_a, **_kw):
        raise AssertionError("acquire_summary_lock should not be called for cache_only requests")

    monkeypatch.setattr(ss, "acquire_summary_lock", _should_not_be_called)

    # Force a small stale window so the lock is considered stale deterministically.
    monkeypatch.setattr(ss.settings.lock, "summary_lock_stale_sec", 0.001)

    with pytest.raises(ss.SummaryCacheMiss):
        ss.generate_paper_summary(
            "p1",
            model="m",
            cache_only=True,
            metas_getter=lambda: {},
            paper_exists_fn=lambda _pid: True,
        )

    # Best-effort cleanup: stale lock should be removed.
    assert not Path(lock_file).exists()
