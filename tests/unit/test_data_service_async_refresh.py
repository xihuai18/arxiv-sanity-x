"""Unit tests for non-blocking cache refresh in data_service."""

from __future__ import annotations

import os


def test_get_data_cached_serves_stale_cache_and_schedules_refresh(monkeypatch, tmp_path):
    """Warm caches should not trigger a full reload on the request thread when DB mtime advances."""
    import backend.services.data_service as ds

    db_path = tmp_path / "papers.db"
    db_path.write_text("placeholder")

    # Point data_service to the temp db path (only existence matters for this test).
    monkeypatch.setattr(ds, "PAPERS_DB_FILE", str(db_path))

    # Force paper caching in memory and seed warm caches.
    monkeypatch.setattr(ds, "_cache_papers_in_memory", lambda: True)
    monkeypatch.setattr(ds, "_PAPERS_CACHE", {"p1": {"_id": "p1"}})
    monkeypatch.setattr(ds, "_METAS_CACHE", {"p1": {"_id": "p1"}})
    monkeypatch.setattr(ds, "_PIDS_CACHE", ["p1"])
    monkeypatch.setattr(ds, "_PAPERS_DB_FILE_MTIME", 1.0)

    # Make DB appear changed.
    monkeypatch.setattr(ds, "_sqlite_effective_mtime", lambda _path: 2.0)

    scheduled = {"n": 0}

    def _fake_schedule(_mtime: float):
        scheduled["n"] += 1

    monkeypatch.setattr(ds, "_schedule_data_refresh", _fake_schedule)

    data = ds.get_data_cached()
    assert data["metas"] == {"p1": {"_id": "p1"}}
    assert data["pids"] == ["p1"]
    assert data["papers"] == {"p1": {"_id": "p1"}}
    assert scheduled["n"] == 1


def test_get_features_cached_serves_stale_cache_and_schedules_refresh(monkeypatch, tmp_path):
    """Warm features cache should not block on file changes."""
    import backend.services.data_service as ds

    features_path = tmp_path / "features.p"
    features_path.write_bytes(b"placeholder")
    os.utime(features_path, (2, 2))

    monkeypatch.setattr(ds, "FEATURES_FILE", str(features_path))
    monkeypatch.setattr(ds, "FEATURES_FILE_NEW", str(tmp_path / "features_new.p"))

    # Seed warm cache with an older mtime.
    monkeypatch.setattr(ds, "_FEATURES_CACHE", {"pids": []})
    monkeypatch.setattr(ds, "_FEATURES_FILE_MTIME", 1.0)

    scheduled = {"n": 0}

    def _fake_schedule(_mtime: float):
        scheduled["n"] += 1

    monkeypatch.setattr(ds, "_schedule_features_refresh", _fake_schedule)

    # load_features should not be called on warm cache refresh path.
    monkeypatch.setattr(ds, "load_features", lambda: (_ for _ in ()).throw(RuntimeError("should not load")))

    out = ds.get_features_cached()
    assert out == {"pids": []}
    assert scheduled["n"] == 1
