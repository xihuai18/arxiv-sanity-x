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


def test_get_data_cached_wait_false_is_non_blocking_on_cold_start(monkeypatch, tmp_path):
    """Cold-start peek should not sleep/block, but should trigger the cold loader."""
    import backend.services.data_service as ds

    db_path = tmp_path / "papers.db"
    db_path.write_text("placeholder")

    monkeypatch.setattr(ds, "PAPERS_DB_FILE", str(db_path))
    monkeypatch.setattr(ds, "_sqlite_effective_mtime", lambda _path: 1.0)
    monkeypatch.setattr(ds, "_cache_papers_in_memory", lambda: False)

    # Force cold start.
    monkeypatch.setattr(ds, "_PAPERS_CACHE", None)
    monkeypatch.setattr(ds, "_METAS_CACHE", None)
    monkeypatch.setattr(ds, "_PIDS_CACHE", None)
    monkeypatch.setattr(ds, "_PAPERS_DB_FILE_MTIME", 0.0)

    started = {"n": 0}

    def _fake_ensure(**_kwargs):
        started["n"] += 1

    monkeypatch.setattr(ds, "_ensure_cold_load_started", _fake_ensure)
    monkeypatch.setattr(ds.time, "sleep", lambda _s: (_ for _ in ()).throw(RuntimeError("should not sleep")))

    out = ds.get_data_cached(wait=False)
    assert out["pids"] == []
    assert out["metas"] == {}
    assert out["papers"] is None
    assert started["n"] == 1


def test_get_data_cached_max_wait_s_bounds_cold_start_wait(monkeypatch, tmp_path):
    """Cold-start wait should be bounded when max_wait_s is provided."""
    import backend.services.data_service as ds

    db_path = tmp_path / "papers.db"
    db_path.write_text("placeholder")

    monkeypatch.setattr(ds, "PAPERS_DB_FILE", str(db_path))
    monkeypatch.setattr(ds, "_sqlite_effective_mtime", lambda _path: 1.0)
    monkeypatch.setattr(ds, "_cache_papers_in_memory", lambda: False)

    # Force cold start that never becomes ready.
    monkeypatch.setattr(ds, "_PAPERS_CACHE", None)
    monkeypatch.setattr(ds, "_METAS_CACHE", None)
    monkeypatch.setattr(ds, "_PIDS_CACHE", None)
    monkeypatch.setattr(ds, "_PAPERS_DB_FILE_MTIME", 0.0)
    monkeypatch.setattr(ds, "_data_cache_ready", lambda **_kwargs: False)

    def _fake_ensure(**_kwargs):
        ds._DATA_COLD_LOAD_IN_PROGRESS = True
        ds._DATA_COLD_LOAD_LAST_ERROR = None

    monkeypatch.setattr(ds, "_ensure_cold_load_started", _fake_ensure)
    monkeypatch.setattr(ds.time, "sleep", lambda _s: None)

    t = {"now": 0.0}

    def _fake_time():
        t["now"] += 0.02
        return t["now"]

    monkeypatch.setattr(ds.time, "time", _fake_time)

    out = ds.get_data_cached(wait=True, max_wait_s=0.01)
    assert out["pids"] == []
    assert out["metas"] == {}
    assert out["papers"] is None


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
