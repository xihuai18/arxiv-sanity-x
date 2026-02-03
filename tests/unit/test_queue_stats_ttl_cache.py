"""Unit tests for queue_stats/task_status TTL caching."""

from __future__ import annotations


def test_queue_stats_and_task_status_share_prefix_scan_cache(monkeypatch, client):
    """Repeated calls should avoid repeated task:: full scans within TTL."""
    import backend.legacy as legacy

    calls = {"prefix_scan": 0, "all_items": 0}

    def fake_get_items_with_prefix(prefix: str, limit=None):
        assert prefix == "task::"
        calls["prefix_scan"] += 1
        return [
            ("task::1", {"status": "queued", "priority": 10, "updated_time": 1.0}),
            ("task::2", {"status": "queued", "priority": 1, "updated_time": 2.0}),
            ("task::3", {"status": "running", "priority": 10, "updated_time": 3.0}),
        ]

    def fake_get_all_items(limit=None):
        calls["all_items"] += 1
        return []

    def fake_get_task_status(task_id: str):
        if task_id == "1":
            return {"status": "queued", "priority": 10, "updated_time": 1.0, "user": "u1"}
        return None

    monkeypatch.setattr(legacy, "_QUEUE_STATS_CACHE", legacy._LRUCacheTTL(maxsize=16, ttl_s=60.0))
    monkeypatch.setattr(legacy, "SUMMARY_PRIORITY_HIGH", 5)
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_items_with_prefix", fake_get_items_with_prefix)
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_all_items", fake_get_all_items)
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_task_status", fake_get_task_status)

    # First call computes snapshot (one prefix scan).
    resp1 = client.get("/api/queue_stats")
    assert resp1.status_code == 200
    assert calls["prefix_scan"] == 1

    # Second call should hit TTL cache (no new scan).
    resp2 = client.get("/api/queue_stats")
    assert resp2.status_code == 200
    assert calls["prefix_scan"] == 1
    assert calls["all_items"] == 0

    # task_status should reuse the same cached snapshot for queue_rank computation.
    resp3 = client.get("/api/task_status/1")
    assert resp3.status_code == 200
    assert calls["prefix_scan"] == 1

    data3 = resp3.get_json(silent=True) or {}
    assert data3.get("success") is True
    assert data3.get("queue_rank") == 1
    assert data3.get("queue_total") == 1


def test_queue_stats_does_not_fullscan_when_task_entries_exist_but_queue_empty(monkeypatch, client):
    """When task:: entries exist but none are queued/running, do not fall back to full DB scans."""
    import backend.legacy as legacy

    calls = {"prefix_scan": 0, "all_items": 0}

    def fake_get_items_with_prefix(prefix: str, limit=None):
        assert prefix == "task::"
        calls["prefix_scan"] += 1
        return [
            ("task::a", {"status": "ok", "priority": 10, "updated_time": 1.0}),
            ("task::b", {"status": "failed", "priority": 10, "updated_time": 2.0}),
        ]

    def fake_get_all_items(limit=None):
        calls["all_items"] += 1
        return [("p1::m1", {"status": "ok"})]

    monkeypatch.setattr(legacy, "_QUEUE_STATS_CACHE", legacy._LRUCacheTTL(maxsize=16, ttl_s=60.0))
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_items_with_prefix", fake_get_items_with_prefix)
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_all_items", fake_get_all_items)

    resp = client.get("/api/queue_stats")
    assert resp.status_code == 200
    payload = resp.get_json(silent=True) or {}
    assert payload.get("success") is True
    assert payload.get("queued") == 0
    assert payload.get("running") == 0
    assert calls["prefix_scan"] == 1
    assert calls["all_items"] == 0
