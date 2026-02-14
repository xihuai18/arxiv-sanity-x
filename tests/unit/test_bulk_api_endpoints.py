"""Unit tests for batch API endpoints."""

from __future__ import annotations


def test_tag_feedback_bulk_partial_success(logged_in_client, auth_headers):
    payload = {
        "items": [
            {"pid": "2301.00001", "tag": "t1", "label": 1},
            {"pid": "2301.00002", "tag": "bad/tag", "label": 1},
            {"pid": "2301.00003", "tag": "t3", "label": 0},
        ]
    }
    resp = logged_in_client.post("/api/tag_feedback_bulk", headers=auth_headers, json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    results = data["results"]
    assert len(results) == 3
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "slash" in str(results[1].get("error", "")).lower()
    assert results[2]["success"] is True


def test_trigger_paper_summary_bulk_returns_results(client, auth_headers, monkeypatch):
    import backend.legacy as legacy

    monkeypatch.setattr(legacy, "get_summary_status", lambda pid, model: ("", None))
    monkeypatch.setattr(legacy, "_trigger_summary_async", lambda user, pid, model, priority, force_refresh: "task123")

    payload = {"items": [{"pid": "2301.00001", "model": "m1"}]}
    resp = client.post("/api/trigger_paper_summary_bulk", headers=auth_headers, json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert isinstance(data.get("results"), list)
    assert data["results"][0]["success"] is True
    assert data["results"][0]["status"] == "queued"
    assert data["results"][0]["task_id"] == "task123"
