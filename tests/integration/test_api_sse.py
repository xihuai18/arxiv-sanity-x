"""Integration tests for SSE endpoints."""

from __future__ import annotations

import re
from uuid import uuid4


def test_api_user_stream_enforces_per_user_connection_limit(client, monkeypatch):
    from backend.utils.sse import try_acquire_user_connection_lease
    from config import settings

    # Force local fallback (avoid test flakiness from shared/persistent SQLite sse_events.db).
    monkeypatch.setattr("backend.utils.sse.get_sse_bus", lambda: None)

    # Use a unique username to avoid interference from stale dev data_dir (e.g., reused data/sse_events.db).
    about = client.get("/about")
    m = re.search(r'csrf-token"\s+content="([^"]+)"', about.get_data(as_text=True))
    csrf = m.group(1) if m else ""
    username = f"test_user_{uuid4().hex[:8]}"
    client.post(
        "/login",
        data={"username": username},
        headers={"X-CSRF-Token": csrf},
        follow_redirects=False,
    )

    limit = int(getattr(settings.sse, "max_connections_per_user", 2) or 2)
    if limit <= 0:
        return

    # Reserve `limit` slots (simulate open SSE connections), then ensure the next is rejected.
    leases = []
    try:
        for _ in range(limit):
            ok, lease, _lim = try_acquire_user_connection_lease(username)
            assert ok is True
            assert lease is not None
            assert int(_lim) == limit
            leases.append(lease)

        r3 = client.get("/api/user_stream")
        try:
            assert r3.status_code == 429
            payload = r3.get_json() or {}
            assert payload.get("success") is False
            assert payload.get("limit") == limit
        finally:
            r3.close()
    finally:
        for lease in leases:
            lease.release()
