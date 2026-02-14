"""Integration tests for uploaded-paper permissions on summary-related APIs."""

from __future__ import annotations


def test_trigger_paper_summary_with_csrf_allows_anonymous_for_public_pid(client, csrf_token):
    resp = client.post(
        "/api/trigger_paper_summary",
        json={"pid": "2301.00001", "model": "test-model"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 200
    payload = resp.get_json(silent=True) or {}
    assert payload.get("success") is True
    assert payload.get("pid") == "2301.00001"
    assert payload.get("status") in ("queued", "running", "ok")


def test_trigger_paper_summary_upload_pid_requires_owner(logged_in_client, csrf_token, monkeypatch):
    from aslite.repositories import UploadedPaperRepository

    monkeypatch.setattr(
        UploadedPaperRepository,
        "get",
        staticmethod(lambda _pid: {"owner": "someone_else", "parse_status": "ok"}),
    )

    resp = logged_in_client.post(
        "/api/trigger_paper_summary",
        json={"pid": "up_testpid", "model": "test-model"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 404
    payload = resp.get_json(silent=True) or {}
    assert payload.get("success") is False


def test_clear_model_summary_upload_pid_requires_owner(logged_in_client, csrf_token, monkeypatch):
    from aslite.repositories import UploadedPaperRepository

    monkeypatch.setattr(
        UploadedPaperRepository,
        "get",
        staticmethod(lambda _pid: {"owner": "someone_else", "parse_status": "ok"}),
    )

    resp = logged_in_client.post(
        "/api/clear_model_summary",
        json={"pid": "up_testpid", "model": "test-model"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 404
    payload = resp.get_json(silent=True) or {}
    assert payload.get("success") is False


def test_clear_paper_cache_upload_pid_requires_owner(logged_in_client, csrf_token, monkeypatch):
    from aslite.repositories import UploadedPaperRepository

    monkeypatch.setattr(
        UploadedPaperRepository,
        "get",
        staticmethod(lambda _pid: {"owner": "someone_else", "parse_status": "ok"}),
    )

    resp = logged_in_client.post(
        "/api/clear_paper_cache",
        json={"pid": "up_testpid"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert resp.status_code == 404
    payload = resp.get_json(silent=True) or {}
    assert payload.get("success") is False


def test_paper_image_upload_pid_requires_owner(logged_in_client, monkeypatch):
    from aslite.repositories import UploadedPaperRepository

    monkeypatch.setattr(
        UploadedPaperRepository,
        "get",
        staticmethod(lambda _pid: {"owner": "someone_else", "parse_status": "ok"}),
    )

    resp = logged_in_client.get("/api/paper_image/up_testpid/some.png")
    assert resp.status_code == 404
