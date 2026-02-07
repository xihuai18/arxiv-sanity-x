"""Integration tests for legacy tag/keyword mutation endpoints JSON contract."""

from __future__ import annotations

import uuid

import pytest

MUTATION_ENDPOINTS = [
    "/add_tag/test_tag",
    "/add/2301.00001/test_tag",
    "/sub/2301.00001/test_tag",
    "/del/test_tag",
    "/rename/old_tag/new_tag",
    "/add_ctag/test_ctag",
    "/del_ctag/test_ctag",
    "/rename_ctag/old_ctag/new_ctag",
    "/add_key/test_keyword",
    "/del_key/test_keyword",
    "/rename_key/old_key/new_key",
]


@pytest.mark.parametrize("endpoint", MUTATION_ENDPOINTS)
def test_legacy_mutation_endpoints_return_json_error_when_not_logged_in(client, csrf_token, endpoint):
    """Legacy mutation endpoints should return JSON error payload when unauthenticated."""
    resp = client.post(endpoint, headers={"X-CSRF-Token": csrf_token})
    assert resp.status_code == 200

    payload = resp.get_json()
    assert isinstance(payload, dict)
    assert payload.get("success") is False
    assert isinstance(payload.get("error"), str)
    assert payload["error"]


def test_add_tag_returns_json_success_when_logged_in(logged_in_client, csrf_token):
    """add_tag should return JSON success payload for a unique tag."""
    tag_name = f"json_contract_tag_{uuid.uuid4().hex[:8]}"
    resp = logged_in_client.post(f"/add_tag/{tag_name}", headers={"X-CSRF-Token": csrf_token})
    assert resp.status_code == 200

    payload = resp.get_json()
    assert isinstance(payload, dict)
    assert payload.get("success") is True


def test_add_key_returns_json_success_when_logged_in(logged_in_client, csrf_token):
    """add_key should return JSON success payload for a unique keyword."""
    keyword = f"json_contract_keyword_{uuid.uuid4().hex[:8]}"
    resp = logged_in_client.post(f"/add_key/{keyword}", headers={"X-CSRF-Token": csrf_token})
    assert resp.status_code == 200

    payload = resp.get_json()
    assert isinstance(payload, dict)
    assert payload.get("success") is True


@pytest.mark.parametrize("endpoint", MUTATION_ENDPOINTS)
def test_legacy_mutation_endpoints_without_csrf_return_403(logged_in_client, endpoint):
    """Mutation endpoints still require CSRF protection."""
    resp = logged_in_client.post(endpoint)
    assert resp.status_code == 403


@pytest.mark.parametrize("endpoint", MUTATION_ENDPOINTS)
def test_legacy_mutation_endpoints_reject_get(client, endpoint):
    """Mutation endpoints should reject GET method."""
    resp = client.get(endpoint)
    assert resp.status_code == 405
