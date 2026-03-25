"""Integration tests for search APIs."""

from __future__ import annotations

from contextlib import contextmanager


class TestKeywordSearchApi:
    """Tests for keyword search API."""

    def test_keyword_search_missing_keyword_returns_400(self, client):
        """Test that missing keyword returns 400."""
        resp = client.post("/api/keyword_search", json={})
        assert resp.status_code == 400

        data = resp.get_json(silent=True) or {}
        assert "error" in data

    def test_keyword_search_with_keyword_returns_success(self, client):
        """Test that valid keyword returns success structure."""
        resp = client.post(
            "/api/keyword_search",
            json={"keyword": "transformer", "time_delta": 365, "limit": 5},
        )
        assert resp.status_code == 200

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "pids" in data
        assert isinstance(data["pids"], list)

    def test_keyword_search_invalid_limit_uses_default(self, client):
        """Test that invalid limit uses default value."""
        resp = client.post("/api/keyword_search", json={"keyword": "test", "limit": "invalid"})
        # Should not crash, uses default limit
        assert resp.status_code == 200

    def test_keyword_search_empty_keyword_returns_400(self, client):
        """Test that empty keyword returns 400."""
        resp = client.post("/api/keyword_search", json={"keyword": ""})
        assert resp.status_code == 400

    def test_keyword_search_non_string_keyword_returns_400(self, client):
        resp = client.post("/api/keyword_search", json={"keyword": 123})
        assert resp.status_code == 400

    def test_keyword_search_malformed_json_returns_400(self, client):
        resp = client.post(
            "/api/keyword_search",
            data='{"keyword": ',
            content_type="application/json",
        )
        assert resp.status_code == 400

        data = resp.get_json(silent=True) or {}
        assert data.get("success") is False
        assert data.get("error") == "No JSON data provided"

    def test_keyword_search_non_object_json_returns_400(self, client):
        for body in ("123", '"abc"', '["x"]'):
            resp = client.post(
                "/api/keyword_search",
                data=body,
                content_type="application/json",
            )
            assert resp.status_code == 400

            data = resp.get_json(silent=True) or {}
            assert data.get("success") is False
            assert data.get("error") == "Request body must be a JSON object"

    def test_keyword_search_without_time_filter_does_not_default_to_recent_only(self, client, monkeypatch):
        from backend import legacy

        monkeypatch.setattr(legacy, "enhanced_search_rank", lambda **_kwargs: (["old-paper"], [1.0], {}))
        monkeypatch.setattr(legacy, "get_metas", lambda: {"old-paper": {"_time": 1.0}})

        resp = client.post("/api/keyword_search", json={"keyword": "transformer", "limit": 5})
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("pids") == ["old-paper"]


class TestTagSearchApi:
    """Tests for tag search API."""

    def test_tag_search_without_login_returns_401(self, client):
        """Test that tag search requires login."""
        resp = client.post("/api/tag_search", json={"tag_name": "test_tag"})
        assert resp.status_code == 401

    def test_tag_search_with_api_key_allows_access(self, client):
        """Test that internal API key can be used for non-browser calls."""
        resp = client.post(
            "/api/tag_search",
            json={
                "tag_name": "test_tag",
                "user": "test_user",
                "time_delta": 365,
                "limit": 5,
            },
            headers={"X-ARXIV-SANITY-API-KEY": "test-api-key"},
        )
        assert resp.status_code in [200, 400]

    def test_tag_search_with_api_key_missing_user_returns_400(self, client):
        """Test that API-key auth requires user field for login-required endpoints."""
        resp = client.post(
            "/api/tag_search",
            json={"tag_name": "test_tag"},
            headers={"X-ARXIV-SANITY-API-KEY": "test-api-key"},
        )
        assert resp.status_code == 400

    def test_tag_search_with_wrong_api_key_returns_401(self, client):
        """Test that wrong API key does not bypass login requirement."""
        resp = client.post(
            "/api/tag_search",
            json={"tag_name": "test_tag", "user": "test_user"},
            headers={"X-ARXIV-SANITY-API-KEY": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_tag_search_missing_tag_returns_400(self, logged_in_client, csrf_token):
        """Test that missing tag_name returns 400."""
        resp = logged_in_client.post("/api/tag_search", json={}, headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code == 400

    def test_tag_search_session_without_csrf_returns_403(self, logged_in_client):
        """Test that session auth requires CSRF for tag_search."""
        resp = logged_in_client.post("/api/tag_search", json={"tag_name": "test_tag"})
        assert resp.status_code == 403

    def test_tag_search_with_tag_returns_success(self, logged_in_client, csrf_token):
        """Test that valid tag search returns success structure."""
        resp = logged_in_client.post(
            "/api/tag_search",
            json={
                "tag_name": "test_tag",
                "user": "test_user",
                "time_delta": 365,
                "limit": 5,
            },
            headers={"X-CSRF-Token": csrf_token},
        )
        # May return 200 with empty results or 400 if tag doesn't exist
        assert resp.status_code in [200, 400]

    def test_tag_search_non_string_tag_returns_400(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/api/tag_search",
            json={"tag_name": 123, "user": "test_user"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_tag_search_user_mismatch_returns_403(self, logged_in_client, csrf_token):
        """Test that mismatched user field is rejected."""
        resp = logged_in_client.post(
            "/api/tag_search",
            json={"tag_name": "test_tag", "user": "other_user"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 403


class TestTagsSearchApi:
    """Tests for multi-tag search API."""

    def test_tags_search_without_login_returns_401(self, client):
        """Test that tags search requires login."""
        resp = client.post("/api/tags_search", json={"tags": ["test_tag"]})
        assert resp.status_code == 401

    def test_tags_search_with_api_key_allows_access(self, client):
        """Test that internal API key can be used for non-browser calls."""
        resp = client.post(
            "/api/tags_search",
            json={
                "tags": ["test_tag"],
                "user": "test_user",
                "time_delta": 365,
                "limit": 5,
            },
            headers={"X-ARXIV-SANITY-API-KEY": "test-api-key"},
        )
        assert resp.status_code in [200, 400]

    def test_tags_search_missing_tags_returns_400(self, logged_in_client, csrf_token):
        """Test that missing tags returns 400."""
        resp = logged_in_client.post("/api/tags_search", json={}, headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code == 400

    def test_tags_search_session_without_csrf_returns_403(self, logged_in_client):
        """Test that session auth requires CSRF for tags_search."""
        resp = logged_in_client.post("/api/tags_search", json={"tags": ["test_tag"]})
        assert resp.status_code == 403

    def test_tags_search_with_tags_returns_success(self, logged_in_client, csrf_token):
        """Test that valid tags search returns success structure."""
        resp = logged_in_client.post(
            "/api/tags_search",
            json={
                "tags": ["test_tag"],
                "user": "test_user",
                "time_delta": 365,
                "limit": 5,
            },
            headers={"X-CSRF-Token": csrf_token},
        )
        # May return 200 with empty results or 400 if tags don't exist
        assert resp.status_code in [200, 400]

    def test_tags_search_user_mismatch_returns_403(self, logged_in_client, csrf_token):
        """Test that mismatched user field is rejected."""
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": ["test_tag"], "user": "other_user"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 403

    def test_tags_search_empty_tags_list(self, logged_in_client, csrf_token):
        """Test tags search with empty tags list."""
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": [], "user": "test_user"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_tags_search_invalid_logic_returns_400(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": ["test_tag"], "logic": "foo", "user": "test_user"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_tags_search_non_string_tag_member_returns_400(self, logged_in_client, csrf_token):
        resp = logged_in_client.post(
            "/api/tags_search",
            json={"tags": ["test_tag", {"bad": "value"}], "user": "test_user"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_tags_search_strict_and_returns_empty_when_any_tag_missing(self, logged_in_client, csrf_token, monkeypatch):
        from backend import legacy

        @contextmanager
        def _fake_user_context(_user):
            yield {"tag_a": {"2301.00001"}}

        monkeypatch.setattr(legacy, "_temporary_user_context", _fake_user_context)
        monkeypatch.setattr(
            legacy,
            "svm_rank",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("svm_rank should not run")),
        )

        resp = logged_in_client.post(
            "/api/tags_search",
            json={
                "tags": ["tag_a", "missing_tag"],
                "logic": "and",
                "user": "test_user",
            },
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("pids") == []
