"""Integration tests for tag and keyword management APIs.

Note: These are legacy endpoints that return plain text responses (e.g., "ok" or
"error, not logged in") with HTTP 200 status code, rather than proper JSON API
responses with appropriate HTTP status codes.

The wide assertions (e.g., `in [200, 302, 401, 403]`) are intentional because:
1. These endpoints return 200 even for "not logged in" errors (as plain text)
2. Some endpoints may redirect (302) in certain configurations
3. CSRF protection returns 403

These tests primarily verify that endpoints exist and don't crash.
For proper API behavior testing, see the newer /api/* endpoints.
"""

from __future__ import annotations


class TestAddTagApi:
    """Tests for add_tag API."""

    def test_add_tag_responds(self, client, csrf_token):
        """Test that add_tag endpoint responds."""
        resp = client.post("/add_tag/test_tag", headers={"X-CSRF-Token": csrf_token})
        # Legacy endpoint returns 200 with "error, not logged in" text when not logged in
        assert resp.status_code in [200, 302, 401, 403]

    def test_add_tag_without_csrf_returns_403(self, logged_in_client):
        """Test that add_tag without CSRF returns 403."""
        resp = logged_in_client.post("/add_tag/test_tag")
        assert resp.status_code == 403


class TestAddPaperToTagApi:
    """Tests for add/<pid>/<tag> API."""

    def test_add_paper_to_tag_responds(self, client, csrf_token):
        """Test that adding paper to tag endpoint responds."""
        resp = client.post("/add/2301.00001/test_tag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestSubPaperFromTagApi:
    """Tests for sub/<pid>/<tag> API."""

    def test_sub_paper_from_tag_responds(self, client, csrf_token):
        """Test that removing paper from tag endpoint responds."""
        resp = client.post("/sub/2301.00001/test_tag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestDeleteTagApi:
    """Tests for del/<tag> API."""

    def test_delete_tag_responds(self, client, csrf_token):
        """Test that deleting tag endpoint responds."""
        resp = client.post("/del/test_tag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestRenameTagApi:
    """Tests for rename/<otag>/<ntag> API."""

    def test_rename_tag_responds(self, client, csrf_token):
        """Test that renaming tag endpoint responds."""
        resp = client.post("/rename/old_tag/new_tag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestAddCombinedTagApi:
    """Tests for add_ctag/<ctag> API."""

    def test_add_ctag_responds(self, client, csrf_token):
        """Test that adding combined tag endpoint responds."""
        resp = client.post("/add_ctag/test_ctag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestDeleteCombinedTagApi:
    """Tests for del_ctag/<ctag> API."""

    def test_del_ctag_responds(self, client, csrf_token):
        """Test that deleting combined tag endpoint responds."""
        resp = client.post("/del_ctag/test_ctag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestRenameCombinedTagApi:
    """Tests for rename_ctag/<otag>/<ntag> API."""

    def test_rename_ctag_responds(self, client, csrf_token):
        """Test that renaming combined tag endpoint responds."""
        resp = client.post("/rename_ctag/old_ctag/new_ctag", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestAddKeywordApi:
    """Tests for add_key/<keyword> API."""

    def test_add_key_responds(self, client, csrf_token):
        """Test that adding keyword endpoint responds."""
        resp = client.post("/add_key/test_keyword", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestDeleteKeywordApi:
    """Tests for del_key/<keyword> API."""

    def test_del_key_responds(self, client, csrf_token):
        """Test that deleting keyword endpoint responds."""
        resp = client.post("/del_key/test_keyword", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]


class TestRenameKeywordApi:
    """Tests for rename_key/<okey>/<nkey> API."""

    def test_rename_key_responds(self, client, csrf_token):
        """Test that renaming keyword endpoint responds."""
        resp = client.post("/rename_key/old_key/new_key", headers={"X-CSRF-Token": csrf_token})
        assert resp.status_code in [200, 302, 401, 403]
