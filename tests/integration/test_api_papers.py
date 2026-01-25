"""Integration tests for paper-related APIs."""

from __future__ import annotations


class TestPaperImageApi:
    """Tests for paper image serving APIs."""

    def test_paper_image_path_traversal_in_pid_returns_400(self, client):
        """Test that path traversal in pid returns 400."""
        resp = client.get("/api/paper_image/test../test.jpg")
        assert resp.status_code == 400

    def test_paper_image_path_traversal_in_filename_returns_400(self, client):
        """Test that path traversal in filename returns 400."""
        resp = client.get("/api/paper_image/2501.00001/..test.jpg")
        assert resp.status_code == 400

    def test_paper_image_empty_pid_returns_404(self, client):
        """Test that empty pid returns 404 (route doesn't match)."""
        resp = client.get("/api/paper_image//test.jpg")
        assert resp.status_code == 404

    def test_paper_image_not_found_returns_404(self, client):
        """Test that non-existent image returns 404."""
        resp = client.get("/api/paper_image/nonexistent_pid_12345/test.jpg")
        assert resp.status_code == 404


class TestMineruImageApi:
    """Tests for MinerU image serving APIs."""

    def test_mineru_image_path_traversal_in_pid_returns_400(self, client):
        """Test that path traversal in pid returns 400."""
        resp = client.get("/api/mineru_image/test../test.jpg")
        assert resp.status_code == 400

    def test_mineru_image_path_traversal_in_filename_returns_400(self, client):
        """Test that path traversal in filename returns 400."""
        resp = client.get("/api/mineru_image/2501.00001/..test.jpg")
        assert resp.status_code == 400

    def test_mineru_image_empty_pid_returns_404(self, client):
        """Test that empty pid returns 404 (route doesn't match)."""
        resp = client.get("/api/mineru_image//test.jpg")
        assert resp.status_code == 404

    def test_mineru_image_not_found_returns_404(self, client):
        """Test that non-existent MinerU image returns 404."""
        resp = client.get("/api/mineru_image/nonexistent_pid_12345/test.jpg")
        assert resp.status_code == 404


class TestPaperTitlesApi:
    """Tests for paper titles API."""

    def test_paper_titles_requires_login(self, client, csrf_token):
        """Test that paper_titles requires login."""
        resp = client.post("/api/paper_titles", json={"pids": ["2301.00001"]}, headers={"X-CSRF-Token": csrf_token})
        # API requires login, should return 401
        assert resp.status_code == 401
