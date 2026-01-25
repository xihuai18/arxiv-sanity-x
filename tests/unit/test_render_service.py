"""Unit tests for render service functions."""

from __future__ import annotations


class TestGetThumbUrl:
    """Tests for get_thumb_url function."""

    def test_get_thumb_url_returns_string(self):
        """Test that get_thumb_url returns a string."""
        from backend.services.render_service import get_thumb_url

        result = get_thumb_url("2301.00001")
        assert isinstance(result, str)


class TestBuildPaperTextFields:
    """Tests for build_paper_text_fields function."""

    def test_build_paper_text_fields_returns_dict(self):
        """Test that build_paper_text_fields returns a dict."""
        from backend.services.render_service import build_paper_text_fields

        paper = {
            "title": "Test Paper",
            "summary": "This is a test summary.",
            "authors": [{"name": "Test Author"}],
        }
        result = build_paper_text_fields(paper)
        assert isinstance(result, dict)

    def test_build_paper_text_fields_with_empty_paper(self):
        """Test build_paper_text_fields with empty paper dict."""
        from backend.services.render_service import build_paper_text_fields

        result = build_paper_text_fields({})
        assert isinstance(result, dict)


class TestRenderPid:
    """Tests for render_pid function."""

    def test_render_pid_exists(self):
        """Test that render_pid function exists."""
        from backend.services.render_service import render_pid

        assert callable(render_pid)


class TestServePaperImage:
    """Tests for serve_paper_image function."""

    def test_serve_paper_image_exists(self):
        """Test that serve_paper_image function exists."""
        from backend.services.render_service import serve_paper_image

        assert callable(serve_paper_image)
