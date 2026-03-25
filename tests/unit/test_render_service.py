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

    def test_build_paper_title_fields_returns_only_title_keys(self):
        from backend.services.render_service import build_paper_title_fields

        result = build_paper_title_fields({"title": "Graph Neural Networks"})

        assert set(result.keys()) == {"title_lower", "title_norm", "title_norm_loose"}
        assert result["title_norm"] == "graph neural networks"


class TestRenderPid:
    """Tests for render_pid function."""

    def test_render_pid_exists(self):
        """Test that render_pid function exists."""
        from backend.services.render_service import render_pid

        assert callable(render_pid)

    def test_render_pid_exposes_versioned_id(self, monkeypatch):
        from backend.services import render_service

        monkeypatch.setattr(render_service, "get_thumb_url", lambda _pid: "")
        monkeypatch.setattr(render_service, "extract_tldr_from_summary", lambda _pid: "")
        monkeypatch.setattr(render_service, "get_summary_status", lambda _pid: ("", ""))

        paper = {
            "_id": "2301.00001",
            "_idv": "2301.00001v3",
            "_effective_idv": "2301.00001v2",
            "title": "Test Paper",
            "_time_str": "Jan 01 2024",
            "authors": [{"name": "Alice"}],
            "tags": [{"term": "cs.LG"}],
            "summary": "Abstract",
        }

        rendered = render_service.render_pid("2301.00001", paper=paper)

        assert rendered["id"] == "2301.00001"
        assert rendered["raw_id"] == "2301.00001"
        assert rendered["versioned_id"] == "2301.00001v2"

    def test_render_pid_skips_tldr_probe_when_summary_not_ready(self, monkeypatch):
        from backend.services import render_service

        calls = {"tldr": 0}

        monkeypatch.setattr(render_service, "get_thumb_url", lambda _pid: "")
        monkeypatch.setattr(render_service, "get_summary_status", lambda _pid: ("running", None))

        def _fake_extract_tldr(_pid):
            calls["tldr"] += 1
            return "Should not be used"

        monkeypatch.setattr(render_service, "extract_tldr_from_summary", _fake_extract_tldr)

        rendered = render_service.render_pid(
            "2301.00001",
            paper={
                "_id": "2301.00001",
                "title": "Test Paper",
                "_time_str": "2024-01-01",
                "authors": [{"name": "Alice"}],
                "tags": [{"term": "ml"}],
                "summary": "abstract",
            },
        )

        assert rendered["summary_status"] == "running"
        assert rendered["tldr"] == ""
        assert calls["tldr"] == 0

    def test_render_pid_still_loads_tldr_when_summary_status_check_disabled(self, monkeypatch):
        from backend.services import render_service

        calls = {"tldr": 0}

        monkeypatch.setattr(render_service, "get_thumb_url", lambda _pid: "")
        monkeypatch.setattr(render_service, "get_summary_status", lambda _pid: ("ok", None))

        def _fake_extract_tldr(_pid):
            calls["tldr"] += 1
            return "Short TLDR"

        monkeypatch.setattr(render_service, "extract_tldr_from_summary", _fake_extract_tldr)

        rendered = render_service.render_pid(
            "2301.00002",
            paper={
                "_id": "2301.00002",
                "title": "Test Paper",
                "_time_str": "2024-01-01",
                "authors": [{"name": "Bob"}],
                "tags": [{"term": "ai"}],
                "summary": "abstract",
            },
            include_summary_status=False,
        )

        assert rendered["summary_status"] == ""
        assert rendered["tldr"] == "Short TLDR"
        assert calls["tldr"] == 1

    def test_render_pid_uses_preloaded_summary_snapshot(self, monkeypatch):
        from backend.services import render_service

        monkeypatch.setattr(render_service, "get_thumb_url", lambda _pid: "")
        monkeypatch.setattr(
            render_service,
            "get_summary_status",
            lambda _pid: (_ for _ in ()).throw(AssertionError("should not call get_summary_status")),
        )
        monkeypatch.setattr(
            render_service,
            "extract_tldr_from_summary",
            lambda _pid: (_ for _ in ()).throw(AssertionError("should not call extract_tldr_from_summary")),
        )

        rendered = render_service.render_pid(
            "2301.00003",
            summary_snapshot={
                "status": "ok",
                "last_error": None,
                "tldr": "Snapshot TLDR",
            },
            paper={
                "_id": "2301.00003",
                "title": "Test Paper",
                "_time_str": "2024-01-01",
                "authors": [{"name": "Cara"}],
                "tags": [{"term": "vision"}],
                "summary": "abstract",
            },
        )

        assert rendered["summary_status"] == "ok"
        assert rendered["tldr"] == "Snapshot TLDR"

    def test_legacy_render_pid_forwards_summary_snapshot(self, monkeypatch):
        from backend import legacy

        captured = {}

        def _fake_render_pid(pid, **kwargs):
            captured["pid"] = pid
            captured.update(kwargs)
            return {"id": pid}

        monkeypatch.setattr(
            "backend.services.render_service.render_pid",
            _fake_render_pid,
        )

        rendered = legacy.render_pid(
            "2301.00004",
            summary_snapshot={"status": "ok", "tldr": "Snapshot TLDR"},
        )

        assert rendered == {"id": "2301.00004"}
        assert captured["summary_snapshot"] == {
            "status": "ok",
            "tldr": "Snapshot TLDR",
        }


class TestServePaperImage:
    """Tests for serve_paper_image function."""

    def test_serve_paper_image_exists(self):
        """Test that serve_paper_image function exists."""
        from backend.services.render_service import serve_paper_image

        assert callable(serve_paper_image)
