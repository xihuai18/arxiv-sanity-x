"""Contracts for summary rendering surfaces and image behavior."""

from __future__ import annotations

from pathlib import Path


def test_summary_images_use_lazy_async_loading():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "markdown_summary_utils.js").read_text(encoding="utf-8")

    assert 'loading="lazy" decoding="async"' in text


def test_summary_image_zoom_does_not_replace_image_nodes():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "markdown_summary_dom_utils.js").read_text(encoding="utf-8")

    assert "cloneNode(true)" not in text
    assert "dataset.summaryZoomBound" in text


def test_all_summary_render_surfaces_defer_markdownit_cdn_script():
    repo_root = Path(__file__).resolve().parents[2]
    templates = [
        repo_root / "templates" / "summary.html",
        repo_root / "templates" / "index.html",
        repo_root / "templates" / "readinglist.html",
        repo_root / "templates" / "inspect.html",
    ]

    expected = "<script defer src=\"{{ npm_cdn_url('markdown-it@13.0.1/dist/markdown-it.min.js') }}\""
    for path in templates:
        text = path.read_text(encoding="utf-8")
        assert expected in text, path.name


def test_mathjax_font_preload_hints_are_removed_to_avoid_console_noise():
    repo_root = Path(__file__).resolve().parents[2]
    common_utils = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8")
    summary_js = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8")

    assert "link.rel = 'preload'" not in common_utils
    assert "link.rel = 'preload'" not in summary_js


def test_base_template_does_not_request_google_fonts():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "templates" / "base.html").read_text(encoding="utf-8")

    assert "fonts.googleapis.com" not in text
    assert "fonts.gstatic.com" not in text
