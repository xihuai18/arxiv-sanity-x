"""Lightweight UI accessibility/source contracts.

These tests intentionally avoid importing the Flask app so they can run even
when the local Python environment is missing runtime web dependencies.
"""

from __future__ import annotations

from pathlib import Path


def _read_text(*parts: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.joinpath(*parts).read_text(encoding="utf-8", errors="ignore")


def test_base_template_mobile_nav_keeps_accessible_labels():
    text = _read_text("templates", "base.html")

    assert "aria-label=\"{{ 'Profile' if user else 'Login' }}\"" in text
    assert 'aria-label="Reading list"' in text
    assert 'aria-label="Stats"' in text
    assert 'aria-label="About"' in text
    assert 'aria-hidden="true">📊<' in text


def test_paper_list_uses_real_button_for_readinglist_toggle():
    text = _read_text("static", "paper_list.js")

    assert "<button" in text
    assert "aria-pressed={isInReadingList ? 'true' : 'false'}" in text
    assert "aria-label={btnTitle}" in text


def test_summary_page_readinglist_toggle_and_actions_have_labels():
    text = _read_text("static", "paper_summary.js")

    assert 'data-summary-action="toggle-reading-list"' in text
    assert "aria-pressed=\"${rlActive ? 'true' : 'false'}\"" in text
    assert 'aria-label="${rlBtnTitle}"' in text
    assert "summary-actions-notices" in text


def test_tag_dropdown_remove_action_uses_real_button_and_label():
    text = _read_text("static", "tag_dropdown_shared.js")

    assert "'button'" in text
    assert "class: 'remove-tag'" in text
    assert "'aria-label': `Remove tag ${item.tag}`" in text


def test_homepage_tag_suggestions_expose_listbox_accessibility_hooks():
    text = _read_text("templates", "index.html")

    assert "tagSuggestionIndex: -1" in text
    assert "setAttribute('aria-autocomplete', 'list')" in text
    assert "setAttribute('role', 'listbox')" in text
    assert "setAttribute('role', 'option')" in text
    assert "aria-activedescendant" in text


def test_inspect_template_uses_shared_script_fallback_pattern():
    text = _read_text("templates", "inspect.html")

    assert "data-fallback-src" in text
    assert "document.addEventListener('error'" in text
    assert 'id="wrap"' in text
    assert 'id="wordwrap"' in text


def test_error_template_keeps_primary_recovery_actions():
    text = _read_text("templates", "error.html")

    assert 'class="error-page" role="main"' in text
    assert 'href="/" class="btn btn-primary"' in text
    assert 'href="/profile" class="btn btn-cancel"' in text
