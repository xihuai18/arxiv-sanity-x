"""Reading list UI source contracts."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_readinglist_template_empty_states_are_announced():
    text = (_repo_root() / "templates" / "readinglist.html").read_text(encoding="utf-8")

    assert 'id="rl-empty-state" role="status" aria-live="polite"' in text
    assert 'id="uploaded-empty-state" role="status" aria-live="polite"' in text
    assert 'class="rl-login-message" role="status" aria-live="polite"' in text


def test_readinglist_cards_use_grouped_action_layout():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "paper-actions-group paper-actions-group-primary" in text
    assert "paper-actions-group paper-actions-group-secondary" in text
    assert "rel_summary_trigger" in text


def test_readinglist_css_keeps_mobile_safe_padding_and_focus_states():
    text = (_repo_root() / "static" / "css" / "pages" / "readinglist.css").read_text(encoding="utf-8", errors="ignore")

    assert "padding: 24px !important" not in text
    assert "@media (hover: hover) and (pointer: fine)" in text
    assert ".readinglist-btn:focus-visible" in text
    assert ".upload-btn:focus-visible" in text


def test_readinglist_mutations_use_keepalive_for_fast_navigation():
    repo_root = _repo_root()
    homepage_js = (repo_root / "static" / "paper_list.js").read_text(encoding="utf-8", errors="ignore")
    readinglist_js = (repo_root / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "csrfFetch('/api/readinglist/add'" in homepage_js
    assert "csrfFetch('/api/readinglist/remove'" in homepage_js
    assert "keepalive: true" in homepage_js
    assert "trackReadingListMutation" in homepage_js
    assert "setupReadingListNavigationGuard" in homepage_js
    assert "shouldSuppressMutationError" in homepage_js
    assert "pagehide" in homepage_js
    assert "csrfFetch('/api/readinglist/remove'" in readinglist_js
    assert "keepalive: true" in readinglist_js
    assert "shouldSuppressMutationError" in readinglist_js
