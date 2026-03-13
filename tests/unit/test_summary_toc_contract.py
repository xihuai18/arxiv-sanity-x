"""Summary TOC behavior contracts.

These checks lock down the JS/CSS hooks that keep the mobile TOC usable.
"""

from __future__ import annotations

from pathlib import Path


def test_summary_toc_keeps_aria_and_mobile_autocollapse_logic():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "markdown_summary_dom_utils.js").read_text(encoding="utf-8", errors="ignore")

    assert 'class="toc-toggle" aria-expanded="true"' in text
    assert "toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true')" in text
    assert "window.matchMedia('(max-width: 768px)').matches" in text
    assert "link._tocLinkCollapse = collapseForMobileNavigation" in text
    assert "backTop._tocBackTopCollapse = collapseForMobileNavigation" in text


def test_summary_css_keeps_toc_collapsed_and_sticky_states():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "css" / "pages" / "summary.css").read_text(encoding="utf-8", errors="ignore")

    assert ".summary-toc.is-collapsed .toc-list" in text
    assert ".summary-toc .toc-toggle" in text
    assert "position: sticky;" in text
