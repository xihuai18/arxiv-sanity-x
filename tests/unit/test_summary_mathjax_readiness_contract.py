"""Frontend math rendering readiness contract.

These tests are intentionally lightweight: they assert that the summary-page
markdown math pipeline does not rely on a single MathJax API variant.
"""

from __future__ import annotations

import re
from pathlib import Path


def test_summary_markdown_math_accepts_mathjax_typeset_fallback():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "markdown_summary_utils.js").read_text(encoding="utf-8")

    # The summary markdown renderer prefers tex2chtmlPromise when available, but must also
    # work with bundles that only expose typesetPromise/typeset.
    assert "MathJax.tex2chtmlPromise" in text
    assert "MathJax.typesetPromise" in text
    assert "MathJax.typeset === 'function'" in text
    # Prefer MathJax.startup.document.convert when available (works with browser bundles).
    assert "MathJax.startup.document.convert" in text
    assert "outputJax.styleSheet(MathJax.startup.document)" in text


def test_summary_syncs_mathjax_styles_before_document_update():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "markdown_summary_utils.js").read_text(encoding="utf-8")

    matches = re.findall(
        r"syncMathJaxStyles\(\);\s*\n\s*MathJax\.startup\.document\.updateDocument\(\)",
        text,
    )
    assert len(matches) >= 2


def test_common_utils_mathjax_loaded_detection_accepts_typeset():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8")

    # Avoid treating a fully loaded MathJax (sync typeset only) as "not loaded",
    # otherwise on-demand loader logic can get stuck.
    assert "MathJax.typesetPromise || MathJax.typeset" in text


def test_summary_waits_for_critical_mathjax_fonts_before_first_render():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8")

    assert "function _waitForCriticalMathJaxFonts" in text
    assert "_areCriticalMathJaxFontsReady" in text
    assert "document.fonts.load('1em ' + family)" in text
    assert "_preloadMathJaxFonts" not in text


def test_mathjax_config_does_not_redeclare_loader_extensions_for_full_bundle():
    repo_root = Path(__file__).resolve().parents[2]
    common_utils = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8")
    summary_template = (repo_root / "templates" / "summary.html").read_text(encoding="utf-8")

    assert "loader:" not in common_utils
    assert "loader:" not in summary_template


def test_mathjax_fonts_are_self_hosted_by_default():
    repo_root = Path(__file__).resolve().parents[2]
    common_utils = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8")
    summary_template = (repo_root / "templates" / "summary.html").read_text(encoding="utf-8")

    assert "fontURL: '/static/lib/es5/output/chtml/fonts/woff-v2'" in common_utils
    assert "filename='lib/es5/output/chtml/fonts/woff-v2'" in summary_template
