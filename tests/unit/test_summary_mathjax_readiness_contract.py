"""Frontend math rendering readiness contract.

These tests are intentionally lightweight: they assert that the summary-page
markdown math pipeline does not rely on a single MathJax API variant.
"""

from __future__ import annotations

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


def test_common_utils_mathjax_loaded_detection_accepts_typeset():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8")

    # Avoid treating a fully loaded MathJax (sync typeset only) as "not loaded",
    # otherwise on-demand loader logic can get stuck.
    assert "MathJax.typesetPromise || MathJax.typeset" in text
