"""Guardrails for summary-page generation behavior.

The summary page should not auto-trigger summary generation on cache misses.
Users must explicitly click "Generate".
"""

from __future__ import annotations

from pathlib import Path


def test_summary_page_does_not_auto_trigger_generation():
    repo_root = Path(__file__).resolve().parents[2]
    js_path = repo_root / "static" / "paper_summary.js"
    text = js_path.read_text(encoding="utf-8", errors="ignore")

    assert "auto_trigger: true" not in text
