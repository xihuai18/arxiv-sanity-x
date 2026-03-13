"""Summary-page init flow contracts.

These checks keep the summary page from reintroducing avoidable serial waits on
the critical path.
"""

from __future__ import annotations

from pathlib import Path


def test_summary_init_prefetches_available_summaries_in_parallel():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8")

    assert "initialAvailableSummariesPromise" in text
    assert "await Promise.all([" in text


def test_summary_init_does_not_block_summary_fetch_on_resource_gate():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8")

    start = text.find("async function initSummaryApp()")
    assert start != -1
    end = text.find("function renderTagDropdown()", start)
    assert end != -1
    block = text[start:end]

    load_pos = block.find("summaryApp.loadSummary(")
    await_resources_pos = block.find("await resourcesPromise")
    assert load_pos != -1
    assert await_resources_pos == -1 or await_resources_pos > load_pos
