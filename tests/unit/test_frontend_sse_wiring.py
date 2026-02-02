"""Frontend SSE wiring checks.

This is a lightweight guardrail against regressions where a page forgets to
pass `user` into `setupUserEventStream`, making SSE a no-op.
"""

from __future__ import annotations

from pathlib import Path


def test_summary_page_sets_up_user_event_stream():
    repo_root = Path(__file__).resolve().parents[2]
    js_path = repo_root / "static" / "paper_summary.js"
    text = js_path.read_text(encoding="utf-8", errors="ignore")

    # The CommonUtils.setupUserEventStream signature is (user, applyStateFn).
    assert "setupUserEventStream(user" in text or "_setupUserEventStream(user" in text


def test_summary_page_does_not_auto_trigger_generation_on_cache_miss():
    repo_root = Path(__file__).resolve().parents[2]
    js_path = repo_root / "static" / "paper_summary.js"
    text = js_path.read_text(encoding="utf-8", errors="ignore")

    # The summary page should never auto-trigger generation just because a cache is missing.
    # Users must click "Generate" to enqueue a job.
    assert "auto_trigger: true" not in text
    start = text.find("summaryApp.loadSummary = async function")
    assert start != -1
    end = text.find("summaryApp.loadModels", start)
    assert end != -1
    load_summary_block = text[start:end]
    assert "queueSummary" not in load_summary_block
