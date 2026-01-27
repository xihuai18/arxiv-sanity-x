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
