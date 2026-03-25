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


def test_summary_page_has_business_status_watch_beyond_user_state_polling():
    repo_root = Path(__file__).resolve().parents[2]
    js_path = repo_root / "static" / "paper_summary.js"
    text = js_path.read_text(encoding="utf-8", errors="ignore")

    assert "scheduleStatusWatch(pid" in text
    assert "_fetchSummaryStatusCached(this, pid, targetModel" in text
    assert "this.scheduleStatusWatch(pid, { model: statusModel, immediate: false })" in text


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


def test_summary_page_handles_model_scoped_summary_status_events():
    repo_root = Path(__file__).resolve().parents[2]
    js_text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")
    tasks_text = (repo_root / "tasks.py").read_text(encoding="utf-8", errors="ignore")

    assert "event.type === 'summary_status'" in js_text
    assert "event.model" in js_text
    assert "event.resolved_model" in js_text
    assert "summaryApp.summaryStatusCacheByModel[eventModel]" in js_text
    assert "if (summaryApp.pendingGenerationModel === eventModel)" in js_text
    assert '"model": model' in tasks_text


def test_summary_page_refreshes_when_resolved_model_matches_current_selection():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "summaryApp.summaryStatusCacheByModel[resolvedEventModel]" in text
    assert "resolvedEventModel === currentModel" in text


def test_non_summary_surfaces_ignore_other_models_summary_status_events():
    repo_root = Path(__file__).resolve().parents[2]
    paper_list_text = (repo_root / "static" / "paper_list.js").read_text(encoding="utf-8", errors="ignore")
    readinglist_text = (repo_root / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")
    common_utils_text = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8", errors="ignore")

    assert "isSummaryModelMatch" in common_utils_text
    assert "if (!isSummaryModelMatch(event.model)) return;" in paper_list_text
    assert "if (!isSummaryModelMatch(event.model)) return;" in readinglist_text


def test_shared_summary_status_callback_exposes_model_argument():
    repo_root = Path(__file__).resolve().parents[2]
    common_utils_text = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8", errors="ignore")

    assert "summaryStatusCallback(pid, status, lastError, taskId, model)" in common_utils_text


def test_paper_list_batches_summary_and_tldr_renders():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_list.js").read_text(encoding="utf-8", errors="ignore")

    assert "function schedulePaperListRender(afterRender)" in text
    assert "window.requestAnimationFrame(flushPaperListRender);" in text
    assert "queueMicrotask(flushPaperListRender);" in text or "Promise.resolve().then(flushPaperListRender);" in text

    summary_start = text.find("function updatePaperSummaryStatus")
    summary_end = text.find("const TLDR_REFRESH_STATE", summary_start)
    assert summary_start != -1 and summary_end != -1
    summary_block = text[summary_start:summary_end]
    assert "schedulePaperListRender();" in summary_block
    assert "renderPaperList();" not in summary_block

    tldr_start = text.find("function updatePaperTldr")
    tldr_end = text.find("function fetchPaperTldr", tldr_start)
    assert tldr_start != -1 and tldr_end != -1
    tldr_block = text[tldr_start:tldr_end]
    assert "schedulePaperListRender(() =>" in tldr_block
    assert "triggerMathJax(document.getElementById('paperList'));" in tldr_block
    assert "renderPaperList();" not in tldr_block


def test_paper_list_uses_pid_lookup_for_hot_updates_and_single_initial_mathjax():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_list.js").read_text(encoding="utf-8", errors="ignore")

    assert "const paperById = new Map();" in text
    assert "function rebuildPaperLookup()" in text
    assert "function getPaperById(pid)" in text
    assert text.count("const p = getPaperById(pid);") >= 5
    assert "triggerMathJax(document.getElementById('paperList'));" in text

    mount_start = text.find("componentDidMount()")
    mount_end = text.find("componentDidUpdate(", mount_start)
    assert mount_start != -1 and mount_end != -1
    mount_block = text[mount_start:mount_end]
    assert "triggerMathJax(document.getElementById('paperList'));" not in mount_block


def test_summary_page_tracks_auto_retry_per_model():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "autoRetryCountsByModel" in text
    assert "autoRetryTimersByModel" in text
    assert "getAutoRetryKey(model)" in text
    assert "this.clearAutoRetry(targetModel)" in text


def test_summary_page_timestamp_formatter_preserves_time_components():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "var summaryTimestampFormatter = new Intl.DateTimeFormat(undefined, {" in text
    assert "hour: 'numeric'" in text
    assert "minute: 'numeric'" in text
    assert "second: 'numeric'" in text


def test_summary_page_keeps_retry_budget_across_follow_up_polls():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "this.resetAutoRetry(targetModel);\n    this.clearStatusWatch();" in text
    assert "this.resetAutoRetry(chosenModelStr);\n                    this.scheduleAutoRetry" not in text
    assert "this.resetAutoRetry(statusModel);\n                            this.scheduleAutoRetry" not in text


def test_summary_page_tracks_fallback_aliases_for_requested_models():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "resolvedModelByRequested" in text
    assert "requestedModelsByResolved" in text
    assert "setModelAlias(chosenModelStr, actualModelStr)" in text
    assert "clearModelAndAliasState(currentModel, { removeAvailability: true })" in text
    assert (
        "if (resolved) {\n            this.clearModelLocalState(targetModel, options);\n            this.clearModelAlias(targetModel);\n            return;\n        }"
        in text
    )


def test_summary_page_keeps_requested_model_selected_after_fallback():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "const actualModel = fallback.occurred" in text
    assert "const selectedModel = this.selectedModel || chosenModel || actualModel || '';" in text


def test_summary_page_retry_callback_is_not_blocked_by_other_pending_model():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

    assert "this.pendingGenerationModel &&\n                this.pendingGenerationModel !== targetModel" not in text


def test_sse_reconnect_respects_backoff_when_timer_pending():
    """connectAsLeaderIfNeeded must not bypass exponential backoff.

    When eventSourceReconnectTimer is set (i.e. scheduleReconnect has already
    queued a retry), the leader monitor callback should NOT trigger an additional
    reconnect attempt.  This prevents a '429 storm' where the monitor fires
    every 2 s while the backoff timer says to wait much longer.
    """
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "static" / "common_utils.js").read_text(encoding="utf-8", errors="ignore")

    idx_fn = text.find("const connectAsLeaderIfNeeded")
    assert idx_fn != -1, "connectAsLeaderIfNeeded function not found"

    # Extract the function body (up to the next top-level const/function).
    fn_body = text[idx_fn : idx_fn + 500]

    # The guard must appear in the function body before the leadership check.
    assert (
        "if (eventSourceReconnectTimer) return;" in fn_body
    ), "connectAsLeaderIfNeeded missing eventSourceReconnectTimer guard"

    idx_guard_in_body = fn_body.find("if (eventSourceReconnectTimer) return;")
    idx_leadership_in_body = fn_body.find("tryClaimUserEventLeadership()")
    assert idx_leadership_in_body != -1, "tryClaimUserEventLeadership call not found in connectAsLeaderIfNeeded"
    assert (
        idx_guard_in_body < idx_leadership_in_body
    ), "eventSourceReconnectTimer guard must appear before the leadership claim check"
