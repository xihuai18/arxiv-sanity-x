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
    summary_js = (repo_root / "static" / "paper_summary.js").read_text(encoding="utf-8", errors="ignore")

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
    assert "csrfFetch('/api/readinglist/add'" in summary_js
    assert "csrfFetch('/api/readinglist/remove'" in summary_js
    assert "keepalive: true" in summary_js


def test_homepage_readinglist_add_uses_backend_summary_status():
    text = (_repo_root() / "static" / "paper_list.js").read_text(encoding="utf-8", errors="ignore")

    assert "const responseStatus = String(data.summary_status || '').trim();" in text
    assert "summaryStatus: 'queued'" not in text


def test_uploaded_summary_link_requires_parse_only_not_metadata():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "const summaryDisabled = p.parse_status !== 'ok';" in text
    assert "const summaryLinkDisabled = currentParseStatus !== 'ok';" in text
    assert "if (summaryDisabled) {" in text
    assert "if (summaryLinkDisabled) {" in text


def test_uploaded_readinglist_events_resync_uploaded_section_instead_of_public_card_endpoint():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "String(event.pid).indexOf('up_') === 0" in text
    assert "fetchUploadedPapers(true)" in text


def test_uploaded_parse_failures_can_surface_retry_button_after_live_updates():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "const updateRetryParseState = newStatus => {" in text
    assert "retryWrap.style.display = showRetry ? '' : 'none';" in text
    assert "ui.updateRetryParseState(status);" in text


def test_uploaded_parse_refresh_clears_stale_summary_state_when_backend_omits_it():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "function resetUploadedSummaryState(ui)" in text
    assert text.count("resetUploadedSummaryState(ui);") >= 2


def test_readinglist_empty_state_checks_only_for_any_card_not_all_cards():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "container.querySelector('.rl-paper-card')" in text
    assert "querySelectorAll('.rl-paper-card')" not in text


def test_uploaded_readinglist_ui_stores_hot_dom_refs_for_follow_up_updates():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    assert "const uiState = {" in text
    assert "uploadedSummaryUI.set(p.id, uiState);" in text
    assert "titleDiv: titleDiv" in text
    assert "utagsWrap: utagsWrap" in text
    assert "tldrEl: tldrDiv" in text
    assert "tldrTextEl: tldrTextEl" in text
    assert "abstractDetailsEl: abstractDetailsEl" in text
    assert "abstractEl: abstractEl" in text


def test_uploaded_tldr_and_extract_updates_reuse_cached_dom_refs():
    text = (_repo_root() / "static" / "readinglist.js").read_text(encoding="utf-8", errors="ignore")

    tldr_start = text.find("function updateTldrDisplay(ui, tldr)")
    tldr_end = text.find("function handleReadingListEvent", tldr_start)
    assert tldr_start != -1 and tldr_end != -1
    tldr_block = text[tldr_start:tldr_end]
    assert "ui.tldrEl" in tldr_block
    assert "ui.tldrTextEl" in tldr_block
    assert "ui.abstractDetailsEl" in tldr_block
    assert "ui.abstractEl" in tldr_block
    assert "ui.utagsWrap" in tldr_block

    extract_start = text.find("function handleUploadExtractStatusEvent(event)")
    extract_end = text.find("function handleUploadDeletedEvent", extract_start)
    assert extract_start != -1 and extract_end != -1
    extract_block = text[extract_start:extract_end]
    assert "ui.titleDiv" in extract_block
    assert "ui.tldrEl || null" in extract_block
    assert "ui.abstractEl || null" in extract_block
    assert "ui.utagsWrap || null" in extract_block
