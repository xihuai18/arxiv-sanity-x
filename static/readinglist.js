'use strict';

// Reading list page logic extracted from templates/readinglist.html
// Expects globals injected by template:
// - papers, tags, defaultSummaryModel, user
// And Common utilities loaded via base.html:
// - window.ArxivSanityCommon

(function (global) {
    const CommonUtils = global.ArxivSanityCommon;
    if (!CommonUtils) {
        console.error('ArxivSanityCommon not loaded');
        return;
    }

    // Injected globals (from template)
    // Use var to match existing global injection style.
    // eslint-disable-next-line no-var
    var papers = global.papers;
    // eslint-disable-next-line no-var
    var tags = global.tags;

    // Shared utilities
    const csrfFetch = CommonUtils.csrfFetch;
    const fetchUserState = CommonUtils.fetchUserState;

    // Summary status polling (shared from common_utils.js)
    const markSummaryPending = CommonUtils.markSummaryPending;
    const unmarkSummaryPending = CommonUtils.unmarkSummaryPending;
    const canTriggerSummary = CommonUtils.canTriggerSummary;
    const formatSummaryStatus = CommonUtils.formatSummaryStatus;
    const fetchTaskStatus = CommonUtils.fetchTaskStatus;

    let pendingRemove = null;
    const readinglistDropdowns = new Map();
    const readinglistSummaryUI = new Map();

    // Register callback for summary status updates
    CommonUtils.setSummaryStatusCallback(function (pid, status, lastError, taskId) {
        updateSummaryStatusFromEvent(pid, status, lastError, { task_id: taskId });
    });

    function applyUserState(state) {
        if (!state || !state.success) return;
        if (Array.isArray(state.tags)) {
            tags = state.tags;
            global.tags = tags;
        }
        const available = getAvailableTags();
        readinglistDropdowns.forEach(dropdownApi => {
            if (dropdownApi && typeof dropdownApi.updateAvailableTags === 'function') {
                dropdownApi.updateAvailableTags(available);
            }
        });
    }

    function fetchUserStateAndApply() {
        return fetchUserState().then(applyUserState);
    }

    function updateSummaryStatusFromEvent(pid, status, error, event) {
        const ui = readinglistSummaryUI.get(pid);
        if (!ui) return;
        ui.state.status = status || '';
        ui.state.lastError = error || '';
        if (event && event.task_id !== undefined) {
            ui.state.taskId = event.task_id ? String(event.task_id) : '';
        }
        if (ui.state.status !== 'queued') {
            ui.state.queueRank = 0;
            ui.state.queueTotal = 0;
            ui.state.taskId = '';
            stopQueueRankPolling(pid);
        }
        updateSummaryBadge(
            ui.badge,
            ui.state.status,
            ui.state.lastError,
            ui.state.queueRank,
            ui.state.queueTotal
        );
        ui.syncTriggerState();
    }

    function handleReadingListEvent(event) {
        if (!event || !event.pid) return;
        if (event.action === 'add') {
            const existing = document.querySelector(`.rl-paper-card[data-pid="${event.pid}"]`);
            if (!existing) {
                global.location.reload();
            }
            return;
        }
        if (event.action === 'remove') {
            const ui = readinglistSummaryUI.get(event.pid);
            if (ui && ui.card) {
                ui.card.remove();
                updateEmptyState();
            }
            // Stop queue rank polling to prevent timer leak
            stopQueueRankPolling(event.pid);
            unmarkSummaryPending(event.pid);
            readinglistSummaryUI.delete(event.pid);
            const dropdownApi = readinglistDropdowns.get(event.pid);
            if (dropdownApi && typeof dropdownApi.unregister === 'function') {
                dropdownApi.unregister();
            }
            readinglistDropdowns.delete(event.pid);
        }
    }

    function handleUserEvent(event, options = {}) {
        if (!event || typeof event !== 'object') return;
        if (event.type === 'user_state_changed') {
            if (event.reason === 'rename_tag') {
                readinglistDropdowns.forEach(api => {
                    if (api && typeof api.applyTagRename === 'function') {
                        api.applyTagRename(event.from, event.to);
                    }
                });
            } else if (event.reason === 'delete_tag') {
                readinglistDropdowns.forEach(api => {
                    if (api && typeof api.applyTagDelete === 'function') {
                        api.applyTagDelete(event.tag);
                    }
                });
            } else if (
                event.reason === 'tag_feedback' &&
                event.pid &&
                event.tag &&
                event.label !== undefined
            ) {
                const api = readinglistDropdowns.get(event.pid);
                if (api && typeof api.applyTagFeedback === 'function') {
                    api.applyTagFeedback(event.tag, event.label);
                }
            }
            fetchUserStateAndApply();
        } else if (event.type === 'summary_status') {
            updateSummaryStatusFromEvent(event.pid, event.status, event.error, event);
        } else if (event.type === 'readinglist_changed') {
            handleReadingListEvent(event);
        }
        void options;
    }

    function setupUserEventStream() {
        CommonUtils.registerEventHandler(handleUserEvent);
        CommonUtils.setupUserEventStream(global.user, applyUserState);
    }

    function closeRemoveConfirm() {
        if (!pendingRemove) return;
        const { popup } = pendingRemove;
        if (popup && popup.parentNode) {
            popup.parentNode.removeChild(popup);
        }
        pendingRemove = null;
    }

    function confirmRemoveFromReadingList() {
        if (!pendingRemove) return;
        const { pid, element } = pendingRemove;
        closeRemoveConfirm();
        performRemoveFromReadingList(pid, element);
    }

    function openRemoveConfirm(pid, element) {
        if (!element) return;
        closeRemoveConfirm();

        const wrap = element.closest('.rl-remove-wrap') || element.parentElement;
        const card = element.closest('.rl-paper-card');
        const paperTitle = card
            ? (
                  (card.querySelector('.rel_title a') &&
                      card.querySelector('.rel_title a').textContent) ||
                  ''
              ).trim()
            : '';
        const desc = paperTitle
            ? `Remove “${paperTitle}” from your reading list?`
            : 'Remove this paper from your reading list?';

        const popup = document.createElement('div');
        popup.className = 'confirm-popup';
        popup.setAttribute('role', 'dialog');
        popup.innerHTML = `
            <div class="confirm-content">
                <strong>Remove from reading list?</strong>
                <p>${desc}</p>
                <p>You can add it back anytime.</p>
                <div class="confirm-actions">
                    <button class="confirm-btn confirm-yes" type="button">Remove</button>
                    <button class="confirm-btn confirm-no" type="button">Cancel</button>
                </div>
            </div>
        `;

        const yesBtn = popup.querySelector('.confirm-yes');
        const noBtn = popup.querySelector('.confirm-no');
        if (yesBtn) yesBtn.addEventListener('click', confirmRemoveFromReadingList);
        if (noBtn) noBtn.addEventListener('click', closeRemoveConfirm);

        if (wrap) {
            wrap.appendChild(popup);
        }
        pendingRemove = { pid, element, popup, wrap };
    }

    document.addEventListener('mousedown', event => {
        if (!pendingRemove) return;
        const { wrap } = pendingRemove;
        if (wrap && wrap.contains(event.target)) return;
        closeRemoveConfirm();
    });

    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
            closeRemoveConfirm();
        }
    });

    function performRemoveFromReadingList(pid, element) {
        csrfFetch('/api/readinglist/remove', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const card = element.closest('.rl-paper-card');
                    if (card) {
                        readinglistDropdowns.delete(pid);
                        readinglistSummaryUI.delete(pid);
                        unmarkSummaryPending(pid);
                        card.style.transition = 'opacity 0.3s, transform 0.3s';
                        card.style.opacity = '0';
                        card.style.transform = 'translateX(-20px)';
                        setTimeout(() => {
                            card.remove();
                            updateEmptyState();
                        }, 300);
                    }
                } else {
                    alert('Failed to remove: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Error removing from reading list:', err);
                alert('Failed to remove paper');
            });
    }

    function updateEmptyState() {
        const container = document.getElementById('rl-papers');
        const emptyState = document.getElementById('rl-empty-state');
        const cards = container ? container.querySelectorAll('.rl-paper-card') : [];

        if (emptyState) {
            emptyState.style.display = cards.length === 0 ? 'block' : 'none';
        }
    }

    function formatDate(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp * 1000);
        return (
            date.toLocaleDateString() +
            ' ' +
            date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        );
    }

    function renderTldrMarkdown(text) {
        return CommonUtils.renderTldrMarkdown(text);
    }

    function renderAbstractMarkdown(text) {
        return CommonUtils.renderAbstractMarkdown(text);
    }

    function triggerMathJax(element) {
        return CommonUtils.triggerMathJax(element);
    }

    function buildTagUrl(tagName) {
        return CommonUtils.buildTagUrl(tagName);
    }

    function createTextElement(tag, className, text) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        el.textContent = text;
        return el;
    }

    function createLinkElement(href, className, text, target) {
        const el = document.createElement('a');
        el.href = href;
        if (className) el.className = className;
        el.textContent = text;
        if (target) el.target = target;
        return el;
    }

    function getAvailableTags() {
        if (!Array.isArray(tags)) return [];
        return tags.map(tag => tag.name).filter(name => name && name !== 'all');
    }

    function updateSummaryBadge(badgeEl, status, lastError, queueRank, queueTotal) {
        if (!badgeEl) return;
        const text = formatSummaryStatus(status);
        const rankText =
            status === 'queued' && queueRank
                ? `${queueRank}${queueTotal ? '/' + queueTotal : ''} Queued`
                : '';
        if (!text) {
            badgeEl.style.display = 'none';
            return;
        }
        badgeEl.style.display = 'inline-flex';
        badgeEl.textContent = '';
        badgeEl.appendChild(document.createTextNode(text));
        if (rankText) {
            const rankSpan = document.createElement('span');
            rankSpan.className = 'queue-rank-pill';
            rankSpan.textContent = rankText;
            badgeEl.appendChild(rankSpan);
        }
        badgeEl.className = 'summary-status-badge';
        if (status === 'ok') {
            badgeEl.classList.add('ok');
        } else if (status === 'failed') {
            badgeEl.classList.add('failed');
        }
        const tooltipParts = [];
        if (lastError) tooltipParts.push(lastError);
        if (rankText) tooltipParts.push(rankText + ' (high priority only)');
        if (tooltipParts.length) {
            badgeEl.title = tooltipParts.join(' · ');
        } else {
            badgeEl.removeAttribute('title');
        }
    }

    const summaryQueueRankPollers = new Map();

    function startQueueRankPolling(pid) {
        if (!pid || summaryQueueRankPollers.has(pid)) return;
        const timer = setInterval(() => {
            refreshQueueRank(pid);
        }, 6000);
        summaryQueueRankPollers.set(pid, timer);
        refreshQueueRank(pid);
    }

    function stopQueueRankPolling(pid) {
        const timer = summaryQueueRankPollers.get(pid);
        if (timer) {
            clearInterval(timer);
            summaryQueueRankPollers.delete(pid);
        }
    }

    function refreshQueueRank(pid) {
        const ui = readinglistSummaryUI.get(pid);
        if (!ui || !ui.state || !ui.state.taskId) return;
        fetchTaskStatus(ui.state.taskId).then(data => {
            if (!data) return;
            if (data.status && data.status !== 'queued') {
                ui.state.queueRank = 0;
                ui.state.queueTotal = 0;
                ui.state.taskId = '';
                updateSummaryBadge(
                    ui.badge,
                    ui.state.status,
                    ui.state.lastError,
                    ui.state.queueRank,
                    ui.state.queueTotal
                );
                stopQueueRankPolling(pid);
                return;
            }
            ui.state.queueRank = Number(data.queue_rank || 0);
            ui.state.queueTotal = Number(data.queue_total || 0);
            updateSummaryBadge(
                ui.badge,
                ui.state.status,
                ui.state.lastError,
                ui.state.queueRank,
                ui.state.queueTotal
            );
        });
    }

    function buildAddedTimeLine(addedTime) {
        const line = document.createElement('div');
        line.className = 'rl-added-time-line';

        const label = createTextElement('span', 'rl-meta-label', 'Added at:');
        line.appendChild(label);

        const timeText = addedTime ? formatDate(addedTime) : '-';
        line.appendChild(createTextElement('span', 'rl-added-time-pill', timeText));

        return line;
    }

    function buildRelatedTagsLine(topTags) {
        const line = document.createElement('div');
        line.className = 'rl-related-tags';
        const label = createTextElement('span', 'rl-meta-label', 'Related Tags:');
        line.appendChild(label);

        const tagsList = Array.isArray(topTags) ? topTags.slice(0, 3) : [];
        const tagsWrap = document.createElement('div');
        tagsWrap.className = 'rel_utags rl-related-tags-list';

        if (tagsList.length === 0) {
            tagsWrap.appendChild(createTextElement('span', 'rl-meta-empty', '-'));
            line.appendChild(tagsWrap);
            return line;
        }

        tagsList.forEach(tag => {
            const tagWrap = document.createElement('div');
            tagWrap.className = 'rel_utag rl-related-tag-pill';
            tagWrap.appendChild(createLinkElement(buildTagUrl(tag), null, tag));
            tagsWrap.appendChild(tagWrap);
        });

        line.appendChild(tagsWrap);
        return line;
    }

    function createTagDropdown(paper) {
        const container = document.createElement('div');
        container.className = 'rl-tag-dropdown-root';
        const api =
            global.ArxivSanityTagDropdown && global.ArxivSanityTagDropdown.mount
                ? global.ArxivSanityTagDropdown.mount(container, {
                      pid: paper.id,
                      selectedTags: Array.isArray(paper.utags) ? paper.utags.slice() : [],
                      negativeTags: Array.isArray(paper.ntags) ? paper.ntags.slice() : [],
                      availableTags: getAvailableTags(),
                      onStateChange: st => {
                          paper.utags = Array.isArray(st.selectedTags)
                              ? st.selectedTags.slice()
                              : [];
                          paper.ntags = Array.isArray(st.negativeTags)
                              ? st.negativeTags.slice()
                              : [];
                      },
                  })
                : null;

        if (api) {
            readinglistDropdowns.set(paper.id, {
                updateAvailableTags: nextTags => api.updateAvailableTags(nextTags),
                unregister: () => api.unmount(),
                applyTagFeedback: (tagName, label) => api.applyTagFeedback(tagName, label),
                applyTagRename: (fromTag, toTag) => api.applyTagRename(fromTag, toTag),
                applyTagDelete: tagName => api.applyTagDelete(tagName),
            });
        }

        return container;
    }

    function createReadingListCard(p, container) {
        if (!container || !p) return;

        const card = document.createElement('div');
        card.className = 'rel_paper rl-paper-card';
        card.dataset.pid = p.id;

        const removeWrap = document.createElement('div');
        removeWrap.className = 'rl-remove-wrap summary-btn-group';

        const removeBtn = document.createElement('div');
        removeBtn.className = 'readinglist-btn active rl-remove-btn';
        removeBtn.title = 'Remove from reading list';
        removeBtn.textContent = '✕';
        removeBtn.addEventListener('click', function (event) {
            event.stopPropagation();
            openRemoveConfirm(p.id, this);
        });

        removeWrap.appendChild(removeBtn);
        card.appendChild(removeWrap);

        // Title
        const titleDiv = document.createElement('div');
        titleDiv.className = 'rel_title';
        titleDiv.appendChild(
            createLinkElement(
                'http://arxiv.org/abs/' + encodeURIComponent(p.id),
                null,
                p.title || p.id,
                '_blank'
            )
        );
        card.appendChild(titleDiv);

        // Authors (unified truncation)
        const authorsFull = String(p.authors || '');
        let authorsText = authorsFull;
        try {
            if (global.ArxivSanityAuthors && global.ArxivSanityAuthors.format) {
                authorsText = global.ArxivSanityAuthors.format(authorsFull, {
                    maxAuthors: 10,
                    head: 5,
                    tail: 3,
                }).text;
            }
        } catch (e) {}
        const authorsEl = createTextElement('div', 'rel_authors', authorsText);
        if (authorsFull) authorsEl.title = authorsFull;
        card.appendChild(authorsEl);

        if (p.time) {
            card.appendChild(createTextElement('div', 'rel_time rl-paper-time', p.time));
        }
        card.appendChild(createTextElement('div', 'rel_tags', p.tags || ''));

        const statusBadge = document.createElement('div');
        updateSummaryBadge(statusBadge, p.summary_status || '', p.summary_last_error || '', 0, 0);
        card.appendChild(statusBadge);

        // TL;DR section (prioritize over abstract)
        if (p.tldr) {
            const tldrDiv = document.createElement('div');
            tldrDiv.className = 'rel_tldr';
            const tldrLabel = document.createElement('div');
            tldrLabel.className = 'tldr_label';
            tldrLabel.textContent = '💡 TL;DR';
            const tldrText = document.createElement('div');
            tldrText.className = 'tldr_text';
            tldrText.innerHTML = renderTldrMarkdown(p.tldr);
            tldrDiv.appendChild(tldrLabel);
            tldrDiv.appendChild(tldrText);
            card.appendChild(tldrDiv);
            triggerMathJax(tldrDiv);
        } else if (p.summary) {
            // Abstract (only show if no TL;DR) - now with markdown rendering
            const absDiv = document.createElement('div');
            absDiv.className = 'rel_abs';
            absDiv.innerHTML = renderAbstractMarkdown(p.summary);
            card.appendChild(absDiv);
            triggerMathJax(absDiv);
        }

        card.appendChild(buildAddedTimeLine(p.added_time));
        card.appendChild(buildRelatedTagsLine(p.top_tags));

        if (typeof global.user !== 'undefined' && global.user) {
            const utagsWrap = document.createElement('div');
            utagsWrap.className = 'rel_utags';
            utagsWrap.appendChild(createTagDropdown(p));
            card.appendChild(utagsWrap);
        }

        // Actions
        const actions = document.createElement('div');
        actions.className = 'paper-actions-footer';

        const triggerWrap = document.createElement('div');
        triggerWrap.className = 'rel_summary';
        const triggerBtn = document.createElement('button');
        triggerBtn.className = 'summary-trigger-btn';
        triggerBtn.textContent = '✨ Generate Summary';
        triggerBtn.title = 'Generate summary';
        triggerWrap.appendChild(triggerBtn);

        const similarWrap = document.createElement('div');
        similarWrap.className = 'rel_more';
        similarWrap.appendChild(
            createLinkElement(
                '/?rank=pid&pid=' + encodeURIComponent(p.id),
                null,
                'Similar',
                '_blank'
            )
        );

        const inspectWrap = document.createElement('div');
        inspectWrap.className = 'rel_inspect';
        inspectWrap.appendChild(
            createLinkElement('/inspect?pid=' + encodeURIComponent(p.id), null, 'Inspect', '_blank')
        );

        const summaryWrap = document.createElement('div');
        summaryWrap.className = 'rel_summary';
        summaryWrap.appendChild(
            createLinkElement('/summary?pid=' + encodeURIComponent(p.id), null, 'Summary', '_blank')
        );

        const alphaWrap = document.createElement('div');
        alphaWrap.className = 'rel_alphaxiv';
        alphaWrap.appendChild(
            createLinkElement(
                'https://www.alphaxiv.org/overview/' + encodeURIComponent(p.id),
                null,
                'alphaXiv',
                '_blank'
            )
        );

        const coolWrap = document.createElement('div');
        coolWrap.className = 'rel_cool';
        coolWrap.appendChild(
            createLinkElement(
                'https://papers.cool/arxiv/' + encodeURIComponent(p.id),
                null,
                'Cool',
                '_blank'
            )
        );

        const summaryState = {
            status: p.summary_status || '',
            lastError: p.summary_last_error || '',
            taskId: p.summary_task_id ? String(p.summary_task_id) : '',
            queueRank: 0,
            queueTotal: 0,
        };

        const syncTriggerState = () => {
            triggerBtn.disabled = !canTriggerSummary(summaryState.status);
            triggerBtn.title = triggerBtn.disabled
                ? 'Summary already available or generating'
                : 'Generate summary';
        };

        triggerBtn.addEventListener('click', function () {
            if (!canTriggerSummary(summaryState.status)) return;
            summaryState.status = 'queued';
            summaryState.lastError = '';
            updateSummaryBadge(
                statusBadge,
                summaryState.status,
                summaryState.lastError,
                summaryState.queueRank,
                summaryState.queueTotal
            );
            syncTriggerState();
            markSummaryPending(p.id);

            csrfFetch('/api/trigger_paper_summary', {
                method: 'POST',
                body: JSON.stringify({ pid: p.id }),
            })
                .then(resp => resp.json())
                .then(data => {
                    if (data && data.success) {
                        summaryState.status = data.status || 'queued';
                        summaryState.lastError = data.last_error || '';
                        summaryState.taskId = data.task_id ? String(data.task_id) : '';
                        if (summaryState.taskId && summaryState.status === 'queued') {
                            startQueueRankPolling(p.id);
                        }
                        if (summaryState.status === 'queued' || summaryState.status === 'running') {
                            markSummaryPending(p.id);
                        } else {
                            unmarkSummaryPending(p.id);
                        }
                    } else {
                        summaryState.status = 'failed';
                        summaryState.lastError = (data && data.error) || 'Unknown error';
                        summaryState.taskId = '';
                        summaryState.queueRank = 0;
                        summaryState.queueTotal = 0;
                        stopQueueRankPolling(p.id);
                        unmarkSummaryPending(p.id);
                        alert('Failed to trigger summary: ' + summaryState.lastError);
                    }
                    updateSummaryBadge(
                        statusBadge,
                        summaryState.status,
                        summaryState.lastError,
                        summaryState.queueRank,
                        summaryState.queueTotal
                    );
                    syncTriggerState();
                })
                .catch(err => {
                    console.error('Error triggering summary:', err);
                    summaryState.status = 'failed';
                    summaryState.lastError = String(err);
                    summaryState.taskId = '';
                    summaryState.queueRank = 0;
                    summaryState.queueTotal = 0;
                    stopQueueRankPolling(p.id);
                    unmarkSummaryPending(p.id);
                    updateSummaryBadge(
                        statusBadge,
                        summaryState.status,
                        summaryState.lastError,
                        summaryState.queueRank,
                        summaryState.queueTotal
                    );
                    syncTriggerState();
                    alert('Network error, failed to trigger summary');
                });
        });

        syncTriggerState();

        readinglistSummaryUI.set(p.id, {
            badge: statusBadge,
            state: summaryState,
            syncTriggerState,
            card: card,
        });

        actions.appendChild(triggerWrap);
        actions.appendChild(similarWrap);
        actions.appendChild(inspectWrap);
        actions.appendChild(summaryWrap);
        actions.appendChild(alphaWrap);
        actions.appendChild(coolWrap);
        card.appendChild(actions);

        container.appendChild(card);
    }

    document.addEventListener('DOMContentLoaded', function () {
        const container = document.getElementById('rl-papers');
        if (!container || !papers) {
            setupUserEventStream();
            return;
        }

        papers.forEach(function (p) {
            createReadingListCard(p, container);
            if (p && (p.summary_status === 'queued' || p.summary_status === 'running')) {
                markSummaryPending(p.id);
                if (p.summary_task_id) {
                    startQueueRankPolling(p.id);
                }
            }
        });

        updateEmptyState();

        setupUserEventStream();
    });
})(typeof window !== 'undefined' ? window : this);
