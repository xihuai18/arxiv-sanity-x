'use strict';

// Reading list page logic extracted from templates/readinglist.html
// Version: 2026-01-26 - Added Parse and Extract Info buttons
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
        uploadedDropdowns.forEach(dropdownApi => {
            if (dropdownApi && typeof dropdownApi.updateAvailableTags === 'function') {
                dropdownApi.updateAvailableTags(available);
            }
        });
    }

    function fetchUserStateAndApply() {
        return fetchUserState().then(applyUserState);
    }

    function updateSummaryStatusFromEvent(pid, status, error, event) {
        const targets = [];
        const readingUi = readinglistSummaryUI.get(pid);
        if (readingUi) targets.push(readingUi);
        if (typeof uploadedSummaryUI !== 'undefined') {
            const uploadedUi = uploadedSummaryUI.get(pid);
            if (uploadedUi) targets.push(uploadedUi);
        }
        if (!targets.length) return;
        targets.forEach(ui => {
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
        });
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
                uploadedDropdowns.forEach(api => {
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
                uploadedDropdowns.forEach(api => {
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
                const readinglistApi = readinglistDropdowns.get(event.pid);
                if (readinglistApi && typeof readinglistApi.applyTagFeedback === 'function') {
                    readinglistApi.applyTagFeedback(event.tag, event.label);
                }
                const uploadedApi = uploadedDropdowns.get(event.pid);
                if (uploadedApi && typeof uploadedApi.applyTagFeedback === 'function') {
                    uploadedApi.applyTagFeedback(event.tag, event.label);
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
        const content = document.createElement('div');
        content.className = 'confirm-content';

        const titleEl = document.createElement('strong');
        titleEl.textContent = 'Remove from reading list?';

        const descEl = document.createElement('p');
        descEl.textContent = desc;

        const hintEl = document.createElement('p');
        hintEl.textContent = 'You can add it back anytime.';

        const actionsEl = document.createElement('div');
        actionsEl.className = 'confirm-actions';

        const yesBtn = document.createElement('button');
        yesBtn.className = 'confirm-btn confirm-yes';
        yesBtn.type = 'button';
        yesBtn.textContent = 'Remove';
        yesBtn.addEventListener('click', confirmRemoveFromReadingList);

        const noBtn = document.createElement('button');
        noBtn.className = 'confirm-btn confirm-no';
        noBtn.type = 'button';
        noBtn.textContent = 'Cancel';
        noBtn.addEventListener('click', closeRemoveConfirm);

        actionsEl.appendChild(yesBtn);
        actionsEl.appendChild(noBtn);
        content.appendChild(titleEl);
        content.appendChild(descEl);
        content.appendChild(hintEl);
        content.appendChild(actionsEl);
        popup.appendChild(content);

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
                        stopQueueRankPolling(pid);
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
        if (target) {
            el.target = target;
            if (target === '_blank') el.rel = 'noopener noreferrer';
        }
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
        const ui = readinglistSummaryUI.get(pid) || uploadedSummaryUI.get(pid);
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

    function createUploadedTagDropdown(paper) {
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
            uploadedDropdowns.set(paper.id, {
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

    // =========================================================================
    // Uploaded Papers Section
    // =========================================================================

    let uploadedPapers = [];
    const uploadedDropdowns = new Map();
    const uploadedSummaryUI = new Map();
    let activeUploadDeleteConfirmCleanup = null;
    const uploadedPendingOps = new Map(); // pid -> { kind: 'parse'|'extract', startedAt }
    let uploadedPendingPoller = null;
    let uploadedPendingPollInFlight = false;

    function markUploadedPending(pid, kind) {
        const key = String(pid || '').trim();
        if (!key) return;
        uploadedPendingOps.set(key, { kind: kind || 'parse', startedAt: Date.now() });
        startUploadedPendingPolling();
    }

    function stopUploadedPendingPolling() {
        if (uploadedPendingPoller) {
            clearInterval(uploadedPendingPoller);
            uploadedPendingPoller = null;
        }
    }

    function reconcileUploadedPendingOps(nextPapers) {
        if (!uploadedPendingOps.size) return;
        const byId = new Map();
        (nextPapers || []).forEach(p => {
            if (p && p.id) byId.set(String(p.id), p);
        });

        const now = Date.now();
        uploadedPendingOps.forEach((info, pid) => {
            const paper = byId.get(pid);
            if (!paper) {
                uploadedPendingOps.delete(pid);
                return;
            }
            const kind = info && info.kind ? String(info.kind) : 'parse';
            const startedAt = info && info.startedAt ? Number(info.startedAt) : now;
            if (kind === 'parse') {
                const ps = String(paper.parse_status || '');
                if (ps === 'ok' || ps === 'failed') {
                    uploadedPendingOps.delete(pid);
                }
            } else if (kind === 'extract') {
                if (paper.meta_extracted_ok === true) {
                    uploadedPendingOps.delete(pid);
                } else if (now - startedAt > 3 * 60 * 1000) {
                    uploadedPendingOps.delete(pid);
                }
            }
        });

        if (!uploadedPendingOps.size) {
            stopUploadedPendingPolling();
        }
    }

    function startUploadedPendingPolling() {
        if (uploadedPendingPoller) return;
        uploadedPendingPoller = setInterval(() => {
            if (!uploadedPendingOps.size) {
                stopUploadedPendingPolling();
                return;
            }
            if (uploadedPendingPollInFlight) return;
            uploadedPendingPollInFlight = true;
            fetchUploadedPapers(true).finally(() => {
                uploadedPendingPollInFlight = false;
            });
        }, 10000);

        if (!uploadedPendingPollInFlight) {
            uploadedPendingPollInFlight = true;
            fetchUploadedPapers(true).finally(() => {
                uploadedPendingPollInFlight = false;
            });
        }
    }

    function seedUploadedPendingOpsFromServer(nextPapers) {
        if (!Array.isArray(nextPapers) || nextPapers.length === 0) return;
        let added = false;
        nextPapers.forEach(p => {
            if (!p || !p.id) return;
            const pid = String(p.id);
            if (uploadedPendingOps.has(pid)) return;
            const ps = p.parse_status ? String(p.parse_status) : '';
            if (ps === 'queued' || ps === 'running') {
                uploadedPendingOps.set(pid, { kind: 'parse', startedAt: Date.now() });
                added = true;
            }
        });
        if (added) startUploadedPendingPolling();
    }

    function updateUploadedEmptyState() {
        const container = document.getElementById('uploaded-papers');
        const emptyState = document.getElementById('uploaded-empty-state');
        const cards = container ? container.querySelectorAll('.rl-paper-card') : [];

        if (emptyState) {
            emptyState.style.display = cards.length === 0 ? 'block' : 'none';
        }
    }

    function fetchUploadedPapers(fromPoll = false) {
        return fetch('/api/uploaded_papers/list')
            .then(resp => resp.json())
            .then(data => {
                if (data.success && Array.isArray(data.papers)) {
                    uploadedPapers = data.papers;
                    seedUploadedPendingOpsFromServer(uploadedPapers);
                    renderUploadedPapers();
                    reconcileUploadedPendingOps(uploadedPapers);
                    // Also keep summary status in sync for uploaded papers.
                    uploadedPapers.forEach(p => {
                        if (!p || !p.id) return;
                        const st = p.summary_status ? String(p.summary_status) : '';
                        if (st === 'queued' || st === 'running') {
                            markSummaryPending(p.id);
                        }
                    });
                }
            })
            .catch(err => {
                if (!fromPoll) {
                    console.error('Failed to fetch uploaded papers:', err);
                }
            });
    }

    function renderUploadedPapers() {
        const container = document.getElementById('uploaded-papers');
        if (!container) return;

        uploadedDropdowns.forEach(api => {
            if (api && typeof api.unregister === 'function') {
                api.unregister();
            }
        });
        uploadedDropdowns.clear();
        container.innerHTML = '';
        uploadedSummaryUI.clear();

        uploadedPapers.forEach(p => {
            createUploadedPaperCard(p, container);
        });

        updateUploadedEmptyState();
    }

    function getParseStatusBadge(status, parseError) {
        const badge = document.createElement('span');
        badge.className = 'parse-status-badge';
        if (status === 'ok') {
            badge.textContent = '✓ Parsed';
            badge.classList.add('ok');
        } else if (status === 'running' || status === 'queued') {
            badge.textContent = '⏳ Parsing...';
            badge.classList.add('running');
        } else if (status === 'failed') {
            badge.textContent = '✗ Parse Failed';
            badge.classList.add('failed');
            if (parseError) badge.title = String(parseError);
        } else {
            badge.textContent = status || 'Unknown';
            if (parseError) badge.title = String(parseError);
        }
        return badge;
    }

    function createUploadedPaperCard(p, container) {
        if (!container || !p) return;

        const card = document.createElement('div');
        card.className = 'rel_paper rl-paper-card uploaded-paper-card';
        card.dataset.pid = p.id;

        // Delete button with confirm popup
        const deleteWrap = document.createElement('div');
        deleteWrap.className = 'rl-remove-wrap summary-btn-group';

        const deleteBtn = document.createElement('div');
        deleteBtn.className = 'readinglist-btn active rl-remove-btn';
        deleteBtn.title = 'Delete uploaded paper';
        deleteBtn.textContent = '✕';
        deleteBtn.addEventListener('click', function (event) {
            event.stopPropagation();
            showUploadDeleteConfirm(deleteWrap, p.id, card);
        });

        deleteWrap.appendChild(deleteBtn);
        card.appendChild(deleteWrap);

        // Title (download PDF)
        const titleDiv = document.createElement('div');
        titleDiv.className = 'rel_title';
        const titleLink = createLinkElement(
            '/api/uploaded_papers/pdf/' + encodeURIComponent(p.id),
            null,
            p.title || p.original_filename || p.id,
            '_self'
        );
        titleDiv.appendChild(titleLink);
        card.appendChild(titleDiv);

        // Authors
        if (p.authors) {
            const authorsEl = createTextElement('div', 'rel_authors', p.authors);
            card.appendChild(authorsEl);
        }

        // No year/time display for uploaded papers

        // Parse status badge
        const parseStatusBadge = getParseStatusBadge(p.parse_status, p.parse_error);
        card.appendChild(parseStatusBadge);

        // Original filename (plain text)
        const filenameDiv = document.createElement('div');
        filenameDiv.className = 'rl-original-filename';
        filenameDiv.textContent = '📄 ' + (p.original_filename || 'Unknown file');
        card.appendChild(filenameDiv);

        // Uploaded time
        if (p.created_time) {
            const uploadedTimeLine = buildAddedTimeLine(p.created_time);
            uploadedTimeLine.querySelector('.rl-meta-label').textContent = 'Uploaded at:';
            card.appendChild(uploadedTimeLine);
        }

        // TL;DR section (prioritize over abstract if available)
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
            // Fallback to abstract if no TL;DR
            const absDiv = document.createElement('div');
            absDiv.className = 'rel_abs';
            absDiv.innerHTML = renderAbstractMarkdown(p.summary);
            card.appendChild(absDiv);
            triggerMathJax(absDiv);
        }

        // Tag dropdown for uploaded papers (same 3-state behavior)
        const utagsWrap = document.createElement('div');
        utagsWrap.className = 'rel_utags';
        utagsWrap.appendChild(createUploadedTagDropdown(p));
        card.appendChild(utagsWrap);

        // Actions
        const actions = document.createElement('div');
        actions.className = 'paper-actions-footer';

        // Generate Summary button
        const triggerWrap = document.createElement('div');
        triggerWrap.className = 'rel_summary';
        const triggerBtn = document.createElement('button');
        triggerBtn.className = 'summary-trigger-btn';
        triggerBtn.textContent = '✨ Generate Summary';
        triggerBtn.title = 'Generate summary';
        triggerWrap.appendChild(triggerBtn);

        // Similar and Inspect require both parse and metadata extraction
        const metaExtracted = p.meta_extracted_ok === true;
        const featureDisabled = p.parse_status !== 'ok' || !metaExtracted;

        // Similar button for uploaded papers
        const similarWrap = document.createElement('div');
        similarWrap.className = 'rel_more';
        const similarBtn = document.createElement('button');
        similarBtn.className = 'action-btn similar-btn';
        similarBtn.textContent = 'Similar';
        similarBtn.title = 'Find similar arXiv papers';
        if (featureDisabled) {
            similarBtn.disabled = true;
            similarBtn.classList.add('disabled');
            similarBtn.title =
                p.parse_status !== 'ok'
                    ? 'Parse PDF first to find similar papers'
                    : 'Extract metadata first to find similar papers';
        }
        similarBtn.addEventListener('click', function () {
            if (!similarBtn.disabled) {
                findSimilarPapers(p.id, similarBtn);
            }
        });
        similarWrap.appendChild(similarBtn);

        // Inspect link for uploaded papers
        const inspectWrap = document.createElement('div');
        inspectWrap.className = 'rel_inspect';
        const inspectLink = createLinkElement(
            '/inspect?pid=' + encodeURIComponent(p.id),
            null,
            'Inspect',
            '_blank'
        );
        if (featureDisabled) {
            inspectLink.classList.add('disabled-link');
            inspectLink.title =
                p.parse_status !== 'ok'
                    ? 'Parse PDF first to inspect features'
                    : 'Extract metadata first to inspect features';
            inspectLink.addEventListener('click', function (e) {
                e.preventDefault();
            });
        } else {
            inspectLink.title = 'Inspect TF-IDF features';
        }
        inspectWrap.appendChild(inspectLink);

        // Summary link
        const summaryWrap = document.createElement('div');
        summaryWrap.className = 'rel_summary';
        summaryWrap.appendChild(
            createLinkElement('/summary?pid=' + encodeURIComponent(p.id), null, 'Summary', '_blank')
        );

        // Summary state management (similar to regular papers)
        const summaryState = {
            status: p.summary_status || '',
            lastError: p.summary_last_error || '',
            taskId: p.summary_task_id ? String(p.summary_task_id) : '',
            queueRank: 0,
            queueTotal: 0,
        };

        // Summary status badge
        const statusBadge = document.createElement('div');
        updateSummaryBadge(
            statusBadge,
            summaryState.status,
            summaryState.lastError,
            summaryState.queueRank,
            summaryState.queueTotal
        );
        card.insertBefore(statusBadge, card.querySelector('.rl-original-filename'));

        // Track parse status for dependency management
        let currentParseStatus = p.parse_status;

        const syncTriggerState = () => {
            // Generate Summary requires parse to be completed
            const parseNotReady = currentParseStatus !== 'ok';
            const summaryNotReady = !canTriggerSummary(summaryState.status);
            triggerBtn.disabled = parseNotReady || summaryNotReady;
            if (parseNotReady) {
                triggerBtn.title = 'Parse PDF first before generating summary';
            } else if (summaryNotReady) {
                triggerBtn.title = 'Summary already available or generating';
            } else {
                triggerBtn.title = 'Generate summary';
            }
        };

        // Function to update parse status and sync dependent buttons
        const updateParseStatus = newStatus => {
            currentParseStatus = newStatus;
            syncTriggerState();

            // Similar and Inspect require both parse and metadata extraction
            const metaExtracted = p.meta_extracted_ok === true;
            const featureDisabled = currentParseStatus !== 'ok' || !metaExtracted;

            // Update Similar button state
            if (featureDisabled) {
                similarBtn.disabled = true;
                similarBtn.classList.add('disabled');
                similarBtn.title =
                    currentParseStatus !== 'ok'
                        ? 'Parse PDF first to find similar papers'
                        : 'Extract metadata first to find similar papers';
            } else {
                similarBtn.disabled = false;
                similarBtn.classList.remove('disabled');
                similarBtn.title = 'Find similar arXiv papers';
            }

            // Update Inspect link state
            if (featureDisabled) {
                inspectLink.classList.add('disabled-link');
                inspectLink.title =
                    currentParseStatus !== 'ok'
                        ? 'Parse PDF first to inspect features'
                        : 'Extract metadata first to inspect features';
            } else {
                inspectLink.classList.remove('disabled-link');
                inspectLink.title = 'Inspect TF-IDF features';
            }

            // Update Extract Info button state
            if (extractBtn) {
                if (p.meta_extracted_ok) {
                    extractBtn.disabled = true;
                    extractBtn.title = 'Already extracted';
                } else if (currentParseStatus !== 'ok') {
                    extractBtn.disabled = true;
                    extractBtn.title = 'Parse PDF first before extracting info';
                } else {
                    extractBtn.disabled = false;
                    extractBtn.title = 'Extract metadata with LLM';
                }
            }
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

        // Declare extractBtn early so updateParseStatus can reference it
        let extractBtn = null;

        syncTriggerState();

        uploadedSummaryUI.set(p.id, {
            badge: statusBadge,
            state: summaryState,
            syncTriggerState,
            updateParseStatus,
            card: card,
        });

        actions.appendChild(triggerWrap);
        actions.appendChild(summaryWrap);
        actions.appendChild(similarWrap);
        actions.appendChild(inspectWrap);

        // Parse button (disabled if already parsed)
        const parseWrap = document.createElement('div');
        parseWrap.className = 'rel_parse';
        const parseBtn = document.createElement('button');
        parseBtn.className = 'action-btn parse-btn';
        parseBtn.textContent = '📄 Parse';
        parseBtn.title = 'Parse PDF to Markdown with MinerU';
        if (p.parse_status === 'ok') {
            parseBtn.disabled = true;
            parseBtn.classList.add('disabled');
            parseBtn.title = 'Already parsed';
        } else if (p.parse_status === 'running' || p.parse_status === 'queued') {
            parseBtn.disabled = true;
            parseBtn.classList.add('disabled');
            parseBtn.textContent = '⏳ Parsing...';
        }
        parseBtn.addEventListener('click', function () {
            if (!parseBtn.disabled) {
                triggerParse(p.id, parseStatusBadge, parseBtn);
            }
        });
        parseWrap.appendChild(parseBtn);
        actions.appendChild(parseWrap);

        // Extract Info button (disabled if not parsed or already extracted)
        const extractWrap = document.createElement('div');
        extractWrap.className = 'rel_extract';
        extractBtn = document.createElement('button');
        extractBtn.className = 'action-btn extract-btn';
        extractBtn.textContent = '🔍 Extract Info';
        extractBtn.title = 'Extract title/authors from PDF with LLM';
        if (p.meta_extracted_ok) {
            extractBtn.disabled = true;
            extractBtn.classList.add('disabled');
            extractBtn.title = 'Metadata already extracted';
        } else if (p.parse_status !== 'ok') {
            extractBtn.disabled = true;
            extractBtn.classList.add('disabled');
            extractBtn.title = 'Parse PDF first before extracting info';
        }
        extractBtn.addEventListener('click', function () {
            if (!extractBtn.disabled) {
                triggerExtractInfo(p.id, extractBtn);
            }
        });
        extractWrap.appendChild(extractBtn);
        actions.appendChild(extractWrap);

        // Retry parse (if failed)
        if (p.parse_status === 'failed') {
            const retryWrap = document.createElement('div');
            retryWrap.className = 'rel_retry';
            const retryBtn = document.createElement('button');
            retryBtn.className = 'retry-parse-btn';
            retryBtn.textContent = '🔄 Retry Parse';
            retryBtn.addEventListener('click', function () {
                retryParse(p.id, parseStatusBadge, retryBtn);
            });
            retryWrap.appendChild(retryBtn);
            actions.appendChild(retryWrap);
        }

        card.appendChild(actions);
        container.appendChild(card);
    }

    function showUploadDeleteConfirm(wrapElement, pid, cardElement) {
        if (activeUploadDeleteConfirmCleanup) {
            activeUploadDeleteConfirmCleanup();
        }

        const popup = document.createElement('div');
        popup.className = 'confirm-popup upload-confirm-popup';
        popup.setAttribute('role', 'dialog');

        const content = document.createElement('div');
        content.className = 'confirm-content';

        const titleEl = document.createElement('strong');
        titleEl.textContent = 'Delete this uploaded paper?';

        const descEl = document.createElement('p');
        descEl.textContent = 'This will permanently remove the PDF and all associated data.';

        const warningEl = document.createElement('p');
        warningEl.className = 'confirm-warning';
        warningEl.textContent = 'This action cannot be undone!';

        const actionsEl = document.createElement('div');
        actionsEl.className = 'confirm-actions';

        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'confirm-btn confirm-yes';
        confirmBtn.setAttribute('data-action', 'confirm');
        confirmBtn.type = 'button';
        confirmBtn.textContent = 'Delete';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'confirm-btn confirm-no';
        cancelBtn.setAttribute('data-action', 'cancel');
        cancelBtn.type = 'button';
        cancelBtn.textContent = 'Cancel';

        actionsEl.appendChild(confirmBtn);
        actionsEl.appendChild(cancelBtn);
        content.appendChild(titleEl);
        content.appendChild(descEl);
        content.appendChild(warningEl);
        content.appendChild(actionsEl);
        popup.appendChild(content);

        let closed = false;
        let bindOutsideClickTimer = null;
        function cleanup() {
            if (closed) return;
            closed = true;
            if (bindOutsideClickTimer) {
                clearTimeout(bindOutsideClickTimer);
                bindOutsideClickTimer = null;
            }
            if (popup && popup.parentNode) popup.parentNode.removeChild(popup);
            document.removeEventListener('click', handleClickOutside);
            document.removeEventListener('keydown', handleEscape);
            if (activeUploadDeleteConfirmCleanup === cleanup) {
                activeUploadDeleteConfirmCleanup = null;
            }
        }
        activeUploadDeleteConfirmCleanup = cleanup;

        // Handle button clicks
        confirmBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            cleanup();
            deleteUploadedPaper(pid, cardElement);
        });

        cancelBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            cleanup();
        });

        // Close on click outside
        function handleClickOutside(e) {
            if (!popup.contains(e.target) && !wrapElement.contains(e.target)) {
                cleanup();
            }
        }
        bindOutsideClickTimer = setTimeout(function () {
            if (closed) return;
            document.addEventListener('click', handleClickOutside);
        }, 0);

        // Close on Escape key
        function handleEscape(e) {
            if (e.key === 'Escape') {
                cleanup();
            }
        }
        document.addEventListener('keydown', handleEscape);

        wrapElement.appendChild(popup);
    }

    function deleteUploadedPaper(pid, cardElement) {
        csrfFetch('/api/uploaded_papers/delete', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => resp.json())
            .then(data => {
                if (data.success) {
                    const dropdownApi = uploadedDropdowns.get(pid);
                    if (dropdownApi && typeof dropdownApi.unregister === 'function') {
                        dropdownApi.unregister();
                    }
                    uploadedDropdowns.delete(pid);
                    uploadedSummaryUI.delete(pid);
                    if (cardElement) {
                        cardElement.style.transition = 'opacity 0.3s, transform 0.3s';
                        cardElement.style.opacity = '0';
                        cardElement.style.transform = 'translateX(-20px)';
                        setTimeout(() => {
                            cardElement.remove();
                            updateUploadedEmptyState();
                        }, 300);
                    }
                    uploadedPapers = uploadedPapers.filter(p => p.id !== pid);
                } else {
                    alert('Failed to delete: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Error deleting uploaded paper:', err);
                alert('Failed to delete paper');
            });
    }

    function triggerParse(pid, statusBadge, parseBtn) {
        if (parseBtn) {
            parseBtn.disabled = true;
            parseBtn.textContent = '⏳ Parsing...';
        }

        csrfFetch('/api/uploaded_papers/parse', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => resp.json())
            .then(data => {
                if (data.success) {
                    markUploadedPending(pid, 'parse');
                    if (statusBadge) {
                        statusBadge.textContent = '⏳ Parsing...';
                        statusBadge.className = 'parse-status-badge running';
                    }
                } else {
                    alert('Failed to parse: ' + (data.error || 'Unknown error'));
                    if (parseBtn) {
                        parseBtn.disabled = false;
                        parseBtn.textContent = '📄 Parse';
                    }
                }
            })
            .catch(err => {
                console.error('Error triggering parse:', err);
                alert('Failed to trigger parse');
                if (parseBtn) {
                    parseBtn.disabled = false;
                    parseBtn.textContent = '📄 Parse';
                }
            });
    }

    function triggerExtractInfo(pid, extractBtn) {
        if (extractBtn) {
            extractBtn.disabled = true;
            extractBtn.textContent = '⏳ Extracting...';
        }

        csrfFetch('/api/uploaded_papers/extract_info', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => resp.json())
            .then(data => {
                if (data.success) {
                    markUploadedPending(pid, 'extract');
                    // Refresh the list to show updated info
                    setTimeout(function () {
                        fetchUploadedPapers();
                    }, 2000);
                } else {
                    alert('Failed to extract: ' + (data.error || 'Unknown error'));
                    if (extractBtn) {
                        extractBtn.disabled = false;
                        extractBtn.textContent = '🔍 Extract Info';
                    }
                }
            })
            .catch(err => {
                console.error('Error triggering extract:', err);
                alert('Failed to trigger extraction');
                if (extractBtn) {
                    extractBtn.disabled = false;
                    extractBtn.textContent = '🔍 Extract Info';
                }
            });
    }

    function findSimilarPapers(pid, similarBtn) {
        if (similarBtn) {
            similarBtn.disabled = true;
            similarBtn.textContent = '⏳ Finding...';
        }

        fetch('/api/uploaded_papers/similar/' + encodeURIComponent(pid))
            .then(resp => resp.json())
            .then(data => {
                if (similarBtn) {
                    similarBtn.disabled = false;
                    similarBtn.textContent = 'Similar';
                }
                if (data.success && data.papers && data.papers.length > 0) {
                    showSimilarPapersModal(data.papers);
                } else if (data.success && (!data.papers || data.papers.length === 0)) {
                    alert(
                        'No similar papers found. This may happen if the paper content is too short or unique.'
                    );
                } else {
                    alert('Failed to find similar papers: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Error finding similar papers:', err);
                alert('Failed to find similar papers');
                if (similarBtn) {
                    similarBtn.disabled = false;
                    similarBtn.textContent = 'Similar';
                }
            });
    }

    function showSimilarPapersModal(papers) {
        // Remove existing modal if any
        const existingModal = document.getElementById('similar-papers-modal');
        if (existingModal) {
            if (typeof existingModal._cleanupModal === 'function') {
                existingModal._cleanupModal();
            } else {
                existingModal.remove();
            }
        }

        const modal = document.createElement('div');
        modal.id = 'similar-papers-modal';
        modal.className = 'similar-modal-overlay';

        const buildPaperItem = (p, i) => {
            const titleSafe = escapeHtml(p.title || p.id);
            const authorsSafe = escapeHtml(p.authors || '');
            const timeSafe = escapeHtml(p.time || '');
            // Prefer TL;DR over abstract
            const contentText = p.tldr || p.abstract || '';
            const contentSafe = escapeHtml(contentText);
            const contentLabel = p.tldr ? '💡 TL;DR' : p.abstract ? 'Abstract' : '';

            return `
                <div class="similar-paper-item">
                    <span class="similar-paper-rank">${i + 1}</span>
                    <div class="similar-paper-info">
                        <a href="/summary?pid=${encodeURIComponent(p.id)}" target="_blank" rel="noopener noreferrer" class="similar-paper-title">${titleSafe}</a>
                        ${authorsSafe ? `<div class="similar-paper-authors">${authorsSafe}</div>` : ''}
                        <div class="similar-paper-meta-line">
                            ${timeSafe ? `<span class="similar-paper-time">${timeSafe}</span>` : ''}
                            <span class="similar-paper-score">Score: ${p.score.toFixed(3)}</span>
                        </div>
                        ${
                            contentSafe
                                ? `
                            <div class="similar-paper-content">
                                ${contentLabel ? `<span class="similar-paper-content-label">${contentLabel}</span>` : ''}
                                <span class="similar-paper-content-text">${contentSafe}</span>
                            </div>
                        `
                                : ''
                        }
                    </div>
                </div>
            `;
        };

        modal.innerHTML = `
            <div class="similar-modal-content">
                <div class="similar-modal-header">
                    <h3>Similar Papers (${papers.length})</h3>
                    <button class="similar-modal-close">&times;</button>
                </div>
                <div class="similar-modal-body">
                    ${papers.map((p, i) => buildPaperItem(p, i)).join('')}
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close handlers
        let modalClosed = false;
        function cleanupModal() {
            if (modalClosed) return;
            modalClosed = true;
            if (modal && modal.parentNode) modal.parentNode.removeChild(modal);
            document.removeEventListener('keydown', escHandler);
        }
        function escHandler(e) {
            if (e.key === 'Escape') cleanupModal();
        }
        modal._cleanupModal = cleanupModal;
        modal.querySelector('.similar-modal-close').addEventListener('click', cleanupModal);
        modal.addEventListener('click', e => {
            if (e.target === modal) cleanupModal();
        });
        document.addEventListener('keydown', escHandler);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function retryParse(pid, statusBadge, retryBtn) {
        if (retryBtn) retryBtn.disabled = true;

        csrfFetch('/api/uploaded_papers/retry_parse', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => resp.json())
            .then(data => {
                if (data.success) {
                    markUploadedPending(pid, 'parse');
                    if (statusBadge) {
                        statusBadge.textContent = '⏳ Parsing...';
                        statusBadge.className = 'parse-status-badge running';
                    }
                    if (retryBtn) retryBtn.style.display = 'none';
                } else {
                    alert('Failed to retry: ' + (data.error || 'Unknown error'));
                    if (retryBtn) retryBtn.disabled = false;
                }
            })
            .catch(err => {
                console.error('Error retrying parse:', err);
                alert('Failed to retry parse');
                if (retryBtn) retryBtn.disabled = false;
            });
    }

    function setupUploadUI() {
        const uploadBtn = document.getElementById('upload-btn');
        const uploadInput = document.getElementById('pdf-upload-input');
        const uploadProgress = document.getElementById('upload-progress');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        if (!uploadBtn || !uploadInput) return;

        uploadBtn.addEventListener('click', function () {
            uploadInput.click();
        });

        uploadInput.addEventListener('change', function () {
            const file = this.files[0];
            if (!file) return;

            if (!file.name.toLowerCase().endsWith('.pdf')) {
                alert('Please select a PDF file');
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                alert('File too large (max 50MB)');
                return;
            }

            uploadPdf(file, uploadBtn, uploadProgress, progressFill, progressText);
            this.value = '';
        });
    }

    function uploadPdf(file, uploadBtn, uploadProgress, progressFill, progressText) {
        uploadBtn.style.display = 'none';
        uploadProgress.style.display = 'flex';
        progressFill.style.width = '0%';
        progressText.textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload_pdf', true);

        // Add CSRF token
        const csrfToken = CommonUtils.getCsrfToken();
        if (csrfToken) {
            xhr.setRequestHeader('X-CSRF-Token', csrfToken);
        }

        xhr.upload.onprogress = function (e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = percent + '%';
                progressText.textContent = 'Uploading... ' + percent + '%';
            }
        };

        xhr.onload = function () {
            uploadBtn.style.display = 'inline-flex';
            uploadProgress.style.display = 'none';

            // Handle HTTP error status codes
            if (xhr.status === 413) {
                alert('Upload failed: File too large. Please upload a smaller file.');
                return;
            }

            if (xhr.status >= 400) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    alert('Upload failed: ' + (data.error || `Server error (${xhr.status})`));
                } catch (e) {
                    alert('Upload failed: Server error (' + xhr.status + ')');
                }
                return;
            }

            try {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    if (data.pid) markUploadedPending(data.pid, 'parse');
                    progressText.textContent = 'Processing...';
                    fetchUploadedPapers();
                } else {
                    alert('Upload failed: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Upload failed: Invalid response');
            }
        };

        xhr.onerror = function () {
            uploadBtn.style.display = 'inline-flex';
            uploadProgress.style.display = 'none';
            alert('Upload failed: Network error');
        };

        xhr.send(formData);
    }

    document.addEventListener('DOMContentLoaded', function () {
        const container = document.getElementById('rl-papers');
        if (!container || !papers) {
            setupUserEventStream();
            setupUploadUI();
            if (typeof user !== 'undefined' && user) {
                fetchUploadedPapers();
            }
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
        setupUploadUI();
        if (typeof user !== 'undefined' && user) {
            fetchUploadedPapers();
        }
    });
})(typeof window !== 'undefined' ? window : this);
