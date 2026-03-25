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
    const showToast = CommonUtils.showToast;
    const showConfirm = CommonUtils.showConfirm;
    const sharedShowSimilarPapersModal = CommonUtils.showSimilarPapersModal;

    // Summary status polling (shared from common_utils.js)
    const markSummaryPending = CommonUtils.markSummaryPending;
    const unmarkSummaryPending = CommonUtils.unmarkSummaryPending;
    const canTriggerSummary = CommonUtils.canTriggerSummary;
    const formatSummaryStatus = CommonUtils.formatSummaryStatus;
    const hasMathContent = CommonUtils.hasMathContent;
    const isSummaryModelMatch =
        CommonUtils.isSummaryModelMatch ||
        function () {
            return true;
        };
    const fetchTaskStatus = CommonUtils.fetchTaskStatus;

    const readinglistDropdowns = new Map();
    const readinglistSummaryUI = new Map();
    const pageRefs = {
        rlContainer: null,
        rlEmptyState: null,
        uploadedContainer: null,
        uploadedEmptyState: null,
    };

    function getReadingListContainer() {
        if (!pageRefs.rlContainer) {
            pageRefs.rlContainer = document.getElementById('rl-papers');
        }
        return pageRefs.rlContainer;
    }

    function getReadingListEmptyState() {
        if (!pageRefs.rlEmptyState) {
            pageRefs.rlEmptyState = document.getElementById('rl-empty-state');
        }
        return pageRefs.rlEmptyState;
    }

    function getUploadedContainer() {
        if (!pageRefs.uploadedContainer) {
            pageRefs.uploadedContainer = document.getElementById('uploaded-papers');
        }
        return pageRefs.uploadedContainer;
    }

    function getUploadedEmptyState() {
        if (!pageRefs.uploadedEmptyState) {
            pageRefs.uploadedEmptyState = document.getElementById('uploaded-empty-state');
        }
        return pageRefs.uploadedEmptyState;
    }

    function notify(message, type) {
        if (typeof showToast === 'function') {
            showToast(String(message || ''), { type: type || 'error' });
            return;
        }
        console.warn(String(message || ''));
    }

    let pageIsUnloading = false;
    if (typeof global.addEventListener === 'function') {
        global.addEventListener('pagehide', function () {
            pageIsUnloading = true;
        });
    }

    function shouldSuppressMutationError(error) {
        const message = String((error && error.message) || error || '');
        return pageIsUnloading && /failed to fetch|networkerror|load failed/i.test(message);
    }

    function escapeCssAttrValue(value) {
        const text = String(value == null ? '' : value);
        try {
            if (global.CSS && typeof global.CSS.escape === 'function')
                return global.CSS.escape(text);
        } catch (e) {}
        // Fallback: ensure the value cannot break out of the quoted attribute selector.
        return text
            .replace(/\\/g, '\\\\')
            .replace(/"/g, '\\"')
            .replace(/\n/g, '\\n')
            .replace(/\r/g, '\\r')
            .replace(/\f/g, '\\f')
            .replace(/\0/g, '\ufffd');
    }

    // Register callback for summary status updates
    CommonUtils.setSummaryStatusCallback(function (pid, status, lastError, taskId, model) {
        if (!isSummaryModelMatch(model)) return;
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

    const dateFormatter = new Intl.DateTimeFormat();
    const timeFormatter = new Intl.DateTimeFormat([], {
        hour: '2-digit',
        minute: '2-digit',
    });

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

            // When summary is ready, fetch and display TL;DR for uploaded papers
            if (status === 'ok' && ui.card && pid.startsWith('up_')) {
                fetchAndDisplayTldr(pid, ui);
            }
        });
    }

    // Fetch TL;DR from server and update display
    function fetchAndDisplayTldr(pid, ui) {
        fetch(`/api/uploaded_papers/tldr/${encodeURIComponent(pid)}`)
            .then(resp => resp.json())
            .then(data => {
                if (data && data.success && data.tldr) {
                    updateTldrDisplay(ui, data.tldr);
                }
            })
            .catch(err => {
                console.error('Failed to fetch TL;DR:', err);
            });
    }

    // Update TL;DR display in the card
    function updateTldrDisplay(ui, tldr) {
        if (!ui || !ui.card || !tldr) return;

        const card = ui.card;
        const existingDetails = ui.abstractDetailsEl || null;
        const existingAbs = ui.abstractEl || null;
        const existingAbsHtml = existingAbs ? existingAbs.innerHTML : '';

        // Check if TL;DR already exists
        let tldrDiv = ui.tldrEl || null;
        if (tldrDiv) {
            // Update existing TL;DR
            let tldrText = ui.tldrTextEl || null;
            if (!tldrText) {
                tldrText = tldrDiv.querySelector('.tldr_text');
                if (tldrText) ui.tldrTextEl = tldrText;
            }
            if (tldrText) {
                tldrText.innerHTML = renderTldrMarkdown(tldr);
                triggerMathJax(tldrDiv);
            }
        } else {
            // Create new TL;DR element
            tldrDiv = document.createElement('div');
            tldrDiv.className = 'rel_tldr';
            const tldrLabel = document.createElement('div');
            tldrLabel.className = 'tldr_label';
            tldrLabel.textContent = '💡 TL;DR';
            const tldrText = document.createElement('div');
            tldrText.className = 'tldr_text';
            tldrText.innerHTML = renderTldrMarkdown(tldr);
            tldrDiv.appendChild(tldrLabel);
            tldrDiv.appendChild(tldrText);
            ui.tldrEl = tldrDiv;
            ui.tldrTextEl = tldrText;

            // Insert before tags section
            const utagsWrap = ui.utagsWrap || null;
            if (utagsWrap) {
                utagsWrap.insertAdjacentElement('beforebegin', tldrDiv);
            } else {
                // Fallback: append to card
                card.appendChild(tldrDiv);
            }
            triggerMathJax(tldrDiv);
        }

        // Ensure abstract stays accessible (collapsed) when TL;DR exists.
        // For uploaded papers, TL;DR can arrive asynchronously via live updates.
        const hasAbstractHtml = Boolean(existingAbsHtml && existingAbsHtml.trim());
        const hasAbstractText = Boolean(
            ui.paperData && ui.paperData.summary && String(ui.paperData.summary).trim()
        );
        if (hasAbstractHtml || hasAbstractText) {
            const utagsWrap = ui.utagsWrap || null;
            let details = existingDetails;
            if (!details) {
                details = document.createElement('details');
                details.className = 'rel_abs_details';
                const summaryEl = document.createElement('summary');
                summaryEl.className = 'rel_abs_summary';
                summaryEl.textContent = 'Abstract';
                details.appendChild(summaryEl);
                details.dataset.mathjaxBound = '1';
                details.addEventListener('toggle', function () {
                    if (
                        details.open &&
                        hasMathContent((ui.paperData && ui.paperData.summary) || '')
                    ) {
                        triggerMathJax(details);
                    }
                });

                if (tldrDiv) {
                    tldrDiv.insertAdjacentElement('afterend', details);
                } else if (utagsWrap) {
                    utagsWrap.insertAdjacentElement('beforebegin', details);
                } else {
                    card.appendChild(details);
                }
                ui.abstractDetailsEl = details;
            } else if (!details.dataset.mathjaxBound) {
                details.dataset.mathjaxBound = '1';
                details.addEventListener('toggle', function () {
                    if (
                        details.open &&
                        hasMathContent((ui.paperData && ui.paperData.summary) || '')
                    ) {
                        triggerMathJax(details);
                    }
                });
            }

            let absDiv = ui.abstractEl || null;
            if (!absDiv) {
                const existingInDetails = details.querySelector('.rel_abs');
                if (existingInDetails) {
                    absDiv = existingInDetails;
                } else {
                    absDiv = document.createElement('div');
                    absDiv.className = 'rel_abs';
                    details.appendChild(absDiv);
                }
                ui.abstractEl = absDiv;
            }
            if (existingAbs && existingAbs !== absDiv && !details.contains(existingAbs)) {
                absDiv.innerHTML = existingAbsHtml;
                existingAbs.remove();
                ui.abstractEl = absDiv;
            }
            if (absDiv && absDiv.parentElement !== details) {
                details.appendChild(absDiv);
            }
            if (!existingAbsHtml && hasAbstractText) {
                absDiv.innerHTML = renderAbstractMarkdown(ui.paperData.summary);
            } else if (existingAbsHtml) {
                absDiv.innerHTML = existingAbsHtml;
            }
        }

        // Update paper data reference
        if (ui.paperData) {
            ui.paperData.tldr = tldr;
        }
    }

    function handleReadingListEvent(event) {
        if (!event || !event.pid) return;
        if (event.action === 'add') {
            if (String(event.pid).indexOf('up_') === 0) {
                fetchUploadedPapers(true).catch(function () {});
                return;
            }
            const safePid = escapeCssAttrValue(event.pid);
            const existing = document.querySelector(`.rl-paper-card[data-pid="${safePid}"]`);
            if (!existing) {
                const container = getReadingListContainer();
                if (!container) return;
                fetch('/api/readinglist/paper?pid=' + encodeURIComponent(event.pid))
                    .then(resp => resp.json())
                    .then(data => {
                        if (!data || !data.success || !data.paper) return;
                        if (!Array.isArray(papers)) papers = [];
                        papers = [data.paper].concat(
                            papers.filter(x => x && x.id !== data.paper.id)
                        );
                        const card = createReadingListCard(data.paper, container, {
                            prepend: true,
                        });
                        if (
                            card &&
                            (data.paper.summary_status === 'queued' ||
                                data.paper.summary_status === 'running')
                        ) {
                            markSummaryPending(data.paper.id);
                            if (data.paper.summary_task_id) {
                                startQueueRankPolling(data.paper.id);
                            }
                        }
                        updateEmptyState();
                    })
                    .catch(() => {});
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
            if (!isSummaryModelMatch(event.model)) return;
            updateSummaryStatusFromEvent(event.pid, event.status, event.error, event);
        } else if (event.type === 'readinglist_changed') {
            handleReadingListEvent(event);
        } else if (event.type === 'upload_parse_status') {
            handleUploadParseStatusEvent(event);
        } else if (event.type === 'upload_extract_status') {
            handleUploadExtractStatusEvent(event);
        } else if (event.type === 'upload_deleted') {
            handleUploadDeletedEvent(event);
        }
        void options;
    }

    function setupUserEventStream() {
        CommonUtils.registerEventHandler(handleUserEvent);
        CommonUtils.setupUserEventStream(global.user, applyUserState);
    }

    function openRemoveConfirm(pid, element) {
        if (!element) return;
        const ui = readinglistSummaryUI.get(pid);
        const paperTitle = ui && ui.titleLink ? String(ui.titleLink.textContent || '').trim() : '';
        const desc = paperTitle
            ? `Remove “${paperTitle}” from your reading list?`
            : 'Remove this paper from your reading list?';
        if (typeof showConfirm !== 'function') {
            notify('Confirmation dialog is unavailable. Please refresh and try again.');
            return;
        }
        showConfirm({
            title: 'Remove from reading list?',
            message: desc,
            detail: 'You can add it back anytime.',
            detailTone: 'muted',
            confirmText: 'Remove',
            cancelText: 'Cancel',
            danger: true,
        }).then(confirmed => {
            if (confirmed) {
                performRemoveFromReadingList(pid, element);
            }
        });
    }

    function performRemoveFromReadingList(pid, element) {
        const card = element ? element.closest('.rl-paper-card') : null;
        const ui = readinglistSummaryUI.get(pid);
        const removeBtn =
            (ui && ui.removeBtn) || (card ? card.querySelector('.rl-remove-btn') : null);
        if (removeBtn) {
            removeBtn.classList.add('disabled');
            removeBtn.setAttribute('aria-disabled', 'true');
            removeBtn.title = 'Removing...';
        }
        csrfFetch('/api/readinglist/remove', {
            method: 'POST',
            keepalive: true,
            body: JSON.stringify({ pid: pid }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (card) {
                        stopQueueRankPolling(pid);
                        const dropdownApi = readinglistDropdowns.get(pid);
                        if (dropdownApi && typeof dropdownApi.unregister === 'function') {
                            try {
                                dropdownApi.unregister();
                            } catch (e) {}
                        }
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
                    if (removeBtn) {
                        removeBtn.classList.remove('disabled');
                        removeBtn.removeAttribute('aria-disabled');
                        removeBtn.title = '';
                    }
                    notify('Failed to remove: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                if (shouldSuppressMutationError(err)) {
                    return;
                }
                console.error('Error removing from reading list:', err);
                if (removeBtn) {
                    removeBtn.classList.remove('disabled');
                    removeBtn.removeAttribute('aria-disabled');
                    removeBtn.title = '';
                }
                notify('Failed to remove paper');
            });
    }

    function updateEmptyState() {
        const container = getReadingListContainer();
        const emptyState = getReadingListEmptyState();
        const hasCards = Boolean(container && container.querySelector('.rl-paper-card'));

        if (emptyState) {
            emptyState.style.display = hasCards ? 'none' : 'block';
        }
    }

    function formatDate(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp * 1000);
        return dateFormatter.format(date) + ' ' + timeFormatter.format(date);
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
            tagWrap.style.cursor = 'default';

            const tagText = createTextElement('span', null, tag);
            tagWrap.appendChild(tagText);

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

    function createReadingListCard(p, container, options = {}) {
        if (!container || !p) return;
        const linkPid = p.versioned_id || p.id;

        const card = document.createElement('div');
        card.className = 'rel_paper rl-paper-card';
        card.dataset.pid = p.id;

        const removeWrap = document.createElement('div');
        removeWrap.className = 'rl-remove-wrap summary-btn-group';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'readinglist-btn active rl-remove-btn';
        removeBtn.title = 'Remove from reading list';
        removeBtn.setAttribute('aria-label', 'Remove from reading list');
        removeBtn.textContent = '✕';
        removeBtn.addEventListener('click', function (event) {
            event.stopPropagation();
            if (
                removeBtn.classList.contains('disabled') ||
                removeBtn.getAttribute('aria-disabled') === 'true'
            ) {
                return;
            }
            openRemoveConfirm(p.id, this);
        });

        removeWrap.appendChild(removeBtn);
        card.appendChild(removeWrap);

        // Title
        const titleDiv = document.createElement('div');
        titleDiv.className = 'rel_title';
        const titleLink = createLinkElement(
            'https://arxiv.org/abs/' + encodeURIComponent(linkPid),
            null,
            p.title || p.id,
            '_blank'
        );
        titleDiv.appendChild(titleLink);
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
        const hasTldr = Boolean(p.tldr && String(p.tldr).trim());
        if (hasTldr) {
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

            // Abstract: collapsed by default when TL;DR exists
            if (p.summary) {
                const details = document.createElement('details');
                details.className = 'rel_abs_details';
                const summaryEl = document.createElement('summary');
                summaryEl.className = 'rel_abs_summary';
                summaryEl.textContent = 'Abstract';
                const absDiv = document.createElement('div');
                absDiv.className = 'rel_abs';
                absDiv.innerHTML = renderAbstractMarkdown(p.summary);
                details.appendChild(summaryEl);
                details.appendChild(absDiv);
                details.addEventListener('toggle', function () {
                    if (details.open && hasMathContent(p.summary || '')) {
                        triggerMathJax(details);
                    }
                });
                card.appendChild(details);
            }
        } else if (p.summary) {
            // Abstract (only show if no TL;DR) - now with markdown rendering
            const absDiv = document.createElement('div');
            absDiv.className = 'rel_abs';
            absDiv.innerHTML = renderAbstractMarkdown(p.summary);
            card.appendChild(absDiv);
            if (hasMathContent(p.summary || '')) {
                triggerMathJax(absDiv);
            }
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

        const primaryActions = document.createElement('div');
        primaryActions.className = 'paper-actions-group paper-actions-group-primary';

        const secondaryActions = document.createElement('div');
        secondaryActions.className = 'paper-actions-group paper-actions-group-secondary';

        const triggerWrap = document.createElement('div');
        triggerWrap.className = 'rel_summary_trigger';
        const triggerBtn = document.createElement('button');
        triggerBtn.className = 'summary-trigger-btn';
        triggerBtn.textContent = '✨ Generate Summary';
        triggerBtn.title = 'Generate summary';
        triggerBtn.setAttribute('aria-label', 'Generate summary');
        triggerWrap.appendChild(triggerBtn);

        const similarWrap = document.createElement('div');
        similarWrap.className = 'rel_more';
        similarWrap.appendChild(
            createLinkElement(
                '/?rank=pid&pid=' + encodeURIComponent(linkPid),
                null,
                'Similar',
                '_blank'
            )
        );

        const inspectWrap = document.createElement('div');
        inspectWrap.className = 'rel_inspect';
        inspectWrap.appendChild(
            createLinkElement(
                '/inspect?pid=' + encodeURIComponent(linkPid),
                null,
                'Inspect',
                '_blank'
            )
        );

        const summaryWrap = document.createElement('div');
        summaryWrap.className = 'rel_summary';
        summaryWrap.appendChild(
            createLinkElement(
                '/summary?pid=' + encodeURIComponent(linkPid),
                null,
                'Summary',
                '_blank'
            )
        );

        const alphaWrap = document.createElement('div');
        alphaWrap.className = 'rel_alphaxiv';
        alphaWrap.appendChild(
            createLinkElement(
                'https://www.alphaxiv.org/overview/' + encodeURIComponent(linkPid),
                null,
                'alphaXiv',
                '_blank'
            )
        );

        const coolWrap = document.createElement('div');
        coolWrap.className = 'rel_cool';
        coolWrap.appendChild(
            createLinkElement(
                'https://papers.cool/arxiv/' + encodeURIComponent(linkPid),
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
                            notify('Summary generation started', 'success');
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
                        notify('Failed to trigger summary: ' + summaryState.lastError);
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
                    notify('Network error, failed to trigger summary');
                });
        });

        syncTriggerState();

        readinglistSummaryUI.set(p.id, {
            badge: statusBadge,
            state: summaryState,
            syncTriggerState,
            card: card,
            titleLink: titleLink,
            removeBtn: removeBtn,
            paperData: p,
        });

        primaryActions.appendChild(triggerWrap);
        primaryActions.appendChild(similarWrap);
        primaryActions.appendChild(inspectWrap);
        primaryActions.appendChild(summaryWrap);
        secondaryActions.appendChild(alphaWrap);
        secondaryActions.appendChild(coolWrap);
        actions.appendChild(primaryActions);
        actions.appendChild(secondaryActions);
        card.appendChild(actions);

        if (options && options.prepend && container.firstChild) {
            container.insertBefore(card, container.firstChild);
        } else {
            container.appendChild(card);
        }
        return card;
    }

    // =========================================================================
    // Uploaded Papers Section
    // =========================================================================

    let uploadedPapers = [];
    const uploadedDropdowns = new Map();
    const uploadedSummaryUI = new Map();
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

    // Handle upload_parse_status SSE event
    function handleUploadParseStatusEvent(event) {
        if (!event || !event.pid) return;
        const ui = uploadedSummaryUI.get(event.pid);
        if (!ui) return;

        const status = event.status || '';
        const error = event.error || '';

        // Update paper data reference
        if (ui.paperData) {
            ui.paperData.parse_status = status;
            if (error) ui.paperData.parse_error = error;
        }

        // Update parse status badge
        if (ui.parseStatusBadge) {
            ui.parseStatusBadge.className = 'parse-status-badge';
            if (status === 'ok') {
                ui.parseStatusBadge.textContent = '✓ Parsed';
                ui.parseStatusBadge.classList.add('ok');
                ui.parseStatusBadge.title = '';
            } else if (status === 'running' || status === 'queued') {
                ui.parseStatusBadge.textContent = '⏳ Parsing...';
                ui.parseStatusBadge.classList.add('running');
                ui.parseStatusBadge.title = '';
            } else if (status === 'failed') {
                ui.parseStatusBadge.textContent = '✗ Parse Failed';
                ui.parseStatusBadge.classList.add('failed');
                ui.parseStatusBadge.title = error;
            }
        }

        // Update parse button
        if (ui.parseBtn) {
            if (status === 'ok') {
                ui.parseBtn.disabled = true;
                ui.parseBtn.classList.add('disabled');
                ui.parseBtn.textContent = '⚡ Process';
                ui.parseBtn.title = 'Already parsed';
            } else if (status === 'running' || status === 'queued') {
                ui.parseBtn.disabled = true;
                ui.parseBtn.classList.add('disabled');
                ui.parseBtn.textContent = '⏳ Processing...';
            } else if (status === 'failed') {
                ui.parseBtn.disabled = true;
                ui.parseBtn.classList.add('disabled');
                ui.parseBtn.textContent = '⚡ Process';
                ui.parseBtn.title = 'Parse failed: use Retry Parse';
            }
        }

        // Update dependent buttons via updateParseStatus
        if (typeof ui.updateParseStatus === 'function') {
            ui.updateParseStatus(status);
        }

        // Remove from pending ops if completed
        if (status === 'ok' || status === 'failed') {
            uploadedPendingOps.delete(event.pid);
            if (!uploadedPendingOps.size) {
                stopUploadedPendingPolling();
            }
        }
    }

    // Handle upload_extract_status SSE event
    function handleUploadExtractStatusEvent(event) {
        if (!event || !event.pid) return;
        const ui = uploadedSummaryUI.get(event.pid);
        if (!ui) return;

        const status = event.status || '';

        // Update extract button
        if (ui.extractBtn) {
            if (status === 'ok') {
                ui.extractBtn.disabled = true;
                ui.extractBtn.classList.add('disabled');
                ui.extractBtn.textContent = '🔍 Extract Info';
                ui.extractBtn.title = 'Metadata already extracted';
            } else if (status === 'running') {
                ui.extractBtn.disabled = true;
                ui.extractBtn.classList.add('disabled');
                ui.extractBtn.textContent = '⏳ Extracting...';
            } else if (status === 'failed') {
                ui.extractBtn.disabled = false;
                ui.extractBtn.classList.remove('disabled');
                ui.extractBtn.textContent = '🔍 Extract Info';
                ui.extractBtn.title = 'Extract metadata with LLM';
            }
        }

        // Update paper data and UI if extraction succeeded
        if (status === 'ok' && event.meta_extracted_ok) {
            // Update paper data reference
            if (ui.paperData) {
                ui.paperData.meta_extracted_ok = true;
                if (event.title) ui.paperData.title = event.title;
                if (event.authors) ui.paperData.authors = event.authors;
                if (event.abstract) ui.paperData.summary = event.abstract;
            }

            // Update title display
            if (ui.titleLink && event.title) {
                ui.titleLink.textContent = event.title;
            }

            // Update authors display
            if (ui.authorsEl && event.authors) {
                ui.authorsEl.textContent = event.authors;
            } else if (!ui.authorsEl && event.authors && ui.card) {
                // Create authors element if it doesn't exist
                const titleDiv = ui.titleDiv;
                if (titleDiv) {
                    const authorsEl = document.createElement('div');
                    authorsEl.className = 'rel_authors';
                    authorsEl.textContent = event.authors;
                    titleDiv.insertAdjacentElement('afterend', authorsEl);
                    ui.authorsEl = authorsEl;
                }
            }

            // Update abstract display (only if no TL;DR)
            if (event.abstract && ui.card) {
                const tldrDiv = ui.tldrEl || null;
                if (!tldrDiv) {
                    let absDiv = ui.abstractEl || null;
                    if (!absDiv) {
                        // Create abstract element if it doesn't exist
                        absDiv = document.createElement('div');
                        absDiv.className = 'rel_abs';
                        const utagsWrap = ui.utagsWrap || null;
                        if (utagsWrap) {
                            utagsWrap.insertAdjacentElement('beforebegin', absDiv);
                        } else {
                            ui.card.appendChild(absDiv);
                        }
                        ui.abstractEl = absDiv;
                    }
                    absDiv.innerHTML = renderAbstractMarkdown(event.abstract);
                    triggerMathJax(absDiv);
                }
            }

            // Update Similar and Inspect buttons (now enabled if parse is also ok)
            // Check parse_status from paperData for robustness
            const parseOk = ui.paperData && ui.paperData.parse_status === 'ok';
            if (parseOk) {
                if (ui.similarBtn) {
                    ui.similarBtn.disabled = false;
                    ui.similarBtn.classList.remove('disabled');
                    ui.similarBtn.title = 'Find similar arXiv papers';
                }
                if (ui.inspectLink) {
                    ui.inspectLink.classList.remove('disabled-link');
                    ui.inspectLink.title = 'Inspect TF-IDF features';
                }
                if (ui.summaryLink) {
                    ui.summaryLink.classList.remove('disabled-link');
                    ui.summaryLink.title = 'View summary';
                }
            }
        }

        // Remove from pending ops if completed
        if (status === 'ok' || status === 'failed') {
            uploadedPendingOps.delete(event.pid);
            if (!uploadedPendingOps.size) {
                stopUploadedPendingPolling();
            }
        }
    }

    function handleUploadDeletedEvent(event) {
        if (!event || !event.pid) return;
        const pid = String(event.pid || '').trim();
        if (!pid) return;

        const dropdownApi = uploadedDropdowns.get(pid);
        if (dropdownApi && typeof dropdownApi.unregister === 'function') {
            try {
                dropdownApi.unregister();
            } catch (e) {}
        }
        uploadedDropdowns.delete(pid);

        const ui = uploadedSummaryUI.get(pid);
        uploadedSummaryUI.delete(pid);

        // Stop pollers to prevent timer leaks.
        uploadedPendingOps.delete(pid);
        if (!uploadedPendingOps.size) {
            stopUploadedPendingPolling();
        }
        stopQueueRankPolling(pid);
        unmarkSummaryPending(pid);

        const safePid = escapeCssAttrValue(pid);
        const card =
            (ui && ui.card) ||
            document.querySelector(`.rl-paper-card.uploaded-paper-card[data-pid="${safePid}"]`);
        if (card && card.parentNode) {
            card.parentNode.removeChild(card);
        }

        uploadedPapers = (uploadedPapers || []).filter(p => p && p.id !== pid);
        updateUploadedEmptyState();
    }

    function updateUploadedEmptyState() {
        const container = getUploadedContainer();
        const emptyState = getUploadedEmptyState();
        const hasCards = Boolean(container && container.querySelector('.rl-paper-card'));

        if (emptyState) {
            emptyState.style.display = hasCards ? 'none' : 'block';
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
        const container = getUploadedContainer();
        if (!container) return;
        const nextById = new Map();
        (uploadedPapers || []).forEach(p => {
            if (p && p.id) nextById.set(String(p.id), p);
        });

        // Remove cards that no longer exist.
        Array.from(uploadedSummaryUI.keys()).forEach(pid => {
            if (!nextById.has(String(pid))) {
                handleUploadDeletedEvent({ pid });
            }
        });

        // Add or update cards.
        (uploadedPapers || []).forEach(p => {
            if (!p || !p.id) return;
            const pid = String(p.id);
            const ui = uploadedSummaryUI.get(pid);
            if (!ui) {
                createUploadedPaperCard(p, container);
                return;
            }

            // Best-effort state sync for cases where SSE is delayed/missed.
            if (ui.paperData) {
                ui.paperData.title = p.title;
                ui.paperData.authors = p.authors;
                ui.paperData.parse_status = p.parse_status;
                ui.paperData.parse_error = p.parse_error;
                ui.paperData.meta_extracted_ok = p.meta_extracted_ok;
            }

            // Keep parse/extract UI in sync.
            if (p.parse_status) {
                handleUploadParseStatusEvent({
                    pid,
                    status: p.parse_status,
                    error: p.parse_error || '',
                });
            }
            if (p.meta_extracted_ok === true) {
                handleUploadExtractStatusEvent({
                    pid,
                    status: 'ok',
                    meta_extracted_ok: true,
                    title: p.title || '',
                    authors: p.authors || '',
                    abstract: p.summary || '',
                });
            }

            // Keep summary status badge in sync (shared handler supports uploaded cards).
            if (p.summary_status) {
                updateSummaryStatusFromEvent(pid, p.summary_status, p.summary_last_error || '', {
                    task_id: p.summary_task_id || '',
                });
            }
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

        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'readinglist-btn active rl-remove-btn';
        deleteBtn.title = 'Delete uploaded paper';
        deleteBtn.setAttribute('aria-label', 'Delete uploaded paper');
        deleteBtn.textContent = '✕';
        deleteBtn.addEventListener('click', function (event) {
            event.stopPropagation();
            showUploadDeleteConfirm(p.id, card);
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
        let authorsEl = null;
        if (p.authors) {
            authorsEl = createTextElement('div', 'rel_authors', p.authors);
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
            const uploadedTimeLabel = uploadedTimeLine.querySelector('.rl-meta-label');
            if (uploadedTimeLabel) {
                uploadedTimeLabel.textContent = 'Uploaded at:';
            }
            card.appendChild(uploadedTimeLine);
        }

        // TL;DR section (prioritize over abstract if available)
        const hasTldr = Boolean(p.tldr && String(p.tldr).trim());
        let tldrDiv = null;
        let tldrTextEl = null;
        let abstractDetailsEl = null;
        let abstractEl = null;
        if (hasTldr) {
            tldrDiv = document.createElement('div');
            tldrDiv.className = 'rel_tldr';
            const tldrLabel = document.createElement('div');
            tldrLabel.className = 'tldr_label';
            tldrLabel.textContent = '💡 TL;DR';
            tldrTextEl = document.createElement('div');
            tldrTextEl.className = 'tldr_text';
            tldrTextEl.innerHTML = renderTldrMarkdown(p.tldr);
            tldrDiv.appendChild(tldrLabel);
            tldrDiv.appendChild(tldrTextEl);
            card.appendChild(tldrDiv);
            triggerMathJax(tldrDiv);

            // Abstract: collapsed by default when TL;DR exists
            if (p.summary) {
                abstractDetailsEl = document.createElement('details');
                abstractDetailsEl.className = 'rel_abs_details';
                const summaryEl = document.createElement('summary');
                summaryEl.className = 'rel_abs_summary';
                summaryEl.textContent = 'Abstract';
                abstractEl = document.createElement('div');
                abstractEl.className = 'rel_abs';
                abstractEl.innerHTML = renderAbstractMarkdown(p.summary);
                abstractDetailsEl.appendChild(summaryEl);
                abstractDetailsEl.appendChild(abstractEl);
                abstractDetailsEl.addEventListener('toggle', function () {
                    if (abstractDetailsEl.open) triggerMathJax(abstractDetailsEl);
                });
                card.appendChild(abstractDetailsEl);
            }
        } else if (p.summary) {
            // Fallback to abstract if no TL;DR
            abstractEl = document.createElement('div');
            abstractEl.className = 'rel_abs';
            abstractEl.innerHTML = renderAbstractMarkdown(p.summary);
            card.appendChild(abstractEl);
            triggerMathJax(abstractEl);
        }

        // Tag dropdown for uploaded papers (same 3-state behavior)
        const utagsWrap = document.createElement('div');
        utagsWrap.className = 'rel_utags';
        utagsWrap.appendChild(createUploadedTagDropdown(p));
        card.appendChild(utagsWrap);

        // Actions
        const actions = document.createElement('div');
        actions.className = 'paper-actions-footer';

        const primaryActions = document.createElement('div');
        primaryActions.className = 'paper-actions-group paper-actions-group-primary';

        // Generate Summary button
        const triggerWrap = document.createElement('div');
        triggerWrap.className = 'rel_summary_trigger';
        const triggerBtn = document.createElement('button');
        triggerBtn.className = 'summary-trigger-btn';
        triggerBtn.textContent = '✨ Generate Summary';
        triggerBtn.title = 'Generate summary';
        triggerBtn.setAttribute('aria-label', 'Generate summary');
        triggerWrap.appendChild(triggerBtn);

        // Similar and Inspect require both parse and metadata extraction
        const metaExtracted = p.meta_extracted_ok === true;
        const featureDisabled = p.parse_status !== 'ok' || !metaExtracted;
        const summaryDisabled = p.parse_status !== 'ok';

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
        inspectLink.addEventListener('click', function (e) {
            if (inspectLink.classList.contains('disabled-link')) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
        if (featureDisabled) {
            inspectLink.classList.add('disabled-link');
            inspectLink.title =
                p.parse_status !== 'ok'
                    ? 'Parse PDF first to inspect features'
                    : 'Extract metadata first to inspect features';
        } else {
            inspectLink.title = 'Inspect TF-IDF features';
        }
        inspectWrap.appendChild(inspectLink);

        // Summary link
        const summaryWrap = document.createElement('div');
        summaryWrap.className = 'rel_summary';
        const summaryLink = createLinkElement(
            '/summary?pid=' + encodeURIComponent(p.id),
            null,
            'Summary',
            '_blank'
        );
        summaryLink.addEventListener('click', function (e) {
            if (summaryLink.classList.contains('disabled-link')) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
        if (summaryDisabled) {
            summaryLink.classList.add('disabled-link');
            summaryLink.title =
                p.parse_status !== 'ok' ? 'Parse PDF first to view summary' : 'View summary';
        } else {
            summaryLink.title = 'View summary';
        }
        summaryWrap.appendChild(summaryLink);

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
        card.insertBefore(statusBadge, filenameDiv);

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

            // Update Summary link state
            if (featureDisabled) {
                summaryLink.classList.add('disabled-link');
                summaryLink.title =
                    currentParseStatus !== 'ok'
                        ? 'Parse PDF first to view summary'
                        : 'Extract metadata first to view summary';
            } else {
                summaryLink.classList.remove('disabled-link');
                summaryLink.title = 'View summary';
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
                            notify('Summary generation started', 'success');
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
                        notify('Failed to trigger summary: ' + summaryState.lastError);
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
                    notify('Network error, failed to trigger summary');
                });
        });

        // Declare extractBtn early so updateParseStatus can reference it
        let extractBtn = null;

        syncTriggerState();

        primaryActions.appendChild(triggerWrap);
        primaryActions.appendChild(summaryWrap);
        primaryActions.appendChild(similarWrap);
        primaryActions.appendChild(inspectWrap);

        // Process button (parse + extract + summary). Disabled once parsed.
        const parseWrap = document.createElement('div');
        parseWrap.className = 'rel_parse';
        const parseBtn = document.createElement('button');
        parseBtn.className = 'action-btn parse-btn';
        parseBtn.textContent = '⚡ Process';
        parseBtn.title = 'Parse + Extract Info + Summary (one-click)';
        if (p.parse_status === 'ok') {
            parseBtn.disabled = true;
            parseBtn.classList.add('disabled');
            parseBtn.title = 'Already parsed';
        } else if (p.parse_status === 'failed') {
            // Prefer the dedicated retry button for failed state to avoid confusion.
            parseBtn.disabled = true;
            parseBtn.classList.add('disabled');
            parseBtn.title = 'Parse failed: use Retry Parse';
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
        primaryActions.appendChild(parseWrap);

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
        primaryActions.appendChild(extractWrap);

        // Retry parse (if failed)
        if (p.parse_status === 'failed') {
            const retryWrap = document.createElement('div');
            retryWrap.className = 'rel_retry';
            const retryBtn = document.createElement('button');
            retryBtn.className = 'retry-parse-btn';
            retryBtn.textContent = '🔄 Retry Parse';
            retryBtn.addEventListener('click', function () {
                retryParse(p.id, parseStatusBadge, retryBtn, parseBtn, extractBtn);
            });
            retryWrap.appendChild(retryBtn);
            primaryActions.appendChild(retryWrap);
        }

        actions.appendChild(primaryActions);
        card.appendChild(actions);

        uploadedSummaryUI.set(p.id, {
            badge: statusBadge,
            state: summaryState,
            syncTriggerState,
            updateParseStatus,
            card: card,
            titleDiv: titleDiv,
            parseStatusBadge: parseStatusBadge,
            parseBtn: parseBtn,
            extractBtn: extractBtn,
            similarBtn: similarBtn,
            inspectLink: inspectLink,
            summaryLink: summaryLink,
            titleLink: titleLink,
            authorsEl: authorsEl,
            filenameDiv: filenameDiv,
            utagsWrap: utagsWrap,
            tldrEl: tldrDiv,
            tldrTextEl: tldrTextEl,
            abstractDetailsEl: abstractDetailsEl,
            abstractEl: abstractEl,
            removeBtn: deleteBtn,
            paperData: p,
        });

        container.appendChild(card);
    }

    function showUploadDeleteConfirm(pid, cardElement) {
        if (typeof showConfirm !== 'function') {
            notify('Confirmation dialog is unavailable. Please refresh and try again.');
            return;
        }
        showConfirm({
            title: 'Delete this uploaded paper?',
            message: 'This will permanently remove the PDF and all associated data.',
            detail: 'This action cannot be undone.',
            confirmText: 'Delete',
            cancelText: 'Cancel',
            danger: true,
        }).then(confirmed => {
            if (confirmed) {
                deleteUploadedPaper(pid, cardElement);
            }
        });
    }

    function deleteUploadedPaper(pid, cardElement) {
        let reverted = false;
        const rollback = () => {
            if (reverted) return;
            reverted = true;
            if (!cardElement) return;
            try {
                delete cardElement.dataset.deleting;
            } catch (e) {}
            cardElement.style.opacity = '';
            cardElement.style.pointerEvents = '';
        };

        if (cardElement) {
            try {
                cardElement.dataset.deleting = '1';
            } catch (e) {}
            cardElement.style.opacity = '0.6';
            cardElement.style.pointerEvents = 'none';
        }

        csrfFetch('/api/uploaded_papers/delete', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => {
                if (!resp.ok) {
                    return resp.text().then(text => {
                        throw new Error(`HTTP ${resp.status}: ${text}`);
                    });
                }
                return resp.json();
            })
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
                    notify('Uploaded paper deleted', 'success');
                } else {
                    rollback();
                    notify('Failed to delete: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Error deleting uploaded paper:', err);
                rollback();
                notify('Failed to delete paper: ' + err.message);
            });
    }

    function triggerParse(pid, statusBadge, parseBtn) {
        if (parseBtn) {
            parseBtn.disabled = true;
            parseBtn.textContent = '⏳ Parsing...';
        }

        csrfFetch('/api/uploaded_papers/process', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => resp.json())
            .then(data => {
                if (data.success) {
                    markUploadedPending(pid, 'parse');
                    notify('Processing started', 'success');
                    if (statusBadge) {
                        statusBadge.textContent = '⏳ Parsing...';
                        statusBadge.className = 'parse-status-badge running';
                    }
                } else {
                    notify('Failed to process: ' + (data.error || 'Unknown error'));
                    if (parseBtn) {
                        parseBtn.disabled = false;
                        parseBtn.textContent = '⚡ Process';
                    }
                }
            })
            .catch(err => {
                console.error('Error triggering parse:', err);
                notify('Failed to start processing');
                if (parseBtn) {
                    parseBtn.disabled = false;
                    parseBtn.textContent = '⚡ Process';
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
                    notify('Metadata extraction started', 'success');
                } else {
                    notify('Failed to extract: ' + (data.error || 'Unknown error'));
                    if (extractBtn) {
                        extractBtn.disabled = false;
                        extractBtn.textContent = '🔍 Extract Info';
                    }
                }
            })
            .catch(err => {
                console.error('Error triggering extract:', err);
                notify('Failed to trigger extraction');
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
                    sharedShowSimilarPapersModal(data.papers);
                } else if (data.success && (!data.papers || data.papers.length === 0)) {
                    notify(
                        'No similar papers found. This may happen if the paper content is too short or unique.',
                        'info'
                    );
                } else {
                    notify('Failed to find similar papers: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error('Error finding similar papers:', err);
                notify('Failed to find similar papers');
                if (similarBtn) {
                    similarBtn.disabled = false;
                    similarBtn.textContent = 'Similar';
                }
            });
    }

    function retryParse(pid, statusBadge, retryBtn, parseBtn, extractBtn) {
        if (retryBtn) retryBtn.disabled = true;
        if (parseBtn) {
            parseBtn.disabled = true;
            parseBtn.classList.add('disabled');
            parseBtn.textContent = '⏳ Processing...';
        }
        if (extractBtn) {
            extractBtn.disabled = true;
            extractBtn.classList.add('disabled');
        }

        csrfFetch('/api/uploaded_papers/retry_parse', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        })
            .then(resp => resp.json())
            .then(data => {
                if (data.success) {
                    markUploadedPending(pid, 'parse');
                    notify('Retry started', 'success');
                    if (statusBadge) {
                        statusBadge.textContent = '⏳ Parsing...';
                        statusBadge.className = 'parse-status-badge running';
                    }
                    if (retryBtn) retryBtn.style.display = 'none';
                } else {
                    notify('Failed to retry: ' + (data.error || 'Unknown error'));
                    if (retryBtn) retryBtn.disabled = false;
                    if (parseBtn) {
                        parseBtn.disabled = true;
                        parseBtn.classList.add('disabled');
                        parseBtn.textContent = '⚡ Process';
                        parseBtn.title = 'Parse failed: use Retry Parse';
                    }
                }
            })
            .catch(err => {
                console.error('Error retrying parse:', err);
                notify('Failed to retry parse');
                if (retryBtn) retryBtn.disabled = false;
                if (parseBtn) {
                    parseBtn.disabled = true;
                    parseBtn.classList.add('disabled');
                    parseBtn.textContent = '⚡ Process';
                    parseBtn.title = 'Parse failed: use Retry Parse';
                }
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
                notify('Please select a PDF file', 'warning');
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                notify('File too large (max 50MB)', 'warning');
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
        xhr.timeout = 300000; // 5 minute timeout for large file uploads

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

        xhr.ontimeout = function () {
            uploadBtn.style.display = 'inline-flex';
            uploadProgress.style.display = 'none';
            notify('Upload failed: Request timed out. Please try again.');
        };

        xhr.onload = function () {
            // Handle HTTP error status codes
            if (xhr.status === 413) {
                uploadBtn.style.display = 'inline-flex';
                uploadProgress.style.display = 'none';
                notify('Upload failed: File too large. Please upload a smaller file.');
                return;
            }

            if (xhr.status >= 400) {
                uploadBtn.style.display = 'inline-flex';
                uploadProgress.style.display = 'none';
                try {
                    const data = JSON.parse(xhr.responseText);
                    notify('Upload failed: ' + (data.error || `Server error (${xhr.status})`));
                } catch (e) {
                    notify('Upload failed: Server error (' + xhr.status + ')');
                }
                return;
            }

            try {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    if (data.pid) markUploadedPending(data.pid, 'parse');
                    uploadBtn.style.display = 'inline-flex';
                    // Keep progress visible briefly so "Processing..." is actually visible.
                    progressFill.style.width = '100%';
                    progressText.textContent = 'Processing...';
                    fetchUploadedPapers()
                        .catch(() => {})
                        .finally(() => {
                            uploadProgress.style.display = 'none';
                        });
                    notify('Upload complete. Processing started.', 'success');
                } else {
                    uploadBtn.style.display = 'inline-flex';
                    uploadProgress.style.display = 'none';
                    notify('Upload failed: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                uploadBtn.style.display = 'inline-flex';
                uploadProgress.style.display = 'none';
                notify('Upload failed: Invalid response');
            }
        };

        xhr.onerror = function () {
            uploadBtn.style.display = 'inline-flex';
            uploadProgress.style.display = 'none';
            notify('Upload failed: Network error');
        };

        xhr.send(formData);
    }

    document.addEventListener('DOMContentLoaded', function () {
        const container = getReadingListContainer();
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
