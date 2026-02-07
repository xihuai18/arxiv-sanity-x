'use strict';

// Use shared utilities from common_utils.js
var CommonUtils = window.ArxivSanityCommon;
var csrfFetch = CommonUtils.csrfFetch;
var formatAuthorsText = CommonUtils.formatAuthorsText;
var escapeHtml = CommonUtils.escapeHtml;
var checkSummaryFallback = CommonUtils.checkSummaryFallback;
var renderAbstractMarkdown = CommonUtils.renderAbstractMarkdown;
var triggerMathJax = CommonUtils.triggerMathJax;
var fetchTaskStatus =
    typeof CommonUtils.fetchTaskStatus === 'function'
        ? CommonUtils.fetchTaskStatus
        : function (taskId) {
              if (!taskId) return Promise.resolve(null);
              return fetch(`/api/task_status/${encodeURIComponent(taskId)}`)
                  .then(resp => (resp.ok ? resp.json() : null))
                  .then(data => (data && data.success ? data : null))
                  .catch(() => null);
          };

// Resolve summary markdown renderer at call-time so we can fall back gracefully
// when some frontend bundles fail to load (e.g., transient 404 / cache mismatch).
var _warnedMissingSummaryRenderer = false;
function _getSummaryMarkdownRenderer() {
    try {
        const mod = typeof window !== 'undefined' ? window.ArxivSanitySummaryMarkdown : null;
        if (mod && typeof mod.renderSummaryMarkdown === 'function')
            return mod.renderSummaryMarkdown;
    } catch (e) {}
    return null;
}

function renderSummaryMarkdown(markdown, container, tocContainer) {
    if (!container) return;

    const renderer = _getSummaryMarkdownRenderer();
    if (renderer) {
        try {
            renderer(markdown, container, tocContainer);
            return;
        } catch (e) {
            console.warn('Summary markdown renderer failed, falling back:', e);
        }
    } else if (!_warnedMissingSummaryRenderer) {
        _warnedMissingSummaryRenderer = true;
        console.warn(
            'Summary markdown renderer bundle not loaded (window.ArxivSanitySummaryMarkdown missing). ' +
                'Falling back to basic markdown rendering.'
        );
    }

    // Fallback: render with the abstract markdown renderer (no TOC, best-effort safety).
    // NOTE: renderAbstractMarkdown returns an HTML string.
    let rendered = '';
    try {
        if (renderAbstractMarkdown && typeof renderAbstractMarkdown === 'function') {
            rendered = renderAbstractMarkdown(markdown || '');
        }
    } catch (e) {
        rendered = '';
    }
    try {
        if (rendered) {
            container.innerHTML = rendered;
        } else {
            container.textContent = String(markdown || '');
        }
    } catch (e) {}

    // Clear TOC in fallback mode (to avoid stale TOC from previous renders).
    if (tocContainer) {
        try {
            tocContainer.innerHTML = '';
            if (tocContainer.classList) tocContainer.classList.add('is-empty');
        } catch (e) {}
        try {
            const contentContainer = tocContainer.closest('.summary-content');
            if (contentContainer && contentContainer.classList)
                contentContainer.classList.remove('has-toc');
        } catch (e) {}
    }

    // Mirror a bit of the main renderer cleanup to avoid stale UI elements.
    try {
        const legacy = document.querySelector('.back-to-top');
        if (legacy) legacy.remove();
    } catch (e) {}

    // Best-effort math rendering via delimiter scanning (slower, but better than raw TeX).
    try {
        if (CommonUtils && typeof CommonUtils.triggerMathJax === 'function') {
            CommonUtils.triggerMathJax(container);
        }
    } catch (e) {}
}

// Shared tag dropdown API instance
var sharedTagDropdownApi = null;

// Shared event stream from common_utils
var _setupUserEventStream = CommonUtils.setupUserEventStream;
var _registerEventHandler = CommonUtils.registerEventHandler;

function applyUserState(state) {
    if (!state || !state.success) return;
    if (Array.isArray(state.tags)) {
        summaryApp.availableTags = state.tags
            .filter(t => t && t.name && t.name !== 'all')
            .map(t => t.name)
            .sort();
        renderTagDropdown();
    }
}

function fetchUserStateAndApply() {
    return CommonUtils.fetchUserState().then(applyUserState);
}

function handleUserEvent(event, options = {}) {
    if (!event || typeof event !== 'object') return;
    if (event.type === 'user_state_changed') {
        if (event.reason === 'rename_tag' && event.from && event.to) {
            summaryApp.userTags = (summaryApp.userTags || []).map(t =>
                t === event.from ? event.to : t
            );
            summaryApp.negativeTags = (summaryApp.negativeTags || []).map(t =>
                t === event.from ? event.to : t
            );
            renderTagDropdown();
        } else if (event.reason === 'delete_tag' && event.tag) {
            summaryApp.userTags = (summaryApp.userTags || []).filter(t => t !== event.tag);
            summaryApp.negativeTags = (summaryApp.negativeTags || []).filter(t => t !== event.tag);
            renderTagDropdown();
        } else if (
            event.reason === 'tag_feedback' &&
            event.pid &&
            summaryApp.pid &&
            event.pid === summaryApp.pid
        ) {
            const pos = new Set(summaryApp.userTags || []);
            const neg = new Set(summaryApp.negativeTags || []);
            if (event.label === 1) {
                pos.add(event.tag);
                neg.delete(event.tag);
            } else if (event.label === -1) {
                pos.delete(event.tag);
                neg.add(event.tag);
            } else {
                pos.delete(event.tag);
                neg.delete(event.tag);
            }
            summaryApp.userTags = Array.from(pos);
            summaryApp.negativeTags = Array.from(neg);
            renderTagDropdown();
        }
        fetchUserStateAndApply();
    } else if (event.type === 'readinglist_changed') {
        // Keep reading list button in sync across pages/tabs.
        if (event.pid && summaryApp.pid && String(event.pid) === String(summaryApp.pid)) {
            if (event.action === 'add') {
                summaryApp.setState({ inReadingList: true });
            } else if (event.action === 'remove') {
                summaryApp.setState({ inReadingList: false });
            }
        }
    }
}

// Summary markdown rendering is handled by markdown_summary_utils.js

// Simplified state management
class SummaryState {
    constructor() {
        this.loading = false;
        this.error = null;
        this.content = null;
        this.paper = null;
        this.pid = null;
        this.meta = null;
        this.models = [];
        this.modelsError = null;
        this.regenerating = false;
        this.selectedModel = '';
        this.autoRetryCount = 0;
        this.autoRetryTimer = null;
        this.maxAutoRetries = 20;
        this.notice = '';
        this.clearing = false;
        this.defaultModel = '';
        this.pendingConfirm = null; // 'clearModel' or 'clearAll'

        // Available summaries for this paper (model cache keys that have summaries)
        this.availableSummaries = [];

        this.queueRank = 0;
        this.queueTotal = 0;
        this.lastTaskId = '';

        // Global queue stats
        this.globalQueued = 0;
        this.globalRunning = 0;

        // Track async requests to avoid stale responses overriding UI state
        this.requestSeq = 0;
        this.pendingGenerationModel = '';

        // Per-model in-flight generation tracking
        this.inflightModels = Object.create(null);

        // Track which model the currently rendered content belongs to
        this.contentModel = '';

        // Track forced regenerations so we don't stop polling on stale cached content
        // Shape: { [modelId]: { generatedAt: number|null, contentHash: string } }
        this.pendingRegenerations = Object.create(null);

        // Timer for renderMath to avoid stale callbacks (fix for issue #4)
        this._renderMathTimer = null;

        // Batch setState optimization: pending state updates and render frame
        this._pendingState = null;
        this._renderFrame = null;

        // Reading list state (arXiv papers only; uploaded papers do not use reading list)
        this.inReadingList = false;
        this.readingListPending = false;
        this._readingListRequestInFlight = false;
    }

    setState(newState) {
        // Batch multiple setState calls into a single render using requestAnimationFrame
        if (!this._pendingState) {
            this._pendingState = {};
        }
        Object.assign(this._pendingState, newState);

        // Schedule render on next animation frame if not already scheduled
        if (!this._renderFrame) {
            this._renderFrame = requestAnimationFrame(() => {
                this._renderFrame = null;
                if (this._pendingState) {
                    Object.assign(this, this._pendingState);
                    this._pendingState = null;
                    this.render();
                }
            });
        }
    }

    // Synchronous setState for cases where immediate render is needed
    setStateSync(newState) {
        // Cancel any pending batched updates and apply them first
        if (this._renderFrame) {
            cancelAnimationFrame(this._renderFrame);
            this._renderFrame = null;
        }
        if (this._pendingState) {
            Object.assign(this, this._pendingState);
            this._pendingState = null;
        }
        Object.assign(this, newState);
        this.render();
    }

    render() {
        const container = document.getElementById('wrap');
        if (!container) return;

        // Cancel any pending renderMath timer to avoid stale callbacks (fix for issue #4)
        if (this._renderMathTimer) {
            clearTimeout(this._renderMathTimer);
            this._renderMathTimer = null;
        }

        const htmlContent = this.getHTML();

        if (this.content) {
            container.innerHTML = htmlContent;
            const markdownContainer = container.querySelector('.summary-markdown');
            const tocContainer = container.querySelector('.summary-toc');
            if (markdownContainer) {
                renderSummaryMarkdown(this.content, markdownContainer, tocContainer);
            }
            // Also trigger MathJax for the abstract section
            // NOTE: We use triggerMathJax only for abstract, not for summary content
            // Summary content uses tex2chtmlPromise via renderSummaryMarkdown (fix for issue #3)
            const abstractContainer = container.querySelector('.paper-abstract');
            if (abstractContainer && triggerMathJax) {
                triggerMathJax(abstractContainer);
            }
        } else {
            container.innerHTML = htmlContent;
            this.renderMath();
        }

        // Render tag dropdown (shared implementation) after DOM update
        if (typeof user !== 'undefined' && user) {
            renderTagDropdown();
        }
    }

    renderMath() {
        // Use MathJax to render math formulas (for non-markdown content like loading/error states)
        // Cancel any previous pending timer (fix for issue #4)
        if (this._renderMathTimer) {
            clearTimeout(this._renderMathTimer);
            this._renderMathTimer = null;
        }

        this._renderMathTimer = setTimeout(() => {
            this._renderMathTimer = null;
            try {
                if (CommonUtils && typeof CommonUtils.triggerMathJax === 'function') {
                    CommonUtils.triggerMathJax();
                    return;
                }
            } catch (e) {}

            // Fallback: legacy behavior
            if (typeof MathJax !== 'undefined') {
                if (MathJax.typeset) {
                    MathJax.typeset();
                } else if (MathJax.typesetPromise) {
                    MathJax.typesetPromise().catch(err => {
                        console.warn('MathJax rendering error:', err);
                    });
                }
            }
        }, 100);
    }

    getCurrentModel() {
        return this.selectedModel || '';
    }

    isCurrentModelGenerating() {
        const current = String(this.getCurrentModel() || '').trim();
        if (!current) return false;
        if (this.pendingGenerationModel && String(this.pendingGenerationModel) === current)
            return true;
        if (this.inflightModels && this.inflightModels[current]) return true;
        return false;
    }

    getSummaryModel() {
        return '';
    }

    clearAutoRetry() {
        if (this.autoRetryTimer) {
            clearTimeout(this.autoRetryTimer);
            this.autoRetryTimer = null;
        }
    }

    scheduleAutoRetry(pid, options = {}) {
        if (this.autoRetryCount >= this.maxAutoRetries) {
            // Avoid getting stuck in "generating" state forever if the backend never produces a cache.
            const targetModel = String(options.model || '').trim();
            if (targetModel) {
                if (this.inflightModels) {
                    this.inflightModels[targetModel] = false;
                }
                if (this.pendingRegenerations) {
                    delete this.pendingRegenerations[targetModel];
                }
                if (this.pendingGenerationModel && this.pendingGenerationModel === targetModel) {
                    this.pendingGenerationModel = '';
                }
            } else {
                this.pendingGenerationModel = '';
            }
            this.setState({
                loading: false,
                regenerating: false,
                notice: 'Auto-refresh stopped. Please click Generate/Regenerate again to retry.',
            });
            return;
        }
        this.autoRetryCount += 1;
        const delayMs = Math.min(15000, 5000 + this.autoRetryCount * 2000);
        this.clearAutoRetry();
        this.autoRetryTimer = setTimeout(() => {
            const targetModel = String(options.model || '').trim();
            // Don't steal the UI back if user switched models.
            if (targetModel && String(this.getCurrentModel() || '') !== targetModel) {
                return;
            }
            if (
                targetModel &&
                this.pendingGenerationModel &&
                this.pendingGenerationModel !== targetModel
            ) {
                return;
            }
            this.refreshQueueRank();
            this.loadSummary(pid, {
                model: targetModel || '',
                force_regenerate: false,
                cache_only: true,
                auto_trigger: false,
            });
        }, delayMs);
    }

    formatTimestamp(ts) {
        if (ts === undefined || ts === null || Number.isNaN(Number(ts))) return '';
        const date = new Date(Number(ts) * 1000);
        if (Number.isNaN(date.getTime())) return '';
        return date.toLocaleString();
    }

    renderModelOptions() {
        const current = this.getCurrentModel();
        const availableSummaries = this.availableSummaries || [];

        if (!Array.isArray(this.models) || this.models.length === 0) {
            const fallbackLabel = current
                ? `${escapeHtml(current)}`
                : 'Default (server configured)';
            const value = current ? escapeHtml(current) : '';
            return `<option value="${value}">${fallbackLabel}</option>`;
        }

        let options = this.models
            .map(model => {
                const rawId = String(model.id || '');
                const id = escapeHtml(rawId);
                const selected = rawId === String(current || '') ? ' selected' : '';
                const key = modelCacheKey(rawId);
                const hasSummary = key && availableSummaries.includes(key);
                const checkmark = hasSummary ? '✓ ' : '';
                return `<option value="${id}"${selected}>${checkmark}${id}</option>`;
            })
            .join('');

        const hasCurrent = this.models.some(m => String(m.id || '') === String(current || ''));
        if (current && !hasCurrent) {
            const value = escapeHtml(current);
            const key = modelCacheKey(current);
            const hasSummary = key && availableSummaries.includes(key);
            const checkmark = hasSummary ? '✓ ' : '';
            options += `<option value="${value}" selected>${checkmark}${value}</option>`;
        }

        return options;
    }

    renderActions() {
        const currentModel = this.getCurrentModel();
        const contentModel = String(this.contentModel || '').trim();
        const hasCachedSummary = Boolean(
            this.content &&
            currentModel &&
            contentModel &&
            contentModel === String(currentModel).trim()
        );

        const isUploadedPaper = this.paper && this.paper.kind === 'upload';
        const uploadParseStatus =
            this.paper && this.paper.parse_status ? String(this.paper.parse_status) : '';
        const uploadNotParsed = isUploadedPaper && uploadParseStatus && uploadParseStatus !== 'ok';

        // Allow clearing while generating; only block actions during clearing.
        // Keep Generate disabled during loading to avoid concurrent generations.
        // For uploaded papers, require MinerU parsing to be complete.
        const disableGenerate =
            this.loading || this.clearing || this.isCurrentModelGenerating() || uploadNotParsed
                ? 'disabled'
                : '';
        const disableClear = this.clearing ? 'disabled' : '';
        // Allow switching models even while generating
        const disableSelect = this.clearing ? 'disabled' : '';
        const isGenerating = this.isCurrentModelGenerating();
        const regenLabel = isGenerating
            ? 'Generating...'
            : hasCachedSummary
              ? 'Regenerate'
              : 'Generate';
        const generateTitle = uploadNotParsed
            ? 'Parse PDF first before generating summary.'
            : hasCachedSummary
              ? 'Regenerate summary for this model (overwrites the cached summary).'
              : 'Generate summary';
        const modelOptions = this.renderModelOptions();
        const errorNote = this.modelsError
            ? `<div class="summary-note" style="color: #d9534f;" role="status" aria-live="polite">${escapeHtml(this.modelsError)}</div>`
            : '';
        const notice = this.notice
            ? `<div class="summary-note" style="color: #b8860b;" role="status" aria-live="polite">${escapeHtml(this.notice)}</div>`
            : '';

        // Queue status display - only show when loading/generating and no content
        const showQueueStatus = (this.loading || this.isCurrentModelGenerating()) && !this.content;
        let queueStatusNote = '';
        if (showQueueStatus) {
            const total = this.globalQueued + this.globalRunning;
            if (total > 0) {
                // Show position in queue: "1/3 Queued" means position 1 of 3 total
                const position = this.globalRunning > 0 ? this.globalRunning : 1;
                queueStatusNote = `<div class="summary-note summary-queue-note" style="color: #6c757d; font-size: 12px;" role="status" aria-live="polite">${position}/${total} Queued</div>`;
            }
        }

        const modelLabel = currentModel ? escapeHtml(currentModel) : 'current model';
        const inflightNote = this.isCurrentModelGenerating()
            ? `<p class="confirm-warning">Note: generation for this model is currently running. Clearing will cancel the in-flight job, but it may take a moment to stop.</p>`
            : '';

        return `
            <div class="summary-actions">
                <label class="summary-action-label" for="model-select">Model</label>
                <select id="model-select" class="summary-model-select" onchange="summaryApp.handleModelChange(event)" ${disableSelect}>
                    ${modelOptions}
                </select>
                <button onclick="summaryApp.regenerate()" class="summary-action-btn" ${disableGenerate} title="${generateTitle}">
                    ${regenLabel}
                </button>
                <div class="summary-btn-group">
                    <button onclick="summaryApp.requestClearModel()" class="summary-action-btn summary-btn-warning" ${disableClear} title="Clear summary for current model only">
                        ${this.clearing === 'model' ? 'Clearing...' : 'Clear Current Summary'}
                    </button>
                    ${
                        this.pendingConfirm === 'clearModel'
                            ? `
                        <div class="confirm-popup" role="dialog" aria-labelledby="confirm-title">
                            <div class="confirm-content">
                                <strong id="confirm-title">Clear summary for ${modelLabel}?</strong>
                                <p>This will only remove the summary generated by this model.</p>
                                ${inflightNote}
                                <div class="confirm-actions">
                                    <button onclick="summaryApp.confirmClearModel()" class="confirm-btn confirm-yes">Confirm</button>
                                    <button onclick="summaryApp.cancelConfirm()" class="confirm-btn confirm-no">Cancel</button>
                                </div>
                            </div>
                        </div>
                    `
                            : ''
                    }
                </div>
                <div class="summary-btn-group">
                    <button onclick="summaryApp.requestClearAll()" class="summary-action-btn summary-btn-danger" ${disableClear} title="Clear all caches (all models, HTML, MinerU, etc.)">
                        ${this.clearing === 'all' ? 'Clearing...' : 'Clear All'}
                    </button>
                    ${
                        this.pendingConfirm === 'clearAll'
                            ? `
                        <div class="confirm-popup" role="dialog" aria-labelledby="confirm-all-title">
                            <div class="confirm-content">
                                <strong id="confirm-all-title">Clear ALL caches for this paper?</strong>
                                <p>This will remove:</p>
                                <ul>
                                    <li>All model summaries</li>
                                    <li>HTML/Markdown cache</li>
                                    <li>MinerU cache</li>
                                    <li>All related files</li>
                                </ul>
                                <p class="confirm-warning">This action cannot be undone!</p>
                                ${inflightNote}
                                <div class="confirm-actions">
                                    <button onclick="summaryApp.confirmClearAll()" class="confirm-btn confirm-yes">Confirm</button>
                                    <button onclick="summaryApp.cancelConfirm()" class="confirm-btn confirm-no">Cancel</button>
                                </div>
                            </div>
                        </div>
                    `
                            : ''
                    }
                </div>
                <button onclick="summaryApp.exportMarkdownZip()" class="summary-action-btn summary-btn-export" ${this.content ? '' : 'disabled'} title="Export summary as Markdown ZIP">
                Export
                </button>
                ${errorNote}
                ${notice}
                ${queueStatusNote}
            </div>
        `;
    }

    renderMetaLine() {
        const timeStr = this.meta ? this.formatTimestamp(this.meta.generated_at) : '';
        const timeLabel = timeStr ? `Generated at ${escapeHtml(timeStr)}` : '';
        const note = [timeLabel].filter(Boolean).join(' · ');
        const fallbackNote = 'This summary is automatically generated by AI';

        return `
            <div class="summary-meta">
                <span class="summary-badge">AI</span>
                <span class="summary-note">${note || fallbackNote}</span>
            </div>
        `;
    }

    getHTML() {
        // Paper header section - styled like paper list with full abstract
        const pidSafe = this.paper ? String(this.paper.id || '') : '';
        const titleSafe = this.paper ? escapeHtml(this.paper.title) : '';
        const authorsFull = this.paper ? String(this.paper.authors || '') : '';
        const authorsDisplay = formatAuthorsText(authorsFull, { maxAuthors: 10, head: 5, tail: 3 });
        const authorsSafe = escapeHtml(authorsDisplay);
        const authorsTitleSafe = escapeHtml(authorsFull);
        const timeSafe = this.paper ? escapeHtml(this.paper.time) : '';
        const tagsSafe = this.paper && this.paper.tags ? escapeHtml(this.paper.tags) : '';
        // Render abstract with markdown support for math formulas
        const abstractHtml = this.paper
            ? renderAbstractMarkdown
                ? renderAbstractMarkdown(this.paper.summary || 'No abstract available.')
                : escapeHtml(this.paper.summary || 'No abstract available.')
            : '';

        // Check if user is logged in
        const isLoggedIn = typeof user !== 'undefined' && user;

        // Check if this is an uploaded paper (hide arXiv-specific links)
        const isUploadedPaper = this.paper && this.paper.kind === 'upload';

        // Reading list toggle (hide for uploaded papers; show only for logged in users)
        const showReadingList = Boolean(isLoggedIn && !isUploadedPaper);
        const rlPending = Boolean(this.readingListPending);
        const rlActive = Boolean(this.inReadingList);
        const rlBtnClass =
            'readinglist-toggle-btn' + (rlActive ? ' active' : '') + (rlPending ? ' pending' : '');
        const rlBtnIcon = rlPending ? '⏳' : rlActive ? '📖' : '🔖';
        const rlBtnTitle = rlPending
            ? 'Working...'
            : rlActive
              ? 'In reading list'
              : 'Add to reading list';
        const rlBtnDisabledAttr = rlPending ? 'disabled' : '';
        const readingListHTML = showReadingList
            ? `<div class="rel_readinglist"><button type="button" class="${rlBtnClass}" ${rlBtnDisabledAttr} onclick="summaryApp.toggleReadingList()" title="${rlBtnTitle}">${rlBtnIcon}</button></div>`
            : '';

        // Build navigation links - for uploaded papers, show Similar button; for arXiv, show all links
        const navLinksHTML = isUploadedPaper
            ? (() => {
                  const ps =
                      this.paper && this.paper.parse_status ? String(this.paper.parse_status) : '';
                  const disabled = ps && ps !== 'ok';
                  const metaExtracted = this.paper && this.paper.meta_extracted_ok === true;
                  // Similar and Inspect require both parse and metadata extraction
                  const featureDisabled = disabled || !metaExtracted;
                  const title = featureDisabled
                      ? disabled
                          ? 'Parse PDF first to find similar papers'
                          : 'Extract metadata first to find similar papers'
                      : 'Find similar arXiv papers';
                  const featureDisabledAttr = featureDisabled ? 'disabled' : '';
                  const featureDisabledClass = featureDisabled ? 'disabled' : '';
                  const inspectTitle = featureDisabled
                      ? disabled
                          ? 'Parse PDF first to inspect features'
                          : 'Extract metadata first to inspect features'
                      : 'Inspect TF-IDF features';
                  const extractDisabled = disabled || metaExtracted;
                  const extractTitle = metaExtracted
                      ? 'Metadata already extracted'
                      : disabled
                        ? 'Parse PDF first before extracting info'
                        : 'Extract title/authors from PDF with LLM';
                  const extractDisabledAttr = extractDisabled ? 'disabled' : '';
                  const extractDisabledClass = extractDisabled ? 'disabled' : '';
                  return `
                <div class="paper-nav paper-actions-footer">
                    <div class="rel_more"><button class="upload-similar-btn ${featureDisabledClass}" ${featureDisabledAttr} onclick="summaryApp.findSimilarPapers()" title="${title}">Similar</button></div>
                    <div class="rel_inspect"><a href="/inspect?pid=${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer" class="${featureDisabled ? 'disabled-link' : ''}" ${featureDisabled ? 'onclick="return false;"' : ''} title="${inspectTitle}">Inspect</a></div>
                    <div class="rel_extract"><button class="action-btn extract-btn ${extractDisabledClass}" ${extractDisabledAttr} onclick="summaryApp.extractInfo()" title="${extractTitle}">🔍 Extract Info</button></div>
                </div>`;
              })()
            : `
                <div class="paper-nav paper-actions-footer">
                    <div class="rel_more"><a href="/?rank=pid&pid=${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">Similar</a></div>
                    <div class="rel_inspect"><a href="/inspect?pid=${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">Inspect</a></div>
                    ${readingListHTML}
                    <div class="rel_alphaxiv"><a href="https://www.alphaxiv.org/overview/${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">alphaXiv</a></div>
                    <div class="rel_cool"><a href="https://papers.cool/arxiv/${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">Cool</a></div>
                </div>`;

        // Title link - for uploaded papers, link to PDF download; for arXiv papers, link to arXiv
        const titleLinkHTML = isUploadedPaper
            ? `<a href="/api/uploaded_papers/pdf/${encodeURIComponent(pidSafe)}" title="Download PDF">${titleSafe}</a>`
            : `<a href="https://arxiv.org/abs/${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">${titleSafe}</a>`;

        const headerHTML = this.paper
            ? `
            <div class="paper-header">
                ${navLinksHTML}
                <div class="paper-content-section">
                    <div class="paper-main">
                        <h1 class="paper-title">
                            ${titleLinkHTML}
                        </h1>
                        <div class="paper-authors-line">
                            <span title="${authorsTitleSafe}">${authorsSafe}</span>
                        </div>
                        <div class="paper-meta-line">
                            <span class="paper-time">${timeSafe}</span>
                            ${tagsSafe ? `<span class="paper-tags">${tagsSafe}</span>` : ''}
                        </div>
                        <div class="paper-abstract">
                            ${abstractHtml}
                        </div>
                        ${
                            isLoggedIn
                                ? `
                        <div class="paper-user-tags-section">
                            <div class="rel_utags" id="summary-tag-dropdown"></div>
                        </div>
                        `
                                : ''
                        }
                    </div>
                </div>
            </div>
        `
            : '';

        // Summary content section with logo
        let summaryHTML = '';
        if (this.loading) {
            summaryHTML = `
                <div class="summary-container" aria-busy="true">
                    <div class="summary-header">
                        <h2>Summary</h2>
                        ${this.renderMetaLine()}
                    </div>
                    ${this.renderActions()}
                    <div class="summary-content" aria-live="polite">
                        <div class="loading-indicator">
                            <div class="spinner"></div>
                            <p>Generating paper summary, please wait...</p>
                            <p class="loading-note">This may take a few moments</p>
                        </div>
                    </div>
                </div>
            `;
        } else if (this.error) {
            const err = escapeHtml(this.error);
            summaryHTML = `
                <div class="summary-container" aria-busy="false">
                    <div class="summary-header">
                        <h2>Summary</h2>
                        ${this.renderMetaLine()}
                    </div>
                    <div class="summary-content" aria-live="polite">
                        <div class="error-message" role="alert">
                            <p>⚠️ Error generating summary: ${err}</p>
                            ${this.renderActions()}
                        </div>
                    </div>
                </div>
            `;
        } else if (this.content) {
            // Don't process markdown here, let renderSummaryMarkdown in render() method handle it
            summaryHTML = `
                <div class="summary-container" aria-busy="false">
                    <div class="summary-header">
                        <h2>Summary</h2>
                        ${this.renderMetaLine()}
                    </div>
                    ${this.renderActions()}
                    <div class="summary-content" aria-live="polite">
                        <nav class="summary-toc" aria-label="Table of contents"></nav>
                        <div class="summary-markdown markdown-content"></div>
                    </div>
                </div>
            `;
        } else {
            summaryHTML = `
                <div class="summary-container" aria-busy="false">
                    <div class="summary-header">
                        <h2>Summary</h2>
                        ${this.renderMetaLine()}
                    </div>
                    <div class="summary-content" aria-live="polite">
                        <p>No summary available for this paper at the moment.</p>
                        ${this.renderActions()}
                    </div>
                </div>
            `;
        }

        return headerHTML + summaryHTML;
    }
}

// Global app instance
const summaryApp = new SummaryState();
window.summaryApp = summaryApp;

// API call function
async function fetchSummary(pid, options = {}) {
    try {
        // Create AbortController for timeout control
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout (cache-only fetch should be fast)

        const response = await csrfFetch('/api/get_paper_summary', {
            method: 'POST',
            body: JSON.stringify({
                pid: pid,
                model: options.model || undefined,
                force_regenerate: Boolean(options.force_regenerate),
                cache_only: Boolean(options.cache_only),
            }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        const data = await response.json().catch(() => null);
        if (response.ok && data && data.success) {
            // Handle cache miss (backend returns 200 with cached=false)
            if (data.cached === false) {
                const err = new Error('No cached summary available');
                err.code = 'summary_cache_miss';
                throw err;
            }
            return {
                content: data.summary_content,
                meta: data.summary_meta || {},
            };
        }

        // Extract error info even from non-2xx responses so callers can react to cache misses
        const err = new Error(
            (data && data.error) || `HTTP ${response.status}: ${response.statusText}`
        );
        if (data && data.code) {
            err.code = data.code;
        }
        err.httpStatus = response.status;
        throw err;
    } catch (error) {
        if (error.name === 'AbortError') {
            const timeoutError = new Error(
                'Request timeout, paper summarization takes time, please try again later'
            );
            timeoutError.code = 'summary_timeout';
            throw timeoutError;
        }
        if (error && (error.code === 'summary_cache_miss' || error.httpStatus === 404)) {
            throw error;
        }
        console.error('Failed to fetch summary:', error);
        throw error;
    }
}

async function clearModelSummary(pid, model) {
    const response = await csrfFetch('/api/clear_model_summary', {
        method: 'POST',
        body: JSON.stringify({ pid, model }),
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error || 'Failed to clear model summary');
    }
    return data;
}

async function clearPaperCache(pid) {
    const response = await csrfFetch('/api/clear_paper_cache', {
        method: 'POST',
        body: JSON.stringify({ pid }),
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error || 'Failed to clear cache');
    }
    return data;
}

async function triggerSummary(pid, options = {}) {
    const model = String(options.model || '').trim();
    const payload = { pid };
    if (model) payload.model = model;
    if (options.force_regenerate !== undefined) {
        payload.force_regenerate = Boolean(options.force_regenerate);
    }
    if (options.priority !== undefined && options.priority !== null) {
        payload.priority = options.priority;
    }
    const response = await csrfFetch('/api/trigger_paper_summary', {
        method: 'POST',
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error || 'Failed to trigger summary');
    }
    return data;
}

async function fetchModels() {
    const response = await fetch('/api/llm_models');
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    const models = Array.isArray(data.models) ? data.models : [];
    if (!data.success && models.length === 0) {
        throw new Error(data.error || 'Failed to load models');
    }
    return models;
}

async function fetchAvailableSummaries(pid) {
    const response = await fetch(`/api/check_paper_summaries?pid=${encodeURIComponent(pid)}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error || 'Failed to check available summaries');
    }
    return Array.isArray(data.available_models) ? data.available_models : [];
}

async function fetchSummaryStatus(pid, model) {
    const payload = { pids: [pid] };
    if (model) payload.model = model;
    const response = await csrfFetch('/api/summary_status', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    if (!data || !data.success || !data.statuses) {
        return null;
    }
    const info = data.statuses[pid] || null;
    return info || null;
}

function modelCacheKey(modelId) {
    const raw = String(modelId || '').trim();
    if (!raw) return '';
    return raw.replace(/[^a-zA-Z0-9._-]+/g, '_');
}

function hashString(input) {
    const str = String(input || '');
    // djb2 hash; stable and fast enough for UI polling comparisons
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
    }
    // Force unsigned 32-bit and stringify for easy comparisons
    return String(hash >>> 0);
}

async function fetchReadingListItems() {
    const response = await fetch('/api/readinglist/list');
    if (!response.ok) {
        const err = new Error(`HTTP ${response.status}: ${response.statusText}`);
        err.httpStatus = response.status;
        throw err;
    }
    const data = await response.json().catch(() => null);
    if (!data || !data.success) {
        throw new Error((data && data.error) || 'Failed to fetch reading list');
    }
    return Array.isArray(data.items) ? data.items : [];
}

summaryApp.refreshReadingListState = async function (pidValue) {
    // Only relevant for logged-in, arXiv papers.
    try {
        const items = await fetchReadingListItems();
        const p = String(pidValue || this.pid || '').trim();
        if (!p) return;
        const inList = items.some(it => it && String(it.pid || '') === p);
        this.setState({ inReadingList: Boolean(inList) });
    } catch (e) {
        // Ignore errors (e.g., 401 when not logged in).
    }
};

summaryApp.toggleReadingList = function () {
    const isLoggedIn = typeof user !== 'undefined' && user;
    if (!isLoggedIn) {
        alert('Please log in to use reading list.');
        return;
    }
    if (!this.paper || this.paper.kind === 'upload') return;
    if (!this.pid) return;
    if (this.readingListPending || this._readingListRequestInFlight) return;

    const pidValue = String(this.pid).trim();
    if (!pidValue) return;

    const toast =
        CommonUtils && typeof CommonUtils.showToast === 'function' ? CommonUtils.showToast : null;

    this._readingListRequestInFlight = true;
    const currentlyIn = Boolean(this.inReadingList);
    if (currentlyIn) {
        // Optimistic remove
        this.setState({ readingListPending: true, inReadingList: false });
        csrfFetch('/api/readinglist/remove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pid: pidValue }),
        })
            .then(r => r.json())
            .then(data => {
                if (data && data.success) {
                    this.setStateSync({ inReadingList: false });
                    if (toast) toast('Removed from reading list', { type: 'success' });
                } else {
                    this.setStateSync({ inReadingList: true });
                    const msg =
                        'Failed to remove from reading list: ' + ((data && data.error) || '');
                    if (toast) toast(msg, { type: 'error' });
                }
            })
            .catch(err => {
                this.setStateSync({ inReadingList: true });
                if (toast) {
                    toast('Network error: failed to remove from reading list', { type: 'error' });
                }
                console.error('Error removing from reading list:', err);
            })
            .finally(() => {
                this._readingListRequestInFlight = false;
                this.setState({ readingListPending: false });
            });
    } else {
        // Optimistic add
        this.setState({ readingListPending: true, inReadingList: true });
        csrfFetch('/api/readinglist/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pid: pidValue }),
        })
            .then(r => r.json())
            .then(data => {
                if (data && data.success) {
                    this.setStateSync({ inReadingList: true });
                    if (toast) toast('Added to reading list', { type: 'success' });

                    // Backend may enqueue summary generation when adding.
                    const taskId = data.task_id ? String(data.task_id) : '';
                    if (taskId) {
                        this.lastTaskId = taskId;
                        this.refreshQueueRank();
                    }
                } else {
                    this.setStateSync({ inReadingList: false });
                    const msg = 'Failed to add to reading list: ' + ((data && data.error) || '');
                    if (toast) toast(msg, { type: 'error' });
                }
            })
            .catch(err => {
                this.setStateSync({ inReadingList: false });
                if (toast) toast('Network error: failed to add to reading list', { type: 'error' });
                console.error('Error adding to reading list:', err);
            })
            .finally(() => {
                this._readingListRequestInFlight = false;
                this.setState({ readingListPending: false });
            });
    }
};

// Retry function
summaryApp.retry = function () {
    if (this.pid) {
        this.queueSummary(this.pid, { model: this.getCurrentModel() });
    }
};

summaryApp.handleModelChange = function (event) {
    const value = event && event.target ? event.target.value : '';
    this.lastTaskId = '';
    this.setState({ selectedModel: value, queueRank: 0, queueTotal: 0 });
    if (this.pid) {
        this.loadSummary(this.pid, { model: value, cache_only: true });
    }
};

summaryApp.regenerate = function () {
    if (!this.pid || this.loading || this.isCurrentModelGenerating()) return;

    // Guard: uploaded paper must be parsed before summary generation.
    if (this.paper && this.paper.kind === 'upload') {
        const ps = this.paper.parse_status ? String(this.paper.parse_status) : '';
        if (ps && ps !== 'ok') {
            alert('Parse PDF first before generating summary.');
            return;
        }
    }

    const currentModel = this.getCurrentModel();
    const contentModel = String(this.contentModel || '').trim();
    const hasCachedSummary = Boolean(
        this.content && currentModel && contentModel && contentModel === String(currentModel).trim()
    );
    this.queueSummary(this.pid, { model: currentModel, force_regenerate: hasCachedSummary });
};

summaryApp.requestClearModel = function () {
    if (!this.pid || this.clearing) return;

    const currentModel = this.getCurrentModel();
    if (!currentModel) {
        this.setState({ error: 'No model selected' });
        return;
    }

    this.setState({ pendingConfirm: 'clearModel' });
};

summaryApp.confirmClearModel = async function () {
    const currentModel = this.getCurrentModel();
    this.clearAutoRetry();
    this.setState({ clearing: 'model', notice: '', error: null, pendingConfirm: null });

    try {
        await clearModelSummary(this.pid, currentModel);
        // Cancel local in-flight UI state.
        if (currentModel) {
            this.inflightModels[currentModel] = false;
        }
        this.pendingGenerationModel = '';

        // Remove the cleared model from availableSummaries
        const clearedKey = modelCacheKey(currentModel);
        if (clearedKey) {
            this.availableSummaries = this.availableSummaries.filter(k => k !== clearedKey);
        }
        if (currentModel && this.pendingRegenerations) {
            delete this.pendingRegenerations[currentModel];
        }

        this.setState({
            clearing: false,
            content: null,
            meta: null,
            queueRank: 0,
            queueTotal: 0,
            lastTaskId: '',
            notice: `Summary for model "${currentModel}" cleared. Click Generate to create a new one.`,
        });
    } catch (error) {
        const friendlyMsg = CommonUtils.handleApiError(error, 'Clear Model Summary');
        this.setState({ clearing: false, error: friendlyMsg, pendingConfirm: null });
    }
};

summaryApp.refreshGlobalQueueStats = async function () {
    try {
        const resp = await fetch('/api/queue_stats');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.success) {
            this.setState({
                globalQueued: Number(data.queued || 0),
                globalRunning: Number(data.running || 0),
            });
        }
    } catch (error) {
        console.warn('Failed to fetch global queue stats:', error);
    }
};

summaryApp.refreshQueueRank = async function () {
    // Also refresh global stats
    this.refreshGlobalQueueStats();

    if (!this.lastTaskId) return;
    try {
        const data = await fetchTaskStatus(this.lastTaskId);
        if (!data) return;
        const queueRank = Number(data.queue_rank || 0);
        const queueTotal = Number(data.queue_total || 0);
        if (data.status && data.status !== 'queued') {
            this.setState({ queueRank: 0, queueTotal: 0, lastTaskId: '' });
            return;
        }
        if (queueRank > 0) {
            this.setState({ queueRank, queueTotal });
        } else {
            this.setState({ queueRank: 0, queueTotal: 0 });
        }
    } catch (error) {
        console.warn('Failed to fetch queue rank:', error);
    }
};

summaryApp.queueSummary = async function (pid, options = {}) {
    const targetModel = String(
        options.model || this.getCurrentModel() || this.defaultModel || ''
    ).trim();
    if (!pid) return;

    // Guard: uploaded paper must be parsed before summary generation.
    if (this.paper && this.paper.kind === 'upload') {
        const ps = this.paper.parse_status ? String(this.paper.parse_status) : '';
        if (ps && ps !== 'ok') {
            this.setState({
                loading: false,
                regenerating: false,
                error: 'Parse PDF first before generating summary.',
            });
            return;
        }
    }
    if (!targetModel) {
        this.setState({
            loading: false,
            regenerating: false,
            error: 'No summary model configured. Please configure a default model.',
        });
        return;
    }
    if (targetModel && this.inflightModels[targetModel]) {
        return;
    }

    const forceRegenerate =
        options.force_regenerate !== undefined
            ? Boolean(options.force_regenerate)
            : Boolean(options.force);
    this.clearAutoRetry();
    this.autoRetryCount = 0;
    this.pendingGenerationModel = targetModel || '';
    if (targetModel) {
        this.inflightModels[targetModel] = true;
    }

    const currentContentModel = String(this.contentModel || '').trim();
    const hasRenderedSummaryForTarget = Boolean(
        this.content && currentContentModel && currentContentModel === String(targetModel).trim()
    );
    if (forceRegenerate && targetModel && hasRenderedSummaryForTarget) {
        const generatedAt =
            this.meta && this.meta.generated_at !== undefined && this.meta.generated_at !== null
                ? Number(this.meta.generated_at)
                : null;
        this.pendingRegenerations[targetModel] = {
            generatedAt: Number.isFinite(generatedAt) ? generatedAt : null,
            contentHash: hashString(this.content),
        };
    }

    this.setState({
        loading: !(forceRegenerate && hasRenderedSummaryForTarget),
        regenerating: true,
        notice: forceRegenerate
            ? 'Summary is being regenerated for this model. You can switch models; this view will auto-refresh when ready.'
            : 'Summary is being generated for this model. You can switch models; this view will auto-refresh when ready.',
        error: null,
        // When regenerating, keep showing the cached version until the new one arrives.
        content: forceRegenerate && hasRenderedSummaryForTarget ? this.content : null,
        meta: forceRegenerate && hasRenderedSummaryForTarget ? this.meta : null,
        selectedModel: targetModel || this.selectedModel || '',
    });

    try {
        const triggerOptions = { model: targetModel, force_regenerate: forceRegenerate };
        if (options.priority !== undefined && options.priority !== null) {
            triggerOptions.priority = options.priority;
        }
        const triggerData = await triggerSummary(pid, triggerOptions);
        if (triggerData && triggerData.task_id) {
            this.lastTaskId = String(triggerData.task_id);
            this.refreshQueueRank();
        } else {
            this.setState({ queueRank: 0, queueTotal: 0, lastTaskId: '' });
        }
        this.scheduleAutoRetry(pid, { model: targetModel, cache_only: true });
    } catch (error) {
        const friendlyMsg = CommonUtils.handleApiError(error, 'Trigger Summary');
        if (targetModel) {
            this.inflightModels[targetModel] = false;
        }
        this.pendingGenerationModel = '';
        this.setState({
            loading: false,
            regenerating: false,
            error: friendlyMsg,
            queueRank: 0,
            queueTotal: 0,
            lastTaskId: '',
        });
    }
};

summaryApp.requestClearAll = function () {
    if (!this.pid || this.clearing) return;
    this.setState({ pendingConfirm: 'clearAll' });
};

summaryApp.confirmClearAll = async function () {
    this.clearAutoRetry();
    this.setState({ clearing: 'all', notice: '', error: null, pendingConfirm: null });

    try {
        await clearPaperCache(this.pid);
        if (this.paper && this.paper.kind === 'upload') {
            this.paper.parse_status = 'pending';
            this.paper.parse_error = 'Cache cleared, re-parsing required';
        }
        // Cancel local in-flight UI state for all models.
        this.inflightModels = Object.create(null);
        this.pendingGenerationModel = '';

        // Clear all available summaries since we cleared everything
        this.availableSummaries = [];
        this.pendingRegenerations = Object.create(null);

        this.setState({
            clearing: false,
            content: null,
            meta: null,
            queueRank: 0,
            queueTotal: 0,
            lastTaskId: '',
            notice: 'All caches cleared. Click Generate to fetch a fresh summary.',
        });
    } catch (error) {
        const friendlyMsg = CommonUtils.handleApiError(error, 'Clear All Caches');
        this.setState({ clearing: false, error: friendlyMsg, pendingConfirm: null });
    }
};

summaryApp.cancelConfirm = function () {
    this.setState({ pendingConfirm: null });
};

// Find similar papers for uploaded papers
summaryApp.findSimilarPapers = async function () {
    if (!this.paper || this.paper.kind !== 'upload') {
        return;
    }

    // Uploaded paper must be parsed before similarity can be computed.
    // Keep this aligned with readinglist.js behavior (disable when parse_status != 'ok').
    if (this.paper.parse_status && this.paper.parse_status !== 'ok') {
        alert('Parse PDF first to find similar papers.');
        return;
    }

    const pid = this.pid;
    const btn = document.querySelector('.upload-similar-btn');

    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Finding...';
    }

    try {
        const resp = await fetch('/api/uploaded_papers/similar/' + encodeURIComponent(pid));
        const data = await resp.json();

        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Similar';
        }

        if (data.success && data.papers && data.papers.length > 0) {
            this.showSimilarPapersModal(data.papers);
        } else if (data.success && (!data.papers || data.papers.length === 0)) {
            alert(
                'No similar papers found. This may happen if the paper content is too short or unique.'
            );
        } else {
            alert('Failed to find similar papers: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Error finding similar papers:', err);
        alert('Failed to find similar papers');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Similar';
        }
    }
};

summaryApp.showSimilarPapersModal = function (papers) {
    // Remove existing modal if any
    const existingModal = document.getElementById('similar-papers-modal');
    if (existingModal) {
        if (typeof existingModal._cleanupModal === 'function') {
            existingModal._cleanupModal();
        } else {
            existingModal.remove();
        }
    }

    const escapeHtml = text => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

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

    const modal = document.createElement('div');
    modal.id = 'similar-papers-modal';
    modal.className = 'similar-modal-overlay';
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
};

// Extract metadata for uploaded papers
summaryApp.extractInfo = async function () {
    if (!this.paper || this.paper.kind !== 'upload') {
        return;
    }

    // Check if already extracted
    if (this.paper.meta_extracted_ok === true) {
        alert('Metadata already extracted.');
        return;
    }

    // Check if parsed
    if (this.paper.parse_status && this.paper.parse_status !== 'ok') {
        alert('Parse PDF first before extracting info.');
        return;
    }

    const pid = this.pid;
    const btn = document.querySelector('.extract-btn');

    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Extracting...';
    }

    try {
        const resp = await csrfFetch('/api/uploaded_papers/extract_info', {
            method: 'POST',
            body: JSON.stringify({ pid: pid }),
        });
        const data = await resp.json();

        if (data.success) {
            // Refresh the page to show updated info
            alert('Metadata extraction started. The page will refresh shortly.');
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        } else {
            alert('Failed to extract: ' + (data.error || 'Unknown error'));
            if (btn) {
                btn.disabled = false;
                btn.textContent = '🔍 Extract Info';
            }
        }
    } catch (err) {
        console.error('Error triggering extract:', err);
        alert('Failed to trigger extraction');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🔍 Extract Info';
        }
    }
};

// Export summary as Markdown ZIP
summaryApp.exportMarkdownZip = async function () {
    if (!this.content || !this.paper) {
        alert('No summary content to export');
        return;
    }

    // Show export progress
    const exportBtn = document.querySelector('.summary-btn-export');
    const originalText = exportBtn ? exportBtn.innerHTML : '';
    if (exportBtn) {
        exportBtn.innerHTML = '⏳ Exporting...';
        exportBtn.disabled = true;
    }

    try {
        // Dynamically load JSZip if not available
        if (typeof JSZip === 'undefined') {
            await loadJSZip();
        }

        const zip = new JSZip();
        const pid = this.pid || 'paper';
        const meta = this.meta || {};

        // Create images folder
        const imagesFolder = zip.folder('images');

        // Parse and process images in markdown
        let markdown = this.content;
        const imageMap = new Map(); // url -> local filename
        let imageIndex = 0;

        // Find all image references: ![alt](url) and <img src="url">
        const mdImageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
        const htmlImageRegex = /<img[^>]+src=["']([^"']+)["'][^>]*>/gi;

        // Collect all image URLs
        const imageUrls = new Set();
        let match;

        // Reset regex lastIndex
        mdImageRegex.lastIndex = 0;
        htmlImageRegex.lastIndex = 0;

        while ((match = mdImageRegex.exec(markdown)) !== null) {
            const url = match[2].trim();
            if (url && !url.startsWith('data:')) {
                imageUrls.add(url);
            }
        }

        while ((match = htmlImageRegex.exec(markdown)) !== null) {
            const url = match[1].trim();
            if (url && !url.startsWith('data:')) {
                imageUrls.add(url);
            }
        }

        // Download images and add to zip
        if (exportBtn && imageUrls.size > 0) {
            exportBtn.innerHTML = `⏳ Downloading ${imageUrls.size} images...`;
        }

        for (const url of imageUrls) {
            try {
                const result = await downloadImageAsBlob(url);
                if (result) {
                    imageIndex++;
                    const ext = result.ext || 'png';
                    const filename = `img_${String(imageIndex).padStart(3, '0')}.${ext}`;
                    imageMap.set(url, filename);
                    imagesFolder.file(filename, result.blob);
                }
            } catch (e) {
                console.warn('Failed to download image:', url, e);
                // Keep original URL if download fails
            }
        }

        // Replace image URLs with local paths
        for (const [url, filename] of imageMap) {
            // Escape special regex characters in URL
            const escapedUrl = url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

            // Replace in markdown image syntax
            markdown = markdown.replace(
                new RegExp(`!\\[([^\\)]*)\\]\\(${escapedUrl}\\)`, 'g'),
                `![$1](images/${filename})`
            );

            // Replace in HTML img tags
            markdown = markdown.replace(
                new RegExp(`(<img[^>]+src=["'])${escapedUrl}(["'][^>]*>)`, 'gi'),
                `$1images/${filename}$2`
            );
        }

        // Create summary.md (clean, no YAML header - metadata goes in meta.json)
        zip.file('summary.md', markdown);

        // Create meta.json with essential metadata only
        // Note: URLs use raw pid (without version), which automatically points to latest version on arXiv
        const metaJson = {
            id: pid,
            title: this.paper.title || '',
            published: this.paper.time || '',
            urls: {
                arxiv: `https://arxiv.org/abs/${pid}`,
                pdf: `https://arxiv.org/pdf/${pid}.pdf`,
            },
            summary: {
                model: meta.llm_model || this.selectedModel || '',
                generated_at: meta.generated_at || null,
            },
            exported_at: new Date().toISOString(),
        };
        zip.file('meta.json', JSON.stringify(metaJson, null, 2));

        if (exportBtn) {
            exportBtn.innerHTML = '⏳ Generating ZIP...';
        }

        // Generate and download ZIP
        const blob = await zip.generateAsync({
            type: 'blob',
            compression: 'DEFLATE',
            compressionOptions: { level: 6 },
        });

        // Create filename
        const safeTitle = (this.paper.title || 'summary')
            .replace(/[<>:"/\\|?*]/g, '')
            .substring(0, 40)
            .trim();
        const filename = `${pid}_${safeTitle}.zip`;

        // Download
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);

        // Show success
        if (exportBtn) {
            exportBtn.innerHTML = '✓ Exported!';
            setTimeout(() => {
                exportBtn.innerHTML = originalText;
                exportBtn.disabled = false;
            }, 2000);
        }
    } catch (error) {
        console.error('Export failed:', error);
        alert('Export failed: ' + (error.message || 'Unknown error'));
        if (exportBtn) {
            exportBtn.innerHTML = originalText;
            exportBtn.disabled = false;
        }
    }
};

/**
 * Allowed domains for image download during export (privacy protection)
 * Only download images from trusted sources to avoid exposing user IP to arbitrary servers
 */
const ALLOWED_IMAGE_DOMAINS = [
    window.location.hostname, // Same origin
    'arxiv.org',
    'www.arxiv.org',
    'export.arxiv.org',
    'cdn.jsdelivr.net',
    'raw.githubusercontent.com',
    'i.imgur.com',
    'upload.wikimedia.org',
];

/**
 * Check if URL is from an allowed domain for image download
 */
function isAllowedImageDomain(url) {
    try {
        const urlObj = new URL(url, window.location.origin);
        const hostname = urlObj.hostname.toLowerCase();
        return ALLOWED_IMAGE_DOMAINS.some(
            domain => hostname === domain || hostname.endsWith('.' + domain)
        );
    } catch (e) {
        return false;
    }
}

/**
 * Download image and return as blob with detected extension
 * Only downloads from allowed domains for privacy protection
 */
async function downloadImageAsBlob(url) {
    try {
        // Handle relative URLs
        const fullUrl = url.startsWith('http') ? url : new URL(url, window.location.origin).href;

        // Privacy protection: only download from allowed domains
        if (!isAllowedImageDomain(fullUrl)) {
            console.info('Skipping image from non-allowed domain:', fullUrl);
            return null;
        }

        const response = await fetch(fullUrl, {
            mode: 'cors',
            credentials: 'omit',
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const blob = await response.blob();

        // Detect extension from content-type or URL
        let ext = 'png';
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('jpeg') || contentType.includes('jpg')) {
            ext = 'jpg';
        } else if (contentType.includes('png')) {
            ext = 'png';
        } else if (contentType.includes('gif')) {
            ext = 'gif';
        } else if (contentType.includes('webp')) {
            ext = 'webp';
        } else if (contentType.includes('svg')) {
            ext = 'svg';
        } else {
            // Try to get from URL
            const urlParts = url.split('.');
            const urlExt = urlParts.length > 1 ? urlParts.pop().toLowerCase().split('?')[0] : '';
            if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'bmp'].includes(urlExt)) {
                ext = urlExt === 'jpeg' ? 'jpg' : urlExt;
            }
        }

        return { blob, ext };
    } catch (e) {
        console.warn('Image download failed:', url, e);
        return null;
    }
}

// Helper to escape YAML strings
function escapeYaml(str) {
    return String(str || '')
        .replace(/"/g, '\\"')
        .replace(/\n/g, ' ');
}

// Load JSZip library dynamically
function loadJSZip() {
    return new Promise((resolve, reject) => {
        if (typeof JSZip !== 'undefined') {
            resolve();
            return;
        }
        const script = document.createElement('script');
        // Resolve static base in case the app is deployed under a URL prefix.
        script.src =
            CommonUtils && typeof CommonUtils.staticUrl === 'function'
                ? CommonUtils.staticUrl('lib/jszip.min.js')
                : '/static/lib/jszip.min.js';
        script.onload = resolve;
        script.onerror = () => reject(new Error('Failed to load JSZip library'));
        document.head.appendChild(script);
    });
}

// Deprecated: kept for backward compatibility
summaryApp.clearCache = summaryApp.confirmClearAll;

// Load summary function
summaryApp.loadSummary = async function (pid, options = {}) {
    return await CommonUtils.measurePerformanceAsync(
        `loadSummary(${pid}, model=${options.model || 'default'})`,
        async () => {
            if (this.clearing) {
                return;
            }
            const requestId = (this.requestSeq = (this.requestSeq || 0) + 1);
            const chosenModel = options.model || this.getCurrentModel() || this.defaultModel || '';
            const force = Boolean(options.force_regenerate);
            const cacheOnly = options.cache_only !== undefined ? Boolean(options.cache_only) : true;
            this.clearAutoRetry();

            const chosenModelStr = String(chosenModel || '').trim();
            if (!chosenModelStr) {
                this.setState({
                    loading: false,
                    regenerating: false,
                    error: 'No summary model configured. Please configure a default model.',
                });
                return;
            }

            // Uploaded papers must be parsed before summary generation can work.
            // Avoid making an API call that will fail with 404/400 and instead show
            // a clear action hint to the user.
            if (this.paper && this.paper.kind === 'upload') {
                const ps = this.paper.parse_status ? String(this.paper.parse_status) : '';
                if (ps && ps !== 'ok') {
                    this.setState({
                        loading: false,
                        regenerating: false,
                        notice: 'Parse PDF first before generating summary.',
                        error: null,
                        content: null,
                        meta: null,
                        contentModel: String(chosenModelStr || '').trim(),
                    });
                    return;
                }
            }
            if (force && chosenModelStr) {
                this.inflightModels[chosenModelStr] = true;
            }

            const prevContentModel = String(this.contentModel || '').trim();
            const modelChanged = chosenModelStr && chosenModelStr !== prevContentModel;

            const inFlight = Boolean(chosenModelStr && this.inflightModels[chosenModelStr]);
            // Show loading when: force regenerate, or no content yet, or in-flight generation,
            // or model changed (to avoid showing "No summary" flash before API returns)
            const hasRenderedContent = Boolean(!modelChanged && this.content);
            const shouldShowLoading =
                force ||
                (!cacheOnly && !this.content) ||
                (cacheOnly && inFlight && !hasRenderedContent) ||
                modelChanged;
            this.setState({
                loading: shouldShowLoading,
                error: null,
                regenerating: force,
                selectedModel: chosenModel || '',
                notice: cacheOnly ? (inFlight ? this.notice : '') : this.notice,
                // Prevent showing another model's summary while fetching this model
                content: modelChanged ? null : this.content,
                meta: modelChanged ? null : this.meta,
            });

            try {
                const result = await fetchSummary(pid, {
                    model: chosenModel,
                    force_regenerate: false,
                    cache_only: cacheOnly,
                });

                // Ignore stale responses (user may have switched models)
                if (requestId !== this.requestSeq) {
                    return;
                }

                const meta = result.meta || {};
                const content = result.content;
                if (
                    typeof content === 'string' &&
                    content.startsWith('# Error') &&
                    content.includes('Summary is being generated')
                ) {
                    this.pendingGenerationModel = chosenModel || '';
                    this.setState({
                        loading: true,
                        regenerating: false,
                        notice: 'Summary is being generated for this model. You can switch models; this view will auto-refresh when ready.',
                        error: null,
                        content: null,
                        meta: null,
                    });
                    this.scheduleAutoRetry(pid, { model: chosenModel, cache_only: true });
                    return;
                }

                // Check if fallback occurred using shared utility
                const fallback = checkSummaryFallback(meta, chosenModelStr);
                const selectedModel = fallback.occurred
                    ? fallback.actualModel
                    : this.selectedModel || chosenModel || '';

                // If we requested a forced regeneration and the cache still returns the same content,
                // keep polling instead of stopping at the stale cached version.
                const regen = chosenModelStr ? this.pendingRegenerations[chosenModelStr] : null;
                if (regen && String(selectedModel || '').trim() === chosenModelStr) {
                    const prevGenAt = regen.generatedAt;
                    const newGenAt =
                        meta && meta.generated_at !== undefined && meta.generated_at !== null
                            ? Number(meta.generated_at)
                            : null;
                    const sameGeneratedAt =
                        Number.isFinite(prevGenAt) &&
                        Number.isFinite(newGenAt) &&
                        Number(prevGenAt) === Number(newGenAt);
                    const sameContent = hashString(content) === String(regen.contentHash || '');
                    const stillStale =
                        sameGeneratedAt || (!Number.isFinite(prevGenAt) && sameContent);
                    if (stillStale) {
                        this.pendingGenerationModel = chosenModelStr;
                        if (chosenModelStr) {
                            this.inflightModels[chosenModelStr] = true;
                        }
                        this.setState({
                            loading: false,
                            regenerating: true,
                            notice: 'Summary is being regenerated for this model. Showing the cached version until the new one is ready.',
                            error: null,
                            content: content,
                            meta: meta,
                            contentModel: String(selectedModel || '').trim(),
                            selectedModel,
                        });
                        this.scheduleAutoRetry(pid, { model: chosenModelStr, cache_only: true });
                        return;
                    }
                    delete this.pendingRegenerations[chosenModelStr];
                }

                this.autoRetryCount = 0;
                this.pendingGenerationModel = '';
                if (chosenModelStr) {
                    this.inflightModels[chosenModelStr] = false;
                }

                // Update availableSummaries to include the newly generated model
                const newModelKey = modelCacheKey(selectedModel);
                if (newModelKey && !this.availableSummaries.includes(newModelKey)) {
                    this.availableSummaries = [...this.availableSummaries, newModelKey];
                }

                this.setState({
                    loading: false,
                    regenerating: false,
                    content: content,
                    meta: meta,
                    contentModel: String(selectedModel || '').trim(),
                    selectedModel,
                    notice: fallback.notice,
                    queueRank: 0,
                    queueTotal: 0,
                    lastTaskId: '',
                });
            } catch (error) {
                if (requestId !== this.requestSeq) {
                    return;
                }
                if (error.code === 'summary_cache_miss') {
                    const statusModel = chosenModelStr || String(this.defaultModel || '').trim();
                    const stillInFlight = Boolean(statusModel && this.inflightModels[statusModel]);
                    if (stillInFlight) {
                        this.pendingGenerationModel = statusModel;
                        this.setState({
                            loading: true,
                            regenerating: false,
                            notice: 'Summary is being generated for this model. You can switch models; this view will auto-refresh when ready.',
                            error: null,
                            content: null,
                            meta: null,
                        });
                        this.scheduleAutoRetry(pid, { model: statusModel, cache_only: true });
                        return;
                    }
                    // Summary generation is user-initiated on the summary page.
                    // If a job is already queued/running (e.g., triggered elsewhere), show status and poll.
                    try {
                        const statusInfo = await fetchSummaryStatus(pid, statusModel);
                        if (requestId !== this.requestSeq) {
                            return;
                        }
                        if (
                            statusInfo &&
                            (statusInfo.status === 'queued' || statusInfo.status === 'running')
                        ) {
                            if (statusInfo.task_id) {
                                this.lastTaskId = String(statusInfo.task_id);
                                this.refreshQueueRank();
                            }
                            this.pendingGenerationModel = statusModel;
                            this.setState({
                                loading: true,
                                regenerating: false,
                                notice: 'Summary is being generated for this model. You can switch models; this view will auto-refresh when ready.',
                                error: null,
                                content: null,
                                meta: null,
                            });
                            this.scheduleAutoRetry(pid, {
                                model: statusModel,
                                cache_only: true,
                            });
                            return;
                        }
                    } catch (statusError) {
                        console.warn('Failed to check summary status:', statusError);
                    }
                    this.setState({
                        loading: false,
                        regenerating: false,
                        notice: 'No cached summary for this model. Click Generate to create one.',
                        error: null,
                        content: null,
                        meta: null,
                        contentModel: String(chosenModelStr || '').trim(),
                    });
                    return;
                }
                const friendlyMsg = CommonUtils.handleApiError(error, 'Load Summary');
                this.setState({ loading: false, regenerating: false, error: friendlyMsg });
                if (
                    error.code === 'summary_timeout' ||
                    String(error.message || '').includes('Failed to fetch')
                ) {
                    this.scheduleAutoRetry(pid, { model: chosenModel, cache_only: true });
                }
                if (force && chosenModelStr) {
                    this.inflightModels[chosenModelStr] = false;
                }
            }
        }
    );
};

summaryApp.loadModels = async function () {
    return await CommonUtils.measurePerformanceAsync('loadModels', async () => {
        try {
            const models = await fetchModels();
            // Keep model data immediately readable for init sequence
            // (selectInitialModel runs right after loadModels resolves).
            this.setStateSync({ models, modelsError: null });
        } catch (error) {
            const friendlyMsg = CommonUtils.handleApiError(error, 'Load Models');
            this.setStateSync({ modelsError: friendlyMsg });
        }
    });
};

summaryApp.selectInitialModel = async function (pid) {
    try {
        // Get available summaries for this paper
        const availableSummaries = await fetchAvailableSummaries(pid);

        // Store available summaries in state for UI rendering
        this.availableSummaries = availableSummaries;

        let selectedModel = '';

        // Prefer default model if it already has a cached summary.
        const preferred = String(this.defaultModel || '').trim();
        if (preferred && availableSummaries.length > 0) {
            const preferredKey = modelCacheKey(preferred);
            if (preferredKey && availableSummaries.includes(preferredKey)) {
                selectedModel = preferred;
            }
        }

        // If there are available summaries, select the first one from model list that has a summary
        if (!selectedModel && availableSummaries.length > 0 && this.models.length > 0) {
            for (const model of this.models) {
                const modelId = String(model.id || '');
                const key = modelCacheKey(modelId);
                if (key && availableSummaries.includes(key)) {
                    selectedModel = modelId;
                    break;
                }
            }
        }

        // If no available summary found, use default model
        if (!selectedModel) {
            if (preferred) {
                const matched = this.models.find(m => String(m.id || '') === preferred);
                selectedModel = matched ? matched.id || '' : preferred;
            }
            if (!selectedModel && this.models.length > 0) {
                selectedModel = this.models[0].id || '';
            }
        }

        this.setState({ selectedModel });
        return selectedModel;
    } catch (error) {
        console.error('Failed to check available summaries:', error);
        // Fallback to default model selection on error
        let selectedModel = '';
        const preferred = String(this.defaultModel || '').trim();
        if (preferred) {
            const matched = this.models.find(m => String(m.id || '') === preferred);
            selectedModel = matched ? matched.id || '' : preferred;
        }
        if (!selectedModel && this.models.length > 0) {
            selectedModel = this.models[0].id || '';
        }
        this.setState({ selectedModel });
        return selectedModel;
    }
};

// Initialize app
async function initSummaryApp() {
    console.log('Initializing summary page...');

    // Check required variables
    if (typeof paper === 'undefined') {
        console.error('Paper data is missing!');
        document.getElementById('wrap').innerHTML =
            '<div style="padding: 20px; color: red;">Error: Paper data is missing</div>';
        return;
    }

    if (typeof pid === 'undefined') {
        console.error('Paper ID is missing!');
        document.getElementById('wrap').innerHTML =
            '<div style="padding: 20px; color: red;">Error: Paper ID is missing</div>';
        return;
    }

    // Set initial state
    const initialDefaultModel =
        typeof defaultSummaryModel !== 'undefined' ? String(defaultSummaryModel || '').trim() : '';

    // Use sync state init so defaultModel/selectedModel are available
    // immediately to subsequent initialization steps.
    summaryApp.setStateSync({
        paper: paper,
        pid: pid,
        loading: true,
        meta: null,
        models: [],
        modelsError: null,
        selectedModel: initialDefaultModel,
        notice: '',
        clearing: false,
        defaultModel: initialDefaultModel,
        userTags: paper.utags || [],
        negativeTags: paper.ntags || [],
        availableTags: [],
        tagDropdownOpen: false,
        tagSearchValue: '',
        newTagValue: '',

        // Reading list state (only used for logged-in arXiv papers)
        inReadingList: false,
        readingListPending: false,
    });

    await summaryApp.loadModels();
    // Select initial model based on available summaries
    const initialModel = await summaryApp.selectInitialModel(pid);
    // Start loading summary after model selection
    summaryApp.loadSummary(pid, {
        model: initialModel,
        cache_only: true,
    });

    // Fetch global queue stats
    summaryApp.refreshGlobalQueueStats();

    // Initialize tag management if user is logged in
    if (typeof user !== 'undefined' && user) {
        await initTagManagement();
        _registerEventHandler(handleUserEvent);
        _setupUserEventStream(user, applyUserState);

        // Refresh reading list membership for arXiv papers (do not block summary loading).
        try {
            const isUploaded = summaryApp.paper && summaryApp.paper.kind === 'upload';
            if (!isUploaded) {
                summaryApp.refreshReadingListState(pid);
            }
        } catch (e) {}
    }

    // Setup back to top button
    const DomUtils = window.ArxivSanitySummaryMarkdownDom;
    if (DomUtils && typeof DomUtils.setupBackToTop === 'function') {
        DomUtils.setupBackToTop();
    }
}

// Tag dropdown UI is provided by static/tag_dropdown_shared.js (shared React implementation)

function renderTagDropdown() {
    const container = document.getElementById('summary-tag-dropdown');
    if (!container) return;
    if (!window.ArxivSanityTagDropdown || typeof window.ArxivSanityTagDropdown.mount !== 'function')
        return;

    const pidValue = summaryApp.paper && summaryApp.paper.id ? String(summaryApp.paper.id) : '';
    if (!pidValue) return;

    const prevUi =
        sharedTagDropdownApi && sharedTagDropdownApi.getUiState
            ? sharedTagDropdownApi.getUiState()
            : {};
    if (sharedTagDropdownApi && sharedTagDropdownApi.unmount) {
        sharedTagDropdownApi.unmount();
        sharedTagDropdownApi = null;
    }

    sharedTagDropdownApi = window.ArxivSanityTagDropdown.mount(container, {
        pid: pidValue,
        selectedTags: summaryApp.userTags || [],
        negativeTags: summaryApp.negativeTags || [],
        availableTags: summaryApp.availableTags || [],
        open: summaryApp.tagDropdownOpen || prevUi.open,
        searchValue: summaryApp.tagSearchValue || prevUi.searchValue,
        newTagValue: summaryApp.newTagValue || prevUi.newTagValue,
        onStateChange: st => {
            summaryApp.userTags = st.selectedTags || [];
            summaryApp.negativeTags = st.negativeTags || [];
            summaryApp.availableTags = st.availableTags || [];
            summaryApp.tagDropdownOpen = !!st.open;
            summaryApp.tagSearchValue = st.searchValue || '';
            summaryApp.newTagValue = st.newTagValue || '';
        },
    });
}

// Tag management functions
let tagDropdownListenersBound = false;

async function initTagManagement() {
    // Load available tags from global tags variable if available
    if (typeof tags !== 'undefined' && Array.isArray(tags)) {
        summaryApp.availableTags = tags
            .filter(t => t && t.name && t.name !== 'all')
            .map(t => t.name)
            .sort();
    }

    if (tagDropdownListenersBound) return;
    tagDropdownListenersBound = true;

    // Close confirm popup on Escape
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && summaryApp.pendingConfirm) {
            summaryApp.cancelConfirm();
        }
    });

    // Close confirm popup when clicking outside
    document.addEventListener('mousedown', e => {
        if (summaryApp.pendingConfirm) {
            const confirmPopup = document.querySelector('.confirm-popup');
            if (confirmPopup && !confirmPopup.contains(e.target)) {
                // Check if click is not on the button that triggered it
                const btnGroup = confirmPopup.closest('.summary-btn-group');
                const triggerBtn = btnGroup ? btnGroup.querySelector('.summary-action-btn') : null;
                if (!triggerBtn || !triggerBtn.contains(e.target)) {
                    summaryApp.cancelConfirm();
                }
            }
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initSummaryApp);
