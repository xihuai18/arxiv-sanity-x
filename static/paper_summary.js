'use strict';

// Use shared utilities from common_utils.js
var CommonUtils = window.ArxivSanityCommon;
var getCsrfToken = CommonUtils.getCsrfToken;
var csrfFetch = CommonUtils.csrfFetch;
var formatAuthorsText = CommonUtils.formatAuthorsText;
var escapeHtml = CommonUtils.escapeHtml;
var SummaryMarkdown = window.ArxivSanitySummaryMarkdown;
var renderSummaryMarkdown = SummaryMarkdown.renderSummaryMarkdown;

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
            summaryApp.userTags = (summaryApp.userTags || []).map(t => (t === event.from ? event.to : t));
            summaryApp.negativeTags = (summaryApp.negativeTags || []).map(t => (t === event.from ? event.to : t));
            renderTagDropdown();
        } else if (event.reason === 'delete_tag' && event.tag) {
            summaryApp.userTags = (summaryApp.userTags || []).filter(t => t !== event.tag);
            summaryApp.negativeTags = (summaryApp.negativeTags || []).filter(t => t !== event.tag);
            renderTagDropdown();
        } else if (event.reason === 'tag_feedback' && event.pid && summaryApp.pid && event.pid === summaryApp.pid) {
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
        this.maxAutoRetries = 5;
        this.notice = '';
        this.clearing = false;
        this.defaultModel = '';
        this.pendingConfirm = null; // 'clearModel' or 'clearAll'

        // Track async requests to avoid stale responses overriding UI state
        this.requestSeq = 0;
        this.pendingGenerationModel = '';

        // Per-model in-flight generation tracking
        this.inflightModels = Object.create(null);

        // Track which model the currently rendered content belongs to
        this.contentModel = '';
    }

    setState(newState) {
        Object.assign(this, newState);
        this.render();
    }

    render() {
        const container = document.getElementById('wrap');
        if (!container) return;

        const htmlContent = this.getHTML();

        if (this.content) {
            container.innerHTML = htmlContent;
            const markdownContainer = container.querySelector('.summary-markdown');
            const tocContainer = container.querySelector('.summary-toc');
            if (markdownContainer) {
                renderSummaryMarkdown(this.content, markdownContainer, tocContainer);
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
        // 使用 MathJax 渲染數學公式（用於非 markdown 內容）
        if (typeof MathJax !== 'undefined') {
            setTimeout(() => {
                if (MathJax.typeset) {
                    MathJax.typeset();
                } else if (MathJax.typesetPromise) {
                    MathJax.typesetPromise().catch((err) => {
                        console.warn('MathJax rendering error:', err);
                    });
                }
            }, 100);
        }
    }

    getCurrentModel() {
        return this.selectedModel || '';
    }

    isCurrentModelGenerating() {
        const current = String(this.getCurrentModel() || '').trim();
        if (!current) return false;
        if (this.pendingGenerationModel && String(this.pendingGenerationModel) === current) return true;
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
            if (targetModel && this.pendingGenerationModel && this.pendingGenerationModel !== targetModel) {
                return;
            }
            this.loadSummary(pid, { model: targetModel || '', force_regenerate: false });
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
        if (!Array.isArray(this.models) || this.models.length === 0) {
            const fallbackLabel = current ? `Use ${escapeHtml(current)}` : 'Default (server configured)';
            const value = current ? escapeHtml(current) : '';
            return `<option value="${value}">${fallbackLabel}</option>`;
        }

        let options = this.models
            .map((model) => {
                const rawId = String(model.id || '');
                const id = escapeHtml(rawId);
                const selected = rawId === String(current || '') ? ' selected' : '';
                return `<option value="${id}"${selected}>${id}</option>`;
            })
            .join('');

        const hasCurrent = this.models.some((m) => String(m.id || '') === String(current || ''));
        if (current && !hasCurrent) {
            const value = escapeHtml(current);
            options += `<option value="${value}" selected>${value}</option>`;
        }

        return options;
    }

    renderActions() {
        // Allow clearing while generating; only block actions during clearing.
        // Keep Generate disabled during loading to avoid concurrent generations.
        const disableGenerate = this.loading || this.clearing || this.isCurrentModelGenerating() ? 'disabled' : '';
        const disableClear = this.clearing ? 'disabled' : '';
        // Allow switching models even while generating
        const disableSelect = this.clearing ? 'disabled' : '';
        const regenLabel = this.regenerating ? 'Generating...' : 'Generate';
        const modelOptions = this.renderModelOptions();
        const errorNote = this.modelsError
            ? `<div class="summary-note" style="color: #d9534f;" role="status" aria-live="polite">${escapeHtml(this.modelsError)}</div>`
            : '';
        const notice = this.notice
            ? `<div class="summary-note" style="color: #b8860b;" role="status" aria-live="polite">${escapeHtml(this.notice)}</div>`
            : '';

        const currentModel = this.getCurrentModel();
        const modelLabel = currentModel ? escapeHtml(currentModel) : 'current model';
        const inflightNote =
            this.isCurrentModelGenerating()
                ? `<p class="confirm-warning">Note: generation for this model is currently running. Clearing does not cancel it; it may recreate cache when finished.</p>`
                : '';

        return `
            <div class="summary-actions">
                <label class="summary-action-label" for="model-select">Model</label>
                <select id="model-select" class="summary-model-select" onchange="summaryApp.handleModelChange(event)" ${disableSelect}>
                    ${modelOptions}
                </select>
                <button onclick="summaryApp.regenerate()" class="summary-action-btn" ${disableGenerate}>
                    ${regenLabel}
                </button>
                <div class="summary-btn-group">
                    <button onclick="summaryApp.requestClearModel()" class="summary-action-btn summary-btn-warning" ${disableClear} title="Clear summary for current model only">
                        ${this.clearing === 'model' ? 'Clearing...' : 'Clear Current Summary'}
                    </button>
                    ${this.pendingConfirm === 'clearModel' ? `
                        <div class="confirm-popup" role="dialog" aria-labelledby="confirm-title">
                            <div class="confirm-content">
                                <strong id="confirm-title">Clear summary for ${modelLabel}?</strong>
                                <p>This will only remove the summary generated by this model.</p>
                                ${inflightNote}
                                <div class="confirm-actions">
                                    <button onclick="summaryApp.confirmClearModel()" class="confirm-btn confirm-yes">确定</button>
                                    <button onclick="summaryApp.cancelConfirm()" class="confirm-btn confirm-no">取消</button>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                </div>
                <div class="summary-btn-group">
                    <button onclick="summaryApp.requestClearAll()" class="summary-action-btn summary-btn-danger" ${disableClear} title="Clear all caches (all models, HTML, MinerU, etc.)">
                        ${this.clearing === 'all' ? 'Clearing...' : 'Clear All'}
                    </button>
                    ${this.pendingConfirm === 'clearAll' ? `
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
                                    <button onclick="summaryApp.confirmClearAll()" class="confirm-btn confirm-yes">确定</button>
                                    <button onclick="summaryApp.cancelConfirm()" class="confirm-btn confirm-no">取消</button>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                </div>
                ${errorNote}
                ${notice}
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
        const tagsSafe = (this.paper && this.paper.tags) ? escapeHtml(this.paper.tags) : '';
        const abstractSafe = this.paper ? escapeHtml(this.paper.summary || 'No abstract available.') : '';

        // Check if user is logged in
        const isLoggedIn = typeof user !== 'undefined' && user;

        const headerHTML = this.paper ? `
            <div class="paper-header">
                <div class="paper-nav paper-actions-footer">
                    <div class="rel_more"><a href="/?rank=pid&pid=${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">Similar</a></div>
                    <div class="rel_inspect"><a href="/inspect?pid=${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">Inspect</a></div>
                    <div class="rel_alphaxiv"><a href="https://www.alphaxiv.org/overview/${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">alphaXiv</a></div>
                    <div class="rel_cool"><a href="https://papers.cool/arxiv/${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">Cool</a></div>
                </div>
                <div class="paper-content-section">
                    <div class="paper-main">
                        <h1 class="paper-title">
                            <a href="https://arxiv.org/abs/${encodeURIComponent(pidSafe)}" target="_blank" rel="noopener noreferrer">
                                ${titleSafe}
                            </a>
                        </h1>
                        <div class="paper-authors-line">
                            <span title="${authorsTitleSafe}">${authorsSafe}</span>
                        </div>
                        <div class="paper-meta-line">
                            <span class="paper-time">${timeSafe}</span>
                            ${tagsSafe ? `<span class="paper-tags">${tagsSafe}</span>` : ''}
                        </div>
                        <div class="paper-abstract">
                            ${abstractSafe}
                        </div>
                        ${isLoggedIn ? `
                        <div class="paper-user-tags-section">
                            <div class="rel_utags" id="summary-tag-dropdown"></div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        ` : '';

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
            // 不在這裡處理 markdown，讓 render() 方法中的 renderSummaryMarkdown 處理
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

// API call function
async function fetchSummary(pid, options = {}) {
    try {
        // 創建 AbortController 來控制超時
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000); // 10分鐘超時

        const response = await fetch('/api/get_paper_summary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': getCsrfToken(),
            },
            body: JSON.stringify({
                pid: pid,
                model: options.model || undefined,
                force_regenerate: Boolean(options.force_regenerate),
                cache_only: Boolean(options.cache_only),
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        const data = await response.json().catch(() => null);
        if (response.ok && data && data.success) {
            return {
                content: data.summary_content,
                meta: data.summary_meta || {},
            };
        }

        // Extract error info even from non-2xx responses so callers can react to cache misses
        const err = new Error((data && data.error) || `HTTP ${response.status}: ${response.statusText}`);
        if (data && data.code) {
            err.code = data.code;
        }
        throw err;
    } catch (error) {
        if (error.name === 'AbortError') {
            const timeoutError = new Error('請求超時，論文總結過程較長，請稍後重試');
            timeoutError.code = 'summary_timeout';
            throw timeoutError;
        }
        console.error('Failed to fetch summary:', error);
        throw error;
    }
}

async function clearModelSummary(pid, model) {
    const response = await fetch('/api/clear_model_summary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getCsrfToken(),
        },
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
    const response = await fetch('/api/clear_paper_cache', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getCsrfToken(),
        },
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

function modelCacheKey(modelId) {
    const raw = String(modelId || '').trim();
    if (!raw) return '';
    return raw.replace(/[^a-zA-Z0-9._-]+/g, '_');
}

// Retry function
summaryApp.retry = function() {
    if (this.pid) {
        this.loadSummary(this.pid, { force_regenerate: true });
    }
};

summaryApp.handleModelChange = function(event) {
    const value = event && event.target ? event.target.value : '';
    this.setState({ selectedModel: value });
    if (this.pid) {
        this.loadSummary(this.pid, { model: value, cache_only: true });
    }
};

summaryApp.regenerate = function() {
    if (!this.pid || this.loading || this.isCurrentModelGenerating()) return;
    this.loadSummary(this.pid, { force_regenerate: true, model: this.getCurrentModel() });
};

summaryApp.requestClearModel = function() {
    if (!this.pid || this.clearing) return;

    const currentModel = this.getCurrentModel();
    if (!currentModel) {
        this.setState({ error: 'No model selected' });
        return;
    }

    this.setState({ pendingConfirm: 'clearModel' });
};

summaryApp.confirmClearModel = async function() {
    const currentModel = this.getCurrentModel();
    this.clearAutoRetry();
    this.setState({ clearing: 'model', notice: '', error: null, pendingConfirm: null });

    try {
        await clearModelSummary(this.pid, currentModel);
        this.setState({
            clearing: false,
            content: null,
            meta: null,
            notice: `Summary for model "${currentModel}" cleared. Click Generate to create a new one.`,
        });
    } catch (error) {
        const friendlyMsg = CommonUtils.handleApiError(error, 'Clear Model Summary');
        this.setState({ clearing: false, error: friendlyMsg });
    }
};

summaryApp.requestClearAll = function() {
    if (!this.pid || this.clearing) return;
    this.setState({ pendingConfirm: 'clearAll' });
};

summaryApp.confirmClearAll = async function() {
    this.clearAutoRetry();
    this.setState({ clearing: 'all', notice: '', error: null, pendingConfirm: null });

    try {
        await clearPaperCache(this.pid);
        this.setState({
            clearing: false,
            content: null,
            meta: null,
            notice: 'All caches cleared. Click Generate to fetch a fresh summary.',
        });
    } catch (error) {
        const friendlyMsg = CommonUtils.handleApiError(error, 'Clear All Caches');
        this.setState({ clearing: false, error: friendlyMsg });
    }
};

summaryApp.cancelConfirm = function() {
    this.setState({ pendingConfirm: null });
};

// Deprecated: kept for backward compatibility
summaryApp.clearCache = summaryApp.confirmClearAll;

// Load summary function
summaryApp.loadSummary = async function(pid, options = {}) {
    return await CommonUtils.measurePerformanceAsync(
        `loadSummary(${pid}, model=${options.model || 'default'})`,
        async () => {
            const requestId = (this.requestSeq = (this.requestSeq || 0) + 1);
            const chosenModel = options.model || this.getCurrentModel();
            const force = Boolean(options.force_regenerate);
            const cacheOnly = Boolean(options.cache_only);
            this.clearAutoRetry();

            const chosenModelStr = String(chosenModel || '').trim();
            if (force && chosenModelStr) {
                this.inflightModels[chosenModelStr] = true;
            }

            const prevContentModel = String(this.contentModel || '').trim();
            const modelChanged = chosenModelStr && chosenModelStr !== prevContentModel;

            const inFlight = Boolean(chosenModelStr && this.inflightModels[chosenModelStr]);
            const shouldShowLoading = force || (!cacheOnly && !this.content) || (cacheOnly && inFlight);
            this.setState({
                loading: shouldShowLoading,
                error: null,
                regenerating: force,
                selectedModel: chosenModel || '',
                notice: cacheOnly ? '' : this.notice,
                // Prevent showing another model's summary while fetching this model
                content: modelChanged ? null : this.content,
                meta: modelChanged ? null : this.meta,
            });

            try {
                const result = await fetchSummary(pid, {
                    model: chosenModel,
                    force_regenerate: force,
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
                    this.scheduleAutoRetry(pid, { model: chosenModel });
                    return;
                }

                const selectedModel = this.selectedModel || chosenModel || '';
                this.autoRetryCount = 0;
                this.pendingGenerationModel = '';
                if (chosenModelStr) {
                    this.inflightModels[chosenModelStr] = false;
                }
                this.setState({
                    loading: false,
                    regenerating: false,
                    content: content,
                    meta: meta,
                    contentModel: String(selectedModel || '').trim(),
                    selectedModel,
                    notice: '',
                });
            } catch (error) {
                if (requestId !== this.requestSeq) {
                    return;
                }
                if (error.code === 'summary_cache_miss' && cacheOnly) {
                    const stillInFlight = Boolean(chosenModelStr && this.inflightModels[chosenModelStr]);
                    if (stillInFlight) {
                        this.pendingGenerationModel = chosenModelStr;
                        this.setState({
                            loading: true,
                            regenerating: false,
                            notice: 'Summary is being generated for this model. You can switch models; this view will auto-refresh when ready.',
                            error: null,
                            content: null,
                            meta: null,
                        });
                        this.scheduleAutoRetry(pid, { model: chosenModelStr });
                        return;
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
                if (error.code === 'summary_timeout' || String(error.message || '').includes('Failed to fetch')) {
                    this.scheduleAutoRetry(pid, { model: chosenModel });
                }
                if (force && chosenModelStr) {
                    this.inflightModels[chosenModelStr] = false;
                }
            }
        }
    );
};

summaryApp.loadModels = async function() {
    return await CommonUtils.measurePerformanceAsync('loadModels', async () => {
        try {
            const models = await fetchModels();
            this.setState({ models, modelsError: null });
        } catch (error) {
            const friendlyMsg = CommonUtils.handleApiError(error, 'Load Models');
            this.setState({ modelsError: friendlyMsg });
        }
    });
};

summaryApp.selectInitialModel = async function(pid) {
    try {
        // Get available summaries for this paper
        const availableSummaries = await fetchAvailableSummaries(pid);

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
                const matched = this.models.find((m) => String(m.id || '') === preferred);
                selectedModel = matched ? matched.id || '' : '';
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
            const matched = this.models.find((m) => String(m.id || '') === preferred);
            selectedModel = matched ? matched.id || '' : '';
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
    summaryApp.setState({
        paper: paper,
        pid: pid,
        loading: true,
        meta: null,
        models: [],
        modelsError: null,
        selectedModel: '',
        notice: '',
        clearing: false,
        defaultModel: typeof defaultSummaryModel !== 'undefined' ? defaultSummaryModel : '',
        userTags: paper.utags || [],
        negativeTags: paper.ntags || [],
        availableTags: [],
        tagDropdownOpen: false,
        tagSearchValue: '',
        newTagValue: ''
    });

    await summaryApp.loadModels();
    // Select initial model based on available summaries
    const initialModel = await summaryApp.selectInitialModel(pid);
    // Start loading summary after model selection
    summaryApp.loadSummary(pid, { model: initialModel });

    // Initialize tag management if user is logged in
    if (typeof user !== 'undefined' && user) {
        await initTagManagement();
        setupUserEventStream();
    }
}

// Tag dropdown UI is provided by static/tag_dropdown_shared.js (shared React implementation)

function renderTagDropdown() {
    const container = document.getElementById('summary-tag-dropdown');
    if (!container) return;
    if (!window.ArxivSanityTagDropdown || typeof window.ArxivSanityTagDropdown.mount !== 'function') return;

    const pidValue = summaryApp.paper && summaryApp.paper.id ? String(summaryApp.paper.id) : '';
    if (!pidValue) return;

    const prevUi = sharedTagDropdownApi && sharedTagDropdownApi.getUiState ? sharedTagDropdownApi.getUiState() : {};
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
        onStateChange: (st) => {
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
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && summaryApp.pendingConfirm) {
            summaryApp.cancelConfirm();
        }
    });

    // Close confirm popup when clicking outside
    document.addEventListener('mousedown', (e) => {
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
