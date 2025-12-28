'use strict';

function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? (meta.getAttribute('content') || '') : '';
}

function escapeHtml(text) {
    return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function isSafeUrl(href) {
    try {
        const u = new URL(String(href || ''), window.location.href);
        return u.protocol === 'http:' || u.protocol === 'https:' || u.protocol === 'mailto:';
    } catch (e) {
        return false;
    }
}

function configureMarkedOnce() {
    if (typeof marked === 'undefined' || !marked.use) return;
    if (configureMarkedOnce._done) return;

    const safeRenderer = {
        html() {
            return '';
        },
        image() {
            return '';
        },
        link(hrefOrToken, title, text) {
            // marked 9+: link(token); older: link(href, title, text)
            const token = (hrefOrToken && typeof hrefOrToken === 'object') ? hrefOrToken : null;
            const href = token ? token.href : hrefOrToken;
            const linkText = token ? token.text : text;
            const linkTitle = token ? token.title : title;

            if (!isSafeUrl(href)) {
                return escapeHtml(linkText);
            }

            const t = escapeHtml(linkText);
            const h = escapeHtml(href);
            const ttl = linkTitle ? ` title="${escapeHtml(linkTitle)}"` : '';
            return `<a href="${h}"${ttl}>${t}</a>`;
        },
    };

    marked.use({ renderer: safeRenderer });
    if (marked.setOptions) {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false,
            pedantic: false,
            smartLists: true,
            smartypants: false,
            xhtml: false
        });
    }

    configureMarkedOnce._done = true;
}

// 先 Marked 後 MathJax 的渲染函數（完全重寫版）
function renderMarkdownWithMath(text, container) {
    if (!text || !container) return Promise.resolve();

    return new Promise((resolve) => {
        try {
            console.log('Starting markdown rendering, text length:', text.length);

            let htmlContent = '';

            // 檢查 marked 是否可用
            if (typeof marked !== 'undefined') {
                configureMarkedOnce();

                // 保存所有需要保護的內容
                const protectedContent = {
                    mathBlocks: [],
                    mathInlines: [],
                    codeBlocks: []
                };

                // 第一步：保護所有特殊內容
                let processedText = text;

                // 1. 保護代碼塊
                processedText = processedText.replace(/```[\s\S]*?```/g, (match) => {
                    const index = protectedContent.codeBlocks.length;
                    protectedContent.codeBlocks.push(match);
                    return `\n\nCODEBLOCK${index}CODEBLOCK\n\n`;
                });

                // 2. 保護塊級數學公式
                processedText = processedText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
                    const index = protectedContent.mathBlocks.length;
                    protectedContent.mathBlocks.push(match);
                    return `MATHBLOCK${index}MATHBLOCK`;
                });

                // 2.1 保護 \[ \] 塊級數學公式
                processedText = processedText.replace(/\\\[([\s\S]*?)\\\]/g, (match, formula, offset, string) => {
                    const prevChar = offset > 0 ? string[offset - 1] : '';
                    if (prevChar === '\\') {
                        return match;
                    }
                    const index = protectedContent.mathBlocks.length;
                    protectedContent.mathBlocks.push(match);
                    return `MATHBLOCK${index}MATHBLOCK`;
                });

                // 3. 保護行內數學公式（兼容性更好的版本）
                // 避免使用 lookbehind，因為某些瀏覽器不支持
                processedText = processedText.replace(/\$([^\$\n]+?)\$/g, (match, formula, offset, string) => {
                    // 檢查前後是否是 $（避免匹配 $$）
                    const prevChar = offset > 0 ? string[offset - 1] : '';
                    const nextChar = offset + match.length < string.length ? string[offset + match.length] : '';

                    if (prevChar === '$' || nextChar === '$') {
                        return match; // 這是 $$ 的一部分，不處理
                    }

                    // 檢查是否包含數學符號特徵
                    if (formula.match(/[a-zA-Z_\\{}\^\(\)\[\]]/)) {
                        const index = protectedContent.mathInlines.length;
                        protectedContent.mathInlines.push(match);
                        return `MATHINLINE${index}MATHINLINE`;
                    }
                    return match;
                });

                // 3.1 保護 \( \) 行內數學公式
                processedText = processedText.replace(/\\\(([^\r\n]*?)\\\)/g, (match, formula, offset, string) => {
                    const prevChar = offset > 0 ? string[offset - 1] : '';
                    if (prevChar === '\\') {
                        return match;
                    }
                    const index = protectedContent.mathInlines.length;
                    protectedContent.mathInlines.push(match);
                    return `MATHINLINE${index}MATHINLINE`;
                });

                console.log('Protected content counts:', {
                    codeBlocks: protectedContent.codeBlocks.length,
                    mathBlocks: protectedContent.mathBlocks.length,
                    mathInlines: protectedContent.mathInlines.length
                });

                // 第二步：處理 Markdown
                try {
                    // 嘗試直接解析
                    htmlContent = marked.parse(processedText);
                    console.log('Marked parsing successful, output length:', htmlContent.length);
                } catch (parseError) {
                    console.error('Marked parsing failed:', parseError);
                    // 如果失敗，嘗試分段處理
                    const segments = processedText.split(/\n\n+/);
                    const processedSegments = [];

                    for (let segment of segments) {
                        try {
                            processedSegments.push(marked.parse(segment));
                        } catch (segmentError) {
                            console.warn('Segment parsing failed, using fallback:', segmentError);
                            // 對失敗的段落使用簡單處理
                            processedSegments.push(
                                segment
                                    .replace(/&/g, '&amp;')
                                    .replace(/</g, '&lt;')
                                    .replace(/>/g, '&gt;')
                                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                    .replace(/`(.*?)`/g, '<code>$1</code>')
                            );
                        }
                    }

                    htmlContent = processedSegments.join('</p><p>');
                    // 確保段落標籤正確
                    if (!htmlContent.startsWith('<p>')) {
                        htmlContent = '<p>' + htmlContent;
                    }
                    if (!htmlContent.endsWith('</p>')) {
                        htmlContent = htmlContent + '</p>';
                    }
                }

                // 第三步：恢復保護的內容
                // 注意：不要全局替換HTML實體，只在占位符周圍處理

                // 恢復代碼塊
                htmlContent = htmlContent.replace(/CODEBLOCK(\d+)CODEBLOCK/g, (match, index) => {
                    const code = protectedContent.codeBlocks[parseInt(index)];
                    if (code) {
                        // 更靈活的代碼塊匹配
                        const codeMatch = code.match(/^```([^\n]*)\n?([\s\S]*?)\n?```$/);
                        if (codeMatch) {
                            const lang = (codeMatch[1] || '').trim();
                            const content = codeMatch[2] || '';
                            const escapedContent = content
                                .replace(/&/g, '&amp;')
                                .replace(/</g, '&lt;')
                                .replace(/>/g, '&gt;');
                            return `<pre><code${lang ? ` class="language-${lang}"` : ''}>${escapedContent}</code></pre>`;
                        } else {
                            // 如果正則匹配失敗，直接返回原始代碼塊（去掉```）
                            const simpleContent = code.replace(/^```[^\n]*\n?/, '').replace(/\n?```$/, '');
                            const escapedContent = simpleContent
                                .replace(/&/g, '&amp;')
                                .replace(/</g, '&lt;')
                                .replace(/>/g, '&gt;');
                            return `<pre><code>${escapedContent}</code></pre>`;
                        }
                    }
                    return match;
                });

                // 恢復數學公式 - 處理可能被包裹在標籤中或被HTML編碼的情況
                // 先處理被<p>標籤包裹的塊級數學公式
                htmlContent = htmlContent.replace(/<p>MATHBLOCK(\d+)MATHBLOCK<\/p>/g, (match, index) => {
                    const math = protectedContent.mathBlocks[parseInt(index)];
                    return math ? `<p>${math}</p>` : match;
                });

                // 處理普通的塊級數學公式
                htmlContent = htmlContent.replace(/MATHBLOCK(\d+)MATHBLOCK/g, (match, index) => {
                    return protectedContent.mathBlocks[parseInt(index)] || match;
                });

                // 處理行內數學公式
                htmlContent = htmlContent.replace(/MATHINLINE(\d+)MATHINLINE/g, (match, index) => {
                    return protectedContent.mathInlines[parseInt(index)] || match;
                });

                // 如果還有被HTML編碼的占位符，嘗試恢復
                if (htmlContent.includes('&lt;') || htmlContent.includes('&gt;') || htmlContent.includes('&amp;')) {
                    // 只在占位符區域進行HTML解碼
                    htmlContent = htmlContent.replace(/(&lt;|&gt;|&amp;)?(CODEBLOCK|MATHBLOCK|MATHINLINE)(\d+)\2(&lt;|&gt;|&amp;)?/g,
                        (match, pre, type, index, post) => {
                            const realType = type;
                            const realIndex = parseInt(index);

                            if (realType === 'CODEBLOCK' && protectedContent.codeBlocks[realIndex]) {
                                const code = protectedContent.codeBlocks[realIndex];
                                const codeMatch = code.match(/^```([^\n]*)\n?([\s\S]*?)\n?```$/);
                                if (codeMatch) {
                                    const lang = (codeMatch[1] || '').trim();
                                    const content = codeMatch[2] || '';
                                    const escapedContent = content
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;');
                                    return `<pre><code${lang ? ` class="language-${lang}"` : ''}>${escapedContent}</code></pre>`;
                                }
                            } else if (realType === 'MATHBLOCK' && protectedContent.mathBlocks[realIndex]) {
                                return protectedContent.mathBlocks[realIndex];
                            } else if (realType === 'MATHINLINE' && protectedContent.mathInlines[realIndex]) {
                                return protectedContent.mathInlines[realIndex];
                            }

                            return match;
                        }
                    );
                }

                // 檢查恢復情況 - 更準確的檢查
                const codeBlocksRemaining = (htmlContent.match(/CODEBLOCK\d+CODEBLOCK/g) || []).length;
                const mathBlocksRemaining = (htmlContent.match(/MATHBLOCK\d+MATHBLOCK/g) || []).length;
                const mathInlinesRemaining = (htmlContent.match(/MATHINLINE\d+MATHINLINE/g) || []).length;
                const totalRemaining = codeBlocksRemaining + mathBlocksRemaining + mathInlinesRemaining;

                if (totalRemaining > 0) {
                    console.warn(`Warning: ${totalRemaining} placeholders were not restored:`, {
                        codeBlocks: codeBlocksRemaining,
                        mathBlocks: mathBlocksRemaining,
                        mathInlines: mathInlinesRemaining
                    });
                }

            } else {
                // marked 不可用時的後備方案
                console.log('Marked not available, using fallback');
                htmlContent = text
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`(.*?)`/g, '<code>$1</code>')
                    .replace(/\n\n+/g, '</p><p>')
                    .replace(/\n/g, '<br>');
                htmlContent = '<p>' + htmlContent + '</p>';
            }

            console.log('Final HTML length:', htmlContent.length);

            // 設置 HTML 內容
            container.innerHTML = htmlContent;

            // 第四步：渲染數學公式
            if (typeof MathJax !== 'undefined' && MathJax.startup) {
                MathJax.startup.promise.then(() => {
                    console.log('Starting MathJax rendering');
                    return MathJax.typesetPromise([container]);
                }).then(() => {
                    console.log('MathJax rendering completed');
                    resolve();
                }).catch((err) => {
                    console.warn('MathJax rendering error:', err);
                    resolve();
                });
            } else {
                console.log('MathJax not available');
                resolve();
            }

        } catch (err) {
            console.error('Critical rendering error:', err);
            console.error('Error stack:', err.stack);

            // 最後的後備方案：顯示原始文本
            const lines = text.split('\n');
            const escapedLines = lines.map(line => {
                return line
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
            });
            container.innerHTML = '<pre>' + escapedLines.join('\n') + '</pre>';
            resolve();
        }
    });
}

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
    }

    setState(newState) {
        Object.assign(this, newState);
        this.render();
    }

    render() {
        const container = document.getElementById('wrap');
        if (!container) return;

        const htmlContent = this.getHTML();

        // 如果內容包含 summary，使用新的渲染方式處理數學公式
        if (this.content && htmlContent.includes('markdown-content')) {
            // 先設置基本 HTML 結構
            container.innerHTML = htmlContent;

            // 然後對 markdown 內容進行特殊處理
            const markdownContainer = container.querySelector('.markdown-content');
            if (markdownContainer) {
                renderMarkdownWithMath(this.content, markdownContainer);
            }
        } else {
            // 對於不含 markdown 的內容，直接設置
            container.innerHTML = htmlContent;
            this.renderMath();
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
        return this.selectedModel || (this.meta && this.meta.model) || '';
    }

    getSummaryModel() {
        return (this.meta && this.meta.model) || '';
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
            this.loadSummary(pid, { model: options.model || '', force_regenerate: false });
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
        const disabled = this.loading || this.clearing ? 'disabled' : '';
        const regenLabel = this.regenerating ? 'Regenerating...' : 'Regenerate';
        const modelOptions = this.renderModelOptions();
        const errorNote = this.modelsError
            ? `<div class="summary-note" style="color: #d9534f;" role="status" aria-live="polite">${escapeHtml(this.modelsError)}</div>`
            : '';
        const notice = this.notice
            ? `<div class="summary-note" style="color: #b8860b;" role="status" aria-live="polite">${escapeHtml(this.notice)}</div>`
            : '';

        return `
            <div class="summary-actions">
                <label class="summary-action-label" for="model-select">Model</label>
                <select id="model-select" class="summary-model-select" onchange="summaryApp.handleModelChange(event)" ${disabled}>
                    ${modelOptions}
                </select>
                <button onclick="summaryApp.regenerate()" class="summary-action-btn" ${disabled}>
                    ${regenLabel}
                </button>
                <button onclick="summaryApp.clearCache()" class="summary-action-btn" ${disabled}>
                    ${this.clearing ? 'Clearing...' : 'Clear Cache'}
                </button>
                ${errorNote}
                ${notice}
            </div>
        `;
    }

    renderMetaLine() {
        const model = this.getSummaryModel();
        const selected = this.getCurrentModel();
        const timeStr = this.meta ? this.formatTimestamp(this.meta.generated_at || this.meta.updated_at) : '';
        let modelLabel = model ? `Model: ${escapeHtml(model)}` : '';
        if (model && selected && selected !== model) {
            modelLabel = `Showing cached summary from ${escapeHtml(model)} (selected ${escapeHtml(selected)})`;
        }
        const timeLabel = timeStr ? `Generated at ${escapeHtml(timeStr)}` : '';
        const note = [timeLabel, modelLabel].filter(Boolean).join(' · ');
        const fallbackNote = 'This summary is automatically generated by AI';

        return `
            <div class="summary-meta">
                <span class="summary-badge">AI Generated</span>
                <span class="summary-note">${note || fallbackNote}</span>
            </div>
        `;
    }

    getHTML() {
        // Paper header section - styled like paper list with full abstract
        const pidSafe = this.paper ? String(this.paper.id || '') : '';
        const titleSafe = this.paper ? escapeHtml(this.paper.title) : '';
        const authorsSafe = this.paper ? escapeHtml(this.paper.authors) : '';
        const timeSafe = this.paper ? escapeHtml(this.paper.time) : '';
        const tagsSafe = (this.paper && this.paper.tags) ? escapeHtml(this.paper.tags) : '';
        const abstractSafe = this.paper ? escapeHtml(this.paper.summary || 'No abstract available.') : '';

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
                            ${authorsSafe}
                        </div>
                        <div class="paper-meta-line">
                            <span class="paper-time">${timeSafe}</span>
                            ${tagsSafe ? `<span class="paper-tags">${tagsSafe}</span>` : ''}
                        </div>
                        <div class="paper-abstract">
                            ${abstractSafe}
                        </div>
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
            // 不在這裡處理 markdown，讓 render() 方法中的 renderMarkdownWithMath 函數處理
            summaryHTML = `
                <div class="summary-container" aria-busy="false">
                    <div class="summary-header">
                        <h2>Summary</h2>
                        ${this.renderMetaLine()}
                    </div>
                    ${this.renderActions()}
                    <div class="summary-content markdown-content" aria-live="polite">
                        <!-- 內容將由 renderMarkdownWithMath 函數處理 -->
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
    if (!this.pid || this.loading) return;
    this.loadSummary(this.pid, { force_regenerate: true, model: this.getCurrentModel() });
};

summaryApp.clearCache = async function() {
    if (!this.pid || this.clearing) return;
    this.clearAutoRetry();
    this.setState({ clearing: true, notice: '', error: null });
    try {
        await clearPaperCache(this.pid);
        this.setState({
            clearing: false,
            content: null,
            meta: null,
            notice: 'Cache cleared. Click Regenerate to fetch a fresh summary.',
        });
    } catch (error) {
        this.setState({ clearing: false, error: error.message });
    }
};

// Load summary function
summaryApp.loadSummary = async function(pid, options = {}) {
    const chosenModel = options.model || this.getCurrentModel();
    const force = Boolean(options.force_regenerate);
    const cacheOnly = Boolean(options.cache_only);
    this.clearAutoRetry();

    const shouldShowLoading = force || (!cacheOnly && !this.content);
    this.setState({
        loading: shouldShowLoading,
        error: null,
        regenerating: force,
        selectedModel: chosenModel || '',
        notice: cacheOnly ? '' : this.notice,
    });

    try {
        const result = await fetchSummary(pid, {
            model: chosenModel,
            force_regenerate: force,
            cache_only: cacheOnly,
        });
        const meta = result.meta || {};
        const content = result.content;
        if (
            typeof content === 'string' &&
            content.startsWith('# Error') &&
            content.includes('Summary is being generated')
        ) {
            this.setState({
                loading: false,
                regenerating: false,
                error: 'Summary is being generated, will refresh automatically...',
            });
            this.scheduleAutoRetry(pid, { model: chosenModel });
            return;
        }

        const selectedModel = meta.model || this.selectedModel || '';
        this.autoRetryCount = 0;
        this.setState({
            loading: false,
            regenerating: false,
            content: content,
            meta: meta,
            selectedModel,
            notice: '',
        });
    } catch (error) {
        if (error.code === 'summary_cache_miss' && cacheOnly) {
            this.setState({
                loading: false,
                regenerating: false,
                notice: 'No cached summary for this model. Click Regenerate to create one.',
                error: null,
            });
            return;
        }
        this.setState({ loading: false, regenerating: false, error: error.message });
        if (error.code === 'summary_timeout' || String(error.message || '').includes('Failed to fetch')) {
            this.scheduleAutoRetry(pid, { model: chosenModel });
        }
    }
};

summaryApp.loadModels = async function() {
    try {
        const models = await fetchModels();
        let selectedModel = this.selectedModel;
        if (!selectedModel && models.length > 0) {
            const preferred = String(this.defaultModel || '').trim();
            if (preferred) {
                const matched = models.find((m) => String(m.id || '') === preferred);
                selectedModel = matched ? matched.id || '' : '';
            }
            if (!selectedModel) {
                selectedModel = models[0].id || '';
            }
        }
        this.setState({ models, modelsError: null, selectedModel });
    } catch (error) {
        this.setState({ modelsError: error.message });
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
    });

    await summaryApp.loadModels();
    // Start loading summary after model list is available
    summaryApp.loadSummary(pid, { model: summaryApp.getCurrentModel() });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initSummaryApp);
