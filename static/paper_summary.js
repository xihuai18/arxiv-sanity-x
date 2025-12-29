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

function markdownItMathPlugin(md) {
    function isValidDelim(state, pos) {
        const max = state.posMax;
        const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1;
        const nextChar = pos + 1 <= max ? state.src.charCodeAt(pos + 1) : -1;
        let canOpen = true;
        let canClose = true;

        if (prevChar === 0x20 || prevChar === 0x09 || (nextChar >= 0x30 && nextChar <= 0x39)) {
            canClose = false;
        }
        if (nextChar === 0x20 || nextChar === 0x09) {
            canOpen = false;
        }

        return { can_open: canOpen, can_close: canClose };
    }

    function htmlLineBreak(state, silent) {
        if (state.src[state.pos] !== '<') return false;
        const slice = state.src.slice(state.pos);
        const match = slice.match(/^<br\s*\/?>/i);
        if (!match) return false;

        if (!silent) {
            const token = state.push('hardbreak', 'br', 0);
            token.markup = match[0];
        }
        state.pos += match[0].length;
        return true;
    }

    function mathInlineDollar(state, silent) {
        if (state.src[state.pos] !== '$') return false;

        const res = isValidDelim(state, state.pos);
        if (!res.can_open) {
            if (!silent) {
                state.pending += '$';
            }
            state.pos += 1;
            return true;
        }

        const start = state.pos + 1;
        let match = start;
        let pos;

        while ((match = state.src.indexOf('$', match)) !== -1) {
            pos = match - 1;
            while (state.src[pos] === '\\') {
                pos -= 1;
            }
            if ((match - pos) % 2 === 1) {
                break;
            }
            match += 1;
        }

        if (match === -1) {
            if (!silent) {
                state.pending += '$';
            }
            state.pos = start;
            return true;
        }

        if (match - start === 0) {
            if (!silent) {
                state.pending += '$$';
            }
            state.pos = start + 1;
            return true;
        }

        const resClose = isValidDelim(state, match);
        if (!resClose.can_close) {
            if (!silent) {
                state.pending += '$';
            }
            state.pos = start;
            return true;
        }

        if (!silent) {
            const token = state.push('math_inline', 'math', 0);
            token.content = state.src.slice(start, match);
            token.markup = '$';
        }

        state.pos = match + 1;
        return true;
    }

    function mathInlineParen(state, silent) {
        if (state.src.slice(state.pos, state.pos + 2) !== '\\(') return false;

        const start = state.pos + 2;
        let match = start;
        let pos;

        while ((match = state.src.indexOf('\\)', match)) !== -1) {
            pos = match - 1;
            while (pos >= 0 && state.src[pos] === '\\') {
                pos -= 1;
            }
            if ((match - pos) % 2 === 1) {
                break;
            }
            match += 2;
        }

        if (match === -1) {
            return false;
        }

        if (!silent) {
            const token = state.push('math_inline', 'math', 0);
            token.content = state.src.slice(start, match);
            token.markup = '\\(';
        }

        state.pos = match + 2;
        return true;
    }

    function mathInlineBracket(state, silent) {
        if (state.src.slice(state.pos, state.pos + 2) !== '\\[') return false;

        const start = state.pos + 2;
        let match = start;
        let pos;

        while ((match = state.src.indexOf('\\]', match)) !== -1) {
            pos = match - 1;
            while (pos >= 0 && state.src[pos] === '\\') {
                pos -= 1;
            }
            if ((match - pos) % 2 === 1) {
                break;
            }
            match += 2;
        }

        if (match === -1) {
            return false;
        }

        if (!silent) {
            const token = state.push('math_inline', 'math', 0);
            token.content = state.src.slice(start, match);
            token.markup = '\\[';
            token.displayMode = true;
        }

        state.pos = match + 2;
        return true;
    }

    function mathBlock(state, startLine, endLine, silent) {
        let pos = state.bMarks[startLine] + state.tShift[startLine];
        let max = state.eMarks[startLine];

        if (pos + 2 > max) return false;

        const opener = state.src.slice(pos, pos + 2);
        if (opener !== '$$' && opener !== '\\[') return false;

        const closer = opener === '$$' ? '$$' : '\\]';
        pos += 2;

        let firstLine = state.src.slice(pos, max);
        let found = false;
        let lastLine = '';
        let nextLine = startLine;

        if (silent) return true;

        if (firstLine.trim().endsWith(closer)) {
            firstLine = firstLine.trim().slice(0, -closer.length);
            found = true;
        }

        for (nextLine = startLine; !found;) {
            nextLine += 1;
            if (nextLine >= endLine) {
                break;
            }

            pos = state.bMarks[nextLine] + state.tShift[nextLine];
            max = state.eMarks[nextLine];

            if (pos < max && state.tShift[nextLine] < state.blkIndent) {
                break;
            }

            const line = state.src.slice(pos, max);
            if (line.trim().endsWith(closer)) {
                const lastPos = line.lastIndexOf(closer);
                lastLine = line.slice(0, lastPos);
                found = true;
            }
        }

        state.line = nextLine + 1;

        const token = state.push('math_block', 'math', 0);
        token.block = true;
        token.content = (firstLine && firstLine.trim() ? `${firstLine}\n` : '')
            + state.getLines(startLine + 1, nextLine, state.tShift[startLine], true)
            + (lastLine && lastLine.trim() ? lastLine : '');
        token.map = [startLine, state.line];
        token.markup = opener;

        return true;
    }

    md.inline.ruler.before('escape', 'html_line_break', htmlLineBreak);
    md.inline.ruler.before('escape', 'math_inline_paren', mathInlineParen);
    md.inline.ruler.before('escape', 'math_inline_bracket', mathInlineBracket);
    md.inline.ruler.after('escape', 'math_inline_dollar', mathInlineDollar);
    md.block.ruler.after('blockquote', 'math_block', mathBlock, {
        alt: ['paragraph', 'reference', 'blockquote', 'list']
    });

    md.renderer.rules.math_inline = function(tokens, idx) {
        const content = escapeHtml(tokens[idx].content);
        if (tokens[idx].displayMode || tokens[idx].markup === '\\[') {
            return `<span class="math-display">\\[${content}\\]</span>`;
        }
        return `<span class="math-inline">\\(${content}\\)</span>`;
    };

    md.renderer.rules.math_block = function(tokens, idx) {
        const content = escapeHtml(tokens[idx].content);
        return `<div class="math-display">\\[${content}\\]</div>`;
    };
}

let markdownRenderer = null;
let tocObserver = null;
let tocCollapsed = null;

function getMarkdownRenderer() {
    if (markdownRenderer) return markdownRenderer;
    if (typeof markdownit === 'undefined') return null;

    const md = markdownit({
        html: false,
        linkify: true,
        typographer: false,
        breaks: true
    });

    md.validateLink = (url) => isSafeUrl(url);
    md.use(markdownItMathPlugin);

    const defaultLinkRender = md.renderer.rules.link_open || function(tokens, idx, options, env, self) {
        return self.renderToken(tokens, idx, options);
    };

    md.renderer.rules.link_open = function(tokens, idx, options, env, self) {
        const href = tokens[idx].attrGet('href') || '';
        if (isSafeUrl(href)) {
            try {
                const url = new URL(href, window.location.href);
                if ((url.protocol === 'http:' || url.protocol === 'https:') && url.origin !== window.location.origin) {
                    tokens[idx].attrSet('target', '_blank');
                    tokens[idx].attrSet('rel', 'noopener noreferrer');
                }
            } catch (e) {
                // ignore invalid URLs
            }
        }
        return defaultLinkRender(tokens, idx, options, env, self);
    };

    markdownRenderer = md;
    return markdownRenderer;
}

function slugifyHeading(text, slugCounts) {
    const cleaned = String(text || '')
        .trim()
        .toLowerCase()
        .replace(/[!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~]/g, '')
        .replace(/\s+/g, '-');
    const base = cleaned || 'section';
    const count = slugCounts[base] || 0;
    slugCounts[base] = count + 1;
    return count > 0 ? `${base}-${count + 1}` : base;
}

function extractHeadingText(token) {
    if (!token) return '';
    if (token.type === 'inline' && Array.isArray(token.children)) {
        return token.children.map((child) => {
            if (child.type === 'text' || child.type === 'code_inline') {
                return child.content;
            }
            return '';
        }).join('');
    }
    return token.content || '';
}

function buildTocHtml(items) {
    if (!items || items.length < 2) return '';
    const list = items.map((item) => {
        const title = escapeHtml(item.title);
        const slug = escapeHtml(item.slug);
        return `<li class="toc-item toc-level-${item.level}"><a href="#${slug}">${title}</a></li>`;
    }).join('');
    return `
        <div class="toc-header">
            <div class="toc-title">Contents</div>
            <div class="toc-actions">
                <span class="toc-count">${items.length}</span>
                <button type="button" class="toc-toggle" aria-expanded="true">Collapse</button>
            </div>
        </div>
        <ul class="toc-list">
            ${list}
        </ul>
    `;
}

function setActiveTocLink(tocContainer, link) {
    if (!tocContainer) return;
    const active = tocContainer.querySelector('.toc-item a.is-active');
    if (active) {
        active.classList.remove('is-active');
    }
    if (link) {
        link.classList.add('is-active');
    }
}

function setupTocObserver(tocContainer, markdownContainer) {
    if (tocObserver) {
        tocObserver.disconnect();
        tocObserver = null;
    }
    if (!tocContainer || !markdownContainer) return;

    const headings = markdownContainer.querySelectorAll('h1, h2, h3, h4');
    if (!headings.length) return;

    const linkMap = new Map();
    const links = tocContainer.querySelectorAll('a[href^="#"]');
    links.forEach((link) => {
        const href = link.getAttribute('href') || '';
        const id = href.slice(1);
        if (id) {
            linkMap.set(id, link);
        }
    });

    tocObserver = new IntersectionObserver((entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting);
        if (!visible.length) return;
        visible.sort((a, b) => {
            if (b.intersectionRatio !== a.intersectionRatio) {
                return b.intersectionRatio - a.intersectionRatio;
            }
            return a.boundingClientRect.top - b.boundingClientRect.top;
        });
        const target = visible[0].target;
        const link = linkMap.get(target.id);
        setActiveTocLink(tocContainer, link);
    }, {
        rootMargin: '0px 0px -70% 0px',
        threshold: [0, 1]
    });

    headings.forEach((heading) => {
        tocObserver.observe(heading);
    });
}

function setupTocToggle(tocContainer) {
    if (!tocContainer) return;
    const toggle = tocContainer.querySelector('.toc-toggle');
    if (!toggle) return;

    if (tocCollapsed === null) {
        tocCollapsed = window.matchMedia('(max-width: 960px)').matches;
    }

    const applyState = (collapsed) => {
        tocContainer.classList.toggle('is-collapsed', collapsed);
        toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
        toggle.textContent = collapsed ? 'Expand' : 'Collapse';
    };

    applyState(tocCollapsed);

    toggle.addEventListener('click', () => {
        tocCollapsed = !tocCollapsed;
        applyState(tocCollapsed);
    });
}

function wrapMarkdownTables(container) {
    const tables = container.querySelectorAll('table');
    tables.forEach((table) => {
        const parent = table.parentElement;
        if (parent && parent.classList.contains('table-wrap')) return;
        const wrapper = document.createElement('div');
        wrapper.className = 'table-wrap';
        if (parent) {
            parent.insertBefore(wrapper, table);
        }
        wrapper.appendChild(table);
    });
}

function renderSummaryMarkdown(text, markdownContainer, tocContainer) {
    if (!text || !markdownContainer) return;

    const md = getMarkdownRenderer();
    if (!md) {
        markdownContainer.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
        return;
    }

    try {
        if (tocObserver) {
            tocObserver.disconnect();
            tocObserver = null;
        }
        const env = {};
        const tokens = md.parse(text, env);
        const slugCounts = {};
        const tocItems = [];

        for (let i = 0; i < tokens.length; i += 1) {
            const token = tokens[i];
            if (token.type !== 'heading_open') continue;
            const level = Number(token.tag.slice(1));
            if (Number.isNaN(level) || level > 4) continue;

            const inlineToken = tokens[i + 1];
            const title = extractHeadingText(inlineToken);
            if (!title) continue;

            const slug = slugifyHeading(title, slugCounts);
            token.attrSet('id', slug);
            tocItems.push({ level, title, slug });
        }

        markdownContainer.innerHTML = md.renderer.render(tokens, md.options, env);
        wrapMarkdownTables(markdownContainer);

        if (tocContainer) {
            const tocHtml = buildTocHtml(tocItems);
            tocContainer.innerHTML = tocHtml;
            tocContainer.classList.toggle('is-empty', !tocHtml);
            const contentContainer = tocContainer.closest('.summary-content');
            if (contentContainer) {
                contentContainer.classList.toggle('has-toc', Boolean(tocHtml));
            }
            if (tocHtml) {
                setupTocToggle(tocContainer);
                setupTocObserver(tocContainer, markdownContainer);
                const firstLink = tocContainer.querySelector('.toc-item a');
                if (firstLink) {
                    setActiveTocLink(tocContainer, firstLink);
                }
            }
        }

        if (typeof MathJax !== 'undefined' && MathJax.startup) {
            MathJax.startup.promise
                .then(() => MathJax.typesetPromise([markdownContainer]))
                .catch((err) => {
                    console.warn('MathJax rendering error:', err);
                });
        }
    } catch (err) {
        console.error('Markdown render error:', err);
        markdownContainer.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
    }
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
