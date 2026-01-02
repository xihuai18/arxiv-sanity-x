'use strict';

function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? (meta.getAttribute('content') || '') : '';
}

function csrfFetch(url, options) {
    const opts = options || {};
    const method = (opts.method || 'POST').toUpperCase();
    const headers = new Headers(opts.headers || {});
    const tok = getCsrfToken();
    if (tok) headers.set('X-CSRF-Token', tok);
    return fetch(url, { ...opts, method, headers, credentials: 'same-origin' });
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
    // Add math_block before 'code' rule to prevent indented math from being parsed as code blocks
    md.block.ruler.before('code', 'math_block', mathBlock, {
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

    // Disable indented code blocks to prevent math blocks from being parsed as code
    // Fenced code blocks (```) still work
    md.disable('code');

    md.validateLink = (url) => isSafeUrl(url);
    md.use(markdownItMathPlugin);

    // Custom image renderer to use figure/figcaption
    md.renderer.rules.image = function(tokens, idx, options, env, self) {
        const token = tokens[idx];
        const src = token.attrGet('src');
        const alt = token.content;
        const title = token.attrGet('title');

        if (!isSafeUrl(src)) return '';

        let caption = alt;
        if (title) caption = title;

        // Use figure for images with captions
        if (caption) {
            return `<figure>
                <img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy">
                <figcaption>${escapeHtml(caption)}</figcaption>
            </figure>`;
        }

        return `<figure><img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy"></figure>`;
    };

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

function setupImageZoom(container) {
    const images = container.querySelectorAll('img');
    images.forEach((img) => {
        // Remove any existing click listeners to prevent duplicates if called multiple times
        const newImg = img.cloneNode(true);
        img.parentNode.replaceChild(newImg, img);

        newImg.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Create overlay
            const overlay = document.createElement('div');
            overlay.className = 'image-zoom-overlay';
            overlay.innerHTML = `
                <div class="image-zoom-container">
                    <img src="${newImg.src}" alt="${newImg.alt || ''}" />
                    <button class="image-zoom-close" aria-label="Close">&times;</button>
                </div>
            `;
            document.body.appendChild(overlay);
            document.body.style.overflow = 'hidden';

            // Close on click
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay || e.target.classList.contains('image-zoom-close')) {
                    overlay.remove();
                    document.body.style.overflow = '';
                }
            });

            // Close on Escape key
            const handleEscape = (e) => {
                if (e.key === 'Escape') {
                    overlay.remove();
                    document.body.style.overflow = '';
                    document.removeEventListener('keydown', handleEscape);
                }
            };
            document.addEventListener('keydown', handleEscape);
        });

        // Add error handling for broken images
        newImg.addEventListener('error', () => {
            newImg.style.display = 'none';
            const errorMsg = document.createElement('span');
            errorMsg.className = 'image-load-error';
            errorMsg.textContent = '[Image failed to load]';
            newImg.parentNode.insertBefore(errorMsg, newImg.nextSibling);
        });
    });
}

/**
 * Normalize indented display math blocks to prevent them from being parsed as code blocks.
 * Also ensures multi-line math blocks are properly formatted for parsing.
 */
function normalizeIndentedDisplayMath(markdownText) {
    const text = String(markdownText || '').replace(/\r\n/g, '\n');
    const lines = text.split('\n');
    let inMathBlock = false;
    let mathOpener = '';
    let mathCloser = '';

    for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const trimmed = String(line).trim();

        if (!inMathBlock) {
            // Check if this line starts a deeply indented math block (4+ spaces)
            const mathStart = line.match(/^([ \t]{4,})(\\\[|\$\$)/);
            if (mathStart) {
                mathOpener = mathStart[2];
                mathCloser = mathOpener === '$$' ? '$$' : '\\]';
                inMathBlock = true;
                // Reduce indentation to 0-2 spaces to avoid code block parsing
                lines[i] = line.replace(/^[ \t]{4,}/, '');
                // Check if closer is on the same line (after opener)
                const afterOpener = trimmed.slice(mathOpener.length);
                if (afterOpener.includes(mathCloser)) {
                    inMathBlock = false;
                    mathCloser = '';
                }
            }
            continue;
        }

        // Inside math block - reduce indentation
        lines[i] = line.replace(/^[ \t]{4,}/, '');

        if (trimmed.includes(mathCloser)) {
            inMathBlock = false;
            mathCloser = '';
        }
    }

    return lines.join('\n');
}

/**
 * Fix \tag placement in aligned environments for MathJax compatibility.
 * MathJax doesn't allow \tag inside aligned environment, but LaTeX does.
 * This function moves \tag from inside aligned to after \end{aligned}.
 */
function fixAlignedTags(text) {
    // Match aligned environments with \tag inside, and move \tag to after \end{aligned}
    // Handles: \begin{aligned}...\tag{N}...\end{aligned}
    // Converts to: \begin{aligned}...\end{aligned}\tag{N}
    return text.replace(
        /(\\begin\{aligned\})([\s\S]*?)(\\tag\{[^}]+\})([\s\S]*?)(\\end\{aligned\})/g,
        (match, begin, content1, tag, content2, end) => {
            // Remove the tag from inside and place it after \end{aligned}
            return begin + content1 + content2 + end + ' ' + tag;
        }
    );
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
        // Preprocess: reduce indentation of deeply indented math blocks
        let normalizedText = normalizeIndentedDisplayMath(text);
        // Fix \tag placement in aligned environments for MathJax compatibility
        normalizedText = fixAlignedTags(normalizedText);
        const env = {};
        const tokens = md.parse(normalizedText, env);
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
        setupImageZoom(markdownContainer);

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
        this.pendingConfirm = null; // 'clearModel' or 'clearAll'
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

        // Render tag dropdown and bind events (once) after DOM update
        if (typeof user !== 'undefined' && user) {
            attachTagEventListeners(); // Only binds once due to tagEventsBound flag
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

        return `
            <div class="summary-actions">
                <label class="summary-action-label" for="model-select">Model</label>
                <select id="model-select" class="summary-model-select" onchange="summaryApp.handleModelChange(event)" ${disabled}>
                    ${modelOptions}
                </select>
                <button onclick="summaryApp.regenerate()" class="summary-action-btn" ${disabled}>
                    ${regenLabel}
                </button>
                <div class="summary-btn-group">
                    <button onclick="summaryApp.requestClearModel()" class="summary-action-btn summary-btn-warning" ${disabled} title="Clear summary for current model only">
                        ${this.clearing === 'model' ? 'Clearing...' : 'Clear Current Summary'}
                    </button>
                    ${this.pendingConfirm === 'clearModel' ? `
                        <div class="confirm-popup" role="dialog" aria-labelledby="confirm-title">
                            <div class="confirm-content">
                                <strong id="confirm-title">Clear summary for ${modelLabel}?</strong>
                                <p>This will only remove the summary generated by this model.</p>
                                <div class="confirm-actions">
                                    <button onclick="summaryApp.confirmClearModel()" class="confirm-btn confirm-yes">确定</button>
                                    <button onclick="summaryApp.cancelConfirm()" class="confirm-btn confirm-no">取消</button>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                </div>
                <div class="summary-btn-group">
                    <button onclick="summaryApp.requestClearAll()" class="summary-action-btn summary-btn-danger" ${disabled} title="Clear all caches (all models, HTML, MinerU, etc.)">
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
                            ${authorsSafe}
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
        this.setState({ clearing: false, error: error.message });
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
        this.setState({ clearing: false, error: error.message });
    }
};

summaryApp.cancelConfirm = function() {
    this.setState({ pendingConfirm: null });
};

// Deprecated: kept for backward compatibility
summaryApp.clearCache = summaryApp.confirmClearAll;

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
                notice: 'No cached summary for this model. Click Generate to create one.',
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
        this.setState({ models, modelsError: null });
    } catch (error) {
        this.setState({ modelsError: error.message });
    }
};

summaryApp.selectInitialModel = async function(pid) {
    try {
        // Get available summaries for this paper
        const availableSummaries = await fetchAvailableSummaries(pid);

        let selectedModel = '';

        // If there are available summaries, select the first one from model list that has a summary
        if (availableSummaries.length > 0 && this.models.length > 0) {
            for (const model of this.models) {
                const modelId = String(model.id || '');
                if (availableSummaries.includes(modelId)) {
                    selectedModel = modelId;
                    break;
                }
            }
        }

        // If no available summary found, use default model
        if (!selectedModel) {
            const preferred = String(this.defaultModel || '').trim();
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
    }
}

// MultiSelectDropdown component for tags
function MultiSelectDropdown(selectedTags, availableTags, isOpen, callbacks) {
    const { onToggle, onTagToggle, onRemoveTag, onNewTagChange, onAddNewTag, onSearchChange } = callbacks;
    const { newTagValue, searchValue } = summaryApp;

    const selectedTagElements = selectedTags.map((tag, ix) =>
        `<div class="multi-select-selected-tag">
            <span>${escapeHtml(tag)}</span>
            <span class="remove-tag" data-tag="${escapeHtml(tag)}">×</span>
        </div>`
    ).join('');

    const triggerContent = selectedTags.length > 0 ?
        `<div class="multi-select-selected-tags">${selectedTagElements}</div>` :
        '<div class="multi-select-placeholder">Select tags...</div>';

    // Filter tags by search value
    const searchLower = (searchValue || '').toLowerCase();
    const filteredTags = availableTags.filter(tag =>
        tag.toLowerCase().includes(searchLower)
    );

    const optionElements = filteredTags.map((tag, ix) => {
        const isSelected = selectedTags.includes(tag);
        return `<div class="multi-select-option" data-tag="${escapeHtml(tag)}">
            <input type="checkbox" class="multi-select-checkbox" ${isSelected ? 'checked' : ''} />
            <span class="multi-select-option-text">${escapeHtml(tag)}</span>
        </div>`;
    }).join('');

    const dropdownMenu = isOpen ? `
        <div class="multi-select-dropdown-menu">
            <div class="multi-select-search">
                <input type="text"
                       id="tag-search-input"
                       placeholder="Search tags..."
                       value="${escapeHtml(searchValue)}"
                       autocomplete="off" />
            </div>
            ${optionElements}
            <div class="multi-select-new-tag">
                <input type="text"
                       id="new-tag-input"
                       placeholder="Enter new tag..."
                       value="${escapeHtml(newTagValue)}"
                       autocomplete="off" />
                <button id="add-new-tag-btn" ${!newTagValue.trim() ? 'disabled' : ''}>Add</button>
            </div>
        </div>
    ` : '';

    return `<div class="multi-select-dropdown ${isOpen ? 'open' : ''}" id="summary-tag-dropdown-inner">
        <div class="multi-select-trigger ${isOpen ? 'active' : ''}">
            <div class="multi-select-content">
                ${triggerContent}
            </div>
            <span class="multi-select-arrow">${isOpen ? '▲' : '▼'}</span>
        </div>
        ${dropdownMenu}
    </div>`;
}

function renderTagDropdown() {
    const container = document.getElementById('summary-tag-dropdown');
    if (!container) return;

    const html = MultiSelectDropdown(
        summaryApp.userTags || [],
        summaryApp.availableTags || [],
        summaryApp.tagDropdownOpen,
        {}
    );

    container.innerHTML = html;

    // Restore input values after re-render
    const searchInput = container.querySelector('#tag-search-input');
    if (searchInput && summaryApp.tagSearchValue) {
        searchInput.value = summaryApp.tagSearchValue;
    }

    const newTagInput = container.querySelector('#new-tag-input');
    if (newTagInput && summaryApp.newTagValue) {
        newTagInput.value = summaryApp.newTagValue;
    }
}

// Use event delegation - bind once on document body
let tagEventsBound = false;

function attachTagEventListeners() {
    if (tagEventsBound) return;

    tagEventsBound = true;

    // Use event delegation on document body (persists through render cycles)
    document.body.addEventListener('click', (e) => {
        const target = e.target;

        // Only handle events within summary-tag-dropdown
        if (!target.closest('#summary-tag-dropdown')) return;

        // Handle trigger click
        const trigger = target.closest('.multi-select-trigger');
        if (trigger) {
            e.preventDefault();
            e.stopPropagation();
            handleToggleTagDropdown();
            return;
        }

        // Handle remove tag
        const removeBtn = target.closest('.remove-tag');
        if (removeBtn) {
            e.preventDefault();
            e.stopPropagation();
            const tagName = removeBtn.getAttribute('data-tag');
            if (tagName) handleRemoveTag(tagName);
            return;
        }

        // Handle option click
        const option = target.closest('.multi-select-option');
        if (option) {
            e.preventDefault();
            e.stopPropagation();
            const tagName = option.getAttribute('data-tag');
            if (tagName) handleTagToggle(tagName);
            return;
        }

        // Handle add new tag button
        const addBtn = target.closest('#add-new-tag-btn');
        if (addBtn) {
            e.preventDefault();
            e.stopPropagation();
            handleAddNewTag();
            return;
        }

        // Stop propagation for input clicks
        if (target.closest('#tag-search-input') || target.closest('#new-tag-input')) {
            e.stopPropagation();
            return;
        }
    });

    // Handle input events
    document.body.addEventListener('input', (e) => {
        const target = e.target;

        if (target.id === 'tag-search-input' && target.closest('#summary-tag-dropdown')) {
            summaryApp.tagSearchValue = target.value;
            renderTagDropdown();
            // Restore focus
            setTimeout(() => {
                const input = document.getElementById('tag-search-input');
                if (input) input.focus();
            }, 0);
        }

        if (target.id === 'new-tag-input' && target.closest('#summary-tag-dropdown')) {
            summaryApp.newTagValue = target.value;
            renderTagDropdown();
            // Restore focus
            setTimeout(() => {
                const input = document.getElementById('new-tag-input');
                if (input) input.focus();
            }, 0);
        }
    });

    // Handle keypress for new tag input
    document.body.addEventListener('keypress', (e) => {
        if (e.target.id === 'new-tag-input' && e.target.closest('#summary-tag-dropdown')) {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleAddNewTag();
            }
        }
    });
}

function handleToggleTagDropdown() {
    summaryApp.tagDropdownOpen = !summaryApp.tagDropdownOpen;
    if (summaryApp.tagDropdownOpen) {
        summaryApp.tagSearchValue = '';
    }
    renderTagDropdown();
}

function handleTagToggle(tagName) {
    const isSelected = summaryApp.userTags.includes(tagName);

    if (isSelected) {
        handleRemoveTag(tagName);
    } else {
        handleAddTag(tagName);
    }
}

async function handleAddTag(tagName) {
    if (!summaryApp.paper || !summaryApp.paper.id) {
        alert('Paper ID not found');
        return;
    }

    try {
        const response = await csrfFetch(`/add/${summaryApp.paper.id}/${encodeURIComponent(tagName)}`);
        const text = await response.text();

        if (text.startsWith('ok')) {
            if (!summaryApp.userTags.includes(tagName)) {
                summaryApp.userTags = [...summaryApp.userTags, tagName];
            }
            renderTagDropdown();
            console.log(`Added tag: ${tagName}`);
        } else {
            console.error('Server error adding tag:', text);
            alert('Failed to add tag: ' + text);
        }
    } catch (error) {
        console.error('Error adding tag:', error);
        alert('Network error, failed to add tag');
    }
}

async function handleRemoveTag(tagName) {
    if (!summaryApp.paper || !summaryApp.paper.id) {
        alert('Paper ID not found');
        return;
    }

    try {
        const response = await csrfFetch(`/sub/${summaryApp.paper.id}/${encodeURIComponent(tagName)}`);
        const text = await response.text();

        if (text.startsWith('ok')) {
            summaryApp.userTags = summaryApp.userTags.filter(tag => tag !== tagName);
            renderTagDropdown();
            console.log(`Removed tag: ${tagName}`);
        } else {
            console.error('Server error removing tag:', text);
            alert('Failed to remove tag: ' + text);
        }
    } catch (error) {
        console.error('Error removing tag:', error);
        alert('Network error, failed to remove tag');
    }
}

async function handleAddNewTag() {
    const trimmedTag = summaryApp.newTagValue.trim();

    if (!trimmedTag) return;

    if (summaryApp.userTags.includes(trimmedTag)) {
        alert('Tag already exists');
        return;
    }

    try {
        const response = await csrfFetch(`/add/${summaryApp.paper.id}/${encodeURIComponent(trimmedTag)}`);
        const text = await response.text();

        if (text.startsWith('ok')) {
            summaryApp.userTags = [...summaryApp.userTags, trimmedTag];
            summaryApp.newTagValue = '';

            // Add to available tags if not already there
            if (!summaryApp.availableTags.includes(trimmedTag)) {
                summaryApp.availableTags = [...summaryApp.availableTags, trimmedTag].sort();
            }

            renderTagDropdown();
            console.log(`Added new tag: ${trimmedTag}`);
        } else {
            console.error('Server error adding new tag:', text);
            alert('Failed to add new tag: ' + text);
        }
    } catch (error) {
        console.error('Error adding new tag:', error);
        alert('Network error, failed to add new tag');
    }
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

    // Close dropdown when clicking outside (use mousedown like main page)
    document.addEventListener('mousedown', (e) => {
        const dropdown = document.getElementById('summary-tag-dropdown');
        if (dropdown && !dropdown.contains(e.target) && summaryApp.tagDropdownOpen) {
            summaryApp.tagDropdownOpen = false;
            renderTagDropdown();
        }
    });

    // Close dropdown on Escape key (like main page)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && summaryApp.tagDropdownOpen) {
            summaryApp.tagDropdownOpen = false;
            renderTagDropdown();
        }
        // Close confirm popup on Escape
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
