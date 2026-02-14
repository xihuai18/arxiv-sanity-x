'use strict';

// Shared summary markdown renderer utilities
// Exposes: window.ArxivSanitySummaryMarkdown

(function (global) {
    if (typeof window === 'undefined') return;

    const NS = 'ArxivSanitySummaryMarkdown';
    function _getCommonUtils() {
        return global.ArxivSanityCommon || null;
    }
    function _getRenderer() {
        return global.ArxivSanityMarkdownRenderer || null;
    }
    function _getDomUtils() {
        return global.ArxivSanitySummaryMarkdownDom || null;
    }

    function escapeHtml(text) {
        const CommonUtils = _getCommonUtils();
        if (CommonUtils && typeof CommonUtils.escapeHtml === 'function')
            return CommonUtils.escapeHtml(text);
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function isSafeUrl(href) {
        const CommonUtils = _getCommonUtils();
        if (CommonUtils && typeof CommonUtils.isSafeUrl === 'function')
            return CommonUtils.isSafeUrl(href);
        try {
            const u = new URL(String(href || ''), window.location.href);
            return u.protocol === 'http:' || u.protocol === 'https:' || u.protocol === 'mailto:';
        } catch (e) {
            return false;
        }
    }

    function isNodeConnected(node) {
        if (!node) return false;
        try {
            if (typeof node.isConnected === 'boolean') return node.isConnected;
        } catch (e) {}
        try {
            return !!(document && document.body && document.body.contains(node));
        } catch (e) {
            return true;
        }
    }

    // Whitelist of safe HTML tags allowed in markdown content
    // These are commonly used for formatting and don't pose security risks
    const SAFE_HTML_TAGS = new Set([
        'sub',
        'sup', // Subscript/superscript (common in scientific text)
        'mark',
        'del',
        'ins', // Highlighting, strikethrough, insertion
        'small',
        'big', // Text size
        'abbr',
        'dfn', // Abbreviations and definitions
        'kbd',
        'samp',
        'var', // Keyboard, sample output, variables
        'cite',
        'q', // Citations and inline quotes
        'time',
        'data', // Semantic elements
        'ruby',
        'rt',
        'rp', // Ruby annotations (for CJK text)
        'bdi',
        'bdo', // Bidirectional text
        'wbr', // Word break opportunity
        'br',
        'hr', // Line/horizontal breaks
        'span',
        'div', // Generic containers (with limited attributes)
        'strong',
        'em',
        'b',
        'i',
        'u',
        's', // Basic formatting
        'table',
        'thead',
        'tbody',
        'tfoot',
        'tr',
        'th',
        'td',
        'caption',
        'colgroup',
        'col', // Tables
        'ul',
        'ol',
        'li',
        'dl',
        'dt',
        'dd', // Lists
        'details',
        'summary', // Collapsible content
        'figure',
        'figcaption', // Figures
        'blockquote', // Block quotes
    ]);

    // Safe attributes for HTML tags
    const SAFE_ATTRS = new Set([
        'class',
        'id',
        'title',
        'lang',
        'dir',
        'colspan',
        'rowspan',
        'headers',
        'scope', // Table attributes
        'datetime',
        'value', // time/data attributes
        'open', // details element
        'start',
        'type',
        'reversed', // list attributes
        'align',
        'valign', // alignment (deprecated but harmless)
    ]);

    /**
     * Escape HTML for math content - only escape < and > to prevent HTML injection,
     * but preserve & which is used in LaTeX for column separators (e.g., in cases, align environments)
     * @param {string} text - Math content to escape
     * @returns {string} Escaped string safe for HTML but preserving LaTeX syntax
     */
    function escapeMathHtml(text) {
        // IMPORTANT: summary markdown files may contain hard line-wrapping that
        // can split TeX control sequences (e.g. "\\mathcal" -> "\\m\nathcal").
        // Remove newlines inside math content to keep TeX parseable.
        return String(text || '')
            .replace(/\r\n|\r|\n/g, '')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    /**
     * Normalize TeX string for MathJax parsing.
     * Summary markdown may be hard-wrapped; newlines can break TeX control sequences.
     */
    function normalizeTex(text) {
        return String(text || '').replace(/\r\n|\r|\n/g, '');
    }

    /**
     * Escape for HTML attribute values.
     * We intentionally escape '&' here because browsers will decode it back when reading attributes,
     * while keeping the HTML safe. This preserves LaTeX '&' separators via getAttribute().
     */
    function escapeHtmlAttr(text) {
        return escapeHtml(String(text || ''));
    }

    /**
     * Sanitize HTML content - only allow safe tags and attributes
     * @param {string} html - Raw HTML content
     * @returns {string} Sanitized HTML
     */
    function sanitizeHtmlContent(html) {
        if (!html) return '';

        const DROP_TAGS = new Set([
            'script',
            'style',
            'iframe',
            'object',
            'embed',
            'link',
            'meta',
            'base',
            'form',
            'input',
            'textarea',
            'button',
            'select',
            'option',
        ]);

        const stripNullBytes = value =>
            String(value || '')
                .split('\u0000')
                .join('');

        const sanitizeInto = (parentOut, nodeIn) => {
            if (!nodeIn) return;

            // Text nodes
            if (nodeIn.nodeType === Node.TEXT_NODE) {
                parentOut.appendChild(document.createTextNode(nodeIn.nodeValue || ''));
                return;
            }

            // Ignore comments and other nodes.
            if (nodeIn.nodeType !== Node.ELEMENT_NODE) {
                return;
            }

            const elIn = /** @type {Element} */ (nodeIn);
            const tag = String(elIn.tagName || '').toLowerCase();

            // Hard-drop high-risk tags.
            if (DROP_TAGS.has(tag)) return;

            // Unwrap unknown tags: keep their (sanitized) children.
            if (!SAFE_HTML_TAGS.has(tag)) {
                const children = Array.from(elIn.childNodes || []);
                children.forEach(child => sanitizeInto(parentOut, child));
                return;
            }

            const elOut = document.createElement(tag);

            // Copy allowlisted attributes only.
            try {
                const attrs = elIn.attributes ? Array.from(elIn.attributes) : [];
                attrs.forEach(a => {
                    const name = String(a && a.name ? a.name : '').toLowerCase();
                    if (!name) return;
                    if (!(SAFE_ATTRS.has(name) || name.startsWith('data-'))) return;
                    // Disallow inline event handlers regardless.
                    if (name.startsWith('on')) return;
                    const value = stripNullBytes(a && a.value ? a.value : '');
                    elOut.setAttribute(name, value);
                });
            } catch (e) {}

            // Sanitize children.
            const children = Array.from(elIn.childNodes || []);
            children.forEach(child => sanitizeInto(elOut, child));

            parentOut.appendChild(elOut);
        };

        // DOM-based allowlist sanitization.
        // Regex-based sanitizers are brittle against malformed HTML; DOM parsing is more robust.
        try {
            if (typeof DOMParser === 'undefined' || typeof document === 'undefined') {
                return escapeHtml(String(html || ''));
            }

            const parser = new DOMParser();
            // Wrap into a container to preserve top-level nodes.
            const doc = parser.parseFromString(`<div>${String(html)}</div>`, 'text/html');
            const container = doc && doc.body ? doc.body.firstElementChild : null;
            if (!container) return '';

            const out = document.createElement('div');

            Array.from(container.childNodes || []).forEach(n => sanitizeInto(out, n));
            return out.innerHTML;
        } catch (e) {
            return escapeHtml(String(html || ''));
        }
    }

    /**
     * LaTeX theorem-like environment definitions
     * Maps environment names to their display labels and CSS classes
     */
    const THEOREM_ENVIRONMENTS = {
        // Standard theorem environments
        theorem: { label: 'Theorem', class: 'latex-theorem' },
        lemma: { label: 'Lemma', class: 'latex-lemma' },
        proposition: { label: 'Proposition', class: 'latex-proposition' },
        corollary: { label: 'Corollary', class: 'latex-corollary' },
        conjecture: { label: 'Conjecture', class: 'latex-conjecture' },

        // Definition-like environments
        definition: { label: 'Definition', class: 'latex-definition' },
        axiom: { label: 'Axiom', class: 'latex-axiom' },

        // Remark-like environments
        remark: { label: 'Remark', class: 'latex-remark' },
        note: { label: 'Note', class: 'latex-note' },
        observation: { label: 'Observation', class: 'latex-observation' },

        // Example environments
        example: { label: 'Example', class: 'latex-example' },
        exercise: { label: 'Exercise', class: 'latex-exercise' },
        problem: { label: 'Problem', class: 'latex-problem' },
        question: { label: 'Question', class: 'latex-question' },

        // Proof environment
        proof: { label: 'Proof', class: 'latex-proof', isProof: true },

        // Algorithm environments
        algorithm: { label: 'Algorithm', class: 'latex-algorithm' },

        // Claim and case
        claim: { label: 'Claim', class: 'latex-claim' },
        case: { label: 'Case', class: 'latex-case' },

        // Abstract (sometimes used in papers)
        abstract: { label: 'Abstract', class: 'latex-abstract' },

        // Assumption
        assumption: { label: 'Assumption', class: 'latex-assumption' },
        hypothesis: { label: 'Hypothesis', class: 'latex-hypothesis' },
    };

    /**
     * Pre-process LaTeX theorem-like environments into HTML
     * Converts \begin{theorem}...\end{theorem} etc. to styled HTML blocks
     * @param {string} text - Input text
     * @returns {string} Text with theorem environments converted to HTML
     */
    function preprocessTheoremEnvironments(text) {
        if (!text) return text;
        let s = String(text);

        // Process each theorem environment type
        for (const [envName, config] of Object.entries(THEOREM_ENVIRONMENTS)) {
            // Match \begin{envName}[optional title]...\end{envName}
            // Use a non-greedy match and handle nested content carefully
            const regex = new RegExp(
                `\\\\begin\\{${envName}\\}(?:\\[([^\\]]+)\\])?([\\s\\S]*?)\\\\end\\{${envName}\\}`,
                'gi'
            );

            s = s.replace(regex, (match, optionalTitle, content) => {
                // Escape title to prevent XSS
                const safeTitle = optionalTitle ? ` (${escapeHtml(optionalTitle.trim())})` : '';
                const label = config.label;
                const cssClass = config.class;
                const isProof = config.isProof;

                // Clean up the content - trim leading/trailing whitespace
                // Note: content will be processed by markdown-it later, which handles sanitization
                // But we escape any raw HTML that might be in the content for safety
                let cleanContent = content.trim();

                // For proof environments, add QED symbol at the end
                const qed = isProof ? '<span class="latex-qed">□</span>' : '';

                // Use data attribute to mark this as a theorem block for later processing
                // The content is wrapped in a special marker that markdown-it will process
                return `<div class="latex-env ${escapeHtml(cssClass)}">
<div class="latex-env-header"><strong>${escapeHtml(label)}${safeTitle}</strong></div>
<div class="latex-env-content">${cleanContent}</div>
${qed}</div>`;
            });
        }

        return s;
    }

    /**
     * Pre-process LaTeX text formatting commands outside of math mode
     * Converts \textbf{...}, \textit{...}, etc. to HTML
     * @param {string} text - Input text
     * @returns {string} Text with LaTeX text commands converted to HTML
     */
    function preprocessLatexTextCommands(text) {
        if (!text) return text;
        const input = String(text);

        function applyOnText(s) {
            let out = String(s);

            out = out.replace(
                /\\textbf\{([^}]*)\}/g,
                (m, inner) => `<strong>${escapeHtml(inner)}</strong>`
            );
            out = out.replace(
                /\\textit\{([^}]*)\}/g,
                (m, inner) => `<em>${escapeHtml(inner)}</em>`
            );
            out = out.replace(/\\emph\{([^}]*)\}/g, (m, inner) => `<em>${escapeHtml(inner)}</em>`);
            out = out.replace(
                /\\texttt\{([^}]*)\}/g,
                (m, inner) => `<code>${escapeHtml(inner)}</code>`
            );

            out = out.replace(/\\textrm\{([^}]*)\}/g, (m, inner) => escapeHtml(inner));
            out = out.replace(/\\textsf\{([^}]*)\}/g, (m, inner) => escapeHtml(inner));

            out = out.replace(
                /\\underline\{([^}]*)\}/g,
                (m, inner) => `<u>${escapeHtml(inner)}</u>`
            );
            out = out.replace(
                /\\textsc\{([^}]*)\}/g,
                (m, inner) => `<span class="latex-smallcaps">${escapeHtml(inner)}</span>`
            );
            out = out.replace(
                /\\textsuperscript\{([^}]*)\}/g,
                (m, inner) => `<sup>${escapeHtml(inner)}</sup>`
            );
            out = out.replace(
                /\\textsubscript\{([^}]*)\}/g,
                (m, inner) => `<sub>${escapeHtml(inner)}</sub>`
            );

            out = out.replace(/\\url\{([^}]*)\}/g, (match, url) => {
                const u = String(url || '');
                if (isSafeUrl(u)) {
                    const safe = escapeHtml(u);
                    return `<a href="${safe}" target="_blank" rel="noopener noreferrer">${safe}</a>`;
                }
                return escapeHtml(u);
            });
            out = out.replace(/\\href\{([^}]*)\}\{([^}]*)\}/g, (match, url, t) => {
                const u = String(url || '');
                const txt = String(t || '');
                if (isSafeUrl(u)) {
                    return `<a href="${escapeHtml(u)}" target="_blank" rel="noopener noreferrer">${escapeHtml(txt)}</a>`;
                }
                return escapeHtml(txt);
            });

            out = out.replace(
                /\\footnote\{([^}]*)\}/g,
                (m, inner) =>
                    `<span class="latex-footnote" title="${escapeHtmlAttr(inner)}">†</span>`
            );
            out = out.replace(
                /\\cite\{([^}]*)\}/g,
                (m, inner) => `<span class="latex-cite">[${escapeHtml(inner)}]</span>`
            );
            out = out.replace(
                /\\eqref\{([^}]*)\}/g,
                (m, inner) => `<span class="latex-ref">(${escapeHtml(inner)})</span>`
            );
            out = out.replace(
                /\\ref\{([^}]*)\}/g,
                (m, inner) => `<span class="latex-ref">${escapeHtml(inner)}</span>`
            );

            out = out.replace(/\\textcolor\{([^}]*)\}\{([^}]*)\}/g, (match, color, inner) => {
                const colorLower = String(color || '').toLowerCase();
                const colorClass = /^[a-zA-Z]+$/.test(colorLower)
                    ? `latex-color-${colorLower}`
                    : 'latex-color-default';
                return `<span class="${colorClass}">${escapeHtml(inner)}</span>`;
            });
            out = out.replace(/\\colorbox\{([^}]*)\}\{([^}]*)\}/g, (match, color, inner) => {
                const colorLower = String(color || '').toLowerCase();
                const colorClass = /^[a-zA-Z]+$/.test(colorLower)
                    ? `latex-colorbox-${colorLower}`
                    : 'latex-colorbox-default';
                return `<span class="${colorClass}">${escapeHtml(inner)}</span>`;
            });

            out = out.replace(
                /\\sout\{([^}]*)\}/g,
                (m, inner) => `<del>${escapeHtml(inner)}</del>`
            );
            out = out.replace(/\\uline\{([^}]*)\}/g, (m, inner) => `<u>${escapeHtml(inner)}</u>`);

            out = out.replace(
                /\\paragraph\{([^}]*)\}/g,
                (m, inner) => `<strong class="latex-paragraph">${escapeHtml(inner)}</strong> `
            );
            out = out.replace(
                /\\subparagraph\{([^}]*)\}/g,
                (m, inner) => `<strong class="latex-subparagraph">${escapeHtml(inner)}</strong> `
            );

            return out;
        }

        // Apply replacements only outside math segments ($...$, $$...$$, \\(...\\), \\[...\\]).
        let i = 0;
        let out = '';
        while (i < input.length) {
            const rest = input.slice(i);
            const candidates = [
                { k: 'dollar2', s: '$$', ix: rest.indexOf('$$') },
                { k: 'paren', s: '\\\\(', ix: rest.indexOf('\\(') },
                { k: 'bracket', s: '\\\\[', ix: rest.indexOf('\\[') },
                { k: 'dollar1', s: '$', ix: rest.indexOf('$') },
            ].filter(x => x.ix >= 0);
            if (!candidates.length) {
                out += applyOnText(input.slice(i));
                break;
            }
            // Prefer longer delimiters when ties occur (e.g., '$$' vs '$') for deterministic parsing.
            candidates.sort((a, b) => a.ix - b.ix || b.s.length - a.s.length);
            const next = i + candidates[0].ix;
            const kind = candidates[0].k;
            out += applyOnText(input.slice(i, next));

            if (kind === 'dollar2') {
                const end = input.indexOf('$$', next + 2);
                if (end < 0) {
                    out += applyOnText(input.slice(next));
                    break;
                }
                out += input.slice(next, end + 2);
                i = end + 2;
                continue;
            }
            if (kind === 'paren') {
                const end = input.indexOf('\\)', next + 2);
                if (end < 0) {
                    out += applyOnText(input.slice(next));
                    break;
                }
                out += input.slice(next, end + 2);
                i = end + 2;
                continue;
            }
            if (kind === 'bracket') {
                const end = input.indexOf('\\]', next + 2);
                if (end < 0) {
                    out += applyOnText(input.slice(next));
                    break;
                }
                out += input.slice(next, end + 2);
                i = end + 2;
                continue;
            }
            // Single '$' math: ignore escaped '$' and '$$' start.
            if (input.startsWith('$$', next)) {
                out += '$$';
                i = next + 2;
                continue;
            }
            if (next > 0 && input[next - 1] === '\\') {
                out += '$';
                i = next + 1;
                continue;
            }
            let j = next + 1;
            let end = -1;
            while (j < input.length) {
                const k = input.indexOf('$', j);
                if (k < 0) break;
                if (k > 0 && input[k - 1] === '\\') {
                    j = k + 1;
                    continue;
                }
                if (input.startsWith('$$', k)) {
                    j = k + 2;
                    continue;
                }
                end = k;
                break;
            }
            if (end < 0) {
                out += applyOnText(input.slice(next));
                break;
            }
            out += input.slice(next, end + 1);
            i = end + 1;
        }

        return out;
    }

    /**
     * Pre-process LaTeX list environments (itemize, enumerate)
     * @param {string} text - Input text
     * @returns {string} Text with list environments converted to HTML
     */
    function preprocessLatexLists(text) {
        if (!text) return text;
        let s = String(text);

        // Convert \begin{itemize}...\end{itemize} to <ul>...</ul>
        s = s.replace(/\\begin\{itemize\}([\s\S]*?)\\end\{itemize\}/gi, (match, content) => {
            let items = content.replace(/\\item\s*/g, '</li><li>');
            items = items.replace(/^<\/li>/, ''); // Remove leading </li>
            if (items.trim()) {
                items = items + '</li>';
            }
            return `<ul>${items}</ul>`;
        });

        // Convert \begin{enumerate}...\end{enumerate} to <ol>...</ol>
        s = s.replace(/\\begin\{enumerate\}([\s\S]*?)\\end\{enumerate\}/gi, (match, content) => {
            let items = content.replace(/\\item\s*/g, '</li><li>');
            items = items.replace(/^<\/li>/, '');
            if (items.trim()) {
                items = items + '</li>';
            }
            return `<ol>${items}</ol>`;
        });

        // Convert \begin{description}...\end{description} to <dl>...</dl>
        s = s.replace(
            /\\begin\{description\}([\s\S]*?)\\end\{description\}/gi,
            (match, content) => {
                let items = content.replace(
                    /\\item\[([^\]]*)\]\s*/g,
                    (m, term) => `</dd><dt>${escapeHtml(term)}</dt><dd>`
                );
                items = items.replace(/^<\/dd>/, '');
                if (items.trim()) {
                    items = items + '</dd>';
                }
                return `<dl>${items}</dl>`;
            }
        );

        return s;
    }

    function typesetMathPlaceholders(container) {
        if (!container) return;
        // Avoid work for detached nodes (e.g., during rapid re-renders)
        if (!isNodeConnected(container)) return;

        // Render generation guard: if the container is re-rendered while a typeset
        // is in-flight, we should not write results back into stale DOM.
        const gen = container.dataset ? String(container.dataset.mjxGen || '') : '';

        // Prevent timer storms when MathJax is not ready yet.
        // This flag is per-container DOM node; it is naturally reset when the DOM is replaced.
        if (container.dataset && container.dataset.mjxPending === '1') {
            // Still allow a later explicit call when MathJax becomes ready.
            // If MathJax is now fully ready (has typeset methods), we clear the flag and proceed.
            const isFullyReady =
                typeof MathJax !== 'undefined' &&
                (typeof MathJax.tex2chtmlPromise === 'function' ||
                    (MathJax.startup &&
                        MathJax.startup.document &&
                        typeof MathJax.startup.document.convert === 'function') ||
                    typeof MathJax.typesetPromise === 'function' ||
                    typeof MathJax.typeset === 'function');
            if (!isFullyReady) {
                // Keep retrying, but ensure only one pending retry per container to avoid storms.
                try {
                    if (container.dataset.mjxPendingScheduled !== '1') {
                        container.dataset.mjxPendingScheduled = '1';
                        const tries = Number(container.dataset.mjxPendingTries || 0) + 1;
                        container.dataset.mjxPendingTries = String(tries);
                        const delay = Math.min(2000, 150 + tries * 150);
                        setTimeout(() => {
                            if (!container || !isNodeConnected(container)) return;
                            try {
                                if (container.dataset) delete container.dataset.mjxPendingScheduled;
                            } catch (e) {}
                            typesetMathPlaceholders(container);
                        }, delay);
                    }
                } catch (e) {}
                return;
            }
            delete container.dataset.mjxPending;
            try {
                delete container.dataset.mjxPendingScheduled;
                delete container.dataset.mjxPendingTries;
            } catch (e) {}
        }

        // MathJax may not be fully ready when markdown is rendered (async script, startup in progress).
        // In that case, schedule a retry and (if available) trigger on-demand loading.
        // IMPORTANT: Check for concrete APIs (typeset/convert) rather than `startup` alone,
        // since some MathJax bundles don't expose `startup.promise`/`tex2chtmlPromise`.
        const mathJaxReady =
            typeof MathJax !== 'undefined' &&
            (typeof MathJax.tex2chtmlPromise === 'function' ||
                (MathJax.startup &&
                    MathJax.startup.document &&
                    typeof MathJax.startup.document.convert === 'function') ||
                typeof MathJax.typesetPromise === 'function' ||
                typeof MathJax.typeset === 'function');

        if (!mathJaxReady) {
            if (container.dataset) container.dataset.mjxPending = '1';
            try {
                const CommonUtils = global.ArxivSanityCommon;
                if (CommonUtils && typeof CommonUtils.loadMathJaxOnDemand === 'function') {
                    CommonUtils.loadMathJaxOnDemand(() => {
                        setTimeout(() => typesetMathPlaceholders(container), 0);
                    });
                }
            } catch (e) {}
            // Use MathJax.startup.promise if available for more reliable timing
            if (typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.promise) {
                MathJax.startup.promise
                    .then(() => {
                        setTimeout(() => typesetMathPlaceholders(container), 0);
                    })
                    .catch(() => {
                        setTimeout(() => typesetMathPlaceholders(container), 200);
                    });
            } else {
                setTimeout(() => typesetMathPlaceholders(container), 200);
            }
            return;
        }

        // Avoid concurrent runs on the same container.
        // IMPORTANT: only lock once MathJax is ready; otherwise early returns could leave a stale lock.
        if (container.dataset && container.dataset.mjxRunning === '1') {
            container.dataset.mjxNeedsRetry = '1';
            return;
        }
        if (container.dataset) container.dataset.mjxRunning = '1';

        const nodes = Array.from(
            container.querySelectorAll('[data-tex].math-inline, [data-tex].math-display')
        ).filter(el => !(el.classList && el.classList.contains('mjx-typeset')));
        if (!nodes.length) {
            if (container.dataset) delete container.dataset.mjxRunning;
            return;
        }

        const run = async () => {
            if (MathJax.startup && MathJax.startup.promise) {
                try {
                    await MathJax.startup.promise;
                } catch (e) {}
            }

            // If container was re-rendered or detached, abort.
            if (!isNodeConnected(container)) return;
            if (gen && container.dataset && String(container.dataset.mjxGen || '') !== gen) return;

            if (typeof MathJax.tex2chtmlPromise !== 'function') {
                // Most MathJax browser bundles do NOT expose tex2chtmlPromise.
                // Prefer the internal document.convert() path when available, as it is deterministic
                // and avoids delimiter-scanning edge cases.
                if (
                    MathJax.startup &&
                    MathJax.startup.document &&
                    typeof MathJax.startup.document.convert === 'function'
                ) {
                    let didConvert = false;
                    for (const el of nodes) {
                        if (!el || !isNodeConnected(el)) continue;
                        if (
                            gen &&
                            container.dataset &&
                            String(container.dataset.mjxGen || '') !== gen
                        )
                            return;

                        const tex = normalizeTex(el.getAttribute('data-tex') || '');
                        const display =
                            el.classList.contains('math-display') || el.tagName === 'DIV';
                        if (!tex.trim()) continue;
                        try {
                            const out = MathJax.startup.document.convert(tex, { display: display });
                            el.textContent = '';
                            el.appendChild(out);
                            el.classList.add('mjx-typeset');
                            el.classList.remove('mjx-error');
                            didConvert = true;
                        } catch (e) {
                            el.classList.add('mjx-error');
                            el.classList.remove('mjx-typeset');
                            const errorMsg = extractMathJaxError(e);
                            el.innerHTML = `
                                <span class="math-error-container">
                                    <span class="math-error-icon" title="Math rendering error">⚠️</span>
                                    <code class="math-error-tex">${escapeHtml(tex.substring(0, 100))}${tex.length > 100 ? '...' : ''}</code>
                                    <span class="math-error-msg">${escapeHtml(errorMsg)}</span>
                                    <button type="button" class="math-error-copy" title="Copy LaTeX source">📋</button>
                                </span>
                            `;
                        }
                    }
                    // Ensure CHTML styles and layout are updated after manual conversions.
                    // Some MathJax bundles require this for the output to display correctly.
                    try {
                        if (
                            didConvert &&
                            MathJax.startup &&
                            MathJax.startup.document &&
                            typeof MathJax.startup.document.updateDocument === 'function'
                        ) {
                            MathJax.startup.document.updateDocument();
                        }
                    } catch (e) {}
                    return;
                }

                // Final fallback: delimiter scanning via typeset/typesetPromise.
                // Clear previous math state in the container to avoid "already typeset" no-ops.
                try {
                    if (typeof MathJax.typesetClear === 'function') {
                        MathJax.typesetClear([container]);
                    }
                } catch (e) {}
                if (typeof MathJax.typesetPromise === 'function') {
                    try {
                        await MathJax.typesetPromise([container]);
                        for (const el of nodes) {
                            if (el && el.classList) el.classList.add('mjx-typeset');
                        }
                    } catch (e) {}
                } else if (typeof MathJax.typeset === 'function') {
                    try {
                        MathJax.typeset([container]);
                        for (const el of nodes) {
                            if (el && el.classList) el.classList.add('mjx-typeset');
                        }
                    } catch (e) {}
                }
                return;
            }

            for (const el of nodes) {
                // Skip stale nodes when re-render happens mid-flight.
                if (!el || !isNodeConnected(el)) continue;
                if (gen && container.dataset && String(container.dataset.mjxGen || '') !== gen)
                    return;

                const tex = normalizeTex(el.getAttribute('data-tex') || '');
                const display = el.classList.contains('math-display') || el.tagName === 'DIV';
                if (!tex.trim()) continue;
                try {
                    const out = await MathJax.tex2chtmlPromise(tex, { display: display });
                    // Replace fallback TeX with rendered output
                    el.textContent = '';
                    el.appendChild(out);
                    el.classList.add('mjx-typeset');
                    el.classList.remove('mjx-error');
                } catch (e) {
                    // Show user-friendly error message
                    el.classList.add('mjx-error');
                    el.classList.remove('mjx-typeset');

                    // Create error display
                    const errorMsg = extractMathJaxError(e);
                    el.innerHTML = `
                        <span class="math-error-container">
                            <span class="math-error-icon" title="Math rendering error">⚠️</span>
                            <code class="math-error-tex">${escapeHtml(tex.substring(0, 100))}${tex.length > 100 ? '...' : ''}</code>
                            <span class="math-error-msg">${escapeHtml(errorMsg)}</span>
                            <button type="button" class="math-error-copy" title="Copy LaTeX source">📋</button>
                        </span>
                    `;

                    // Setup copy button
                    const copyBtn = el.querySelector('.math-error-copy');
                    if (copyBtn) {
                        copyBtn.addEventListener('click', evt => {
                            evt.stopPropagation();
                            const doCopy =
                                CommonUtils && typeof CommonUtils.copyTextToClipboard === 'function'
                                    ? CommonUtils.copyTextToClipboard
                                    : null;

                            if (!doCopy) return;

                            doCopy(tex).then(ok => {
                                if (!ok) return;
                                copyBtn.textContent = '✓';
                                setTimeout(() => {
                                    copyBtn.textContent = '📋';
                                }, 1500);
                            });
                        });
                    }

                    if (typeof console !== 'undefined' && console.warn) {
                        console.warn('MathJax tex2chtml error:', e);
                    }
                }
            }
        };

        /**
         * Extract user-friendly error message from MathJax error
         */
        function extractMathJaxError(error) {
            if (!error) return 'Unknown error';
            const msg = error.message || String(error);

            // Common MathJax error patterns
            if (msg.includes('Undefined control sequence')) {
                const match = msg.match(/Undefined control sequence[:\s]*\\?(\w+)/i);
                return match ? `Unknown command: \\${match[1]}` : 'Unknown LaTeX command';
            }
            if (msg.includes('Missing')) {
                return msg.replace(/^.*Missing/i, 'Missing');
            }
            if (msg.includes('Extra')) {
                return msg.replace(/^.*Extra/i, 'Extra');
            }
            if (msg.includes('Double')) {
                return 'Double subscript/superscript';
            }
            if (msg.includes('Misplaced')) {
                return 'Misplaced symbol';
            }

            // Truncate long messages
            return msg.length > 50 ? msg.substring(0, 50) + '...' : msg;
        }

        // Fire-and-forget, with concurrency bookkeeping.
        run().finally(() => {
            if (container.dataset) {
                delete container.dataset.mjxRunning;
                if (container.dataset.mjxNeedsRetry === '1') {
                    delete container.dataset.mjxNeedsRetry;
                    // Retry after current microtasks
                    setTimeout(() => typesetMathPlaceholders(container), 0);
                }
            }
        });
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

        // Display math with $$...$$ even when it appears inline/in a paragraph.
        // Also prevents the single-$ rule from consuming the first $ of $$.
        function mathInlineDoubleDollar(state, silent) {
            if (state.src.slice(state.pos, state.pos + 2) !== '$$') return false;

            const start = state.pos + 2;
            const max = state.posMax;
            let searchPos = start;
            let end = -1;

            while (searchPos < max) {
                const found = state.src.indexOf('$$', searchPos);
                if (found === -1) break;
                if (found > 0 && state.src[found - 1] === '\\') {
                    searchPos = found + 2;
                    continue;
                }
                end = found;
                break;
            }

            if (end === -1) return false;

            if (!silent) {
                const token = state.push('math_inline', 'math', 0);
                token.content = state.src.slice(start, end);
                token.markup = '$$';
                token.displayMode = true;
            }

            state.pos = end + 2;
            return true;
        }

        function mathInlineDollar(state, silent) {
            // IMPORTANT: do not treat '$$' as empty '$...$'
            if (state.src.slice(state.pos, state.pos + 2) === '$$') return false;
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
            let pos = start;
            let found = false;

            while ((pos = state.src.indexOf('$', match)) !== -1) {
                match = pos;
                if (state.src[pos - 1] === '\\') {
                    match += 1;
                    continue;
                }
                found = true;
                break;
            }

            if (!found) {
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
            const end = state.src.indexOf('\\)', start);
            if (end === -1) return false;

            if (!silent) {
                const token = state.push('math_inline', 'math', 0);
                token.content = state.src.slice(start, end);
                token.markup = '\\(';
            }
            state.pos = end + 2;
            return true;
        }

        function mathInlineBracket(state, silent) {
            if (state.src.slice(state.pos, state.pos + 2) !== '\\[') return false;
            const start = state.pos + 2;
            const end = state.src.indexOf('\\]', start);
            if (end === -1) return false;

            if (!silent) {
                const token = state.push('math_inline', 'math', 0);
                token.content = state.src.slice(start, end);
                token.markup = '\\[';
                token.displayMode = true;
            }
            state.pos = end + 2;
            return true;
        }

        function mathBlock(state, startLine, endLine, silent) {
            const start = state.bMarks[startLine] + state.tShift[startLine];
            const max = state.eMarks[startLine];
            const line = state.src.slice(start, max);

            if (!line || line.length < 2) return false;

            const opener = line.trim().startsWith('$$')
                ? '$$'
                : line.trim().startsWith('\\[')
                  ? '\\['
                  : null;
            if (!opener) return false;

            const closer = opener === '$$' ? '$$' : '\\]';
            const startPos = line.indexOf(opener) + opener.length;

            let firstLine = state.src.slice(start + startPos, max);
            let found = false;
            let lastLine = '';
            let nextLine = startLine;

            if (silent) return true;

            if (firstLine.trim().endsWith(closer)) {
                firstLine = firstLine.trim().slice(0, -closer.length);
                found = true;
            }

            for (nextLine = startLine; !found; ) {
                nextLine += 1;
                if (nextLine >= endLine) {
                    break;
                }

                const pos = state.bMarks[nextLine] + state.tShift[nextLine];
                const maxPos = state.eMarks[nextLine];

                if (pos < maxPos && state.tShift[nextLine] < state.blkIndent) {
                    break;
                }

                const nextText = state.src.slice(pos, maxPos);
                if (nextText.trim().endsWith(closer)) {
                    const lastPos = nextText.lastIndexOf(closer);
                    lastLine = nextText.slice(0, lastPos);
                    found = true;
                }
            }

            state.line = nextLine + 1;

            const token = state.push('math_block', 'math', 0);
            token.block = true;
            token.content =
                (firstLine && firstLine.trim() ? `${firstLine}\n` : '') +
                state.getLines(startLine + 1, nextLine, state.tShift[startLine], true) +
                (lastLine && lastLine.trim() ? lastLine : '');
            token.map = [startLine, state.line];
            token.markup = opener;

            return true;
        }

        md.inline.ruler.before('escape', 'html_line_break', htmlLineBreak);
        md.inline.ruler.before('escape', 'math_inline_paren', mathInlineParen);
        md.inline.ruler.before('escape', 'math_inline_bracket', mathInlineBracket);
        // Handle $$ first, then $...
        md.inline.ruler.after('escape', 'math_inline_doubledollar', mathInlineDoubleDollar);
        md.inline.ruler.after('math_inline_doubledollar', 'math_inline_dollar', mathInlineDollar);
        md.block.ruler.before('code', 'math_block', mathBlock, {
            alt: ['paragraph', 'reference', 'blockquote', 'list'],
        });

        md.renderer.rules.math_inline = function (tokens, idx) {
            const raw = tokens[idx].content;
            const tex = normalizeTex(raw);
            const texAttr = escapeHtmlAttr(tex);
            if (tokens[idx].displayMode || tokens[idx].markup === '\\[') {
                const fallback = escapeHtml(tex);
                return `<span class="math-display" data-tex="${texAttr}">\\[${fallback}\\]</span>`;
            }
            const fallback = escapeHtml(tex);
            return `<span class="math-inline" data-tex="${texAttr}">\\(${fallback}\\)</span>`;
        };

        md.renderer.rules.math_block = function (tokens, idx) {
            const raw = tokens[idx].content;
            const tex = normalizeTex(raw);
            const texAttr = escapeHtmlAttr(tex);
            const fallback = escapeHtml(tex);
            return `<div class="math-display" data-tex="${texAttr}">\\[${fallback}\\]</div>`;
        };
    }

    let markdownRenderer = null;
    function getMarkdownRenderer() {
        if (markdownRenderer) return markdownRenderer;
        const Renderer = _getRenderer();
        if (!Renderer) return null;

        const md = Renderer.createMarkdownIt({
            html: true, // Enable HTML to support <sub>, <sup>, etc.
            linkify: true,
            typographer: false,
            breaks: true,
        });
        if (!md) return null;

        md.disable('code');
        Renderer.setSafeLinkValidator(md, { baseValidator: isSafeUrl });
        md.use(markdownItMathPlugin);

        // Add HTML sanitization - filter unsafe tags
        const defaultHtmlBlockRender =
            md.renderer.rules.html_block ||
            function (tokens, idx) {
                return tokens[idx].content;
            };
        const defaultHtmlInlineRender =
            md.renderer.rules.html_inline ||
            function (tokens, idx) {
                return tokens[idx].content;
            };

        md.renderer.rules.html_block = function (tokens, idx, options, env, self) {
            const content = tokens[idx].content;
            // Parse and sanitize HTML content
            const sanitized = sanitizeHtmlContent(content);
            return sanitized;
        };

        md.renderer.rules.html_inline = function (tokens, idx, options, env, self) {
            const content = tokens[idx].content;
            // Parse and sanitize HTML content
            const sanitized = sanitizeHtmlContent(content);
            return sanitized;
        };

        md.renderer.rules.image = function (tokens, idx) {
            const token = tokens[idx];
            const src = token.attrGet('src');
            const alt = token.content;
            const title = token.attrGet('title');

            if (!isSafeUrl(src)) return '';

            let caption = alt;
            if (title) caption = title;

            if (caption) {
                return `<figure>
                <img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy">
                <figcaption>${escapeHtml(caption)}</figcaption>
            </figure>`;
            }

            return `<figure><img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy"></figure>`;
        };

        markdownRenderer = md;
        return markdownRenderer;
    }

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
                const mathStart = line.match(/^([ \t]{4,})(\\\[|\$\$)/);
                if (mathStart) {
                    mathOpener = mathStart[2];
                    mathCloser = mathOpener === '$$' ? '$$' : '\\]';
                    inMathBlock = true;
                    lines[i] = line.replace(/^[ \t]{4,}/, '');
                    const afterOpener = trimmed.slice(mathOpener.length);
                    if (afterOpener.includes(mathCloser)) {
                        inMathBlock = false;
                        mathCloser = '';
                    }
                }
                continue;
            }

            lines[i] = line.replace(/^[ \t]{4,}/, '');

            if (trimmed.includes(mathCloser)) {
                inMathBlock = false;
                mathCloser = '';
            }
        }

        return lines.join('\n');
    }

    function fixAlignedTags(text) {
        return text.replace(
            /(\\begin\{aligned\})([\s\S]*?)(\\tag\{[^}]+\})([\s\S]*?)(\\end\{aligned\})/g,
            (match, begin, content1, tag, content2, end) => {
                return begin + content1 + content2 + end + ' ' + tag;
            }
        );
    }

    /**
     * Fix display math blocks that have been hard-wrapped (line breaks inside $$...$$).
     * This is common when markdown files are generated with fixed-width line wrapping.
     * The function joins lines within $$ blocks to ensure proper parsing.
     */
    function fixHardWrappedDisplayMath(text) {
        const lines = String(text || '').split('\n');
        const result = [];
        let inMathBlock = false;
        let mathBuffer = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const trimmed = line.trim();

            if (!inMathBlock) {
                // Check if line starts a $$ block (possibly with content after)
                if (trimmed.startsWith('$$')) {
                    // Check if it's a single-line $$ block (opens and closes on same line)
                    const afterOpener = trimmed.slice(2);
                    const closerPos = afterOpener.indexOf('$$');
                    if (closerPos !== -1) {
                        // Single line block, pass through
                        result.push(line);
                    } else {
                        // Multi-line block starts
                        inMathBlock = true;
                        mathBuffer = [line];
                    }
                } else {
                    result.push(line);
                }
            } else {
                // Inside math block
                mathBuffer.push(line);
                if (trimmed.endsWith('$$') || trimmed.includes('$$')) {
                    // Math block ends - join all lines with space (not newline)
                    // to prevent TeX control sequences from being broken
                    const joined = mathBuffer.join(' ').replace(/\s+/g, ' ');
                    result.push(joined);
                    inMathBlock = false;
                    mathBuffer = [];
                }
            }
        }

        // If we're still in a math block at the end, just output what we have
        if (mathBuffer.length > 0) {
            result.push(mathBuffer.join(' ').replace(/\s+/g, ' '));
        }

        return result.join('\n');
    }

    function renderSummaryMarkdown(text, markdownContainer, tocContainer) {
        if (!text || !markdownContainer) return;

        const md = getMarkdownRenderer();
        if (!md) {
            markdownContainer.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
            return;
        }

        try {
            const DomUtils = _getDomUtils();
            let normalizedText = normalizeIndentedDisplayMath(text);
            normalizedText = fixHardWrappedDisplayMath(normalizedText);
            normalizedText = fixAlignedTags(normalizedText);
            // Pre-process LaTeX theorem environments before markdown parsing
            normalizedText = preprocessTheoremEnvironments(normalizedText);
            // Pre-process LaTeX list environments
            normalizedText = preprocessLatexLists(normalizedText);
            // Pre-process LaTeX text commands (outside math mode)
            normalizedText = preprocessLatexTextCommands(normalizedText);
            const env = {};
            const tokens = md.parse(normalizedText, env);
            const slugCounts = {};
            const tocItems = [];

            if (
                DomUtils &&
                typeof DomUtils.extractHeadingText === 'function' &&
                typeof DomUtils.slugifyHeading === 'function'
            ) {
                for (let i = 0; i < tokens.length; i += 1) {
                    const token = tokens[i];
                    if (token.type !== 'heading_open') continue;
                    const level = Number(token.tag.slice(1));
                    if (Number.isNaN(level) || level > 4) continue;

                    const inlineToken = tokens[i + 1];
                    const title = DomUtils.extractHeadingText(inlineToken);
                    if (!title) continue;

                    const slug = DomUtils.slugifyHeading(title, slugCounts);
                    token.attrSet('id', slug);
                    tocItems.push({ level, title, slug });
                }
            }

            markdownContainer.innerHTML = md.renderer.render(tokens, md.options, env);

            // Increment generation so in-flight typeset won't write into stale DOM.
            try {
                if (markdownContainer && markdownContainer.dataset) {
                    const cur = Number(markdownContainer.dataset.mjxGen || 0);
                    markdownContainer.dataset.mjxGen = String(cur + 1);
                }
            } catch (e) {}

            if (DomUtils) {
                try {
                    if (typeof DomUtils.wrapMarkdownTables === 'function')
                        DomUtils.wrapMarkdownTables(markdownContainer);
                } catch (e) {}
                try {
                    if (typeof DomUtils.setupImageZoom === 'function')
                        DomUtils.setupImageZoom(markdownContainer);
                } catch (e) {}
            }

            // Typeset math placeholders explicitly (more reliable than delimiter scanning in dynamic HTML).
            try {
                typesetMathPlaceholders(markdownContainer);
            } catch (e) {}

            if (tocContainer && DomUtils && typeof DomUtils.buildTocHtml === 'function') {
                const tocHtml = DomUtils.buildTocHtml(tocItems);
                tocContainer.innerHTML = tocHtml;
                tocContainer.classList.toggle('is-empty', !tocHtml);
                const contentContainer = tocContainer.closest('.summary-content');
                if (contentContainer) {
                    contentContainer.classList.toggle('has-toc', Boolean(tocHtml));
                }
                if (tocHtml) {
                    // Remove legacy floating back-to-top (it may have been created before TOC rendered)
                    try {
                        const legacy = document.querySelector('.back-to-top');
                        if (legacy) legacy.remove();
                    } catch (e) {}

                    if (typeof DomUtils.setupTocToggle === 'function')
                        DomUtils.setupTocToggle(tocContainer);
                    if (typeof DomUtils.setupTocBackToTop === 'function') {
                        DomUtils.setupTocBackToTop(tocContainer);
                    }
                    if (typeof DomUtils.setupTocObserver === 'function') {
                        DomUtils.setupTocObserver(tocContainer, markdownContainer);
                    }
                    const firstLink = tocContainer.querySelector('.toc-item a');
                    if (firstLink) {
                        if (typeof DomUtils.setActiveTocLink === 'function') {
                            DomUtils.setActiveTocLink(tocContainer, firstLink);
                        }
                    }
                }
            }

            // NOTE: We intentionally do not call full-container MathJax scanning here.
            // Math is rendered via tex2chtmlPromise above.
        } catch (err) {
            console.error('Markdown render error:', err);
            markdownContainer.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
        }
    }

    global[NS] = {
        getMarkdownRenderer: getMarkdownRenderer,
        renderSummaryMarkdown: renderSummaryMarkdown,
    };
})(typeof window !== 'undefined' ? window : this);
