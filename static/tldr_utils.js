'use strict';

// Shared TL;DR renderer utilities used by both main paper list and reading list.
// Exposes a small global API: window.ArxivSanityTldr.{ getRenderer, render, triggerMathJax }

(function (global) {
    const NS = 'ArxivSanityTldr';
    const MarkdownCore = global.ArxivSanityMarkdownCore;
    const Renderer = global.ArxivSanityMarkdownRenderer;
    const Sanitizer = global.ArxivSanityMarkdownSanitizer;

    const state = {
        renderer: null,
    };

    /**
     * Escape HTML for math content - only escape < and > to prevent HTML injection,
     * but preserve & which is used in LaTeX for column separators
     */
    function escapeMathHtml(text) {
        // Some texts may be hard-wrapped and split TeX control sequences.
        // Remove newlines inside math content to keep TeX parseable.
        return String(text || '')
            .replace(/\r\n|\r|\n/g, '')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    /**
     * Math plugin for markdown-it to handle inline and display math
     * Supports: $...$, \(...\), \[...\], $$...$$
     */
    function markdownItMathPlugin(md) {
        function isValidDelim(state, pos) {
            const max = state.posMax;
            const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1;
            const nextChar = pos + 1 <= max ? state.src.charCodeAt(pos + 1) : -1;
            let canOpen = true;
            let canClose = true;

            // Check for spaces around delimiter
            if (prevChar === 0x20 || prevChar === 0x09 || (nextChar >= 0x30 && nextChar <= 0x39)) {
                canClose = false;
            }
            if (nextChar === 0x20 || nextChar === 0x09) {
                canOpen = false;
            }

            return { can_open: canOpen, can_close: canClose };
        }

        // Inline math with single $
        function mathInlineDollar(state, silent) {
            // IMPORTANT: do not treat '$$' as empty '$...$'
            // Let the double-dollar rule handle it.
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

        // Display math with $$...$$ even when it appears inline/in a paragraph.
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

        // Inline math with \(...\)
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

        // Inline display math with \[...\] (treated as display mode)
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

        // Register inline rules
        md.inline.ruler.before('escape', 'math_inline_paren', mathInlineParen);
        md.inline.ruler.before('escape', 'math_inline_bracket', mathInlineBracket);
        // Handle $$ first, then $...
        md.inline.ruler.after('escape', 'math_inline_doubledollar', mathInlineDoubleDollar);
        md.inline.ruler.after('math_inline_doubledollar', 'math_inline_dollar', mathInlineDollar);

        // Render math tokens
        md.renderer.rules.math_inline = function (tokens, idx) {
            const content = escapeMathHtml(tokens[idx].content);
            if (tokens[idx].displayMode || tokens[idx].markup === '\\[') {
                return '<span class="math-display">\\[' + content + '\\]</span>';
            }
            return '<span class="math-inline">\\(' + content + '\\)</span>';
        };
    }

    /**
     * Escape HTML special characters for security
     * @param {string} text - Input text
     * @returns {string} Escaped text
     */
    function escapeHtmlForSafety(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    /**
     * Pre-process LaTeX text commands that appear outside of math mode
     * Converts \textbf{...}, \textit{...}, \emph{...}, \texttt{...} etc. to HTML
     * This is safe because we first escape HTML, then only add our own safe tags
     * @param {string} text - Input text (should be HTML-escaped first)
     * @returns {string} Text with LaTeX text commands converted to HTML
     */
    function preprocessLatexTextCommands(text) {
        if (!text) return text;
        let s = String(text);

        // Convert \textbf{...} to <strong>...</strong>
        s = s.replace(/\\textbf\{([^}]*)\}/g, '<strong>$1</strong>');

        // Convert \textit{...} and \emph{...} to <em>...</em>
        s = s.replace(/\\textit\{([^}]*)\}/g, '<em>$1</em>');
        s = s.replace(/\\emph\{([^}]*)\}/g, '<em>$1</em>');

        // Convert \texttt{...} to <code>...</code>
        s = s.replace(/\\texttt\{([^}]*)\}/g, '<code>$1</code>');

        // Convert \textrm{...} and \textsf{...} to plain text (just remove the command)
        s = s.replace(/\\textrm\{([^}]*)\}/g, '$1');
        s = s.replace(/\\textsf\{([^}]*)\}/g, '$1');

        // Convert \underline{...} to <u>...</u>
        s = s.replace(/\\underline\{([^}]*)\}/g, '<u>$1</u>');

        // Convert \textsc{...} (small caps) to span with style
        s = s.replace(/\\textsc\{([^}]*)\}/g, '<span class="latex-smallcaps">$1</span>');

        // Convert \textsuperscript{...} and \textsubscript{...}
        s = s.replace(/\\textsuperscript\{([^}]*)\}/g, '<sup>$1</sup>');
        s = s.replace(/\\textsubscript\{([^}]*)\}/g, '<sub>$1</sub>');

        // Convert \cite{...} to a styled citation reference
        s = s.replace(/\\cite\{([^}]*)\}/g, '<span class="latex-cite">[$1]</span>');

        // Convert \ref{...} and \eqref{...} to reference placeholders
        s = s.replace(/\\eqref\{([^}]*)\}/g, '<span class="latex-ref">($1)</span>');
        s = s.replace(/\\ref\{([^}]*)\}/g, '<span class="latex-ref">$1</span>');

        return s;
    }

    function buildRenderer() {
        if (!Renderer) return null;
        const md = Renderer.createMarkdownIt({
            html: true, // Enable HTML to allow our converted tags (safe because we escape first)
            breaks: true,
            linkify: true,
        });
        if (!md) return null;
        Renderer.setSafeLinkValidator(md, { allowRelative: true, allowHash: true });
        // Add math plugin for formula support
        md.use(markdownItMathPlugin);
        return md;
    }

    function getRenderer() {
        if (state.renderer) return state.renderer;
        state.renderer = buildRenderer();
        return state.renderer;
    }

    function render(text) {
        if (!text) return '';
        let s = Sanitizer ? Sanitizer.stripMarkdownImages(text) : String(text);

        // First escape HTML for security, then process LaTeX commands
        // This ensures user input can't inject malicious HTML
        s = escapeHtmlForSafety(s);

        // Pre-process LaTeX text commands (safe because HTML is already escaped)
        s = preprocessLatexTextCommands(s);

        const md = getRenderer();
        if (!md) {
            return s.replace(/\n/g, '<br>');
        }
        return md.render(s);
    }

    function triggerMathJax(element) {
        if (typeof global.MathJax !== 'undefined' && global.MathJax.typesetPromise) {
            global.MathJax.typesetPromise(element ? [element] : undefined).catch(function (err) {
                console.warn('MathJax typeset error:', err);
            });
        }
    }

    global[NS] = {
        getRenderer: getRenderer,
        render: render,
        triggerMathJax: triggerMathJax,
    };
})(typeof window !== 'undefined' ? window : this);
