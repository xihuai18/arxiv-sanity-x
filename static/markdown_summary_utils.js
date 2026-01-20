'use strict';

// Shared summary markdown renderer utilities
// Exposes: window.ArxivSanitySummaryMarkdown

(function (global) {
    if (typeof window === 'undefined') return;

    const NS = 'ArxivSanitySummaryMarkdown';
    const CommonUtils = global.ArxivSanityCommon;
    const Renderer = global.ArxivSanityMarkdownRenderer;
    const Sanitizer = global.ArxivSanityMarkdownSanitizer;
    const DomUtils = global.ArxivSanitySummaryMarkdownDom;
    const escapeHtml = CommonUtils.escapeHtml;
    const isSafeUrl = CommonUtils.isSafeUrl;

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

            const opener = line.trim().startsWith('$$') ? '$$' : line.trim().startsWith('\\[') ? '\\[' : null;
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

            for (nextLine = startLine; !found;) {
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
        md.block.ruler.before('code', 'math_block', mathBlock, {
            alt: ['paragraph', 'reference', 'blockquote', 'list']
        });

        md.renderer.rules.math_inline = function (tokens, idx) {
            const content = escapeHtml(tokens[idx].content);
            if (tokens[idx].displayMode || tokens[idx].markup === '\\[') {
                return `<span class="math-display">\\[${content}\\]</span>`;
            }
            return `<span class="math-inline">\\(${content}\\)</span>`;
        };

        md.renderer.rules.math_block = function (tokens, idx) {
            const content = escapeHtml(tokens[idx].content);
            return `<div class="math-display">\\[${content}\\]</div>`;
        };
    }

    let markdownRenderer = null;
    function getMarkdownRenderer() {
        if (markdownRenderer) return markdownRenderer;
        if (!Renderer) return null;

        const md = Renderer.createMarkdownIt({
            html: false,
            linkify: true,
            typographer: false,
            breaks: true
        });
        if (!md) return null;

        md.disable('code');
        Renderer.setSafeLinkValidator(md, { baseValidator: isSafeUrl });
        md.use(markdownItMathPlugin);

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

    function renderSummaryMarkdown(text, markdownContainer, tocContainer) {
        if (!text || !markdownContainer) return;

        const md = getMarkdownRenderer();
        if (!md) {
            markdownContainer.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
            return;
        }

        try {
            let normalizedText = normalizeIndentedDisplayMath(text);
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
                const title = DomUtils.extractHeadingText(inlineToken);
                if (!title) continue;

                const slug = DomUtils.slugifyHeading(title, slugCounts);
                token.attrSet('id', slug);
                tocItems.push({ level, title, slug });
            }

            markdownContainer.innerHTML = md.renderer.render(tokens, md.options, env);
            DomUtils.wrapMarkdownTables(markdownContainer);
            DomUtils.setupImageZoom(markdownContainer);

            if (tocContainer) {
                const tocHtml = DomUtils.buildTocHtml(tocItems);
                tocContainer.innerHTML = tocHtml;
                tocContainer.classList.toggle('is-empty', !tocHtml);
                const contentContainer = tocContainer.closest('.summary-content');
                if (contentContainer) {
                    contentContainer.classList.toggle('has-toc', Boolean(tocHtml));
                }
                if (tocHtml) {
                    DomUtils.setupTocToggle(tocContainer);
                    DomUtils.setupTocObserver(tocContainer, markdownContainer);
                    const firstLink = tocContainer.querySelector('.toc-item a');
                    if (firstLink) {
                        DomUtils.setActiveTocLink(tocContainer, firstLink);
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

    global[NS] = {
        getMarkdownRenderer: getMarkdownRenderer,
        renderSummaryMarkdown: renderSummaryMarkdown
    };
})(typeof window !== 'undefined' ? window : this);
