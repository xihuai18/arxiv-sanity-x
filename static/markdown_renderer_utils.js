'use strict';

// Shared Markdown renderer utilities
// Exposes: window.ArxivSanityMarkdownRenderer

(function (global) {
    const NS = 'ArxivSanityMarkdownRenderer';
    function _getSanitizer() {
        return global.ArxivSanityMarkdownSanitizer || null;
    }

    function createMarkdownIt(options) {
        if (typeof global.markdownit === 'undefined') return null;
        return global.markdownit(options || {});
    }

    function setSafeLinkValidator(md, options) {
        const Sanitizer = _getSanitizer();
        if (!md || !Sanitizer) return md;
        md.validateLink = Sanitizer.buildLinkValidator(options);
        return md;
    }

    function applyTextEscapeRenderer(md) {
        const Sanitizer = _getSanitizer();
        if (!md || !Sanitizer) return md;
        md.renderer.rules.text = function (tokens, idx) {
            return Sanitizer.escapeText(tokens[idx].content);
        };
        return md;
    }

    global[NS] = {
        createMarkdownIt: createMarkdownIt,
        setSafeLinkValidator: setSafeLinkValidator,
        applyTextEscapeRenderer: applyTextEscapeRenderer,
    };
})(typeof window !== 'undefined' ? window : this);
