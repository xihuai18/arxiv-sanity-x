'use strict';

// Shared Markdown core utilities (compatibility wrapper)
// Exposes: window.ArxivSanityMarkdownCore

(function (global) {
    const NS = 'ArxivSanityMarkdownCore';
    function _getSanitizer() {
        return global.ArxivSanityMarkdownSanitizer || null;
    }
    function _getRenderer() {
        return global.ArxivSanityMarkdownRenderer || null;
    }
    function _getCommonUtils() {
        return global.ArxivSanityCommon || null;
    }

    global[NS] = {
        createMarkdownIt: function (options) {
            const Renderer = _getRenderer();
            if (!Renderer || typeof Renderer.createMarkdownIt !== 'function') return null;
            return Renderer.createMarkdownIt(options);
        },
        buildLinkValidator: function (options) {
            const Sanitizer = _getSanitizer();
            if (!Sanitizer || typeof Sanitizer.buildLinkValidator !== 'function') return null;
            return Sanitizer.buildLinkValidator(options);
        },
        setSafeLinkValidator: function (md, options) {
            const Renderer = _getRenderer();
            if (!Renderer || typeof Renderer.setSafeLinkValidator !== 'function') return md;
            return Renderer.setSafeLinkValidator(md, options);
        },
        applyTextEscapeRenderer: function (md) {
            const Renderer = _getRenderer();
            if (!Renderer || typeof Renderer.applyTextEscapeRenderer !== 'function') return md;
            return Renderer.applyTextEscapeRenderer(md);
        },
        stripMarkdownImages: function (text) {
            const Sanitizer = _getSanitizer();
            if (!Sanitizer || typeof Sanitizer.stripMarkdownImages !== 'function')
                return String(text || '');
            return Sanitizer.stripMarkdownImages(text);
        },
        escapeText: function (text) {
            const Sanitizer = _getSanitizer();
            if (!Sanitizer || typeof Sanitizer.escapeText !== 'function') return String(text || '');
            return Sanitizer.escapeText(text);
        },
        escapeHtml: function (text) {
            const CommonUtils = _getCommonUtils();
            if (!CommonUtils || typeof CommonUtils.escapeHtml !== 'function')
                return String(text || '');
            return CommonUtils.escapeHtml(text);
        },
    };
})(typeof window !== 'undefined' ? window : this);
