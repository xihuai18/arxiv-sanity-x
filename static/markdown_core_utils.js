'use strict';

// Shared Markdown core utilities (compatibility wrapper)
// Exposes: window.ArxivSanityMarkdownCore

(function (global) {
    const NS = 'ArxivSanityMarkdownCore';
    const Sanitizer = global.ArxivSanityMarkdownSanitizer;
    const Renderer = global.ArxivSanityMarkdownRenderer;
    const CommonUtils = global.ArxivSanityCommon;

    global[NS] = {
        createMarkdownIt: Renderer && Renderer.createMarkdownIt,
        buildLinkValidator: Sanitizer && Sanitizer.buildLinkValidator,
        setSafeLinkValidator: Renderer && Renderer.setSafeLinkValidator,
        applyTextEscapeRenderer: Renderer && Renderer.applyTextEscapeRenderer,
        stripMarkdownImages: Sanitizer && Sanitizer.stripMarkdownImages,
        escapeText: Sanitizer && Sanitizer.escapeText,
        escapeHtml: CommonUtils && CommonUtils.escapeHtml
    };
})(typeof window !== 'undefined' ? window : this);
