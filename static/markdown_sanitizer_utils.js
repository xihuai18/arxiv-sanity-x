'use strict';

// Shared Markdown sanitization utilities
// Exposes: window.ArxivSanityMarkdownSanitizer

(function (global) {
    const NS = 'ArxivSanityMarkdownSanitizer';

    function escapeText(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    function buildLinkValidator(options) {
        const opts = options || {};
        const allowRelative = !!opts.allowRelative;
        const allowHash = !!opts.allowHash;
        const baseValidator = typeof opts.baseValidator === 'function' ? opts.baseValidator : null;

        return function (url) {
            if (!url) return false;
            const s = String(url).trim();
            if (!s) return false;

            if (allowHash && s.startsWith('#')) return true;
            if (allowRelative && s.startsWith('/')) return true;

            if (baseValidator) return baseValidator(s);

            const lower = s.toLowerCase();
            if (lower.startsWith('javascript:')) return false;
            if (lower.startsWith('vbscript:')) return false;
            if (lower.startsWith('file:')) return false;
            if (lower.startsWith('data:')) return false;

            return lower.startsWith('http://') || lower.startsWith('https://') || lower.startsWith('mailto:');
        };
    }

    function stripMarkdownImages(text) {
        let s = String(text || '');
        s = s.replace(/^\s*!\[[^\]]*\]\([^)]*\)\s*$/gm, '');
        s = s.replace(/<img\b[^>]*>/gi, '');
        return s;
    }

    global[NS] = {
        escapeText: escapeText,
        buildLinkValidator: buildLinkValidator,
        stripMarkdownImages: stripMarkdownImages
    };
})(typeof window !== 'undefined' ? window : this);
