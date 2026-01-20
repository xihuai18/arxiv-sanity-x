'use strict';

// Shared TL;DR renderer utilities used by both main paper list and reading list.
// Exposes a small global API: window.ArxivSanityTldr.{ getRenderer, render, triggerMathJax }

(function (global) {
    const NS = 'ArxivSanityTldr';
    const MarkdownCore = global.ArxivSanityMarkdownCore;
    const Renderer = global.ArxivSanityMarkdownRenderer;
    const Sanitizer = global.ArxivSanityMarkdownSanitizer;

    const state = {
        renderer: null
    };

    function buildRenderer() {
        if (!Renderer) return null;
        const md = Renderer.createMarkdownIt({
            html: false,
            breaks: true,
            linkify: true
        });
        if (!md) return null;
        Renderer.setSafeLinkValidator(md, { allowRelative: true, allowHash: true });
        Renderer.applyTextEscapeRenderer(md);
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

        const md = getRenderer();
        if (!md) {
            return String(s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\n/g, '<br>');
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
        triggerMathJax: triggerMathJax
    };
})(typeof window !== 'undefined' ? window : this);
