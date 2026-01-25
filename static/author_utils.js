// Shared author formatting utilities
// Exposes: window.ArxivSanityAuthors
// - format(authorsText, options?) -> { text, full, omitted }
//
// Goal: unify author truncation across all paper views.

'use strict';

(function () {
    if (typeof window === 'undefined') return;

    function splitAuthors(authorsText) {
        if (!authorsText) return [];
        // Backend renders authors as: "A, B, C". Be defensive about separators.
        const raw = String(authorsText);
        return raw
            .split(/\s*,\s*/g)
            .map(s => s.trim())
            .filter(Boolean);
    }

    function format(authorsText, options) {
        const opts = options || {};
        const full = String(authorsText || '').trim();
        if (!full) return { text: '', full: '', omitted: 0 };

        const list = splitAuthors(full);
        // If we can't confidently split, fall back to raw.
        if (list.length <= 1) return { text: full, full, omitted: 0 };

        const maxAuthors = Number.isFinite(opts.maxAuthors) ? opts.maxAuthors : 10;
        const head = Number.isFinite(opts.head) ? opts.head : 5;
        const tail = Number.isFinite(opts.tail) ? opts.tail : 3;
        const ellipsis = opts.ellipsis !== undefined ? String(opts.ellipsis) : '…';

        // Sanity clamps
        const maxA = Math.max(1, Math.floor(maxAuthors));
        const h = Math.max(1, Math.floor(head));
        const t = Math.max(1, Math.floor(tail));

        if (list.length <= maxA) {
            return { text: list.join(', '), full, omitted: 0 };
        }

        // Ensure head+tail < list.length
        let useHead = h;
        let useTail = t;
        if (useHead + useTail >= list.length) {
            useHead = Math.max(1, Math.floor(list.length / 2));
            useTail = Math.max(1, list.length - useHead - 1);
        }

        const omitted = Math.max(0, list.length - useHead - useTail);
        const first = list.slice(0, useHead).join(', ');
        const last = list.slice(list.length - useTail).join(', ');
        const text = `${first}, ${ellipsis}, ${last}`;
        return { text, full, omitted };
    }

    window.ArxivSanityAuthors = {
        splitAuthors,
        format,
    };
})();
