'use strict';

// Common utilities shared across paper_list.js, paper_summary.js, and other modules.
// Exposes: window.ArxivSanityCommon

(function (global) {
    const NS = 'ArxivSanityCommon';

    // =========================================================================
    // CSRF Token Management
    // =========================================================================

    /**
     * Get CSRF token from meta tag
     * @returns {string} CSRF token or empty string if not found
     */
    function getCsrfToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? (meta.getAttribute('content') || '') : '';
    }

    /**
     * Fetch wrapper with automatic CSRF token injection
     * @param {string} url - The URL to fetch
     * @param {Object} [options] - Fetch options
     * @param {string} [options.method='POST'] - HTTP method
     * @param {Object} [options.headers] - Additional headers
     * @param {*} [options.body] - Request body
     * @returns {Promise<Response>} Fetch response
     */
    function csrfFetch(url, options) {
        const opts = options || {};
        const method = (opts.method || 'POST').toUpperCase();
        const headers = new Headers(opts.headers || {});
        const tok = getCsrfToken();
        if (tok) headers.set('X-CSRF-Token', tok);
        if (method !== 'GET' && !headers.has('Content-Type') && opts.body) {
            headers.set('Content-Type', 'application/json');
        }
        return fetch(url, { ...opts, method, headers, credentials: 'same-origin' });
    }

    // =========================================================================
    // User Event Stream (SSE + Polling fallback)
    // =========================================================================

    const USER_EVENT_CHANNEL = 'arxiv-sanity-user-events';
    let userEventChannel = null;
    let userStatePoller = null;
    let eventSource = null;
    let eventHandlers = [];

    /**
     * Initialize BroadcastChannel for cross-tab communication
     * @returns {BroadcastChannel|null} BroadcastChannel instance or null if not supported
     */
    function initBroadcastChannel() {
        if (userEventChannel) return userEventChannel;
        if (typeof BroadcastChannel !== 'undefined') {
            userEventChannel = new BroadcastChannel(USER_EVENT_CHANNEL);
        }
        return userEventChannel;
    }

    /**
     * Fetch user state from server
     * @returns {Promise<Object|null>} User state object or null on error
     */
    function fetchUserState() {
        return fetch('/api/user_state', { credentials: 'same-origin' })
            .then(resp => resp.json())
            .catch(() => null);
    }

    /**
     * Stop user state polling
     */
    function stopUserStatePolling() {
        if (userStatePoller) {
            clearInterval(userStatePoller);
            userStatePoller = null;
        }
    }

    /**
     * Start user state polling with specified interval
     * @param {Function} applyFn - Function to apply user state updates
     */
    function startUserStatePolling(applyFn) {
        if (userStatePoller) return;
        userStatePoller = setInterval(() => {
            fetchUserState().then(state => {
                if (applyFn) applyFn(state);
            });
        }, 15000);
    }

    /**
     * Register an event handler for user events
     * @param {Function} handler - Event handler function
     */
    function registerEventHandler(handler) {
        if (typeof handler === 'function' && !eventHandlers.includes(handler)) {
            eventHandlers.push(handler);
        }
    }

    /**
     * Unregister an event handler
     * @param {Function} handler - Event handler function to remove
     */
    function unregisterEventHandler(handler) {
        const idx = eventHandlers.indexOf(handler);
        if (idx >= 0) eventHandlers.splice(idx, 1);
    }

    /**
     * Dispatch user event to all registered handlers and broadcast to other tabs
     * @param {Object} event - Event object to dispatch
     * @param {Object} [options] - Dispatch options
     * @param {boolean} [options.fromBroadcast=false] - Whether event is from broadcast channel
     */
    function dispatchUserEvent(event, options) {
        if (!event || typeof event !== 'object') return;
        const opts = options || {};

        // Broadcast to other tabs
        if (!opts.fromBroadcast && userEventChannel) {
            userEventChannel.postMessage(event);
        }

        // Notify all registered handlers
        eventHandlers.forEach(handler => {
            try {
                handler(event, opts);
            } catch (e) {
                console.warn('Event handler error:', e);
            }
        });
    }

    /**
     * Setup user event stream with SSE and polling fallback
     * @param {Object} user - User object (must be truthy to enable)
     * @param {Function} applyStateFn - Function to apply state updates
     */
    function setupUserEventStream(user, applyStateFn) {
        if (!user) return;

        const channel = initBroadcastChannel();
        if (channel) {
            channel.onmessage = (e) => {
                dispatchUserEvent(e.data || {}, { fromBroadcast: true });
            };
        }

        if (typeof EventSource === 'undefined') {
            startUserStatePolling(applyStateFn);
            return;
        }

        const connect = () => {
            eventSource = new EventSource('/api/user_stream');
            eventSource.onopen = () => {
                stopUserStatePolling();
            };
            eventSource.onmessage = (evt) => {
                if (!evt || !evt.data) return;
                try {
                    const payload = JSON.parse(evt.data);
                    dispatchUserEvent(payload);
                } catch (e) {
                    console.warn('Failed to parse user event:', e);
                }
            };
            eventSource.onerror = () => {
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
                startUserStatePolling(applyStateFn);
                setTimeout(connect, 8000);
            };
        };
        connect();
    }

    // =========================================================================
    // Dropdown Registry (shared close-on-click-outside / Escape logic)
    // =========================================================================

    const dropdownRegistry = new Map();
    let dropdownListenersBound = false;

    /**
     * Bind global dropdown listeners (click outside and Escape key)
     */
    function bindDropdownListeners() {
        if (dropdownListenersBound) return;
        dropdownListenersBound = true;

        document.addEventListener('mousedown', (event) => {
            dropdownRegistry.forEach((api, id) => {
                if (!api || !api.isOpen || !api.isOpen()) return;
                const dropdown = document.getElementById(id);
                if (dropdown && !dropdown.contains(event.target)) {
                    api.close();
                }
            });
        });

        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape') return;
            dropdownRegistry.forEach((api) => {
                if (api && api.isOpen && api.isOpen()) api.close();
            });
        });
    }

    /**
     * Register a dropdown for global close handling
     * @param {string} id - Dropdown element ID
     * @param {Object} api - Dropdown API with isOpen() and close() methods
     */
    function registerDropdown(id, api) {
        dropdownRegistry.set(id, api);
        bindDropdownListeners();
    }

    /**
     * Unregister a dropdown from global close handling
     * @param {string} id - Dropdown element ID
     */
    function unregisterDropdown(id) {
        dropdownRegistry.delete(id);
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    /**
     * Handle API errors and return user-friendly error messages
     * @param {Error|string} error - Error object or message
     * @param {string} [context='API'] - Context for error logging
     * @returns {string} User-friendly error message
     */
    function handleApiError(error, context) {
        const ctx = context || 'API';
        const msg = error && error.message ? error.message : String(error || 'Unknown error');
        console.error(`[${ctx}]`, error);

        // Return user-friendly error message
        if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
            return 'Network error. Please check your connection and try again.';
        }
        if (msg.includes('timeout') || msg.includes('Timeout')) {
            return 'Request timeout. The operation is taking longer than expected.';
        }
        if (msg.includes('401') || msg.includes('Unauthorized')) {
            return 'Authentication required. Please log in again.';
        }
        if (msg.includes('403') || msg.includes('Forbidden')) {
            return 'Access denied. You do not have permission for this operation.';
        }
        if (msg.includes('404') || msg.includes('Not Found')) {
            return 'Resource not found.';
        }
        if (msg.includes('500') || msg.includes('Internal Server Error')) {
            return 'Server error. Please try again later.';
        }

        return msg;
    }

    // =========================================================================
    // HTML Escaping
    // =========================================================================

    /**
     * Escape HTML special characters
     * @param {string} text - Text to escape
     * @returns {string} Escaped HTML string
     */
    function escapeHtml(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // =========================================================================
    // URL Safety Check
    // =========================================================================

    /**
     * Check if URL is safe (http/https/mailto only)
     * @param {string} href - URL to check
     * @returns {boolean} True if URL is safe
     */
    function isSafeUrl(href) {
        try {
            const u = new URL(String(href || ''), window.location.href);
            return u.protocol === 'http:' || u.protocol === 'https:' || u.protocol === 'mailto:';
        } catch (e) {
            return false;
        }
    }

    // =========================================================================
    // URL Builders (paper list/navigation)
    // =========================================================================

    /**
     * Append parameter to URLSearchParams if value is valid
     * @param {URLSearchParams} params - URLSearchParams object
     * @param {string} key - Parameter key
     * @param {*} value - Parameter value
     */
    function appendParam(params, key, value) {
        if (value === undefined || value === null) return;
        const str = String(value).trim();
        if (!str) return;
        params.set(key, str);
    }

    /**
     * Append common filter parameters to URLSearchParams
     * @param {URLSearchParams} params - URLSearchParams object
     * @param {Object} [options] - Filter options
     * @param {boolean} [options.includeLogic] - Include logic parameter
     * @param {boolean} [options.includeSvmC] - Include SVM C parameter
     * @param {boolean} [options.includeSearchMode] - Include search mode parameter
     * @param {boolean} [options.includeSemanticWeight] - Include semantic weight parameter
     * @param {Object} [gvarsOverride] - Override global vars (defaults to window.gvars)
     */
    function appendCommonFilters(params, options, gvarsOverride) {
        const opts = options || {};
        const vars = gvarsOverride || window.gvars;
        if (!vars) return;
        appendParam(params, 'time_filter', vars.time_filter);
        if (vars.skip_have) {
            params.set('skip_have', vars.skip_have);
        }
        if (opts.includeLogic && vars.logic) {
            params.set('logic', vars.logic);
        }
        if (opts.includeSvmC) {
            appendParam(params, 'svm_c', vars.svm_c);
        }
        if (opts.includeSearchMode && vars.search_mode) {
            params.set('search_mode', vars.search_mode);
        }
        if (opts.includeSemanticWeight) {
            appendParam(params, 'semantic_weight', vars.semantic_weight);
        }
    }

    /**
     * Build URL for tag filtering
     * @param {string} tagName - Tag name
     * @param {Object} [options] - URL options
     * @param {string} [options.logic] - Logic mode (and/or)
     * @param {Object} [options.gvars] - Override global vars
     * @returns {string} Tag filter URL
     */
    function buildTagUrl(tagName, options) {
        const opts = options || {};
        const params = new URLSearchParams();
        params.set('rank', 'tags');
        appendParam(params, 'tags', tagName);
        appendCommonFilters(params, { includeLogic: true, includeSvmC: true }, opts.gvars);
        if (opts.logic) {
            params.set('logic', opts.logic);
        }
        return '/?' + params.toString();
    }

    /**
     * Build URL for keyword search
     * @param {string} keyword - Search keyword
     * @param {Object} [options] - URL options
     * @param {Object} [options.gvars] - Override global vars
     * @returns {string} Keyword search URL
     */
    function buildKeywordUrl(keyword, options) {
        const opts = options || {};
        const params = new URLSearchParams();
        params.set('rank', 'search');
        appendParam(params, 'q', keyword);
        appendCommonFilters(params, { includeSearchMode: true, includeSemanticWeight: true }, opts.gvars);
        return '/?' + params.toString();
    }

    // =========================================================================
    // Author Formatting
    // =========================================================================

    /**
     * Format authors text with truncation
     * @param {string} authorsText - Comma-separated authors
     * @param {Object} [options] - Formatting options
     * @returns {string} Formatted authors text
     */
    function formatAuthorsText(authorsText, options) {
        if (typeof window !== 'undefined' && window.ArxivSanityAuthors && window.ArxivSanityAuthors.format) {
            return window.ArxivSanityAuthors.format(authorsText, options).text;
        }
        return String(authorsText || '');
    }

    // =========================================================================
    // TL;DR Rendering Helpers
    // =========================================================================

    /**
     * Render TL;DR markdown to HTML
     * @param {string} text - Markdown text
     * @returns {string} Rendered HTML
     */
    function renderTldrMarkdown(text) {
        if (typeof window !== 'undefined' && window.ArxivSanityTldr && window.ArxivSanityTldr.render) {
            return window.ArxivSanityTldr.render(text);
        }
        if (!text) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }

    /**
     * Trigger MathJax typesetting for element
     * @param {HTMLElement} [element] - Element to typeset (or entire document if omitted)
     */
    function triggerMathJax(element) {
        if (typeof window !== 'undefined' && window.ArxivSanityTldr && window.ArxivSanityTldr.triggerMathJax) {
            return window.ArxivSanityTldr.triggerMathJax(element);
        }
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            MathJax.typesetPromise(element ? [element] : undefined).catch(function (err) {
                console.warn('MathJax typeset error:', err);
            });
        }
    }

    // =========================================================================
    // Performance Monitoring (Development only)
    // =========================================================================

    const isDevelopment = typeof window !== 'undefined' &&
                         (window.location.hostname === 'localhost' ||
                          window.location.hostname === '127.0.0.1');

    /**
     * Measure synchronous function performance (development only)
     * @param {string} name - Operation name for logging
     * @param {Function} fn - Function to measure
     * @returns {*} Function result
     */
    function measurePerformance(name, fn) {
        if (!isDevelopment || typeof performance === 'undefined') {
            return fn();
        }

        const start = performance.now();
        try {
            const result = fn();
            const duration = performance.now() - start;

            // Only log if it takes more than 100ms
            if (duration > 100) {
                console.warn(`⚠️ Performance: ${name} took ${duration.toFixed(2)}ms`);
            } else if (duration > 10) {
                console.log(`⏱️ Performance: ${name} took ${duration.toFixed(2)}ms`);
            }

            return result;
        } catch (error) {
            const duration = performance.now() - start;
            console.error(`❌ Performance: ${name} failed after ${duration.toFixed(2)}ms`, error);
            throw error;
        }
    }

    /**
     * Measure asynchronous function performance (development only)
     * @param {string} name - Operation name for logging
     * @param {Function} fn - Async function to measure
     * @returns {Promise<*>} Function result
     */
    async function measurePerformanceAsync(name, fn) {
        if (!isDevelopment || typeof performance === 'undefined') {
            return await fn();
        }

        const start = performance.now();
        try {
            const result = await fn();
            const duration = performance.now() - start;

            if (duration > 100) {
                console.warn(`⚠️ Performance: ${name} took ${duration.toFixed(2)}ms`);
            } else if (duration > 10) {
                console.log(`⏱️ Performance: ${name} took ${duration.toFixed(2)}ms`);
            }

            return result;
        } catch (error) {
            const duration = performance.now() - start;
            console.error(`❌ Performance: ${name} failed after ${duration.toFixed(2)}ms`, error);
            throw error;
        }
    }

    // =========================================================================
    // Export API
    // =========================================================================

    global[NS] = {
        // CSRF
        getCsrfToken,
        csrfFetch,
        // User events
        fetchUserState,
        setupUserEventStream,
        registerEventHandler,
        unregisterEventHandler,
        dispatchUserEvent,
        stopUserStatePolling,
        startUserStatePolling,
        // Dropdown
        registerDropdown,
        unregisterDropdown,
        // Error handling
        handleApiError,
        // Performance monitoring
        measurePerformance,
        measurePerformanceAsync,
        // Utilities
        escapeHtml,
        isSafeUrl,
        // URL builders
        appendParam,
        appendCommonFilters,
        buildTagUrl,
        buildKeywordUrl,
        // Authors
        formatAuthorsText,
        // TL;DR
        renderTldrMarkdown,
        triggerMathJax,
    };
})(typeof window !== 'undefined' ? window : this);
