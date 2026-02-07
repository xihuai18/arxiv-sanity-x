'use strict';

// Common utilities shared across paper_list.js, paper_summary.js, and other modules.
// Exposes: window.ArxivSanityCommon

// Debug flag - set to true to enable render timing logs
const RENDER_DEBUG =
    typeof localStorage !== 'undefined' && localStorage.getItem('arxiv_render_debug') === '1';

function debugLog(category, message, data) {
    if (!RENDER_DEBUG) return;
    const timestamp = performance.now().toFixed(2);
    console.log(`[${timestamp}ms] [${category}] ${message}`, data || '');
}

(function (global) {
    const NS = 'ArxivSanityCommon';

    // =========================================================================
    // Toast Notifications (non-blocking replacement for window.alert)
    // =========================================================================

    const TOAST_CONTAINER_ID = 'as-toast-container';

    function ensureToastContainer() {
        if (typeof document === 'undefined') return null;
        const existing = document.getElementById(TOAST_CONTAINER_ID);
        if (existing) return existing;

        const create = () => {
            const c = document.createElement('div');
            c.id = TOAST_CONTAINER_ID;
            c.className = 'as-toast-container';
            document.body.appendChild(c);
            return c;
        };

        if (document.body) return create();
        // If called very early, wait until DOM is ready.
        document.addEventListener(
            'DOMContentLoaded',
            () => {
                if (!document.getElementById(TOAST_CONTAINER_ID) && document.body) create();
            },
            { once: true }
        );
        return null;
    }

    /**
     * Show a transient, non-blocking toast.
     * @param {string} message
     * @param {{type?: 'info'|'success'|'warning'|'error', durationMs?: number}} [opts]
     */
    function showToast(message, opts) {
        if (typeof document === 'undefined') return;
        const o = opts || {};
        const type = String(o.type || 'info');
        const durationMs = Number.isFinite(o.durationMs) ? o.durationMs : 2500;

        const container = ensureToastContainer() || document.getElementById(TOAST_CONTAINER_ID);
        if (!container) return;

        // Keep the UI tidy: cap to last 3 toasts.
        while (container.children.length >= 3) {
            container.removeChild(container.firstChild);
        }

        const el = document.createElement('div');
        el.className = 'as-toast as-toast--' + type;
        el.textContent = String(message || '');
        container.appendChild(el);

        const remove = () => {
            if (el && el.parentNode) el.parentNode.removeChild(el);
        };

        // Let CSS animate out if present.
        setTimeout(
            () => {
                el.classList.add('as-toast--hide');
                setTimeout(remove, 250);
            },
            Math.max(250, durationMs)
        );
    }

    // =========================================================================
    // CSRF Token Management
    // =========================================================================

    /**
     * Get CSRF token from meta tag
     * @returns {string} CSRF token or empty string if not found
     */
    function getCsrfToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') || '' : '';
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
    // Static URL Helpers
    // =========================================================================

    /**
     * Resolve the static base URL (useful when the app is deployed under a URL prefix).
     * Prefers the injected `window.__arxivSanityStaticBase` from templates/base.html.
     * @returns {string} Base URL that ends with '/'
     */
    function getStaticBaseUrl() {
        const injected =
            global && typeof global.__arxivSanityStaticBase === 'string'
                ? global.__arxivSanityStaticBase
                : '';
        if (injected) {
            return injected.endsWith('/') ? injected : injected + '/';
        }

        // Best-effort inference from existing asset URLs.
        try {
            const el =
                document.querySelector('link[href*="/static/"]') ||
                document.querySelector('script[src*="/static/"]');
            if (el) {
                const raw = el.getAttribute('href') || el.getAttribute('src') || '';
                const idx = raw.indexOf('/static/');
                if (idx >= 0) return raw.slice(0, idx + '/static/'.length);
            }
        } catch (e) {}

        return '/static/';
    }

    /**
     * Build a full URL to a static file.
     * @param {string} filename - Path relative to the static root (e.g., 'lib/jszip.min.js')
     * @returns {string}
     */
    function staticUrl(filename) {
        const base = getStaticBaseUrl();
        const rel = String(filename || '').replace(/^\/+/, '');
        return base + rel;
    }

    // =========================================================================
    // User Event Stream (SSE + Polling fallback)
    // =========================================================================

    const USER_EVENT_CHANNEL_PREFIX = 'arxiv-sanity-user-events';
    const USER_EVENT_LEADER_KEY_PREFIX = 'arxiv-sanity-user-events-leader';
    const USER_EVENT_LEADER_STALE_MS = 20000;
    const USER_EVENT_LEADER_HEARTBEAT_MS = 3000;
    const USER_EVENT_LEADER_MONITOR_MS = 2000;
    const USER_STATE_POLL_FAST_MS = 5000;
    const USER_STATE_POLL_SLOW_MS = 15000;
    let userEventChannel = null;
    let userEventChannelName = null;
    let userStatePoller = null;
    let userStatePollIntervalMs = null;
    let eventSource = null;
    let eventHandlers = [];
    let eventSourceReconnectTimer = null;
    let eventSourceConnecting = false;
    let eventSourceConnectWatchdogTimer = null;
    let userStateApplyFns = [];
    let userEventTabId = null;
    let userEventLeaderTimer = null;
    let userEventLeaderMonitorTimer = null;
    let userEventLeaderMonitorFn = null;
    let userEventUnloadBound = false;
    let userEventStorageAvailable = true;
    let userEventNamespace = null;
    let userEventStorageListenerBound = false;

    function normalizeUserKey(user) {
        if (!user) return '';
        if (typeof user === 'string') return user;
        if (typeof user === 'object') {
            return user.username || user.user || user.name || user.id || '';
        }
        return String(user);
    }

    function setUserEventNamespace(user) {
        const key = normalizeUserKey(user);
        if (key && userEventNamespace === key) return;
        userEventNamespace = key || 'anon';

        // If user changes (e.g. logout/login without full reload), reset cross-tab channel.
        if (userEventChannel) {
            try {
                userEventChannel.close();
            } catch (e) {}
            userEventChannel = null;
            userEventChannelName = null;
        }
    }

    function getUserEventChannelName() {
        return USER_EVENT_CHANNEL_PREFIX + ':' + (userEventNamespace || 'anon');
    }

    function getUserEventLeaderKey() {
        return USER_EVENT_LEADER_KEY_PREFIX + ':' + (userEventNamespace || 'anon');
    }

    /**
     * Initialize BroadcastChannel for cross-tab communication
     * @returns {BroadcastChannel|null} BroadcastChannel instance or null if not supported
     */
    function initBroadcastChannel() {
        const name = getUserEventChannelName();
        if (userEventChannel && userEventChannelName === name) return userEventChannel;
        if (typeof BroadcastChannel !== 'undefined') {
            if (userEventChannel) {
                try {
                    userEventChannel.close();
                } catch (e) {}
            }
            userEventChannel = new BroadcastChannel(name);
            userEventChannelName = name;
            startUserEventLeaderElection();
        }
        return userEventChannel;
    }

    function isSseOpen() {
        try {
            return (
                eventSource &&
                typeof EventSource !== 'undefined' &&
                eventSource.readyState === EventSource.OPEN
            );
        } catch (e) {
            return false;
        }
    }

    function getUserEventTabId() {
        if (userEventTabId) return userEventTabId;
        userEventTabId = Math.random().toString(36).slice(2) + Date.now().toString(36);
        return userEventTabId;
    }

    function readUserEventLeader() {
        try {
            if (typeof localStorage === 'undefined') return null;
            const raw = localStorage.getItem(getUserEventLeaderKey());
            if (!raw) return null;
            const parsed = JSON.parse(raw);
            if (!parsed || !parsed.id || (!parsed.ts && !parsed.leaseUntil)) return null;
            return parsed;
        } catch (e) {
            userEventStorageAvailable = false;
            return null;
        }
    }

    function leaderIsStale(leader) {
        if (!leader) return true;
        const now = Date.now();
        if (leader.leaseUntil) return now > Number(leader.leaseUntil);
        if (!leader.ts) return true;
        return now - Number(leader.ts) > USER_EVENT_LEADER_STALE_MS;
    }

    function writeUserEventLeader(id) {
        try {
            if (typeof localStorage === 'undefined') return false;
            const record = {
                id,
                ts: Date.now(),
                leaseUntil: Date.now() + USER_EVENT_LEADER_STALE_MS,
            };
            const key = getUserEventLeaderKey();
            localStorage.setItem(key, JSON.stringify(record));
            // Read-after-write verification reduces split-brain on some browsers/storage backends.
            const raw = localStorage.getItem(key);
            if (!raw) return false;
            const parsed = JSON.parse(raw);
            return parsed && parsed.id === id;
        } catch (e) {
            userEventStorageAvailable = false;
            return false;
        }
    }

    function shouldBroadcastFromThisTab() {
        // If localStorage is unavailable, keep legacy behavior (best-effort broadcast).
        if (typeof localStorage === 'undefined' || !userEventStorageAvailable) return true;

        const tabId = getUserEventTabId();
        const leader = readUserEventLeader();

        if (!leader || leaderIsStale(leader)) {
            // Only a tab with an open SSE connection may (re)claim leadership.
            if (isSseOpen() && writeUserEventLeader(tabId)) return true;
        }
        const current = readUserEventLeader();
        if (!current) return true;
        return current.id === tabId;
    }

    function isCurrentTabLeader() {
        try {
            if (typeof localStorage === 'undefined' || !userEventStorageAvailable) return true;
            const tabId = getUserEventTabId();
            const leader = readUserEventLeader();
            if (!leader || leaderIsStale(leader)) return false;
            return leader.id === tabId;
        } catch (e) {
            return true;
        }
    }

    function relinquishLeadershipIfOwned() {
        try {
            if (typeof localStorage === 'undefined' || !userEventStorageAvailable) return;
            if (isCurrentTabLeader()) {
                localStorage.removeItem(getUserEventLeaderKey());
            }
        } catch (e) {}
    }

    function tryClaimUserEventLeadership() {
        // If localStorage is unavailable, keep legacy behavior (allow every tab to connect SSE).
        if (typeof localStorage === 'undefined' || !userEventStorageAvailable) return true;
        const tabId = getUserEventTabId();
        const leader = readUserEventLeader();
        if (!leader || leaderIsStale(leader) || leader.id === tabId) {
            // Try claim (last writer wins).
            if (!writeUserEventLeader(tabId)) return true;
        }
        const current = readUserEventLeader();
        if (!current || !current.id) return true;
        return current.id === tabId;
    }

    function startUserEventLeaderElection() {
        if (userEventLeaderTimer) return;
        if (typeof localStorage === 'undefined' || !userEventStorageAvailable) return;
        userEventLeaderTimer = setInterval(() => {
            if (!isSseOpen()) return;
            const tabId = getUserEventTabId();
            const leader = readUserEventLeader();
            if (!leader || leaderIsStale(leader) || leader.id === tabId) {
                writeUserEventLeader(tabId);
            }
        }, USER_EVENT_LEADER_HEARTBEAT_MS);
    }

    function startUserEventLeaderMonitor(fn) {
        userEventLeaderMonitorFn = fn;
        if (userEventLeaderMonitorTimer) return;
        userEventLeaderMonitorTimer = setInterval(() => {
            if (typeof userEventLeaderMonitorFn !== 'function') return;
            try {
                userEventLeaderMonitorFn();
            } catch (e) {}
        }, USER_EVENT_LEADER_MONITOR_MS);
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

    function addUserStateApplyFn(applyFn) {
        if (typeof applyFn !== 'function') return;
        if (userStateApplyFns.includes(applyFn)) return;
        userStateApplyFns.push(applyFn);
    }

    function applyUserStateToAll(state) {
        userStateApplyFns.forEach(fn => {
            try {
                fn(state);
            } catch (e) {
                console.warn('User state apply error:', e);
            }
        });
    }

    function computeUserStatePollIntervalMs() {
        // If we don't have BroadcastChannel/localStorage coordination, poll faster for freshness.
        if (!userEventChannel) return USER_STATE_POLL_FAST_MS;
        if (typeof localStorage === 'undefined' || !userEventStorageAvailable)
            return USER_STATE_POLL_FAST_MS;

        // If another tab is leader, we expect broadcasts to be timely; poll slowly as a safety net.
        const leader = readUserEventLeader();
        if (leader && !leaderIsStale(leader) && leader.id !== getUserEventTabId()) {
            return USER_STATE_POLL_SLOW_MS;
        }
        return USER_STATE_POLL_FAST_MS;
    }

    /**
     * Stop user state polling
     */
    function stopUserStatePolling() {
        if (userStatePoller) {
            clearInterval(userStatePoller);
            userStatePoller = null;
        }
        userStatePollIntervalMs = null;
    }

    /**
     * Start user state polling with specified interval
     * @param {Function} applyFn - Function to apply user state updates
     */
    function startUserStatePolling(applyFn) {
        addUserStateApplyFn(applyFn);
        const intervalMs = computeUserStatePollIntervalMs();
        if (userStatePoller && userStatePollIntervalMs === intervalMs) return;
        if (userStatePoller) {
            clearInterval(userStatePoller);
            userStatePoller = null;
        }
        userStatePollIntervalMs = intervalMs;
        // Apply once immediately to reduce perceived lag when falling back to polling.
        fetchUserState().then(state => {
            applyUserStateToAll(state);
        });
        userStatePoller = setInterval(() => {
            fetchUserState().then(state => {
                applyUserStateToAll(state);
            });
        }, intervalMs);
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
        if (!opts.fromBroadcast && userEventChannel && shouldBroadcastFromThisTab()) {
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
        setUserEventNamespace(user);
        addUserStateApplyFn(applyStateFn);

        const channel = initBroadcastChannel();
        if (channel) {
            channel.onmessage = e => {
                // If SSE is connected in this tab, ignore broadcast to avoid duplicates.
                if (isSseOpen()) return;
                dispatchUserEvent(e.data || {}, { fromBroadcast: true });
            };
        }

        if (typeof EventSource === 'undefined') {
            startUserStatePolling(applyStateFn);
            return;
        }

        const scheduleReconnect = () => {
            if (eventSourceReconnectTimer) return;
            eventSourceReconnectTimer = setTimeout(() => {
                eventSourceReconnectTimer = null;
                connectAsLeaderIfNeeded();
            }, 8000);
        };

        const clearConnectWatchdog = () => {
            if (!eventSourceConnectWatchdogTimer) return;
            clearTimeout(eventSourceConnectWatchdogTimer);
            eventSourceConnectWatchdogTimer = null;
        };

        const startConnectWatchdog = () => {
            clearConnectWatchdog();
            eventSourceConnectWatchdogTimer = setTimeout(() => {
                // If EventSource gets stuck in CONNECTING (no open/error), reset and retry.
                try {
                    if (
                        eventSource &&
                        typeof EventSource !== 'undefined' &&
                        eventSource.readyState === EventSource.CONNECTING
                    ) {
                        try {
                            eventSource.close();
                        } catch (e) {}
                        eventSource = null;
                        eventSourceConnecting = false;
                        relinquishLeadershipIfOwned();
                        startUserStatePolling();
                        scheduleReconnect();
                    }
                } catch (e) {
                    // Best-effort: avoid wedging reconnect loop.
                    eventSourceConnecting = false;
                } finally {
                    eventSourceConnectWatchdogTimer = null;
                }
            }, 15000);
        };

        const connect = () => {
            if (eventSource || eventSourceConnecting) return;
            // Avoid opening SSE in every tab: only the leader tab should connect.
            if (!tryClaimUserEventLeadership()) {
                startUserStatePolling();
                return;
            }
            eventSourceConnecting = true;
            try {
                eventSource = new EventSource('/api/user_stream');
            } catch (e) {
                eventSource = null;
                eventSourceConnecting = false;
                relinquishLeadershipIfOwned();
                startUserStatePolling();
                scheduleReconnect();
                return;
            }
            startConnectWatchdog();
            eventSource.onopen = () => {
                stopUserStatePolling();
                // Prefer an SSE-connected tab to lead BroadcastChannel fanout.
                startUserEventLeaderElection();
                eventSourceConnecting = false;
                clearConnectWatchdog();
                if (eventSourceReconnectTimer) {
                    clearTimeout(eventSourceReconnectTimer);
                    eventSourceReconnectTimer = null;
                }
            };
            eventSource.onmessage = evt => {
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
                eventSourceConnecting = false;
                clearConnectWatchdog();
                // If we were the leader, relinquish so another tab can take over quickly.
                relinquishLeadershipIfOwned();
                startUserStatePolling();
                scheduleReconnect();
            };
        };

        const connectAsLeaderIfNeeded = () => {
            // If another tab is the leader, don't connect; rely on broadcast + polling.
            if (!tryClaimUserEventLeadership()) {
                if (eventSource) {
                    try {
                        eventSource.close();
                    } catch (e) {}
                    eventSource = null;
                }
                eventSourceConnecting = false;
                clearConnectWatchdog();
                startUserStatePolling();
                return;
            }
            connect();
        };

        // Periodically monitor leader staleness and avoid multiple SSE connections per user.
        startUserEventLeaderMonitor(() => {
            // If SSE exists (including CONNECTING) but leadership moved elsewhere, close to free server resources.
            if (eventSource && !isCurrentTabLeader()) {
                try {
                    eventSource.close();
                } catch (e) {}
                eventSource = null;
                eventSourceConnecting = false;
                clearConnectWatchdog();
                startUserStatePolling();
                return;
            }
            // If SSE is not open, try to become leader when the current leader is stale.
            if (!isSseOpen()) {
                const leader = readUserEventLeader();
                if (!leader || leaderIsStale(leader) || leader.id === getUserEventTabId()) {
                    connectAsLeaderIfNeeded();
                } else {
                    startUserStatePolling();
                }
            }
        });

        // Best-effort: on tab close, release leadership so other tabs can connect immediately.
        if (!userEventUnloadBound) {
            userEventUnloadBound = true;
            try {
                window.addEventListener('beforeunload', () => {
                    relinquishLeadershipIfOwned();
                });
            } catch (e) {}
        }

        // React quickly to leadership changes across tabs.
        if (!userEventStorageListenerBound) {
            userEventStorageListenerBound = true;
            try {
                window.addEventListener('storage', e => {
                    if (!e || !e.key) return;
                    if (e.key !== getUserEventLeaderKey()) return;
                    if (eventSource && !isCurrentTabLeader()) {
                        try {
                            eventSource.close();
                        } catch (err) {}
                        eventSource = null;
                        eventSourceConnecting = false;
                        clearConnectWatchdog();
                        startUserStatePolling();
                    }
                });
            } catch (e) {}
        }

        connectAsLeaderIfNeeded();
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

        document.addEventListener('mousedown', event => {
            dropdownRegistry.forEach((api, id) => {
                if (!api || !api.isOpen || !api.isOpen()) return;
                const dropdown = document.getElementById(id);
                if (dropdown && !dropdown.contains(event.target)) {
                    api.close();
                }
            });
        });

        document.addEventListener('keydown', event => {
            if (event.key !== 'Escape') return;
            dropdownRegistry.forEach(api => {
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

    function tryReloadForCsrf() {
        // Reload at most once per tab session (with a cooldown) to prevent reload storms
        // when the server keeps returning CSRF-related 403 responses.
        let didReload = false;
        try {
            const key = '__arxivSanityCsrfReloadedAt';
            const now = Date.now();
            const last = Number(sessionStorage.getItem(key) || 0);
            const cooldownMs = 60 * 1000;
            const canReload =
                !global.__arxivSanityCsrfReloaded && (!last || now - last > cooldownMs);
            if (canReload) {
                global.__arxivSanityCsrfReloaded = true;
                sessionStorage.setItem(key, String(now));
                didReload = true;
            }
        } catch (e) {
            // Fallback: only guard within the current page instance.
            if (!global.__arxivSanityCsrfReloaded) {
                global.__arxivSanityCsrfReloaded = true;
                didReload = true;
            }
        }
        if (didReload) {
            try {
                global.location.reload();
            } catch (e) {}
        }
        return didReload;
    }

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
            // CSRF token mismatch after server restart - auto reload page
            if (msg.includes('CSRF') || msg.includes('csrf')) {
                if (tryReloadForCsrf()) {
                    console.warn('CSRF token invalid, reloading page...');
                    return 'Session expired. Reloading page...';
                }
                return 'Session expired. Please refresh the page.';
            }
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
        appendCommonFilters(
            params,
            { includeSearchMode: true, includeSemanticWeight: true },
            opts.gvars
        );
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
        if (
            typeof window !== 'undefined' &&
            window.ArxivSanityAuthors &&
            window.ArxivSanityAuthors.format
        ) {
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
        if (
            typeof window !== 'undefined' &&
            window.ArxivSanityTldr &&
            window.ArxivSanityTldr.render
        ) {
            return window.ArxivSanityTldr.render(text);
        }
        if (!text) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }

    // =========================================================================
    // Abstract Markdown Rendering Helpers
    // =========================================================================

    let abstractMarkdownIt = null;

    function getAbstractMarkdownIt() {
        if (abstractMarkdownIt) return abstractMarkdownIt;

        // Prefer shared markdown-it wrapper (when available)
        try {
            const Renderer =
                typeof window !== 'undefined' ? window.ArxivSanityMarkdownRenderer : null;
            if (Renderer && typeof Renderer.createMarkdownIt === 'function') {
                const md = Renderer.createMarkdownIt({
                    html: false,
                    breaks: true,
                    linkify: true,
                    typographer: false,
                });
                if (md) {
                    // Enforce safe links (prevent javascript:, data:, file:, etc.)
                    if (typeof Renderer.setSafeLinkValidator === 'function') {
                        Renderer.setSafeLinkValidator(md, {
                            allowRelative: true,
                            allowHash: true,
                            baseValidator: isSafeUrl,
                        });
                    }
                    abstractMarkdownIt = md;
                    return abstractMarkdownIt;
                }
            }
        } catch (e) {
            // ignore
        }

        // Fallback to global markdownit if present
        try {
            if (typeof window !== 'undefined' && typeof window.markdownit === 'function') {
                const md = window.markdownit({
                    html: false,
                    breaks: true,
                    linkify: true,
                    typographer: false,
                });
                // Best-effort safe link validation
                const Sanitizer =
                    typeof window !== 'undefined' ? window.ArxivSanityMarkdownSanitizer : null;
                if (Sanitizer && typeof Sanitizer.buildLinkValidator === 'function') {
                    md.validateLink = Sanitizer.buildLinkValidator({
                        allowRelative: true,
                        allowHash: true,
                        baseValidator: isSafeUrl,
                    });
                }
                abstractMarkdownIt = md;
                return abstractMarkdownIt;
            }
        } catch (e) {
            // ignore
        }

        return null;
    }

    /**
     * Render abstract markdown to HTML.
     *
     * Strategy:
     * 1) Prefer the shared TL;DR renderer (adds math parsing and safe escaping) when available.
     * 2) Otherwise, use a local markdown-it instance (html disabled + safe link validator).
     * 3) Fallback to escaped plain text.
     *
     * @param {string} text - Abstract text (may include markdown and LaTeX)
     * @returns {string} Rendered HTML
     */
    function renderAbstractMarkdown(text) {
        if (!text) return '';

        let s = String(text);

        // Remove images for safety/UX (abstracts should be text-only)
        try {
            if (
                typeof window !== 'undefined' &&
                window.ArxivSanityMarkdownSanitizer &&
                typeof window.ArxivSanityMarkdownSanitizer.stripMarkdownImages === 'function'
            ) {
                s = window.ArxivSanityMarkdownSanitizer.stripMarkdownImages(s);
            }
        } catch (e) {
            // ignore
        }

        // Prefer TL;DR renderer if present (consistent markdown + math handling)
        if (
            typeof window !== 'undefined' &&
            window.ArxivSanityTldr &&
            typeof window.ArxivSanityTldr.render === 'function'
        ) {
            return window.ArxivSanityTldr.render(s);
        }

        const md = getAbstractMarkdownIt();
        if (md && typeof md.render === 'function') {
            try {
                return md.render(s);
            } catch (e) {
                // fall through
            }
        }

        // Final fallback: escape + preserve line breaks
        return escapeHtml(s).replace(/\n/g, '<br>');
    }

    /**
     * Trigger MathJax typesetting for element
     * @param {HTMLElement} [element] - Element to typeset (or entire document if omitted)
     */
    function triggerMathJax(element) {
        function _typesetNow() {
            if (typeof MathJax === 'undefined') return;
            const nodes = element ? [element] : undefined;
            try {
                if (MathJax.startup && MathJax.startup.promise) {
                    MathJax.startup.promise
                        .then(() => {
                            if (MathJax.typesetPromise) return MathJax.typesetPromise(nodes);
                            if (MathJax.typeset) MathJax.typeset(nodes);
                            return null;
                        })
                        .catch(err => {
                            console.warn('MathJax typeset error:', err);
                        });
                    return;
                }
                if (MathJax.typesetPromise) {
                    MathJax.typesetPromise(nodes).catch(function (err) {
                        console.warn('MathJax typeset error:', err);
                    });
                    return;
                }
                if (MathJax.typeset) {
                    MathJax.typeset(nodes);
                }
            } catch (err) {
                console.warn('MathJax typeset error:', err);
            }
        }

        // If MathJax is fully loaded (has typeset methods), typeset immediately.
        if (typeof MathJax !== 'undefined' && (MathJax.typesetPromise || MathJax.typeset)) {
            _typesetNow();
            return;
        }

        // Only lazy-load MathJax if we detect math content.
        // Note: MathJax config object may exist but script not loaded yet
        try {
            // If summary markdown placeholders exist, load MathJax even if textContent is empty.
            if (
                element &&
                typeof element.querySelector === 'function' &&
                element.querySelector('[data-tex]')
            ) {
                // fall through
            } else {
                const text = element
                    ? element.textContent || ''
                    : (document.body && document.body.textContent) || '';
                if (!hasMathContent(text)) return;
            }
        } catch (e) {
            // If detection fails, be conservative and load MathJax.
        }

        loadMathJaxOnDemand(function () {
            _typesetNow();
        });
    }

    // =========================================================================
    // MathJax Lazy Loading
    // =========================================================================

    let mathJaxLoading = false;
    let mathJaxLoaded = false;
    const mathJaxCallbacks = [];
    let mathJaxLoadFailures = 0;
    let mathJaxRetryTimer = null;
    let mathJaxAttempt = 0;
    let mathJaxWatchdogTimer = null;

    function isMathJaxAvailable() {
        try {
            if (typeof MathJax === 'undefined') return false;
            if (typeof MathJax.typesetPromise === 'function') return true;
            if (typeof MathJax.typeset === 'function') return true;
            if (typeof MathJax.tex2chtmlPromise === 'function') return true;
            if (
                MathJax.startup &&
                MathJax.startup.document &&
                typeof MathJax.startup.document.convert === 'function'
            )
                return true;
        } catch (e) {}
        return false;
    }

    function runMathJaxCallbacks() {
        while (mathJaxCallbacks.length > 0) {
            const cb = mathJaxCallbacks.shift();
            if (typeof cb !== 'function') continue;
            try {
                cb();
            } catch (e) {
                console.warn('MathJax callback error:', e);
            }
        }
    }

    /**
     * Default MathJax configuration
     */
    const defaultMathJaxConfig = {
        options: {
            enableMenu: false,
            enableAssistiveMml: false,
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process',
        },
        tex: {
            inlineMath: [
                ['\\(', '\\)'],
                ['$', '$'],
            ],
            displayMath: [
                ['\\[', '\\]'],
                ['$$', '$$'],
            ],
            processEscapes: true,
            processEnvironments: true,
            processRefs: true,
            packages: {
                '[+]': [
                    'base',
                    'ams',
                    'boldsymbol',
                    'braket',
                    'cancel',
                    'cases',
                    'color',
                    'enclose',
                    'mathtools',
                    'newcommand',
                    'noerrors',
                    'noundefined',
                    'physics',
                    'tagformat',
                    'textmacros',
                    'unicode',
                    'verb',
                ],
            },
            tags: 'ams',
            macros: {
                // Bold and emphasis
                bm: ['\\mathbf{#1}', 1],
                boldsymbol: ['\\mathbf{#1}', 1],
                bold: ['\\mathbf{#1}', 1],
                pmb: ['\\mathbf{#1}', 1],

                // Common sets (blackboard bold)
                RR: '\\mathbb{R}',
                NN: '\\mathbb{N}',
                ZZ: '\\mathbb{Z}',
                QQ: '\\mathbb{Q}',
                CC: '\\mathbb{C}',
                PP: '\\mathbb{P}',
                EE: '\\mathbb{E}',
                HH: '\\mathbb{H}',
                FF: '\\mathbb{F}',

                // Delimiters
                abs: ['\\left|#1\\right|', 1],
                norm: ['\\left\\|#1\\right\\|', 1],
                inner: ['\\left\\langle#1,#2\\right\\rangle', 2],
                set: ['\\left\\{#1\\right\\}', 1],
                floor: ['\\left\\lfloor#1\\right\\rfloor', 1],
                ceil: ['\\left\\lceil#1\\right\\rceil', 1],

                // Common operators
                argmax: '\\operatorname*{arg\\,max}',
                argmin: '\\operatorname*{arg\\,min}',

                // Indicator function (mathbbm package support)
                ind: '\\mathbb{1}',
                indicator: '\\mathbb{1}',
                one: '\\mathbb{1}',
                mathbbm: ['\\mathbb{#1}', 1],
                mathds: ['\\mathbb{#1}', 1],

                // Script fonts
                mathscr: ['\\mathcal{#1}', 1],
                mathpzc: ['\\mathcal{#1}', 1],

                // Probability and statistics
                Var: '\\mathrm{Var}',
                Cov: '\\mathrm{Cov}',
                Corr: '\\mathrm{Corr}',
                Pr: '\\mathrm{Pr}',
                E: '\\mathbb{E}',

                // Linear algebra
                rank: '\\mathrm{rank}',
                tr: '\\mathrm{tr}',
                Tr: '\\mathrm{Tr}',
                diag: '\\mathrm{diag}',
                vec: ['\\mathbf{#1}', 1],
                mat: ['\\mathbf{#1}', 1],

                // Text in math mode - use math font variants for compatibility
                textit: ['\\mathit{#1}', 1],
                textbf: ['\\mathbf{#1}', 1],
                textrm: ['\\mathrm{#1}', 1],
                textsf: ['\\mathsf{#1}', 1],
                texttt: ['\\mathtt{#1}', 1],

                // Common spacing (note: do NOT redefine \mid as it's a standard LaTeX relation symbol)
                given: '\\,|\\,',

                // Transpose and related
                T: '^{\\mathsf{T}}',
                transpose: '^{\\mathsf{T}}',
                inv: '^{-1}',

                // Common decorations
                hat: ['\\widehat{#1}', 1],
                tilde: ['\\widetilde{#1}', 1],
                bar: ['\\overline{#1}', 1],

                // Machine learning
                Loss: '\\mathcal{L}',
                Data: '\\mathcal{D}',
                Model: '\\mathcal{M}',

                // Optimization
                prox: '\\mathrm{prox}',
                proj: '\\mathrm{proj}',
                dom: '\\mathrm{dom}',

                // Additional symbols
                eps: '\\varepsilon',
                vphi: '\\varphi',

                // Probability distributions
                Normal: '\\mathcal{N}',
            },
        },
        // (moved to options above)
        chtml: {
            scale: 1.0,
            displayAlign: 'center',
            // Ensure MathJax web fonts resolve correctly in self-hosted deployments.
            // Without an explicit fontURL, some setups may request fonts from a wrong relative path,
            // causing missing glyphs in rendered formulas.
            fontURL: staticUrl('lib/es5/output/chtml/fonts/woff-v2'),
        },
    };

    /**
     * Check if text contains math expressions
     * @param {string} text - Text to check
     * @returns {boolean} True if math expressions found
     */
    function hasMathContent(text) {
        if (!text) return false;
        // Check for common LaTeX patterns
        return /\$\$|\$[^$]+\$|\\\[|\\\(|\\begin\{|\\frac|\\sum|\\int|\\alpha|\\beta|\\gamma/.test(
            text
        );
    }

    /**
     * Check if page has any math content that needs rendering
     * @returns {boolean} True if math content found
     */
    function pageHasMathContent() {
        const body = document.body;
        if (!body) return false;
        // Check text content for math patterns
        const text = body.textContent || '';
        return hasMathContent(text);
    }

    /**
     * Load MathJax on demand
     * @param {Function} [callback] - Called when MathJax is ready
     * @param {string} [scriptUrl] - Custom MathJax script URL
     */
    function loadMathJaxOnDemand(callback, scriptUrl) {
        // Queue callback first so we can consistently drain the queue when MathJax becomes ready.
        if (callback) mathJaxCallbacks.push(callback);

        // Already loaded / ready (covers cases where a script tag was loaded outside this helper)
        if (mathJaxLoaded || isMathJaxAvailable()) {
            mathJaxLoaded = true;
            mathJaxLoading = false;
            mathJaxLoadFailures = 0;
            try {
                if (mathJaxRetryTimer) {
                    clearTimeout(mathJaxRetryTimer);
                    mathJaxRetryTimer = null;
                }
            } catch (e) {}
            try {
                if (mathJaxWatchdogTimer) {
                    clearTimeout(mathJaxWatchdogTimer);
                    mathJaxWatchdogTimer = null;
                }
            } catch (e) {}

            // Wait for MathJax startup (if available) before running callbacks.
            try {
                if (typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.promise) {
                    MathJax.startup.promise.then(runMathJaxCallbacks).catch(runMathJaxCallbacks);
                    return;
                }
            } catch (e) {}
            runMathJaxCallbacks();
            return;
        }

        // Already loading
        if (mathJaxLoading) return;
        mathJaxLoading = true;
        mathJaxAttempt += 1;
        const attempt = mathJaxAttempt;

        try {
            if (mathJaxWatchdogTimer) {
                clearTimeout(mathJaxWatchdogTimer);
                mathJaxWatchdogTimer = null;
            }
        } catch (e) {}

        // Set config before loading script
        if (typeof window.MathJax === 'undefined') {
            window.MathJax = defaultMathJaxConfig;
        }

        // Create and load script (avoid duplicates)
        let script = document.getElementById('MathJax-script');
        let isExternalScript = false;
        try {
            isExternalScript =
                !!script && (!script.dataset || String(script.dataset.mjxCreated || '') !== '1');
        } catch (e) {
            isExternalScript = !!script;
        }

        // If a previous attempt failed, remove the script so we can retry.
        // This helps with transient network errors and multi-backend deployments where
        // a subsequent request may hit a different static server successfully.
        try {
            if (script && script.dataset && script.dataset.mjxFailed === '1') {
                script.remove();
                script = null;
            }
        } catch (e) {}

        let finished = false;
        const finishOk = function () {
            if (finished) return;
            if (attempt !== mathJaxAttempt) return;
            finished = true;

            mathJaxLoaded = true;
            mathJaxLoading = false;
            mathJaxLoadFailures = 0;
            try {
                if (mathJaxRetryTimer) {
                    clearTimeout(mathJaxRetryTimer);
                    mathJaxRetryTimer = null;
                }
            } catch (e) {}
            try {
                if (mathJaxWatchdogTimer) {
                    clearTimeout(mathJaxWatchdogTimer);
                    mathJaxWatchdogTimer = null;
                }
            } catch (e) {}
            try {
                if (script && script.dataset) {
                    script.dataset.mjxFailed = '0';
                    script.dataset.mjxLoaded = '1';
                }
            } catch (e) {}

            // Wait for MathJax startup (if available) before running callbacks.
            try {
                if (typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.promise) {
                    MathJax.startup.promise.then(runMathJaxCallbacks).catch(runMathJaxCallbacks);
                    return;
                }
            } catch (e) {}
            runMathJaxCallbacks();
        };

        const finishFail = function () {
            if (finished) return;
            if (attempt !== mathJaxAttempt) return;
            finished = true;

            mathJaxLoading = false;
            mathJaxLoadFailures += 1;
            try {
                if (mathJaxWatchdogTimer) {
                    clearTimeout(mathJaxWatchdogTimer);
                    mathJaxWatchdogTimer = null;
                }
            } catch (e) {}
            try {
                if (script && script.dataset) script.dataset.mjxFailed = '1';
            } catch (e) {}
            try {
                if (script && script.parentNode) script.parentNode.removeChild(script);
            } catch (e) {}
            console.warn('Failed to load MathJax');

            if (mathJaxLoadFailures >= 3) {
                // Give up auto-retry; clear pending callbacks to avoid hanging forever.
                mathJaxCallbacks.length = 0;
                return;
            }

            // Best-effort retry with exponential backoff if there are pending callbacks.
            try {
                if (mathJaxCallbacks.length === 0) return;
                if (mathJaxRetryTimer) return;
                const delay = Math.min(8000, 300 * Math.pow(2, mathJaxLoadFailures - 1));
                mathJaxRetryTimer = setTimeout(() => {
                    mathJaxRetryTimer = null;
                    loadMathJaxOnDemand(null, scriptUrl);
                }, delay);
            } catch (e) {}
        };

        if (!script) {
            script = document.createElement('script');
            script.id = 'MathJax-script';
            script.async = true;
            try {
                if (script.dataset) script.dataset.mjxCreated = '1';
            } catch (e) {}
            // Bind handlers before setting src/append to avoid missing fast load events.
            script.onload = finishOk;
            script.onerror = finishFail;
            script.src = scriptUrl || staticUrl('lib/es5/tex-chtml-full.js');
            document.head.appendChild(script);
        } else {
            // Attach handlers to an existing script tag (e.g., summary.html).
            try {
                if (script.dataset) {
                    if (String(script.dataset.mjxManaged || '') !== '1') {
                        script.dataset.mjxManaged = '1';
                        script.addEventListener('load', finishOk, { once: true });
                        script.addEventListener('error', finishFail, { once: true });
                    }
                } else {
                    script.addEventListener('load', finishOk, { once: true });
                    script.addEventListener('error', finishFail, { once: true });
                }
            } catch (e) {
                // Older browsers: fall back to property assignment (may overwrite).
                script.onload = finishOk;
                script.onerror = finishFail;
            }
        }

        // In some cases (cached, already-failed, or onload missed), the load/error events may not fire.
        // Use a watchdog to avoid leaving the loader stuck in a "loading" state forever.
        try {
            const watchdogMs = isExternalScript ? 2000 : 30000;
            mathJaxWatchdogTimer = setTimeout(() => {
                if (attempt !== mathJaxAttempt) return;
                // If MathJax became available, finalize success.
                if (isMathJaxAvailable()) {
                    finishOk();
                    return;
                }
                finishFail();
            }, watchdogMs);
        } catch (e) {}
    }

    /**
     * Load MathJax only if page has math content
     * @param {Function} [callback] - Called when MathJax is ready (only if loaded)
     */
    function loadMathJaxIfNeeded(callback) {
        if (pageHasMathContent()) {
            loadMathJaxOnDemand(callback);
            return true;
        }
        return false;
    }

    // =========================================================================
    // Performance Monitoring (Development only)
    // =========================================================================

    const isDevelopment =
        typeof window !== 'undefined' &&
        (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

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
    // Summary Fallback Detection
    // =========================================================================

    /**
     * Check if a summary response indicates a model fallback occurred.
     * @param {Object} meta - The summary_meta from API response
     * @param {string} requestedModel - The model that was originally requested
     * @returns {{occurred: boolean, actualModel: string, notice: string}}
     */
    function checkSummaryFallback(meta, requestedModel) {
        const actualModel = String((meta && meta.llm_model) || '').trim();
        const requested = String(requestedModel || '').trim();

        if (actualModel && requested && actualModel !== requested) {
            return {
                occurred: true,
                actualModel: actualModel,
                notice: `Note: Fallback occurred. Summary generated by "${actualModel}" instead of "${requested}".`,
            };
        }
        return { occurred: false, actualModel: actualModel || requested, notice: '' };
    }

    // =========================================================================
    // Summary Status Polling (shared across paper_list.js and readinglist.html)
    // =========================================================================

    const SUMMARY_PENDING = new Set();
    let summaryStatusPoller = null;
    let summaryStatusCallback = null;
    let summaryStatusPollInFlight = false;

    /**
     * Normalize a paper ID to a consistent string format.
     * @param {string} pid - Paper ID
     * @returns {string} Normalized paper ID
     */
    function normalizePid(pid) {
        return String(pid || '').trim();
    }

    /**
     * Get the default summary model from global variable.
     * @returns {string} Model name or empty string
     */
    function getSummaryModel() {
        if (typeof defaultSummaryModel !== 'undefined') {
            return String(defaultSummaryModel || '').trim();
        }
        return '';
    }

    /**
     * Mark a paper as pending summary generation.
     * @param {string} pid - Paper ID
     */
    function markSummaryPending(pid) {
        const key = normalizePid(pid);
        if (!key) return;
        SUMMARY_PENDING.add(key);
        startSummaryStatusPolling();
    }

    /**
     * Unmark a paper from pending summary generation.
     * @param {string} pid - Paper ID
     */
    function unmarkSummaryPending(pid) {
        const key = normalizePid(pid);
        if (!key) return;
        SUMMARY_PENDING.delete(key);
        if (SUMMARY_PENDING.size === 0) {
            stopSummaryStatusPolling();
        }
    }

    /**
     * Stop the summary status polling interval.
     */
    function stopSummaryStatusPolling() {
        if (summaryStatusPoller) {
            clearInterval(summaryStatusPoller);
            summaryStatusPoller = null;
        }
    }

    /**
     * Start polling for summary status updates.
     */
    function startSummaryStatusPolling() {
        if (summaryStatusPoller) return;
        summaryStatusPoller = setInterval(() => {
            if (SUMMARY_PENDING.size === 0) {
                stopSummaryStatusPolling();
                return;
            }
            pollSummaryStatuses();
        }, 6000);
        pollSummaryStatuses();
    }

    /**
     * Poll the server for summary status of pending papers.
     */
    function pollSummaryStatuses() {
        if (SUMMARY_PENDING.size === 0) return;
        if (summaryStatusPollInFlight) return;
        summaryStatusPollInFlight = true;
        const model = getSummaryModel();
        if (!model) {
            // No summary model configured; avoid spamming the API with 400s.
            summaryStatusPollInFlight = false;
            stopSummaryStatusPolling();
            return;
        }
        const pids = Array.from(SUMMARY_PENDING);
        csrfFetch('/api/summary_status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pids, model }),
        })
            .then(resp => {
                if (!resp) return null;
                if (resp.status === 403) {
                    // Likely CSRF token stale (e.g., server restart / session reset).
                    // Reload once to refresh token and session.
                    stopSummaryStatusPolling();
                    try {
                        tryReloadForCsrf();
                    } catch (e) {}
                    return null;
                }
                if (!resp.ok) return null;
                return resp.json().catch(() => null);
            })
            .then(data => {
                if (!data || !data.success || !data.statuses) return;
                const statuses = data.statuses || {};
                Object.keys(statuses).forEach(pid => {
                    const info = statuses[pid] || {};
                    const status = info.status || '';
                    const lastError = info.last_error || '';
                    const taskId = info.task_id ? String(info.task_id) : '';
                    // Call the registered callback if available
                    if (summaryStatusCallback) {
                        summaryStatusCallback(pid, status, lastError, taskId);
                    }
                    if (status && status !== 'queued' && status !== 'running') {
                        unmarkSummaryPending(pid);
                    }
                });
            })
            .catch(err => {
                console.warn('Failed to poll summary status:', err);
            })
            .finally(() => {
                summaryStatusPollInFlight = false;
            });
    }

    /**
     * Register a callback to be called when summary status updates are received.
     * @param {Function} callback - Function(pid, status, lastError, taskId)
     */
    function setSummaryStatusCallback(callback) {
        summaryStatusCallback = callback;
    }

    /**
     * Fetch task status from the server.
     * @param {string} taskId - Task ID
     * @returns {Promise<Object|null>} Task status or null
     */
    function fetchTaskStatus(taskId) {
        if (!taskId) return Promise.resolve(null);
        return fetch(`/api/task_status/${encodeURIComponent(taskId)}`)
            .then(resp => (resp.ok ? resp.json() : null))
            .then(data => (data && data.success ? data : null))
            .catch(() => null);
    }

    /**
     * Check if a summary can be triggered based on current status.
     * @param {string} status - Current summary status
     * @returns {boolean} True if summary can be triggered
     */
    function canTriggerSummary(status) {
        return !(status === 'ok' || status === 'running' || status === 'queued');
    }

    /**
     * Format summary status for display.
     * @param {string} status - Summary status
     * @returns {string} Formatted status text
     */
    function formatSummaryStatus(status) {
        if (!status) return '';
        if (status === 'queued') return 'Summary Queued';
        if (status === 'running') return 'Summary Generating';
        if (status === 'ok') return 'Summary Ready';
        if (status === 'failed') return 'Summary Failed';
        if (status === 'canceled') return 'Summary Canceled';
        return 'Summary ' + status.charAt(0).toUpperCase() + status.slice(1);
    }

    // =========================================================================
    // Clipboard
    // =========================================================================

    /**
     * Copy text to clipboard with fallbacks.
     *
     * Notes:
     * - navigator.clipboard may not be available in insecure contexts (HTTP) or some browsers
     * - execCommand('copy') still works in older environments but may require user gesture
     *
     * @param {string} text - text to copy
     * @returns {Promise<boolean>} resolves true if copy succeeded, else false
     */
    function copyTextToClipboard(text) {
        const value = String(text == null ? '' : text);

        // Preferred: Async Clipboard API
        try {
            if (
                typeof navigator !== 'undefined' &&
                navigator.clipboard &&
                typeof navigator.clipboard.writeText === 'function'
            ) {
                return navigator.clipboard
                    .writeText(value)
                    .then(() => true)
                    .catch(() => false);
            }
        } catch (e) {}

        // Fallback: temporary textarea + execCommand
        return new Promise(resolve => {
            try {
                if (typeof document === 'undefined' || !document.body) {
                    resolve(false);
                    return;
                }

                const ta = document.createElement('textarea');
                ta.value = value;
                ta.setAttribute('readonly', '');
                // Avoid iOS zoom and keep off-screen
                ta.style.position = 'fixed';
                ta.style.top = '-1000px';
                ta.style.left = '-1000px';
                ta.style.width = '1px';
                ta.style.height = '1px';
                ta.style.opacity = '0';
                document.body.appendChild(ta);

                ta.focus();
                ta.select();
                ta.setSelectionRange(0, ta.value.length);

                let ok = false;
                try {
                    ok = document.execCommand && document.execCommand('copy');
                } catch (e) {
                    ok = false;
                }

                document.body.removeChild(ta);
                resolve(!!ok);
            } catch (e) {
                resolve(false);
            }
        });
    }

    // =========================================================================
    // Export API
    // =========================================================================

    global[NS] = {
        // CSRF
        getCsrfToken,
        csrfFetch,
        // Static URL
        getStaticBaseUrl,
        staticUrl,
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
        // Abstract
        renderAbstractMarkdown,
        triggerMathJax,
        // MathJax lazy loading
        hasMathContent,
        pageHasMathContent,
        loadMathJaxOnDemand,
        loadMathJaxIfNeeded,
        // Summary
        checkSummaryFallback,
        // Summary status polling (shared)
        normalizePid,
        getSummaryModel,
        markSummaryPending,
        unmarkSummaryPending,
        startSummaryStatusPolling,
        stopSummaryStatusPolling,
        pollSummaryStatuses,
        setSummaryStatusCallback,
        fetchTaskStatus,
        canTriggerSummary,
        formatSummaryStatus,
        // Clipboard
        copyTextToClipboard,
        // Toast
        showToast,
    };
})(typeof window !== 'undefined' ? window : this);
