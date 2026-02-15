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
        if (tok && method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS') {
            headers.set('X-CSRF-Token', tok);
        }
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

    // =========================================================================
    // Frontend Asset CDN Helpers
    // =========================================================================

    function isAssetCdnEnabled() {
        try {
            if (typeof global === 'undefined') return true;
            if (typeof global.__arxivSanityAssetCdnEnabled === 'undefined') return true;
            return !!global.__arxivSanityAssetCdnEnabled;
        } catch (e) {
            return true;
        }
    }

    function getAssetNpmCdnBase() {
        try {
            const raw =
                global && typeof global.__arxivSanityAssetNpmCdnBase === 'string'
                    ? global.__arxivSanityAssetNpmCdnBase
                    : '';
            const base = String(raw || '').trim();
            return base || 'https://cdn.jsdelivr.net/npm';
        } catch (e) {
            return 'https://cdn.jsdelivr.net/npm';
        }
    }

    function npmCdnUrl(path) {
        const base = getAssetNpmCdnBase().replace(/\/+$/, '');
        const rel = String(path || '').replace(/^\/+/, '');
        return base + '/' + rel;
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
    const SSE_RECONNECT_BASE_MS = 8000;
    const SSE_RECONNECT_MAX_MS = 120000; // cap at 2 minutes
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
    let sseConsecutiveErrors = 0;
    let pollConsecutiveErrors = 0;
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
     * Compute exponential backoff delay with jitter.
     * @param {number} consecutiveErrors - Number of consecutive errors
     * @param {number} baseMs - Base delay in ms
     * @param {number} maxMs - Maximum delay in ms
     * @returns {number} Delay in ms
     */
    function computeBackoffMs(consecutiveErrors, baseMs, maxMs) {
        if (consecutiveErrors <= 0) return baseMs;
        // Exponential: base * 2^(errors-1), capped at max, with ±25% jitter
        const raw = baseMs * Math.pow(2, Math.min(consecutiveErrors - 1, 10));
        const capped = Math.min(raw, maxMs);
        const jitter = capped * (0.75 + Math.random() * 0.5);
        return Math.round(jitter);
    }

    /**
     * Fetch user state from server
     * @returns {Promise<Object|null>} User state object or null on error
     */
    function fetchUserState() {
        return fetch('/api/user_state', { credentials: 'same-origin' })
            .then(resp => {
                if (!resp.ok) {
                    pollConsecutiveErrors++;
                    return null;
                }
                pollConsecutiveErrors = 0;
                return resp.json();
            })
            .catch(() => {
                pollConsecutiveErrors++;
                return null;
            });
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
        let baseMs;
        if (!userEventChannel) {
            baseMs = USER_STATE_POLL_FAST_MS;
        } else if (typeof localStorage === 'undefined' || !userEventStorageAvailable) {
            baseMs = USER_STATE_POLL_FAST_MS;
        } else {
            // If another tab is leader, we expect broadcasts to be timely; poll slowly as a safety net.
            const leader = readUserEventLeader();
            if (leader && !leaderIsStale(leader) && leader.id !== getUserEventTabId()) {
                baseMs = USER_STATE_POLL_SLOW_MS;
            } else {
                baseMs = USER_STATE_POLL_FAST_MS;
            }
        }
        // Apply exponential backoff when server is unreachable
        if (pollConsecutiveErrors > 0) {
            return computeBackoffMs(pollConsecutiveErrors, baseMs, SSE_RECONNECT_MAX_MS);
        }
        return baseMs;
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
            // Re-adjust polling interval after each fetch (backoff may have changed)
            _maybeReadjustPollInterval();
        });
        userStatePoller = setInterval(() => {
            fetchUserState().then(state => {
                applyUserStateToAll(state);
                _maybeReadjustPollInterval();
            });
        }, intervalMs);
    }

    /**
     * Re-adjust polling interval if backoff state changed (e.g. errors resolved or accumulated).
     */
    function _maybeReadjustPollInterval() {
        if (!userStatePoller) return;
        const newInterval = computeUserStatePollIntervalMs();
        if (newInterval !== userStatePollIntervalMs) {
            clearInterval(userStatePoller);
            userStatePollIntervalMs = newInterval;
            userStatePoller = setInterval(() => {
                fetchUserState().then(state => {
                    applyUserStateToAll(state);
                    _maybeReadjustPollInterval();
                });
            }, newInterval);
        }
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
            const delayMs = computeBackoffMs(
                sseConsecutiveErrors,
                SSE_RECONNECT_BASE_MS,
                SSE_RECONNECT_MAX_MS
            );
            eventSourceReconnectTimer = setTimeout(() => {
                eventSourceReconnectTimer = null;
                connectAsLeaderIfNeeded();
            }, delayMs);
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
                sseConsecutiveErrors++;
                eventSource = null;
                eventSourceConnecting = false;
                relinquishLeadershipIfOwned();
                startUserStatePolling();
                scheduleReconnect();
                return;
            }
            startConnectWatchdog();
            eventSource.onopen = () => {
                sseConsecutiveErrors = 0;
                pollConsecutiveErrors = 0;
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
                sseConsecutiveErrors++;
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

        // Best-effort: on tab close, close SSE and release leadership so other tabs can connect immediately.
        if (!userEventUnloadBound) {
            userEventUnloadBound = true;
            try {
                window.addEventListener('beforeunload', () => {
                    if (eventSource) {
                        try {
                            eventSource.close();
                        } catch (e) {}
                        eventSource = null;
                        eventSourceConnecting = false;
                    }
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
                const dropdown = document.getElementById(id);
                // Auto-clean stale registrations to avoid memory leaks when DOM is replaced.
                if (!dropdown || !dropdown.isConnected) {
                    dropdownRegistry.delete(id);
                    return;
                }
                if (!api || !api.isOpen || !api.isOpen()) return;
                if (dropdown && !dropdown.contains(event.target)) {
                    api.close();
                }
            });
        });

        document.addEventListener('keydown', event => {
            if (event.key !== 'Escape') return;
            dropdownRegistry.forEach((api, id) => {
                const dropdown = document.getElementById(id);
                if (!dropdown || !dropdown.isConnected) {
                    dropdownRegistry.delete(id);
                    return;
                }
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

        // Downgrade network errors to warn – they are expected when the server is
        // temporarily unreachable and would otherwise flood the console.
        const isNetworkError =
            msg.includes('Failed to fetch') ||
            msg.includes('NetworkError') ||
            msg.includes('ERR_EMPTY_RESPONSE') ||
            msg.includes('Load failed');
        if (isNetworkError) {
            console.warn(`[${ctx}] Network error (server may be temporarily unavailable)`);
        } else {
            console.error(`[${ctx}]`, error);
        }

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
            _mjxTypesetting++;
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
                        })
                        .then(() => {
                            _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
                        });
                    return;
                }
                if (MathJax.typesetPromise) {
                    MathJax.typesetPromise(nodes)
                        .catch(function (err) {
                            console.warn('MathJax typeset error:', err);
                        })
                        .then(function () {
                            _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
                        });
                    return;
                }
                if (MathJax.typeset) {
                    MathJax.typeset(nodes);
                }
                _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
            } catch (err) {
                _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
                console.warn('MathJax typeset error:', err);
            }
        }

        // If MathJax is fully loaded (has typeset methods), typeset immediately.
        if (typeof MathJax !== 'undefined' && (MathJax.typesetPromise || MathJax.typeset)) {
            _preloadMjxFonts(); // ensure font preload even if MathJax loaded externally
            _typesetNow();
            // Queue element for re-typeset if MJXTEX fonts aren't ready yet
            _queueMjxRetypeset(element);
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
            // Queue element for re-typeset if MJXTEX fonts aren't ready yet
            _queueMjxRetypeset(element);
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
    let mathJaxCandidateIndex = 0;

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

    function _isPlainObject(v) {
        return !!v && typeof v === 'object' && !Array.isArray(v);
    }

    function _mergeDeep(target, source) {
        if (!_isPlainObject(target) || !_isPlainObject(source)) return target;
        Object.keys(source).forEach(k => {
            const sv = source[k];
            const tv = target[k];
            if (_isPlainObject(sv) && _isPlainObject(tv)) {
                _mergeDeep(tv, sv);
            } else {
                target[k] = sv;
            }
        });
        return target;
    }

    function _cloneJson(obj) {
        return JSON.parse(JSON.stringify(obj));
    }

    /**
     * Default MathJax configuration.
     *
     * IMPORTANT:
     * - Do NOT enable inline '$...$' math. It conflicts with markdown-it's '$' parsing on several pages.
     * - Keep this config as the single source of truth; templates should not duplicate MathJax config blocks.
     */
    const defaultMathJaxConfig = {
        loader: {
            // Extensions are included in tex-chtml-full.js, but explicitly listing them makes config
            // consistent across pages and future-proof if the bundle changes.
            load: [
                '[tex]/boldsymbol',
                '[tex]/mathtools',
                '[tex]/physics',
                '[tex]/tagformat',
                '[tex]/textmacros',
            ],
        },
        options: {
            enableMenu: false,
            enableAssistiveMml: false,
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process',
        },
        tex: {
            inlineMath: [['\\(', '\\)']],
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
            // Prefer CDN fonts; can be overridden to local when CDN is unavailable.
            fontURL: 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/output/chtml/fonts/woff-v2',
        },
    };

    /**
     * Ensure window.MathJax is initialised using the default config, optionally overriding parts.
     * This must run before MathJax script loads to take effect.
     *
     * @param {Object} [overrides]
     * @returns {Object} The resulting config object.
     */
    function ensureMathJaxConfig(overrides) {
        if (typeof window === 'undefined') return defaultMathJaxConfig;
        const base = _cloneJson(defaultMathJaxConfig);
        const existing = _isPlainObject(window.MathJax) ? window.MathJax : null;
        if (existing) _mergeDeep(base, existing);
        if (_isPlainObject(overrides)) _mergeDeep(base, overrides);

        // Ensure MathJax fontURL respects runtime CDN settings.
        // Rationale: the CDN base / fallback flags may change at runtime
        // (custom `asset_npm_cdn_base`, probe-triggered local fallback), so we
        // choose the preferred font URL at call-time to stay consistent.
        try {
            if (!base.chtml) base.chtml = {};
            const curFontUrl = String((base.chtml && base.chtml.fontURL) || '');
            const defaultJsdelivrRe =
                /https?:\/\/cdn\.jsdelivr\.net\/npm\/mathjax@[^/]+\/es5\/output\/chtml\/fonts\/woff-v2/;

            // If we already know fonts must be local (e.g. CDN blocked), keep it local.
            // Use the global flag only to avoid temporal-dead-zone hazards with internal `let` vars.
            const forceLocal =
                typeof global !== 'undefined' &&
                (global.__arxivSanityMathJaxForceLocal === 1 ||
                    global.__arxivSanityMathJaxForceLocal === true);

            if (forceLocal) {
                base.chtml.fontURL = staticUrl('lib/es5/output/chtml/fonts/woff-v2');
            } else if (!curFontUrl || defaultJsdelivrRe.test(curFontUrl)) {
                // Prefer the configured npm CDN base when enabled; otherwise local.
                base.chtml.fontURL = isAssetCdnEnabled()
                    ? npmCdnUrl('mathjax@3.2.2/es5/output/chtml/fonts/woff-v2')
                    : staticUrl('lib/es5/output/chtml/fonts/woff-v2');
            }
        } catch (e) {}

        // Defensive: enforce no inline '$...$' even if a page accidentally re-adds it.
        try {
            const im =
                base && base.tex && Array.isArray(base.tex.inlineMath) ? base.tex.inlineMath : [];
            base.tex.inlineMath = im.filter(pair => {
                if (!Array.isArray(pair) || pair.length < 2) return true;
                return !(pair[0] === '$' && pair[1] === '$');
            });
        } catch (e) {}

        window.MathJax = base;
        return base;
    }

    function getMathJaxScriptCandidates() {
        const local = staticUrl('lib/es5/tex-chtml-full.js');
        const cdn = npmCdnUrl('mathjax@3.2.2/es5/tex-chtml-full.js');
        // Prefer CDN, but always keep local as fallback.
        return isAssetCdnEnabled() ? [cdn, local] : [local];
    }

    function getMathJaxFontCandidates() {
        const local = staticUrl('lib/es5/output/chtml/fonts/woff-v2');
        const cdn = npmCdnUrl('mathjax@3.2.2/es5/output/chtml/fonts/woff-v2');
        return isAssetCdnEnabled() ? [cdn, local] : [local];
    }

    function ensureMathJaxFontUrlPreferred(sourceUrl) {
        // If we are forcing local (e.g. CDN blocked), ensure config uses local fonts.
        try {
            const forceLocal =
                global &&
                (global.__arxivSanityMathJaxForceLocal === 1 ||
                    global.__arxivSanityMathJaxForceLocal === true);
            const preferLocal =
                forceLocal || (sourceUrl && String(sourceUrl).indexOf('/static/') >= 0);
            if (!preferLocal) return;

            const localFontUrl = staticUrl('lib/es5/output/chtml/fonts/woff-v2');
            if (typeof window !== 'undefined') {
                if (typeof window.MathJax === 'undefined') {
                    window.MathJax = defaultMathJaxConfig;
                }
                if (!window.MathJax.chtml) window.MathJax.chtml = {};
                window.MathJax.chtml.fontURL = localFontUrl;
            }
        } catch (e) {}
    }

    // ---- CDN MathJax font probe & runtime fallback ----
    // Proactively test whether CDN fonts are reachable.  If the probe fails
    // before MathJax initialises (common_utils loads synchronously, MathJax is
    // deferred), we switch fontURL to local so MathJax never requests CDN fonts.
    // A document.fonts listener acts as a runtime safety-net for edge cases
    // where the probe hasn't finished or fonts fail mid-render.

    let _cdnFontProbeResult = null; // null = pending, true = ok, false = failed
    let _cdnFontProbePromise = null;
    let _fontFallbackApplied = false;

    function _localMathJaxFontUrl() {
        return staticUrl('lib/es5/output/chtml/fonts/woff-v2');
    }

    function _applyLocalMathJaxFonts() {
        if (_fontFallbackApplied) return;
        _fontFallbackApplied = true;
        try {
            if (typeof global !== 'undefined') global.__arxivSanityMathJaxForceLocal = 1;
            const localUrl = _localMathJaxFontUrl();
            if (typeof window !== 'undefined' && window.MathJax) {
                if (!window.MathJax.chtml) window.MathJax.chtml = {};
                window.MathJax.chtml.fontURL = localUrl;
                // Also patch runtime config if MathJax is already initialised
                if (window.MathJax.config && window.MathJax.config.chtml) {
                    window.MathJax.config.chtml.fontURL = localUrl;
                }
            }
            console.warn('[CommonUtils] CDN MathJax fonts unreachable – using local fonts.');
        } catch (e) {}
    }

    /** Rewrite any CDN font URLs already injected into @font-face rules. */
    function _patchFontFaceRules() {
        try {
            const localUrl = _localMathJaxFontUrl();
            const sheets = document.styleSheets;
            // Patch any npm CDN base (not only jsDelivr). This matters when users configure
            // a custom `asset_npm_cdn_base`: MathJax may have already injected @font-face
            // rules pointing at that CDN, and we need to rewrite them to our local fonts.
            // We keep the pattern narrow to MathJax CHTML woff-v2 fonts.
            const cdnRe = /https?:\/\/[^"')]+\/mathjax@[^/]+\/es5\/output\/chtml\/fonts\/woff-v2/g;
            for (let i = 0; i < sheets.length; i++) {
                let rules;
                try {
                    rules = sheets[i].cssRules;
                } catch (_) {
                    continue;
                }
                if (!rules) continue;
                for (let j = 0; j < rules.length; j++) {
                    if (rules[j].type !== CSSRule.FONT_FACE_RULE) continue;
                    const src = rules[j].style.getPropertyValue('src');
                    if (src && cdnRe.test(src)) {
                        cdnRe.lastIndex = 0;
                        rules[j].style.setProperty('src', src.replace(cdnRe, localUrl));
                    }
                }
            }
        } catch (e) {}
    }

    function _startCdnFontProbe() {
        if (_cdnFontProbePromise) return _cdnFontProbePromise;
        if (!isAssetCdnEnabled() || typeof fetch === 'undefined') {
            _cdnFontProbeResult = true;
            _cdnFontProbePromise = Promise.resolve(true);
            return _cdnFontProbePromise;
        }
        const testUrl =
            npmCdnUrl('mathjax@3.2.2/es5/output/chtml/fonts/woff-v2').replace(/\/+$/, '') +
            '/MathJax_Main-Regular.woff';

        _cdnFontProbePromise = new Promise(function (resolve) {
            const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
            const tid = setTimeout(function () {
                try {
                    if (ctrl) ctrl.abort();
                } catch (_) {}
                resolve(false);
            }, 5000);
            const opts = { mode: 'cors' };
            if (ctrl) opts.signal = ctrl.signal;
            fetch(testUrl, opts)
                .then(function (r) {
                    if (!r.ok) throw new Error(r.status);
                    return r.arrayBuffer();
                })
                .then(function (buf) {
                    clearTimeout(tid);
                    resolve(!!(buf && buf.byteLength > 1000));
                })
                .catch(function () {
                    clearTimeout(tid);
                    resolve(false);
                });
        });
        _cdnFontProbePromise.then(function (ok) {
            _cdnFontProbeResult = ok;
            if (!ok) _applyLocalMathJaxFonts();
        });
        return _cdnFontProbePromise;
    }

    /** Runtime listener: catch font-load errors that slip past the probe. */
    function _setupFontErrorListener() {
        try {
            if (typeof document === 'undefined' || !document.fonts) return;
            document.fonts.addEventListener('loadingerror', function (evt) {
                if (_fontFallbackApplied) return;
                const faces = evt.fontfaces || [];
                let isMjx = false;
                for (let i = 0; i < faces.length; i++) {
                    if (String(faces[i].family).indexOf('MathJax') >= 0) {
                        isMjx = true;
                        break;
                    }
                }
                if (!isMjx) return;
                _applyLocalMathJaxFonts();
                _patchFontFaceRules();
                // Re-typeset so MathJax picks up the patched fonts
                setTimeout(function () {
                    try {
                        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                            _mjxTypesetting++;
                            MathJax.typesetPromise()
                                .catch(function () {})
                                .then(function () {
                                    _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
                                });
                        }
                    } catch (_) {}
                }, 200);
            });
        } catch (e) {}
    }

    // Kick off probe & listener immediately
    _startCdnFontProbe();
    _setupFontErrorListener();

    // ---- MathJax font preload & retypeset for non-summary pages ----
    // When triggerMathJax() typesets an element before MJXTEX fonts are ready,
    // the element is queued.  Once fonts load, queued elements are re-typeset
    // so formulas become visible (same MJXZERO issue as the summary page).

    const _CRITICAL_MJX_FONTS = [
        { family: 'MJXTEX', file: 'MathJax_Main-Regular.woff' },
        { family: 'MJXTEX-I', file: 'MathJax_Math-Italic.woff' },
        { family: 'MJXTEX-S1', file: 'MathJax_Size1-Regular.woff' },
        { family: 'MJXTEX-A', file: 'MathJax_AMS-Regular.woff' },
    ];

    let _mjxFontPreloaded = false;
    let _mjxRetypesetQueue = []; // elements to re-typeset when fonts ready
    let _mjxRetypesetCount = 0;
    const _MJX_MAX_RETYPESETS = 3;
    let _mjxFontListenerSetup = false;
    let _mjxFallbackTimers = [];
    let _mjxDebounceTimer = null;
    let _mjxTypesetting = 0; // reference count for concurrent typeset operations
    let _mjxDeferredFlushTimer = null; // single deferred flush to avoid timer storm

    /** Inject <link rel="preload"> hints for critical MathJax fonts. */
    function _preloadMjxFonts() {
        if (_mjxFontPreloaded) return;
        _mjxFontPreloaded = true;
        try {
            const baseUrl = (getMathJaxFontUrl() || '').replace(/\/+$/, '');
            if (!baseUrl) return;
            _CRITICAL_MJX_FONTS.forEach(function (entry) {
                try {
                    const url = baseUrl + '/' + entry.file;
                    if (document.querySelector('link[rel="preload"][href="' + url + '"]')) return;
                    const link = document.createElement('link');
                    link.rel = 'preload';
                    link.as = 'font';
                    link.type = 'font/woff';
                    link.href = url;
                    link.crossOrigin = 'anonymous';
                    document.head.appendChild(link);
                } catch (e) {}
            });
        } catch (e) {}
    }

    function getMathJaxFontUrl() {
        if (
            _fontFallbackApplied ||
            (typeof window !== 'undefined' && window.__arxivSanityMathJaxForceLocal)
        ) {
            return _localMathJaxFontUrl();
        }
        if (_cdnFontProbeResult === false) return _localMathJaxFontUrl();
        if (typeof window !== 'undefined' && window.MathJax && window.MathJax.chtml) {
            return window.MathJax.chtml.fontURL || _localMathJaxFontUrl();
        }
        return _localMathJaxFontUrl();
    }

    /** Check whether any critical MJXTEX font is available. */
    function _isMjxFontReady() {
        try {
            if (typeof document === 'undefined' || !document.fonts || !document.fonts.check) {
                return true; // can't check, assume ready
            }
            return (
                document.fonts.check('1em MJXTEX') ||
                document.fonts.check('1em MJXTEX-I') ||
                document.fonts.check('1em MJXTEX-S1') ||
                document.fonts.check('1em MJXTEX-A')
            );
        } catch (e) {
            return true;
        }
    }

    /** Re-typeset all queued elements and clear the queue. */
    function _flushMjxRetypesetQueue(_reason) {
        // Clear any pending deferred flush
        if (_mjxDeferredFlushTimer) {
            clearTimeout(_mjxDeferredFlushTimer);
            _mjxDeferredFlushTimer = null;
        }
        if (_mjxRetypesetCount >= _MJX_MAX_RETYPESETS) {
            _mjxRetypesetQueue = []; // prevent leak when cap reached
            _cancelMjxFallbacks();
            return;
        }
        // Defer if another typeset is in progress to avoid typesetClear conflicts
        if (_mjxTypesetting > 0) {
            if (!_mjxDeferredFlushTimer) {
                _mjxDeferredFlushTimer = setTimeout(function () {
                    _mjxDeferredFlushTimer = null;
                    _flushMjxRetypesetQueue(_reason);
                }, 200);
            }
            return;
        }
        const elements = _mjxRetypesetQueue.filter(function (el) {
            return el && el.isConnected !== false;
        });
        _mjxRetypesetQueue = [];
        // Clear fallback timers so they can be re-scheduled for future queues
        _cancelMjxFallbacks();
        if (elements.length === 0) return;
        try {
            if (typeof MathJax === 'undefined') return;
            if (!MathJax.typesetPromise && !MathJax.typeset) return; // no typeset method
            // Only consume quota after confirming MathJax can typeset
            _mjxRetypesetCount++;
            _mjxTypesetting++;
            if (MathJax.typesetClear) MathJax.typesetClear(elements);
            if (MathJax.typesetPromise) {
                MathJax.typesetPromise(elements)
                    .catch(function () {})
                    .then(function () {
                        _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
                    });
            } else if (MathJax.typeset) {
                MathJax.typeset(elements);
                _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
            }
        } catch (e) {
            _mjxTypesetting = Math.max(0, _mjxTypesetting - 1);
        }
    }

    function _cancelMjxFallbacks() {
        for (let i = 0; i < _mjxFallbackTimers.length; i++) {
            clearTimeout(_mjxFallbackTimers[i]);
        }
        _mjxFallbackTimers = [];
    }

    /**
     * Set up font-ready listeners (loadingdone + fonts.load).
     * Called once when the first element is queued for retypeset.
     */
    function _setupMjxFontRetypeset() {
        if (_mjxFontListenerSetup) return;
        _mjxFontListenerSetup = true;

        function onFontsLoaded(evt) {
            if (_mjxRetypesetCount >= _MJX_MAX_RETYPESETS) return;
            if (_mjxRetypesetQueue.length === 0) return;

            let faces;
            try {
                faces = evt && evt.fontfaces ? Array.from(evt.fontfaces) : [];
            } catch (e) {
                faces = [];
            }
            if (faces.length > 0) {
                let hasMjxTex = false;
                for (let i = 0; i < faces.length; i++) {
                    if (String(faces[i].family || '').indexOf('MJXTEX') >= 0) {
                        hasMjxTex = true;
                        break;
                    }
                }
                if (!hasMjxTex) return;
            }

            if (_mjxDebounceTimer) clearTimeout(_mjxDebounceTimer);
            _mjxDebounceTimer = setTimeout(function () {
                _mjxDebounceTimer = null;
                if (_isMjxFontReady()) {
                    _cancelMjxFallbacks();
                    _flushMjxRetypesetQueue('loadingdone');
                }
            }, 300);
        }

        // Trigger 1: loadingdone listener
        try {
            document.fonts.addEventListener('loadingdone', onFontsLoaded);
        } catch (e) {}

        // Trigger 2: document.fonts.load() deterministic trigger
        try {
            document.fonts
                .load('1em MJXTEX')
                .then(function () {
                    if (_mjxRetypesetQueue.length > 0 && _isMjxFontReady()) {
                        _cancelMjxFallbacks();
                        _flushMjxRetypesetQueue('fonts.load resolved');
                    }
                })
                .catch(function () {});
        } catch (e) {}
    }

    /**
     * Ensure fallback timers are scheduled.  Called every time an element is
     * queued so that late-arriving elements still get a safety-net flush even
     * if the one-shot listeners/fonts.load have already fired.
     */
    function _ensureMjxFallbacks() {
        // Set up the one-time listeners (loadingdone + fonts.load) if not yet done
        _setupMjxFontRetypeset();

        if (_mjxRetypesetCount >= _MJX_MAX_RETYPESETS) return;

        // Cancel any existing timers and reschedule.  This ensures late-arriving
        // elements get a fresh fallback window and avoids stale timer IDs
        // lingering in the array (which would block future scheduling).
        _cancelMjxFallbacks();

        const DELAYS = [1500, 4000, 9000, 18000];
        DELAYS.forEach(function (delay, idx) {
            const tid = setTimeout(function () {
                if (_mjxRetypesetCount >= _MJX_MAX_RETYPESETS) return;
                if (_mjxRetypesetQueue.length === 0) return;
                const isLast = idx === DELAYS.length - 1;
                if (_isMjxFontReady() || isLast) {
                    _flushMjxRetypesetQueue('fallback ' + delay + 'ms');
                }
            }, delay);
            _mjxFallbackTimers.push(tid);
        });
    }

    /**
     * Queue an element for re-typeset when MJXTEX fonts become ready.
     * Called by triggerMathJax after initial typeset if fonts aren't loaded yet.
     */
    function _queueMjxRetypeset(element) {
        if (_mjxRetypesetCount >= _MJX_MAX_RETYPESETS) return;
        if (_isMjxFontReady()) return; // fonts already ready, no need
        // Use document.body as fallback for full-document typeset calls
        const target = element || document.body;
        if (!target) return;
        // If queueing document.body, collapse queue to just body (avoids
        // redundant typesetClear on overlapping parent+child nodes)
        if (target === document.body) {
            _mjxRetypesetQueue = [document.body];
        } else if (_mjxRetypesetQueue.indexOf(document.body) >= 0) {
            // body already covers everything, skip child
        } else if (_mjxRetypesetQueue.indexOf(target) < 0) {
            _mjxRetypesetQueue.push(target);
        }
        _ensureMjxFallbacks();
    }

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

        // Preload MathJax fonts in parallel with script loading
        _preloadMjxFonts();

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
                    clearInterval(mathJaxWatchdogTimer);
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
                clearInterval(mathJaxWatchdogTimer);
                mathJaxWatchdogTimer = null;
            }
        } catch (e) {}

        // Set config before loading script
        ensureMathJaxConfig();

        // Determine candidate URLs if not explicitly provided.
        const candidates = getMathJaxScriptCandidates();
        if (scriptUrl) {
            // If caller provides explicit URL, treat it as the active choice.
            mathJaxCandidateIndex = 0;
        }
        const chosenUrl =
            scriptUrl || candidates[Math.min(mathJaxCandidateIndex, candidates.length - 1)];
        ensureMathJaxFontUrlPreferred(chosenUrl);

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
                    clearInterval(mathJaxWatchdogTimer);
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
                    clearInterval(mathJaxWatchdogTimer);
                    mathJaxWatchdogTimer = null;
                }
            } catch (e) {}
            try {
                if (script && script.dataset) script.dataset.mjxFailed = '1';
            } catch (e) {}
            // Only remove scripts we created ourselves.  External <script defer> tags
            // (e.g. in summary.html) may still be loading; removing them corrupts
            // MathJax state and prevents recovery even if the script eventually loads.
            try {
                if (script && script.parentNode && !isExternalScript) {
                    script.parentNode.removeChild(script);
                }
            } catch (e) {}
            console.warn('Failed to load MathJax (attempt ' + mathJaxLoadFailures + ')');

            // Switch to the next candidate URL (e.g. CDN -> local) for subsequent retries.
            try {
                if (!scriptUrl && candidates && candidates.length > 1) {
                    mathJaxCandidateIndex = Math.min(
                        candidates.length - 1,
                        Math.max(0, mathJaxCandidateIndex + 1)
                    );
                    if (mathJaxCandidateIndex > 0) {
                        try {
                            if (typeof global !== 'undefined')
                                global.__arxivSanityMathJaxForceLocal = 1;
                        } catch (e) {}
                    }
                }
            } catch (e) {}

            if (mathJaxLoadFailures >= 3) {
                // Give up auto-retry; clear pending callbacks to avoid hanging forever.
                // But schedule one final delayed check – the external script may still
                // finish loading after our watchdog gave up.
                if (isExternalScript) {
                    setTimeout(() => {
                        if (isMathJaxAvailable()) {
                            mathJaxLoaded = true;
                            mathJaxLoading = false;
                            mathJaxLoadFailures = 0;
                            runMathJaxCallbacks();
                        }
                    }, 5000);
                } else {
                    mathJaxCallbacks.length = 0;
                }
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
            script.src = chosenUrl;
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
        // Use a polling watchdog to avoid leaving the loader stuck in a "loading" state forever.
        // For external scripts (e.g. <script defer> in summary.html), the load event may have
        // already fired before we attached our listener, so we poll periodically to detect
        // MathJax becoming available.  The old 2-second single-shot watchdog was far too short
        // for the 1.3 MB tex-chtml-full.js on slower connections and caused "Failed to load
        // MathJax" followed by permanent formula rendering loss after 3 failures.
        try {
            // External scripts (e.g. <script defer> in templates/summary.html) may take a long
            // time on slow networks; keep a longer watchdog to avoid false failures and
            // unnecessary retry logic.
            const watchdogMs = isExternalScript ? 60000 : 30000;
            const pollIntervalMs = 500;
            let elapsed = 0;

            // Immediately check – the script may already be loaded (browser cache hit).
            if (isMathJaxAvailable()) {
                finishOk();
            } else {
                const pollTimer = setInterval(() => {
                    if (finished) {
                        clearInterval(pollTimer);
                        return;
                    }
                    if (attempt !== mathJaxAttempt) {
                        clearInterval(pollTimer);
                        return;
                    }
                    elapsed += pollIntervalMs;
                    if (isMathJaxAvailable()) {
                        clearInterval(pollTimer);
                        finishOk();
                        return;
                    }
                    if (elapsed >= watchdogMs) {
                        clearInterval(pollTimer);
                        finishFail();
                    }
                }, pollIntervalMs);

                // Store the interval so it can be cleared on success/failure from event handlers.
                mathJaxWatchdogTimer = pollTimer;
            }
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
    let summaryStatusPollOffset = 0;
    const SUMMARY_STATUS_POLL_BATCH = 50;

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
        // Avoid sending huge batches that can block the server and cause timeouts.
        // Round-robin through pending pids so large sets eventually get polled.
        const allPids = Array.from(SUMMARY_PENDING);
        let pids = allPids;
        if (allPids.length > SUMMARY_STATUS_POLL_BATCH) {
            const start = summaryStatusPollOffset % allPids.length;
            const end = start + SUMMARY_STATUS_POLL_BATCH;
            if (end <= allPids.length) {
                pids = allPids.slice(start, end);
            } else {
                pids = allPids.slice(start).concat(allPids.slice(0, end - allPids.length));
            }
            summaryStatusPollOffset = (start + SUMMARY_STATUS_POLL_BATCH) % allPids.length;
        }
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
        ensureMathJaxConfig,
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
        // CDN font probe (for summary page resource gate)
        waitForCdnFontProbe: _startCdnFontProbe,
        getMathJaxFontUrl: getMathJaxFontUrl,
    };
})(typeof window !== 'undefined' ? window : this);
