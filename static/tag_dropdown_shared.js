// Shared Tag dropdown (3-state) for index/summary/readinglist
// Exposes: window.ArxivSanityTagDropdown
// - mount(elOrId, { pid, selectedTags, negativeTags, availableTags, onStateChange }) => api
// - MultiSelectDropdown React component (internal)

'use strict';

(function () {
    if (typeof window === 'undefined') return;

    // Use shared utilities from common_utils.js
    var CommonUtils = window.ArxivSanityCommon;
    if (!CommonUtils) {
        console.error('[TagDropdown] common_utils.js not loaded – tag dropdown unavailable.');
        window.ArxivSanityTagDropdown = {
            mount: function () {
                return {};
            },
            unmount: function () {},
        };
        return;
    }
    var csrfFetch = CommonUtils.csrfFetch;

    // Dropdown close registry from common_utils
    var registerDropdown = CommonUtils.registerDropdown;
    var unregisterDropdown = CommonUtils.unregisterDropdown;

    // React component: shared implementation
    const MultiSelectDropdown = props => {
        const {
            selectedTags,
            negativeTags,
            availableTags,
            isOpen,
            onToggle,
            onTagCycle,
            onClearTag,
            newTagValue,
            onNewTagChange,
            onAddNewTag,
            dropdownId,
            searchValue,
            onSearchChange,
            filteredTags = [],
            focusedOptionIndex = -1,
            onTriggerKeyDown,
            onMenuKeyDown,
            showNewTagInput = true,
            pending = false,
            pendingTag = '',
        } = props;

        const getTagState = tag => {
            if ((selectedTags || []).includes(tag)) return 1;
            if ((negativeTags || []).includes(tag)) return -1;
            return 0;
        };

        const selectedTagElements = [
            ...(selectedTags || []).map(tag => ({ tag, state: 1 })),
            ...(negativeTags || []).map(tag => ({ tag, state: -1 })),
        ].map((item, ix) => {
            const isTagPending =
                Boolean(pending) && String(pendingTag || '') === String(item.tag || '');
            return React.createElement(
                'div',
                {
                    key: `${item.tag}-${item.state}-${ix}`,
                    class: `multi-select-selected-tag ${item.state === -1 ? 'tag-negative' : 'tag-positive'}${
                        isTagPending ? ' pending' : ''
                    }`,
                    title: pending
                        ? 'Updating...'
                        : 'Click to cycle: Unlabeled → Positive → Negative → Unlabeled',
                },
                React.createElement(
                    'span',
                    { class: 'tag-state-icon' },
                    isTagPending ? '⏳' : item.state === 1 ? '+' : '−'
                ),
                React.createElement('span', null, item.tag),
                React.createElement(
                    'span',
                    {
                        class: 'remove-tag',
                        onClick: e => {
                            e.stopPropagation();
                            if (pending) return;
                            onClearTag(item.tag);
                        },
                    },
                    '×'
                )
            );
        });

        const hasAny = (selectedTags || []).length > 0 || (negativeTags || []).length > 0;
        const triggerContent = hasAny
            ? React.createElement(
                  'div',
                  { class: 'multi-select-selected-tags' },
                  selectedTagElements
              )
            : React.createElement('div', { class: 'multi-select-placeholder' }, 'Select tags...');

        const focusedOptionId =
            focusedOptionIndex >= 0 ? `${dropdownId}-option-${focusedOptionIndex}` : null;

        const optionElements = filteredTags.map((tag, ix) => {
            const state = getTagState(tag);
            const isTagPending = Boolean(pending) && String(pendingTag || '') === String(tag || '');
            const stateClass =
                state === 1
                    ? 'tag-state-positive'
                    : state === -1
                      ? 'tag-state-negative'
                      : 'tag-state-neutral';
            const isFocused = focusedOptionIndex === ix;
            return React.createElement(
                'div',
                {
                    key: ix,
                    id: `${dropdownId}-option-${ix}`,
                    role: 'option',
                    tabIndex: isFocused ? 0 : -1,
                    'aria-selected': state === 0 ? 'false' : 'true',
                    class: `multi-select-option ${stateClass}${isFocused ? ' is-focused' : ''}${
                        isTagPending ? ' pending' : ''
                    }`,
                    onClick: pending ? null : () => onTagCycle(tag),
                    'aria-disabled': pending ? 'true' : 'false',
                    title: 'Click to cycle: Unlabeled → Positive → Negative → Unlabeled',
                },
                React.createElement(
                    'span',
                    { class: `tag-state-badge ${stateClass}` },
                    isTagPending ? '⏳' : state === 1 ? '+' : state === -1 ? '−' : ''
                ),
                React.createElement('span', { class: 'multi-select-option-text' }, tag)
            );
        });

        const arrowText = pending ? '⏳' : isOpen ? '▲' : '▼';
        const liveMessage =
            pending && pendingTag
                ? `Updating tag ${String(pendingTag)}`
                : pending
                  ? 'Updating tags'
                  : '';

        return React.createElement(
            'div',
            {
                class: `multi-select-dropdown ${isOpen ? 'open' : ''}${pending ? ' pending' : ''}`,
                id: dropdownId,
            },
            React.createElement(
                'div',
                {
                    id: `${dropdownId}-live`,
                    role: 'status',
                    'aria-live': 'polite',
                    'aria-atomic': 'true',
                    style: {
                        position: 'absolute',
                        width: '1px',
                        height: '1px',
                        padding: 0,
                        margin: '-1px',
                        overflow: 'hidden',
                        clip: 'rect(0, 0, 0, 0)',
                        whiteSpace: 'nowrap',
                        border: 0,
                    },
                },
                liveMessage
            ),
            isOpen
                ? React.createElement('div', {
                      class: 'multi-select-backdrop',
                      onClick: pending ? null : onToggle,
                      'aria-hidden': 'true',
                  })
                : null,
            React.createElement(
                'div',
                {
                    class: `multi-select-trigger ${isOpen ? 'active' : ''}${pending ? ' pending' : ''}`,
                    role: 'button',
                    tabIndex: 0,
                    'aria-haspopup': 'listbox',
                    'aria-expanded': isOpen ? 'true' : 'false',
                    'aria-controls': `${dropdownId}-menu`,
                    onClick: pending ? null : onToggle,
                    onKeyDown: pending ? null : onTriggerKeyDown,
                    title: 'Click to cycle: Unlabeled → Positive → Negative → Unlabeled',
                },
                React.createElement('div', { class: 'multi-select-content' }, triggerContent),
                React.createElement('span', { class: 'multi-select-arrow' }, arrowText)
            ),
            isOpen
                ? React.createElement(
                      'div',
                      {
                          class: 'multi-select-dropdown-menu',
                          id: `${dropdownId}-menu`,
                          role: 'listbox',
                          'aria-label': 'Tag suggestions',
                          'aria-activedescendant': focusedOptionId,
                          'aria-multiselectable': 'true',
                          tabIndex: -1,
                          onKeyDown: onMenuKeyDown,
                      },
                      React.createElement(
                          'div',
                          { class: 'multi-select-sheet-header' },
                          React.createElement('div', {
                              class: 'multi-select-sheet-handle',
                              'aria-hidden': 'true',
                          }),
                          React.createElement(
                              'button',
                              {
                                  type: 'button',
                                  class: 'multi-select-sheet-close',
                                  onClick: e => {
                                      e.stopPropagation();
                                      if (pending) return;
                                      onToggle();
                                  },
                                  disabled: pending,
                              },
                              'Close'
                          )
                      ),
                      React.createElement(
                          'div',
                          { class: 'multi-select-search' },
                          React.createElement('input', {
                              type: 'text',
                              placeholder: 'Search tags...',
                              value: searchValue,
                              onChange: onSearchChange,
                              onClick: e => e.stopPropagation(),
                              disabled: pending,
                              'aria-label': 'Search tags',
                              id: `${dropdownId}-search`,
                              onKeyDown: e => {
                                  if (
                                      onMenuKeyDown &&
                                      e &&
                                      (e.key === 'ArrowDown' ||
                                          e.key === 'ArrowUp' ||
                                          e.key === 'Home' ||
                                          e.key === 'End' ||
                                          e.key === 'Escape')
                                  ) {
                                      onMenuKeyDown(e);
                                  }
                              },
                          })
                      ),
                      React.createElement('div', { class: 'multi-select-options' }, optionElements),
                      showNewTagInput
                          ? React.createElement(
                                'div',
                                { class: 'multi-select-new-tag' },
                                React.createElement('input', {
                                    type: 'text',
                                    placeholder: 'Enter new tag...',
                                    value: newTagValue,
                                    onChange: onNewTagChange,
                                    onKeyDown: e => {
                                        if (e && e.key === 'Enter') {
                                            e.preventDefault();
                                            onAddNewTag();
                                            return;
                                        }
                                        if (
                                            onMenuKeyDown &&
                                            e &&
                                            (e.key === 'ArrowDown' ||
                                                e.key === 'ArrowUp' ||
                                                e.key === 'Home' ||
                                                e.key === 'End' ||
                                                e.key === 'Escape')
                                        ) {
                                            onMenuKeyDown(e);
                                        }
                                    },
                                    onClick: e => e.stopPropagation(),
                                    disabled: pending,
                                    'aria-label': 'New tag name',
                                }),
                                React.createElement(
                                    'button',
                                    {
                                        onClick: onAddNewTag,
                                        disabled: pending || !String(newTagValue || '').trim(),
                                    },
                                    'Add'
                                )
                            )
                          : null
                  )
                : null
        );
    };

    function normalizeTagList(list) {
        const out = (Array.isArray(list) ? list : [])
            .map(t => String(t || '').trim())
            .filter(Boolean);
        // de-dup
        const seen = new Set();
        const dedup = [];
        out.forEach(t => {
            if (seen.has(t)) return;
            seen.add(t);
            dedup.push(t);
        });
        dedup.sort((a, b) => a.localeCompare(b));
        return dedup;
    }

    function mount(elOrId, options) {
        const opts = options || {};
        const el = typeof elOrId === 'string' ? document.getElementById(elOrId) : elOrId;
        if (!el) throw new Error('Tag dropdown mount target not found');
        if (typeof React === 'undefined' || typeof ReactDOM === 'undefined') {
            throw new Error('React/ReactDOM not loaded');
        }

        const pid = String(opts.pid || '').trim();
        const onStateChange = typeof opts.onStateChange === 'function' ? opts.onStateChange : null;
        const dropdownId = opts.dropdownId || 'tag-dropdown-' + Math.random().toString(36).slice(2);

        const state = {
            selectedTags: Array.isArray(opts.selectedTags) ? opts.selectedTags.slice() : [],
            negativeTags: Array.isArray(opts.negativeTags) ? opts.negativeTags.slice() : [],
            availableTags: normalizeTagList(opts.availableTags || []),
            open: Boolean(opts.open),
            searchValue: String(opts.searchValue || ''),
            newTagValue: String(opts.newTagValue || ''),
            pending: false,
            pendingTag: '',
            focusedOptionIndex: -1,
        };

        function emit() {
            if (!onStateChange) return;
            onStateChange({
                selectedTags: state.selectedTags.slice(),
                negativeTags: state.negativeTags.slice(),
                availableTags: state.availableTags.slice(),
                open: !!state.open,
                searchValue: String(state.searchValue || ''),
                newTagValue: String(state.newTagValue || ''),
            });
        }

        function ensureAvailable(tagName) {
            const t = String(tagName || '').trim();
            if (!t) return;
            if (!state.availableTags.includes(t)) {
                state.availableTags = normalizeTagList(state.availableTags.concat([t]));
            }
        }

        function applyLocal(tagName, label) {
            const t = String(tagName || '').trim();
            if (!t) return;
            const pos = new Set(state.selectedTags);
            const neg = new Set(state.negativeTags);
            if (label === 1) {
                pos.add(t);
                neg.delete(t);
            } else if (label === -1) {
                pos.delete(t);
                neg.add(t);
            } else {
                pos.delete(t);
                neg.delete(t);
            }
            state.selectedTags = Array.from(pos);
            state.negativeTags = Array.from(neg);
            emit();
        }

        async function applyRemote(tagName, label) {
            if (!pid) throw new Error('pid is required for tag dropdown');
            const t = String(tagName || '').trim();
            if (!t) return;
            if (state.pending) return;
            state.pending = true;
            state.pendingTag = t;
            render();
            try {
                const resp = await csrfFetch('/api/tag_feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pid: pid, tag: t, label: label }),
                });
                const data = await resp.json().catch(() => null);
                if (!data || !data.success) {
                    throw new Error(data && data.error ? data.error : 'Failed to update tag');
                }
                ensureAvailable(t);
                applyLocal(t, label);
            } finally {
                state.pending = false;
                state.pendingTag = '';
                render();
            }
        }

        const requestFrame =
            typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function'
                ? window.requestAnimationFrame
                : callback => callback();

        function focusTrigger() {
            const dropdown = document.getElementById(dropdownId);
            if (!dropdown) return;
            const trigger = dropdown.querySelector('.multi-select-trigger');
            if (!trigger || typeof trigger.focus !== 'function') return;
            requestFrame(() => trigger.focus());
        }

        function clampOptionIndex(index, length) {
            if (!length) return -1;
            if (index < 0) return 0;
            if (index >= length) return length - 1;
            return index;
        }

        function syncFocusedOption(filteredTags) {
            const tags = Array.isArray(filteredTags) ? filteredTags : [];
            if (!state.open || !tags.length) {
                state.focusedOptionIndex = -1;
                return;
            }
            const dropdown = document.getElementById(dropdownId);
            if (!dropdown) return;
            const options = dropdown.querySelectorAll('.multi-select-option');
            if (!options || !options.length) return;
            const nextIndex = clampOptionIndex(state.focusedOptionIndex, tags.length);
            state.focusedOptionIndex = nextIndex;
            if (nextIndex < 0) return;
            const target = options[nextIndex];
            if (target && typeof target.focus === 'function') {
                requestFrame(() => target.focus());
            }
        }

        function setFocusIndex(index, filteredTags) {
            const tags = Array.isArray(filteredTags) ? filteredTags : [];
            const next = clampOptionIndex(index, tags.length);
            state.focusedOptionIndex = next;
            render();
        }

        function moveFocus(delta, filteredTags) {
            const tags = Array.isArray(filteredTags) ? filteredTags : [];
            const len = tags.length;
            if (!len) {
                state.focusedOptionIndex = -1;
                render();
                return;
            }
            let next = state.focusedOptionIndex;
            if (next < 0) {
                next = delta > 0 ? 0 : len - 1;
            } else {
                next = (next + delta + len) % len;
            }
            setFocusIndex(next, filteredTags);
        }

        function selectFocusedOption(filteredTags) {
            if (state.pending) return;
            const tags = Array.isArray(filteredTags) ? filteredTags : [];
            const idx = state.focusedOptionIndex;
            if (idx < 0 || idx >= tags.length) return;
            const tag = tags[idx];
            const isPos = state.selectedTags.includes(tag);
            const isNeg = state.negativeTags.includes(tag);
            const next = isPos ? -1 : isNeg ? 0 : 1;
            applyRemote(tag, next).catch(err => {
                console.error('Failed to update tag feedback:', err);
                const msg =
                    'Failed to update tag feedback: ' +
                    (err && err.message ? err.message : String(err));
                const c = (window && window.ArxivSanityCommon) || {};
                if (typeof c.showToast === 'function') c.showToast(msg, { type: 'error' });
            });
        }

        function setDropdownOpen(nextOpen, focusIndex) {
            if (state.pending) return;
            const target = Boolean(nextOpen);
            if (state.open !== target) {
                state.open = target;
                if (!state.open) {
                    state.searchValue = '';
                    state.newTagValue = '';
                    state.focusedOptionIndex = -1;
                }
            } else if (!state.open) {
                return;
            }
            if (state.open && Number.isFinite(focusIndex) && focusIndex >= 0) {
                state.focusedOptionIndex = focusIndex;
            }
            // Toggle overflow class on parent card so the dropdown menu isn't clipped.
            const dropdown = document.getElementById(dropdownId);
            if (dropdown) {
                const paperCard =
                    dropdown.closest('.rel_paper') || dropdown.closest('.rl-paper-card');
                if (paperCard) {
                    if (state.open) {
                        paperCard.classList.add('dropdown-open');
                    } else {
                        paperCard.classList.remove('dropdown-open');
                    }
                }
            }
            render();
            emit();
            if (!state.open) {
                focusTrigger();
            }
        }

        function toggleDropdown() {
            setDropdownOpen(!state.open);
        }

        function handleTriggerKeyDown(event, filteredTags) {
            if (state.pending) return;
            if (!event || !event.key) return;
            const key = event.key;
            if (key === 'Enter' || key === ' ') {
                event.preventDefault();
                toggleDropdown();
                return;
            }
            if (key === 'ArrowDown' || key === 'ArrowUp') {
                event.preventDefault();
                const tags = Array.isArray(filteredTags) ? filteredTags : [];
                if (!tags.length) {
                    setDropdownOpen(true);
                    return;
                }
                const focusIndex = key === 'ArrowDown' ? 0 : tags.length - 1;
                setDropdownOpen(true, focusIndex);
                return;
            }
            if (key === 'Escape' && state.open) {
                event.preventDefault();
                setDropdownOpen(false);
            }
        }

        function handleOptionsKeyDown(event, filteredTags) {
            if (state.pending) return;
            if (!state.open) return;
            if (!event || !event.key) return;
            const key = event.key;
            const tags = Array.isArray(filteredTags) ? filteredTags : [];
            if (key === 'ArrowDown') {
                event.preventDefault();
                moveFocus(1, tags);
                return;
            }
            if (key === 'ArrowUp') {
                event.preventDefault();
                moveFocus(-1, tags);
                return;
            }
            if (key === 'Home') {
                event.preventDefault();
                setFocusIndex(0, tags);
                return;
            }
            if (key === 'End') {
                event.preventDefault();
                setFocusIndex(tags.length - 1, tags);
                return;
            }
            if (key === 'Enter' || key === ' ') {
                event.preventDefault();
                selectFocusedOption(tags);
                return;
            }
            if (key === 'Escape') {
                event.preventDefault();
                setDropdownOpen(false);
            }
        }

        function render() {
            const normalizedSearch = String(state.searchValue || '').toLowerCase();
            const filteredTags = (state.availableTags || []).filter(tag =>
                String(tag || '')
                    .toLowerCase()
                    .includes(normalizedSearch)
            );

            ReactDOM.render(
                React.createElement(MultiSelectDropdown, {
                    selectedTags: state.selectedTags,
                    negativeTags: state.negativeTags,
                    availableTags: state.availableTags,
                    isOpen: state.open,
                    pending: state.pending,
                    pendingTag: state.pendingTag,
                    onToggle: toggleDropdown,
                    onTagCycle: tag => {
                        const isPos = state.selectedTags.includes(tag);
                        const isNeg = state.negativeTags.includes(tag);
                        const next = isPos ? -1 : isNeg ? 0 : 1;
                        applyRemote(tag, next).catch(err => {
                            console.error('Failed to update tag feedback:', err);
                            const msg =
                                'Failed to update tag feedback: ' +
                                (err && err.message ? err.message : String(err));
                            const c = (window && window.ArxivSanityCommon) || {};
                            if (typeof c.showToast === 'function')
                                c.showToast(msg, { type: 'error' });
                        });
                    },
                    onClearTag: tag => {
                        applyRemote(tag, 0).catch(err => {
                            console.error('Failed to clear tag:', err);
                            const msg =
                                'Failed to clear tag: ' +
                                (err && err.message ? err.message : String(err));
                            const c = (window && window.ArxivSanityCommon) || {};
                            if (typeof c.showToast === 'function')
                                c.showToast(msg, { type: 'error' });
                        });
                    },
                    newTagValue: state.newTagValue,
                    onNewTagChange: e => {
                        if (state.pending) return;
                        state.newTagValue = e && e.target ? e.target.value : '';
                        state.focusedOptionIndex = -1;
                        render();
                        emit();
                    },
                    onAddNewTag: () => {
                        if (state.pending) return;
                        const t = String(state.newTagValue || '').trim();
                        if (!t) return;
                        applyRemote(t, 1)
                            .then(() => {
                                state.newTagValue = '';
                                render();
                                emit();
                            })
                            .catch(err => {
                                console.error('Failed to add new tag:', err);
                                const msg =
                                    'Failed to add new tag: ' +
                                    (err && err.message ? err.message : String(err));
                                const c = (window && window.ArxivSanityCommon) || {};
                                if (typeof c.showToast === 'function')
                                    c.showToast(msg, { type: 'error' });
                            });
                    },
                    dropdownId: dropdownId,
                    searchValue: state.searchValue,
                    onSearchChange: e => {
                        if (state.pending) return;
                        state.searchValue = e && e.target ? e.target.value : '';
                        state.focusedOptionIndex = -1;
                        render();
                        emit();
                    },
                    filteredTags: filteredTags,
                    focusedOptionIndex: state.focusedOptionIndex,
                    onTriggerKeyDown: e => handleTriggerKeyDown(e, filteredTags),
                    onMenuKeyDown: e => handleOptionsKeyDown(e, filteredTags),
                }),
                el
            );

            syncFocusedOption(filteredTags);
        }

        // register close handlers
        registerDropdown(dropdownId, {
            isOpen: () => !!state.open,
            close: () => {
                setDropdownOpen(false);
            },
        });

        // initial render
        render();
        emit();

        const api = {
            getUiState: () => ({
                open: !!state.open,
                searchValue: String(state.searchValue || ''),
                newTagValue: String(state.newTagValue || ''),
            }),
            updateAvailableTags: nextTags => {
                state.availableTags = normalizeTagList(nextTags || []);
                render();
                emit();
            },
            applyTagFeedback: (tagName, label) => {
                ensureAvailable(tagName);
                applyLocal(String(tagName || ''), Number(label));
                render();
            },
            applyTagRename: (fromTag, toTag) => {
                const from = String(fromTag || '').trim();
                const to = String(toTag || '').trim();
                if (!from || !to) return;
                state.selectedTags = state.selectedTags.map(t => (t === from ? to : t));
                state.negativeTags = state.negativeTags.map(t => (t === from ? to : t));
                state.availableTags = normalizeTagList(
                    (state.availableTags || []).map(t => (t === from ? to : t))
                );
                ensureAvailable(to);
                render();
                emit();
            },
            applyTagDelete: tagName => {
                const t = String(tagName || '').trim();
                if (!t) return;
                state.selectedTags = state.selectedTags.filter(x => x !== t);
                state.negativeTags = state.negativeTags.filter(x => x !== t);
                state.availableTags = normalizeTagList(
                    (state.availableTags || []).filter(x => x !== t)
                );
                render();
                emit();
            },
            unmount: () => {
                try {
                    // Clean up dropdown-open class on parent card
                    const dropdown = document.getElementById(dropdownId);
                    if (dropdown) {
                        const paperCard =
                            dropdown.closest('.rel_paper') || dropdown.closest('.rl-paper-card');
                        if (paperCard) paperCard.classList.remove('dropdown-open');
                    }
                    ReactDOM.unmountComponentAtNode(el);
                } catch (e) {}
                unregisterDropdown(dropdownId);
            },
        };

        return api;
    }

    window.ArxivSanityTagDropdown = {
        mount,
        MultiSelectDropdown,
        _registerDropdown: registerDropdown,
        _unregisterDropdown: unregisterDropdown,
    };
})();
