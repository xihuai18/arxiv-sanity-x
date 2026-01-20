// Shared Tag dropdown (3-state) for index/summary/readinglist
// Exposes: window.ArxivSanityTagDropdown
// - mount(elOrId, { pid, selectedTags, negativeTags, availableTags, onStateChange }) => api
// - MultiSelectDropdown React component (internal)

'use strict';

(function () {
	if (typeof window === 'undefined') return;

	// Use shared utilities from common_utils.js
	var CommonUtils = window.ArxivSanityCommon;
	var csrfFetch = CommonUtils.csrfFetch;

	// Dropdown close registry from common_utils
	var registerDropdown = CommonUtils.registerDropdown;
	var unregisterDropdown = CommonUtils.unregisterDropdown;

	// React component: shared implementation
	const MultiSelectDropdown = (props) => {
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
			showNewTagInput = true,
		} = props;

		const getTagState = (tag) => {
			if ((selectedTags || []).includes(tag)) return 1;
			if ((negativeTags || []).includes(tag)) return -1;
			return 0;
		};

		const selectedTagElements = [
			...(selectedTags || []).map((tag) => ({ tag, state: 1 })),
			...(negativeTags || []).map((tag) => ({ tag, state: -1 })),
		].map((item, ix) => (
			React.createElement(
				'div',
				{
					key: `${item.tag}-${item.state}-${ix}`,
					class: `multi-select-selected-tag ${item.state === -1 ? 'tag-negative' : 'tag-positive'}`,
					title: 'Click to cycle: Unlabeled → Positive → Negative → Unlabeled',
				},
				React.createElement('span', { class: 'tag-state-icon' }, item.state === 1 ? '+' : '−'),
				React.createElement('span', null, item.tag),
				React.createElement(
					'span',
					{
						class: 'remove-tag',
						onClick: (e) => {
							e.stopPropagation();
							onClearTag(item.tag);
						},
					},
					'×'
				)
			)
		));

		const hasAny = (selectedTags || []).length > 0 || (negativeTags || []).length > 0;
		const triggerContent = hasAny
			? React.createElement('div', { class: 'multi-select-selected-tags' }, selectedTagElements)
			: React.createElement('div', { class: 'multi-select-placeholder' }, 'Select tags...');

		const filteredTags = (availableTags || []).filter((tag) =>
			String(tag || '').toLowerCase().includes(String(searchValue || '').toLowerCase())
		);

		const optionElements = filteredTags.map((tag, ix) => {
			const state = getTagState(tag);
			const stateClass = state === 1 ? 'tag-state-positive' : state === -1 ? 'tag-state-negative' : 'tag-state-neutral';
			return React.createElement(
				'div',
				{
					key: ix,
					class: `multi-select-option ${stateClass}`,
					onClick: () => onTagCycle(tag),
					title: 'Click to cycle: Unlabeled → Positive → Negative → Unlabeled',
				},
				React.createElement('span', { class: `tag-state-badge ${stateClass}` }, state === 1 ? '+' : state === -1 ? '−' : ''),
				React.createElement('span', { class: 'multi-select-option-text' }, tag)
			);
		});

		return React.createElement(
			'div',
			{ class: `multi-select-dropdown ${isOpen ? 'open' : ''}`, id: dropdownId },
			React.createElement(
				'div',
				{ class: `multi-select-trigger ${isOpen ? 'active' : ''}`, onClick: onToggle, title: 'Click to cycle: Unlabeled → Positive → Negative → Unlabeled' },
				React.createElement('div', { class: 'multi-select-content' }, triggerContent),
				React.createElement('span', { class: 'multi-select-arrow' }, isOpen ? '▲' : '▼')
			),
			isOpen
				? React.createElement(
					'div',
					{ class: 'multi-select-dropdown-menu' },
					React.createElement(
						'div',
						{ class: 'multi-select-search' },
						React.createElement('input', {
							type: 'text',
							placeholder: 'Search tags...',
							value: searchValue,
							onChange: onSearchChange,
							onClick: (e) => e.stopPropagation(),
						})
					),
					optionElements,
					showNewTagInput
						? React.createElement(
							'div',
							{ class: 'multi-select-new-tag' },
							React.createElement('input', {
								type: 'text',
								placeholder: 'Enter new tag...',
								value: newTagValue,
								onChange: onNewTagChange,
								onKeyPress: (e) => {
									if (e.key === 'Enter') onAddNewTag();
								},
								onClick: (e) => e.stopPropagation(),
							}),
							React.createElement(
								'button',
								{ onClick: onAddNewTag, disabled: !String(newTagValue || '').trim() },
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
			.map((t) => String(t || '').trim())
			.filter(Boolean);
		// de-dup
		const seen = new Set();
		const dedup = [];
		out.forEach((t) => {
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
		const dropdownId = opts.dropdownId || ('tag-dropdown-' + Math.random().toString(36).slice(2));

		const state = {
			selectedTags: Array.isArray(opts.selectedTags) ? opts.selectedTags.slice() : [],
			negativeTags: Array.isArray(opts.negativeTags) ? opts.negativeTags.slice() : [],
			availableTags: normalizeTagList(opts.availableTags || []),
			open: Boolean(opts.open),
			searchValue: String(opts.searchValue || ''),
			newTagValue: String(opts.newTagValue || ''),
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
			const resp = await csrfFetch('/api/tag_feedback', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ pid: pid, tag: t, label: label }),
			});
			const data = await resp.json().catch(() => null);
			if (!data || !data.success) {
				throw new Error((data && data.error) ? data.error : 'Failed to update tag');
			}
			ensureAvailable(t);
			applyLocal(t, label);
		}

		function render() {
			ReactDOM.render(
				React.createElement(MultiSelectDropdown, {
					selectedTags: state.selectedTags,
					negativeTags: state.negativeTags,
					availableTags: state.availableTags,
					isOpen: state.open,
					onToggle: () => {
						state.open = !state.open;
						if (!state.open) {
							state.searchValue = '';
							state.newTagValue = '';
						}
						render();
						emit();
					},
					onTagCycle: (tag) => {
						const isPos = state.selectedTags.includes(tag);
						const isNeg = state.negativeTags.includes(tag);
						const next = isPos ? -1 : isNeg ? 0 : 1;
						applyRemote(tag, next).catch((err) => {
							console.error('Failed to update tag feedback:', err);
							alert('Failed to update tag feedback: ' + (err && err.message ? err.message : String(err)));
						});
					},
					onClearTag: (tag) => {
						applyRemote(tag, 0).catch((err) => {
							console.error('Failed to clear tag:', err);
							alert('Failed to clear tag: ' + (err && err.message ? err.message : String(err)));
						});
					},
					newTagValue: state.newTagValue,
					onNewTagChange: (e) => {
						state.newTagValue = e && e.target ? e.target.value : '';
						render();
						emit();
					},
					onAddNewTag: () => {
						const t = String(state.newTagValue || '').trim();
						if (!t) return;
						applyRemote(t, 1)
							.then(() => {
								state.newTagValue = '';
								render();
								emit();
							})
							.catch((err) => {
								console.error('Failed to add new tag:', err);
								alert('Failed to add new tag: ' + (err && err.message ? err.message : String(err)));
							});
					},
					dropdownId: dropdownId,
					searchValue: state.searchValue,
					onSearchChange: (e) => {
						state.searchValue = e && e.target ? e.target.value : '';
						render();
						emit();
					},
				}),
				el
			);
		}

		// register close handlers
		registerDropdown(dropdownId, {
			isOpen: () => !!state.open,
			close: () => {
				if (!state.open) return;
				state.open = false;
				state.searchValue = '';
				state.newTagValue = '';
				render();
				emit();
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
			updateAvailableTags: (nextTags) => {
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
				state.selectedTags = state.selectedTags.map((t) => (t === from ? to : t));
				state.negativeTags = state.negativeTags.map((t) => (t === from ? to : t));
				ensureAvailable(to);
				render();
				emit();
			},
			applyTagDelete: (tagName) => {
				const t = String(tagName || '').trim();
				if (!t) return;
				state.selectedTags = state.selectedTags.filter((x) => x !== t);
				state.negativeTags = state.negativeTags.filter((x) => x !== t);
				render();
				emit();
			},
			unmount: () => {
				try { ReactDOM.unmountComponentAtNode(el); } catch (e) {}
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
