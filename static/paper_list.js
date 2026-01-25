'use strict';

// Use shared utilities from common_utils.js
var CommonUtils = window.ArxivSanityCommon;
var csrfFetch = CommonUtils.csrfFetch;
var _setupUserEventStream = CommonUtils.setupUserEventStream;
var _registerEventHandler = CommonUtils.registerEventHandler;
var renderTldrMarkdown = CommonUtils.renderTldrMarkdown;
var renderAbstractMarkdown = CommonUtils.renderAbstractMarkdown;
var formatAuthorsText = CommonUtils.formatAuthorsText;
var triggerMathJax = CommonUtils.triggerMathJax;
var buildTagUrl = CommonUtils.buildTagUrl;
var buildKeywordUrl = CommonUtils.buildKeywordUrl;
var registerDropdown = CommonUtils.registerDropdown;
var unregisterDropdown = CommonUtils.unregisterDropdown;
var getCsrfToken = CommonUtils.getCsrfToken;
// Summary status polling (shared)
var markSummaryPending = CommonUtils.markSummaryPending;
var unmarkSummaryPending = CommonUtils.unmarkSummaryPending;
var canTriggerSummary = CommonUtils.canTriggerSummary;
var formatSummaryStatus = CommonUtils.formatSummaryStatus;
var fetchTaskStatus = CommonUtils.fetchTaskStatus;

function applyUserState(state) {
    if (!state || !state.success) return;
    if (state.tags) {
        setGlobalTags(state.tags, { renderCombined: false });
    }
    if (state.combined_tags) {
        setGlobalCombinedTags(state.combined_tags);
    }
    if (state.keys) {
        setGlobalKeys(state.keys);
    }
}

function fetchUserStateAndApply() {
    return CommonUtils.fetchUserState().then(applyUserState);
}

// Common helpers are centralized in common_utils.js

function updatePaperSummaryStatus(pid, status, error, queueRank, queueTotal, taskId) {
    if (!pid || !Array.isArray(papers)) return;
    const p = papers.find(item => item && item.id === pid);
    if (!p) return;
    p.summary_status = status || '';
    p.summary_last_error = error || '';
    if (taskId !== undefined) {
        p.summary_task_id = taskId ? String(taskId) : '';
    }
    if (status && status !== 'queued' && status !== 'running') {
        p.summary_task_id = '';
    }
    if (queueRank !== undefined) {
        p.summary_queue_rank = queueRank || 0;
        p.summary_queue_total = queueTotal || 0;
    }
    renderPaperList();
}

// Register callback for summary status updates from shared polling
CommonUtils.setSummaryStatusCallback(function (pid, status, lastError, taskId) {
    updatePaperSummaryStatus(pid, status, lastError, undefined, undefined, taskId);
});

function updatePaperTagsForRename(fromTag, toTag) {
    if (!fromTag || !toTag || !Array.isArray(papers)) return;
    let changed = false;
    papers.forEach(p => {
        if (!p) return;
        if (Array.isArray(p.utags)) {
            const next = p.utags.map(t => (t === fromTag ? toTag : t));
            if (next.join('|') !== p.utags.join('|')) {
                p.utags = next;
                changed = true;
            }
        }
        if (Array.isArray(p.ntags)) {
            const next = p.ntags.map(t => (t === fromTag ? toTag : t));
            if (next.join('|') !== p.ntags.join('|')) {
                p.ntags = next;
                changed = true;
            }
        }
    });
    if (changed) renderPaperList();
}

function updatePaperTagsForDelete(tagName) {
    if (!tagName || !Array.isArray(papers)) return;
    let changed = false;
    papers.forEach(p => {
        if (!p) return;
        if (Array.isArray(p.utags)) {
            const next = p.utags.filter(t => t !== tagName);
            if (next.length !== p.utags.length) {
                p.utags = next;
                changed = true;
            }
        }
        if (Array.isArray(p.ntags)) {
            const next = p.ntags.filter(t => t !== tagName);
            if (next.length !== p.ntags.length) {
                p.ntags = next;
                changed = true;
            }
        }
    });
    if (changed) renderPaperList();
}

function applyTagFeedbackToPaper(pid, tagName, label) {
    if (!pid || !tagName || !Array.isArray(papers)) return;
    const p = papers.find(item => item && item.id === pid);
    if (!p) return;
    const pos = new Set(p.utags || []);
    const neg = new Set(p.ntags || []);
    if (label === 1) {
        pos.add(tagName);
        neg.delete(tagName);
    } else if (label === -1) {
        pos.delete(tagName);
        neg.add(tagName);
    } else {
        pos.delete(tagName);
        neg.delete(tagName);
    }
    p.utags = Array.from(pos);
    p.ntags = Array.from(neg);
    renderPaperList();
}

function handleReadingListEvent(event) {
    if (!event || !event.pid) return;
    if (event.action === 'add') {
        addToReadingListCache(event.pid);
        renderPaperList();
    } else if (event.action === 'remove') {
        removeFromReadingListCache(event.pid);
        renderPaperList();
    }
}

function handleUserEvent(event, options = {}) {
    if (!event || typeof event !== 'object') return;
    if (event.type === 'user_state_changed') {
        if (event.reason === 'rename_tag') {
            updatePaperTagsForRename(event.from, event.to);
        } else if (event.reason === 'delete_tag') {
            updatePaperTagsForDelete(event.tag);
        } else if (
            event.reason === 'tag_feedback' &&
            event.pid &&
            event.tag &&
            event.label !== undefined
        ) {
            applyTagFeedbackToPaper(event.pid, event.tag, event.label);
        }
        fetchUserStateAndApply();
    } else if (event.type === 'summary_status') {
        updatePaperSummaryStatus(
            event.pid,
            event.status,
            event.error,
            undefined,
            undefined,
            event.task_id
        );
        if (event.status === 'queued' || event.status === 'running') {
            markSummaryPending(event.pid);
        } else {
            unmarkSummaryPending(event.pid);
        }
    } else if (event.type === 'readinglist_changed') {
        handleReadingListEvent(event);
    }
}

function setupUserEventStream() {
    _registerEventHandler(handleUserEvent);
    _setupUserEventStream(user, applyUserState);
}

// formatSummaryStatus and canTriggerSummary are now in common_utils.js

// Multi-select dropdown component (shared implementation)
const MultiSelectDropdown =
    typeof window !== 'undefined' &&
    window.ArxivSanityTagDropdown &&
    window.ArxivSanityTagDropdown.MultiSelectDropdown
        ? window.ArxivSanityTagDropdown.MultiSelectDropdown
        : props => {
              // Fallback: should not happen when tag_dropdown_shared.js is loaded.
              return React.createElement('div', null, 'Tag dropdown unavailable');
          };

const Paper = props => {
    const p = props.paper;
    const lst = props.tags;
    const tlst = lst.map(jtag => jtag.name);
    const ulst = p.utags;

    const similar_url = '/?rank=pid&pid=' + encodeURIComponent(p.id);
    const inspect_url = '/inspect?pid=' + encodeURIComponent(p.id);
    const summary_url = '/summary?pid=' + encodeURIComponent(p.id);
    const thumb_img =
        p.thumb_url === '' ? null : (
            <div class="rel_img">
                <img src={p.thumb_url} loading="lazy" alt="Paper thumbnail" />
            </div>
        );

    // if the user is logged in then we can show the multi-select dropdown
    let utag_controls = null;
    if (user) {
        utag_controls = (
            <div class="rel_utags">
                <MultiSelectDropdown
                    selectedTags={ulst}
                    negativeTags={props.negativeTags}
                    availableTags={tlst}
                    isOpen={props.dropdownOpen}
                    onToggle={props.onToggleDropdown}
                    onTagCycle={props.onTagCycle}
                    onClearTag={props.onClearTag}
                    newTagValue={props.newTagValue}
                    onNewTagChange={props.onNewTagChange}
                    onAddNewTag={props.onAddNewTag}
                    dropdownId={props.dropdownId}
                    searchValue={props.searchValue}
                    onSearchChange={props.onSearchChange}
                />
            </div>
        );
    }

    // Render TL;DR if available (with markdown and LaTeX support)
    const tldr_section = p.tldr ? (
        <div class="rel_tldr">
            <div class="tldr_label">💡 TL;DR</div>
            <div
                class="tldr_text"
                dangerouslySetInnerHTML={{ __html: renderTldrMarkdown(p.tldr) }}
            ></div>
        </div>
    ) : null;

    const statusText = formatSummaryStatus(props.summaryStatus);
    const queueRankText =
        props.summaryStatus === 'queued' && props.summaryQueueRank
            ? `${props.summaryQueueRank}${props.summaryQueueTotal ? '/' + props.summaryQueueTotal : ''} Queued`
            : '';
    const queueRankSpan = queueRankText ? (
        <span class="queue-rank-pill">{queueRankText}</span>
    ) : null;
    const statusClass =
        props.summaryStatus === 'ok'
            ? 'summary-status-badge ok'
            : props.summaryStatus === 'failed'
              ? 'summary-status-badge failed'
              : props.summaryStatus
                ? 'summary-status-badge'
                : '';
    const tooltipParts = [];
    if (props.summaryLastError) {
        tooltipParts.push(props.summaryLastError);
    }
    if (queueRankText) {
        tooltipParts.push(`${queueRankText} (high priority only)`);
    }
    const badgeTitle = tooltipParts.length ? tooltipParts.join(' · ') : '';
    const statusBadge = statusText ? (
        <div class={statusClass} title={badgeTitle}>
            {statusText}
            {queueRankSpan}
        </div>
    ) : null;

    // Reading list button (only for logged in users)
    let readinglist_btn = null;
    if (user) {
        const isInReadingList = props.inReadingList;
        const btnClass = isInReadingList ? 'readinglist-btn active' : 'readinglist-btn';
        const btnTitle = isInReadingList ? 'In reading list' : 'Add to reading list';
        const btnIcon = isInReadingList ? '📖' : '🔖';
        readinglist_btn = (
            <div class={btnClass} onClick={props.onToggleReadingList} title={btnTitle}>
                {btnIcon}
            </div>
        );
    }

    const triggerDisabled = !canTriggerSummary(props.summaryStatus);
    const triggerBtn = (
        <button
            class="summary-trigger-btn"
            onClick={props.onTriggerSummary}
            disabled={triggerDisabled}
            title={triggerDisabled ? 'Summary already available or generating' : 'Generate summary'}
        >
            ✨ Generate Summary
        </button>
    );

    return (
        <div class="rel_paper">
            <div class="rel_score">
                {p.weight.toFixed(2)}
                {p.score_breakdown && <div class="score_breakdown">{p.score_breakdown}</div>}
            </div>
            {readinglist_btn}
            <div class="rel_title">
                <a href={'http://arxiv.org/abs/' + p.id} target="_blank" rel="noopener noreferrer">
                    {p.title}
                </a>
            </div>
            <div class="rel_authors" title={p.authors || ''}>
                {formatAuthorsText(p.authors, { maxAuthors: 10, head: 5, tail: 3 })}
            </div>
            <div class="rel_time">{p.time}</div>
            <div class="rel_tags">{p.tags}</div>
            {statusBadge}
            {tldr_section}
            {thumb_img}
            <div
                class="rel_abs"
                dangerouslySetInnerHTML={{ __html: renderAbstractMarkdown(p.summary) }}
            ></div>
            {utag_controls}
            <div class="paper-actions-footer">
                <div class="rel_summary">{triggerBtn}</div>
                <div class="rel_more">
                    <a href={similar_url} target="_blank" rel="noopener noreferrer">
                        Similar
                    </a>
                </div>
                <div class="rel_inspect">
                    <a href={inspect_url} target="_blank" rel="noopener noreferrer">
                        Inspect
                    </a>
                </div>
                <div class="rel_summary">
                    <a href={summary_url} target="_blank" rel="noopener noreferrer">
                        Summary
                    </a>
                </div>
                <div class="rel_alphaxiv">
                    <a
                        href={'https://www.alphaxiv.org/overview/' + p.id}
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        alphaXiv
                    </a>
                </div>
                <div class="rel_cool">
                    <a
                        href={'https://papers.cool/arxiv/' + p.id}
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Cool
                    </a>
                </div>
            </div>
        </div>
    );
};

const PaperList = props => {
    const lst = props.papers;
    const filtered_tags = props.tags.filter(tag => tag.name !== 'all');
    const plst = lst.map((jpaper, ix) => (
        <PaperComponent
            key={jpaper && jpaper.id ? jpaper.id : ix}
            paper={jpaper}
            tags={filtered_tags}
        />
    ));
    return (
        <div>
            <div id="paperList" class="rel_papers">
                {plst}
            </div>
        </div>
    );
};

class PaperComponent extends React.Component {
    constructor(props) {
        super(props);
        if (!Array.isArray(props.paper.utags)) {
            props.paper.utags = [];
        }
        if (!Array.isArray(props.paper.ntags)) {
            props.paper.ntags = [];
        }
        this.state = {
            paper: props.paper,
            tags: props.tags,
            dropdownOpen: false,
            newTagValue: '',
            searchValue: '',
            inReadingList: isInReadingList(props.paper.id),
            summaryStatus: props.paper.summary_status || '',
            summaryLastError: props.paper.summary_last_error || '',
            summaryQueueRank: props.paper.summary_queue_rank || 0,
            summaryQueueTotal: props.paper.summary_queue_total || 0,
            summaryTaskId: props.paper.summary_task_id ? String(props.paper.summary_task_id) : '',
        };
        this.dropdownId = 'dropdown-' + props.paper.id;
        this.queueRankTimer = null;
        this.handleToggleDropdown = this.handleToggleDropdown.bind(this);
        this.handleTagCycle = this.handleTagCycle.bind(this);
        this.handleClearTag = this.handleClearTag.bind(this);
        this.handleNewTagChange = this.handleNewTagChange.bind(this);
        this.handleAddNewTag = this.handleAddNewTag.bind(this);
        this.handleSearchChange = this.handleSearchChange.bind(this);
        this.handleToggleReadingList = this.handleToggleReadingList.bind(this);
        this.handleTriggerSummary = this.handleTriggerSummary.bind(this);
    }

    componentDidMount() {
        registerDropdown(this.dropdownId, {
            isOpen: () => this.state.dropdownOpen,
            close: () => {
                // Remove dropdown-open class when closing
                const dropdown = document.getElementById(this.dropdownId);
                if (dropdown) {
                    const paperCard = dropdown.closest('.rel_paper');
                    if (paperCard) {
                        paperCard.classList.remove('dropdown-open');
                    }
                }
                this.setState({ dropdownOpen: false });
            },
        });
        // Trigger MathJax rendering for TL;DR / abstract content.
        // Scoped to the list container to avoid full-document scans.
        if (this.state.paper.tldr || this.state.paper.summary) {
            triggerMathJax(document.getElementById('paperList'));
        }
    }

    componentDidUpdate(prevProps, prevState) {
        if (prevProps.tags !== this.props.tags) {
            this.setState({ tags: this.props.tags });
        }
        if (
            prevProps.paper.summary_status !== this.props.paper.summary_status ||
            prevProps.paper.summary_last_error !== this.props.paper.summary_last_error ||
            prevProps.paper.summary_queue_rank !== this.props.paper.summary_queue_rank ||
            prevProps.paper.summary_queue_total !== this.props.paper.summary_queue_total ||
            prevProps.paper.summary_task_id !== this.props.paper.summary_task_id
        ) {
            this.setState({
                summaryStatus: this.props.paper.summary_status || '',
                summaryLastError: this.props.paper.summary_last_error || '',
                summaryQueueRank: this.props.paper.summary_queue_rank || 0,
                summaryQueueTotal: this.props.paper.summary_queue_total || 0,
                summaryTaskId: this.props.paper.summary_task_id
                    ? String(this.props.paper.summary_task_id)
                    : '',
            });
        }
        if (this.state.summaryStatus !== prevState.summaryStatus) {
            if (this.state.summaryStatus === 'queued' && this.state.summaryTaskId) {
                this.startQueueRankPolling();
            } else if (this.state.summaryStatus !== 'queued') {
                this.stopQueueRankPolling();
                if (this.state.summaryQueueRank) {
                    this.setState({ summaryQueueRank: 0, summaryQueueTotal: 0 });
                }
            }
        }
    }

    componentWillUnmount() {
        unregisterDropdown(this.dropdownId);
        this.stopQueueRankPolling();
    }

    startQueueRankPolling() {
        if (this.queueRankTimer) return;
        this.queueRankTimer = setInterval(() => {
            this.refreshQueueRank();
        }, 6000);
        this.refreshQueueRank();
    }

    stopQueueRankPolling() {
        if (this.queueRankTimer) {
            clearInterval(this.queueRankTimer);
            this.queueRankTimer = null;
        }
    }

    refreshQueueRank() {
        const taskId = this.state.summaryTaskId;
        if (!taskId) return;
        fetchTaskStatus(taskId).then(data => {
            if (!data) return;
            if (data.status && data.status !== 'queued') {
                this.setState({ summaryQueueRank: 0, summaryQueueTotal: 0, summaryTaskId: '' });
                this.stopQueueRankPolling();
                return;
            }
            const queueRank = Number(data.queue_rank || 0);
            const queueTotal = Number(data.queue_total || 0);
            this.setState({ summaryQueueRank: queueRank, summaryQueueTotal: queueTotal });
            updatePaperSummaryStatus(
                this.state.paper.id,
                this.state.summaryStatus,
                this.state.summaryLastError,
                queueRank,
                queueTotal
            );
        });
    }

    handleToggleDropdown() {
        this.setState(prevState => {
            const newOpen = !prevState.dropdownOpen;
            // Toggle class on parent .rel_paper for z-index (fallback for :has())
            const dropdown = document.getElementById(this.dropdownId);
            if (dropdown) {
                const paperCard = dropdown.closest('.rel_paper');
                if (paperCard) {
                    if (newOpen) {
                        paperCard.classList.add('dropdown-open');
                    } else {
                        paperCard.classList.remove('dropdown-open');
                    }
                }
            }
            return {
                dropdownOpen: newOpen,
                searchValue: newOpen ? '' : prevState.searchValue, // Reset search when opened
            };
        });
    }

    handleSearchChange(event) {
        this.setState({
            searchValue: event.target.value,
        });
    }

    applyTagFeedback(tagName, label) {
        const { paper } = this.state;
        return csrfFetch('/api/tag_feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pid: paper.id, tag: tagName, label: label }),
        })
            .then(response => response.json())
            .then(data => {
                if (!data || !data.success) {
                    const err = data && data.error ? data.error : 'Unknown error';
                    throw new Error(err);
                }
                const nextPos = new Set(paper.utags || []);
                const nextNeg = new Set(paper.ntags || []);
                const wasPos = nextPos.has(tagName);
                const wasNeg = nextNeg.has(tagName);
                let posDelta = 0;
                let negDelta = 0;

                if (label === 1) {
                    if (!wasPos) {
                        nextPos.add(tagName);
                        posDelta += 1;
                    }
                    if (wasNeg) {
                        nextNeg.delete(tagName);
                        negDelta -= 1;
                    }
                } else if (label === -1) {
                    if (wasPos) {
                        nextPos.delete(tagName);
                        posDelta -= 1;
                    }
                    if (!wasNeg) {
                        nextNeg.add(tagName);
                        negDelta += 1;
                    }
                } else {
                    if (wasPos) {
                        nextPos.delete(tagName);
                        posDelta -= 1;
                    }
                    if (wasNeg) {
                        nextNeg.delete(tagName);
                        negDelta -= 1;
                    }
                }

                if (posDelta || negDelta) {
                    adjustTagStats(tagName, posDelta, negDelta);
                }

                paper.utags = Array.from(nextPos);
                paper.ntags = Array.from(nextNeg);
                this.setState({ paper: paper });
                return true;
            });
    }

    handleTagCycle(tagName) {
        const { paper } = this.state;
        const isPositive = (paper.utags || []).includes(tagName);
        const isNegative = (paper.ntags || []).includes(tagName);
        const nextLabel = isPositive ? -1 : isNegative ? 0 : 1;

        this.applyTagFeedback(tagName, nextLabel).catch(error => {
            console.error('Error updating tag feedback:', error);
            alert('Failed to update tag feedback: ' + error.message);
        });
    }

    handleClearTag(tagName) {
        this.applyTagFeedback(tagName, 0).catch(error => {
            console.error('Error clearing tag feedback:', error);
            alert('Failed to clear tag feedback: ' + error.message);
        });
    }

    handleNewTagChange(event) {
        this.setState({
            newTagValue: event.target.value,
        });
    }

    handleAddNewTag() {
        const { paper, newTagValue } = this.state;
        const trimmedTag = newTagValue.trim();

        if (!trimmedTag) return;

        // Check if tag already exists
        if (paper.utags.includes(trimmedTag)) {
            alert('Tag already exists');
            return;
        }

        this.applyTagFeedback(trimmedTag, 1)
            .then(() => {
                this.setState({
                    paper: paper,
                    newTagValue: '',
                });
                console.log(`Added new tag: ${trimmedTag}`);
            })
            .catch(error => {
                console.error('Error adding new tag:', error);
                alert('Failed to add new tag: ' + error.message);
            });
    }

    handleToggleReadingList() {
        const { paper, inReadingList } = this.state;

        if (inReadingList) {
            // Remove from reading list
            csrfFetch('/api/readinglist/remove', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pid: paper.id }),
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        removeFromReadingListCache(paper.id);
                        this.setState({ inReadingList: false });
                        console.log(`Removed ${paper.id} from reading list`);
                    } else {
                        console.error('Failed to remove from reading list:', data.error);
                    }
                })
                .catch(error => {
                    console.error('Error removing from reading list:', error);
                });
        } else {
            // Add to reading list
            csrfFetch('/api/readinglist/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pid: paper.id }),
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        addToReadingListCache(paper.id);
                        const taskId = data.task_id ? String(data.task_id) : '';
                        this.setState({
                            inReadingList: true,
                            summaryStatus: 'queued',
                            summaryLastError: '',
                            summaryTaskId: taskId,
                        });
                        if (taskId) {
                            this.startQueueRankPolling();
                        }
                        markSummaryPending(paper.id);
                        updatePaperSummaryStatus(
                            paper.id,
                            'queued',
                            '',
                            undefined,
                            undefined,
                            taskId
                        );
                        console.log(`Added ${paper.id} to reading list, top_tags:`, data.top_tags);
                    } else {
                        console.error('Failed to add to reading list:', data.error);
                        alert('Failed to add to reading list: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error adding to reading list:', error);
                    alert('Network error, failed to add to reading list');
                });
        }
    }

    handleTriggerSummary() {
        const { paper, summaryStatus } = this.state;
        if (!canTriggerSummary(summaryStatus)) return;

        this.setState({ summaryStatus: 'queued', summaryLastError: '' });
        markSummaryPending(paper.id);
        csrfFetch('/api/trigger_paper_summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pid: paper.id }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const taskId = data.task_id ? String(data.task_id) : '';
                    this.setState({
                        summaryStatus: data.status || 'queued',
                        summaryLastError: data.last_error || '',
                        summaryTaskId: taskId,
                    });
                    if (taskId) {
                        this.startQueueRankPolling();
                    }
                    if (data.status === 'queued' || data.status === 'running') {
                        markSummaryPending(paper.id);
                    } else {
                        unmarkSummaryPending(paper.id);
                    }
                    updatePaperSummaryStatus(
                        paper.id,
                        data.status || 'queued',
                        data.last_error || '',
                        undefined,
                        undefined,
                        taskId
                    );
                } else {
                    this.setState({
                        summaryStatus: 'failed',
                        summaryLastError: data.error || '',
                        summaryTaskId: '',
                        summaryQueueRank: 0,
                        summaryQueueTotal: 0,
                    });
                    unmarkSummaryPending(paper.id);
                    alert('Failed to trigger summary: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error triggering summary:', error);
                this.setState({
                    summaryStatus: 'failed',
                    summaryLastError: String(error),
                    summaryTaskId: '',
                    summaryQueueRank: 0,
                    summaryQueueTotal: 0,
                });
                unmarkSummaryPending(paper.id);
                alert('Network error, failed to trigger summary');
            });
    }

    render() {
        return (
            <Paper
                paper={this.state.paper}
                tags={this.state.tags}
                negativeTags={this.state.paper.ntags || []}
                dropdownOpen={this.state.dropdownOpen}
                onToggleDropdown={this.handleToggleDropdown}
                onTagCycle={this.handleTagCycle}
                onClearTag={this.handleClearTag}
                newTagValue={this.state.newTagValue}
                onNewTagChange={this.handleNewTagChange}
                onAddNewTag={this.handleAddNewTag}
                dropdownId={this.dropdownId}
                searchValue={this.state.searchValue}
                onSearchChange={this.handleSearchChange}
                inReadingList={this.state.inReadingList}
                onToggleReadingList={this.handleToggleReadingList}
                summaryStatus={this.state.summaryStatus}
                summaryLastError={this.state.summaryLastError}
                summaryQueueRank={this.state.summaryQueueRank}
                summaryQueueTotal={this.state.summaryQueueTotal}
                onTriggerSummary={this.handleTriggerSummary}
            />
        );
    }
}

const Tag = props => {
    const t = props.tag;
    const turl = buildTagUrl(t.name);
    const isNegOnly = t.neg_only === true;
    const tag_class =
        'rel_utag' + (t.name === 'all' ? ' rel_utag_all' : '') + (isNegOnly ? ' tag-negative' : '');
    const isEditable = t.name !== 'all';
    const tooltip =
        t.name === 'all'
            ? 'Contains all tags'
            : `Positive: ${t.pos_n || 0} · Negative: ${t.neg_n || 0}`;

    const posCount = Number(t.pos_n || 0);
    const negCount = Number(t.neg_n || 0);

    const handleOpenManage = e => {
        e.preventDefault();
        e.stopPropagation();
        if (props.onManage) props.onManage(t);
    };

    const handleOpenReco = e => {
        e.preventDefault();
        e.stopPropagation();
        if (isNegOnly) {
            alert('This tag only has negative examples and cannot be used for recommendations.');
            return;
        }
        window.location.href = turl;
    };

    return (
        <div class={tag_class + ' enhanced-tag'} title={tooltip}>
            <span
                class={isNegOnly ? 'tag-link tag-link-disabled' : 'tag-link'}
                title={tooltip}
                onClick={handleOpenManage}
            >
                <span class="tag-counts">
                    <span class="tag-count tag-count-pos" title="Positive count">
                        +{posCount}
                    </span>
                    <span class="tag-count tag-count-neg" title="Negative count">
                        −{negCount}
                    </span>
                </span>
                <span class="tag-name">{t.name}</span>
            </span>
            {isEditable && (
                <div class="tag-actions">
                    <span class="tag-reco" onClick={handleOpenReco} title="Open recommendations">
                        ↗
                    </span>
                    <span class="tag-edit" onClick={() => props.onEdit(t)} title="Edit tag">
                        ✎
                    </span>
                    <span class="tag-delete" onClick={() => props.onDelete(t)} title="Delete tag">
                        ×
                    </span>
                </div>
            )}
        </div>
    );
};

const TagList = props => {
    const lst = props.tags;
    const tlst = lst.map((jtag, ix) => (
        <Tag
            key={ix}
            tag={jtag}
            onEdit={props.onEditTag}
            onDelete={props.onDeleteTag}
            onManage={props.onManageTag}
        />
    ));

    // show the #wordwrap element if the user clicks inspect
    const show_inspect = () => {
        const wordwrap = document.getElementById('wordwrap');
        if (wordwrap.style.display === 'block') {
            wordwrap.style.display = 'none';
        } else {
            wordwrap.style.display = 'block';
        }
    };
    const inspect_elt =
        words.length > 0 ? (
            <div id="inspect_svm" onClick={show_inspect}>
                inspect
            </div>
        ) : null;

    return (
        <div class="enhanced-tag-list">
            <div class="tag-list-actions">
                <span class="tag-stats-inline">({lst.length} tags)</span>
                <button class="tag-action-btn add-btn" onClick={props.onAddTag} title="Add new tag">
                    + Add
                </button>
            </div>
            <div id="tagList" class="rel_utags enhanced-tags">
                {tlst}
            </div>
            {inspect_elt}

            {/* Edit Modal */}
            {props.showEditModal && (
                <div class="modal-overlay" onClick={props.onCloseEditModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Edit Tag</h3>
                            <span class="modal-close" onClick={props.onCloseEditModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Tag Name:</label>
                                <input
                                    type="text"
                                    value={props.editingTagName}
                                    onChange={props.onEditingTagNameChange}
                                    class="form-input"
                                    placeholder="Enter new tag name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseEditModal}>
                                Cancel
                            </button>
                            <button class="btn btn-primary" onClick={props.onSaveTagEdit}>
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Add Tag Modal */}
            {props.showAddModal && (
                <div class="modal-overlay" onClick={props.onCloseAddModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Add Tag</h3>
                            <span class="modal-close" onClick={props.onCloseAddModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Tag Name:</label>
                                <input
                                    type="text"
                                    value={props.newTagName}
                                    onChange={props.onNewTagNameChange}
                                    class="form-input"
                                    placeholder="Enter tag name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseAddModal}>
                                Cancel
                            </button>
                            <button class="btn btn-primary" onClick={props.onSaveNewTag}>
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {props.showDeleteModal && (
                <div class="modal-overlay" onClick={props.onCloseDeleteModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Confirm Delete</h3>
                            <span class="modal-close" onClick={props.onCloseDeleteModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <p>
                                Are you sure you want to delete tag "
                                <strong>{props.deletingTag && props.deletingTag.name}</strong>"?
                            </p>
                            <p class="warning-text">
                                This action is irreversible. All papers under this tag will lose the
                                tag.
                            </p>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseDeleteModal}>
                                Cancel
                            </button>
                            <button class="btn btn-danger" onClick={props.onConfirmDelete}>
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Tag Manage Modal */}
            {props.showManageModal && (
                <div class="modal-overlay" onClick={props.onCloseManageModal}>
                    <div
                        class="modal-content wide tag-manage-modal"
                        onClick={e => e.stopPropagation()}
                    >
                        <div class="modal-header">
                            <h3>Manage Tag: {props.managingTagName}</h3>
                            <span class="modal-close" onClick={props.onCloseManageModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <div class="tag-manage-toolbar">
                                <div class="tag-manage-filters">
                                    <label>View:</label>
                                    <select
                                        value={props.manageLabelFilter}
                                        onChange={props.onManageLabelFilterChange}
                                    >
                                        <option value="all">All</option>
                                        <option value="pos">Positive</option>
                                        <option value="neg">Negative</option>
                                    </select>
                                    <input
                                        type="text"
                                        class="form-input"
                                        placeholder="Search title / author..."
                                        value={props.manageSearchValue}
                                        onChange={props.onManageSearchChange}
                                    />
                                </div>
                                <div class="tag-manage-stats">
                                    <span class="tag-manage-stat">pos: {props.managePosTotal}</span>
                                    <span class="tag-manage-stat">neg: {props.manageNegTotal}</span>
                                    <span class="tag-manage-stat">
                                        total: {props.manageTotalCount}
                                    </span>
                                </div>
                            </div>

                            <div class="tag-manage-add">
                                <div class="tag-manage-add-row">
                                    <input
                                        type="text"
                                        class="form-input tag-manage-add-input"
                                        placeholder="Enter PID(s), e.g. 2501.01234"
                                        value={props.manageAddPidsValue}
                                        onChange={props.onManageAddPidsChange}
                                        onKeyDown={e => {
                                            if (
                                                e.key === 'Enter' &&
                                                props.manageAddPidsValue.trim()
                                            ) {
                                                e.preventDefault();
                                                props.onManageAddPids(1);
                                            }
                                        }}
                                    />
                                    <button
                                        class="btn btn-primary tag-manage-add-btn"
                                        onClick={() => props.onManageAddPids(1)}
                                        disabled={!props.manageAddPidsValue.trim()}
                                        title="Add as positive (Enter)"
                                    >
                                        +
                                    </button>
                                    <button
                                        class="btn btn-danger tag-manage-add-btn"
                                        onClick={() => props.onManageAddPids(-1)}
                                        disabled={!props.manageAddPidsValue.trim()}
                                        title="Add as negative"
                                    >
                                        −
                                    </button>
                                </div>
                                <div class="tag-manage-help">
                                    <small>
                                        Multiple PIDs: separate by comma / space / newline. Enter to
                                        add as +. Use buttons for + / −.
                                    </small>
                                </div>

                                {props.managePidPreviewLoading ||
                                (props.managePidPreviewItems &&
                                    props.managePidPreviewItems.length) ||
                                props.managePidPreviewError ? (
                                    <div class="tag-manage-pid-preview">
                                        {props.managePidPreviewLoading ? (
                                            <div class="tag-manage-pid-preview-loading">
                                                Previewing…
                                            </div>
                                        ) : null}
                                        {props.managePidPreviewError ? (
                                            <div class="tag-manage-pid-preview-error">
                                                {props.managePidPreviewError}
                                            </div>
                                        ) : null}
                                        {(props.managePidPreviewItems || [])
                                            .slice(0, 8)
                                            .map((it, ix) => (
                                                <div
                                                    key={it.pid + '-' + ix}
                                                    class={
                                                        'tag-manage-pid-preview-item' +
                                                        (it.title ? '' : ' not-found')
                                                    }
                                                >
                                                    <span class="tag-manage-pid-preview-pid">
                                                        {it.pid}
                                                    </span>
                                                    <span
                                                        class={
                                                            'tag-manage-pid-preview-title' +
                                                            (it.title ? '' : ' not-found')
                                                        }
                                                    >
                                                        {it.title || 'Not found in database'}
                                                    </span>
                                                </div>
                                            ))}
                                        {(props.managePidPreviewItems || []).length > 8 ? (
                                            <div class="tag-manage-pid-preview-more">
                                                …and{' '}
                                                {(props.managePidPreviewItems || []).length - 8}{' '}
                                                more
                                            </div>
                                        ) : null}
                                    </div>
                                ) : null}
                            </div>

                            <div class="tag-manage-list">
                                {props.manageLoading ? (
                                    <div class="tag-manage-loading">Loading…</div>
                                ) : (props.manageItems || []).length === 0 ? (
                                    <div class="tag-manage-empty">No papers in this tag.</div>
                                ) : (
                                    <div class="tag-manage-rows">
                                        {(props.manageItems || []).map((it, ix) => (
                                            <div key={it.pid + '-' + ix} class="tag-manage-row">
                                                <button
                                                    class={
                                                        'tag-manage-label ' +
                                                        (it.label === 1
                                                            ? 'pos'
                                                            : it.label === -1
                                                              ? 'neg'
                                                              : 'none')
                                                    }
                                                    onClick={() =>
                                                        props.onManageCyclePid(it.pid, it.label)
                                                    }
                                                    title="Cycle label"
                                                >
                                                    {it.label === 1
                                                        ? '+'
                                                        : it.label === -1
                                                          ? '−'
                                                          : '·'}
                                                </button>
                                                <div class="tag-manage-main">
                                                    <div class="tag-manage-title">
                                                        <a
                                                            href={
                                                                '/summary?pid=' +
                                                                encodeURIComponent(it.pid)
                                                            }
                                                            target="_blank"
                                                            rel="noreferrer"
                                                        >
                                                            {it.title || it.pid}
                                                        </a>
                                                    </div>
                                                    <div class="tag-manage-meta">
                                                        <span class="tag-manage-time">
                                                            {it.time || ''}
                                                        </span>
                                                        <span
                                                            class="tag-manage-authors"
                                                            title={it.authors || ''}
                                                        >
                                                            {formatAuthorsText(it.authors, {
                                                                maxAuthors: 10,
                                                                head: 5,
                                                                tail: 3,
                                                            })}
                                                        </span>
                                                    </div>
                                                </div>
                                                <button
                                                    class="btn btn-cancel tag-manage-remove"
                                                    onClick={() => props.onManageSetPid(it.pid, 0)}
                                                >
                                                    Remove
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                        <div class="modal-footer">
                            <div class="tag-manage-pagination">
                                <button
                                    class="btn btn-cancel"
                                    onClick={props.onManagePrevPage}
                                    disabled={props.managePageNumber <= 1 || props.manageLoading}
                                >
                                    Prev
                                </button>
                                <span class="tag-manage-page">Page {props.managePageNumber}</span>
                                <button
                                    class="btn btn-primary"
                                    onClick={props.onManageNextPage}
                                    disabled={props.manageLoading || !props.manageHasMore}
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

class TagListComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            tags: props.tags,
            showEditModal: false,
            showDeleteModal: false,
            showAddModal: false,
            showManageModal: false,
            editingTag: null,
            editingTagName: '',
            deletingTag: null,
            newTagName: '',

            managingTag: null,
            manageLabelFilter: 'all',
            manageSearchValue: '',
            manageItems: [],
            manageLoading: false,
            managePageNumber: 1,
            managePageSize: 20,
            manageTotalCount: 0,
            managePosTotal: 0,
            manageNegTotal: 0,
            manageHasMore: false,
            manageAddPidsValue: '',
            managePidPreviewLoading: false,
            managePidPreviewItems: [],
            managePidPreviewError: '',
        };
        this.handleEditTag = this.handleEditTag.bind(this);
        this.handleDeleteTag = this.handleDeleteTag.bind(this);
        this.handleAddTag = this.handleAddTag.bind(this);
        this.handleCloseEditModal = this.handleCloseEditModal.bind(this);
        this.handleCloseDeleteModal = this.handleCloseDeleteModal.bind(this);
        this.handleCloseAddModal = this.handleCloseAddModal.bind(this);
        this.handleEditingTagNameChange = this.handleEditingTagNameChange.bind(this);
        this.handleNewTagNameChange = this.handleNewTagNameChange.bind(this);
        this.handleSaveTagEdit = this.handleSaveTagEdit.bind(this);
        this.handleSaveNewTag = this.handleSaveNewTag.bind(this);
        this.handleConfirmDelete = this.handleConfirmDelete.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);

        this.handleManageTag = this.handleManageTag.bind(this);
        this.handleCloseManageModal = this.handleCloseManageModal.bind(this);
        this.handleManageLabelFilterChange = this.handleManageLabelFilterChange.bind(this);
        this.handleManageSearchChange = this.handleManageSearchChange.bind(this);
        this.handleManagePrevPage = this.handleManagePrevPage.bind(this);
        this.handleManageNextPage = this.handleManageNextPage.bind(this);
        this.handleManageCyclePid = this.handleManageCyclePid.bind(this);
        this.handleManageSetPid = this.handleManageSetPid.bind(this);
        this.handleManageAddPidsChange = this.handleManageAddPidsChange.bind(this);
        this.handleManageAddPids = this.handleManageAddPids.bind(this);

        this._pidPreviewTimeout = null;
    }

    componentDidMount() {
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentDidUpdate(prevProps) {
        if (prevProps.tags !== this.props.tags) {
            this.setState({ tags: this.props.tags });
        }
    }

    componentWillUnmount() {
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (this.state.showEditModal) {
                this.handleCloseEditModal();
            }
            if (this.state.showDeleteModal) {
                this.handleCloseDeleteModal();
            }
            if (this.state.showAddModal) {
                this.handleCloseAddModal();
            }
        }
    }

    handleEditTag(tag) {
        this.setState({
            showEditModal: true,
            editingTag: tag,
            editingTagName: tag.name,
        });
    }

    handleDeleteTag(tag) {
        this.setState({
            showDeleteModal: true,
            deletingTag: tag,
        });
    }

    handleAddTag() {
        this.setState({
            showAddModal: true,
            newTagName: '',
        });
    }

    handleCloseAddModal() {
        this.setState({
            showAddModal: false,
            newTagName: '',
        });
    }

    handleNewTagNameChange(event) {
        this.setState({ newTagName: event.target.value });
    }

    handleSaveNewTag() {
        const { newTagName } = this.state;
        const trimmedTag = newTagName.trim();
        if (!trimmedTag) {
            alert('Tag name cannot be empty');
            return;
        }

        if (trimmedTag === 'all' || trimmedTag === 'null') {
            alert('Tag name is reserved');
            return;
        }

        if (this.state.tags.some(tag => tag.name === trimmedTag)) {
            alert('Tag already exists');
            return;
        }

        csrfFetch('/add_tag/' + encodeURIComponent(trimmedTag))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState(prevState => {
                        const nextTags = normalizeTags(
                            prevState.tags
                                .filter(tag => tag.name !== 'all')
                                .concat([{ name: trimmedTag, n: 0 }])
                        );
                        setGlobalTags(nextTags, { renderTags: false });
                        return {
                            tags: nextTags,
                            showAddModal: false,
                            newTagName: '',
                        };
                    });
                    console.log('Tag added successfully');
                } else {
                    alert('Add failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error adding tag:', error);
                alert('Network error, add failed');
            });
    }

    handleCloseEditModal() {
        this.setState({
            showEditModal: false,
            editingTag: null,
            editingTagName: '',
        });
    }

    handleCloseDeleteModal() {
        this.setState({
            showDeleteModal: false,
            deletingTag: null,
        });
    }

    handleCloseManageModal() {
        this.setState({
            showManageModal: false,
            managingTag: null,
            manageItems: [],
            manageSearchValue: '',
            manageLabelFilter: 'all',
            managePageNumber: 1,
            manageTotalCount: 0,
            managePosTotal: 0,
            manageNegTotal: 0,
            manageHasMore: false,
            manageAddPidsValue: '',
            managePidPreviewLoading: false,
            managePidPreviewItems: [],
            managePidPreviewError: '',
        });
    }

    parsePidInput(raw) {
        const parts = String(raw || '')
            .trim()
            .split(/[\s,\n\r\t;\uFF0C\u3001]+/)
            .map(s => s.trim())
            .filter(Boolean);
        const seen = new Set();
        const out = [];
        for (const p of parts) {
            if (seen.has(p)) continue;
            seen.add(p);
            out.push(p);
            if (out.length >= 50) break;
        }
        return out;
    }

    async fetchPidPreview(pids) {
        const list = Array.isArray(pids) ? pids : [];
        if (list.length === 0) {
            this.setState({
                managePidPreviewLoading: false,
                managePidPreviewItems: [],
                managePidPreviewError: '',
            });
            return;
        }
        this.setState({ managePidPreviewLoading: true, managePidPreviewError: '' });
        try {
            const resp = await csrfFetch('/api/paper_titles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pids: list }),
            });
            const data = await resp.json().catch(() => null);
            if (!data || !data.success) {
                throw new Error(data && data.error ? data.error : 'Failed to preview PIDs');
            }
            const items = Array.isArray(data.items) ? data.items : [];
            this.setState({ managePidPreviewItems: items, managePidPreviewLoading: false });
        } catch (e) {
            const friendlyMsg = CommonUtils.handleApiError(e, 'Preview PIDs');
            this.setState({
                managePidPreviewLoading: false,
                managePidPreviewItems: [],
                managePidPreviewError: friendlyMsg,
            });
        }
    }

    async fetchManageMembers(overrides = {}) {
        const managingTag =
            overrides.managingTag !== undefined ? overrides.managingTag : this.state.managingTag;
        if (!managingTag || !managingTag.name) return;
        const page_number =
            overrides.managePageNumber !== undefined
                ? overrides.managePageNumber
                : this.state.managePageNumber;
        const label =
            overrides.manageLabelFilter !== undefined
                ? overrides.manageLabelFilter
                : this.state.manageLabelFilter;
        const page_size =
            overrides.managePageSize !== undefined
                ? overrides.managePageSize
                : this.state.managePageSize;
        const search =
            overrides.manageSearchValue !== undefined
                ? overrides.manageSearchValue
                : this.state.manageSearchValue;

        this.setState({ manageLoading: true });
        try {
            const params = new URLSearchParams();
            params.set('tag', managingTag.name);
            params.set('label', label);
            params.set('page_number', String(page_number));
            params.set('page_size', String(page_size));
            if (search && search.trim()) {
                params.set('search', search.trim());
            }
            const resp = await fetch('/api/tag_members?' + params.toString(), {
                credentials: 'same-origin',
            });
            const data = await resp.json();
            if (!data || !data.success) {
                throw new Error(data && data.error ? data.error : 'Failed to load tag members');
            }
            const items = Array.isArray(data.items) ? data.items : [];
            this.setState({
                manageItems: items,
                manageTotalCount: data.total_count || 0,
                managePosTotal: data.pos_total || 0,
                manageNegTotal: data.neg_total || 0,
                manageHasMore: page_number * page_size < (data.total_count || 0),
            });
        } catch (e) {
            const friendlyMsg = CommonUtils.handleApiError(e, 'Fetch Tag Members');
            console.error('Failed to fetch tag members:', e);
            alert('Failed to load tag members: ' + friendlyMsg);
        } finally {
            this.setState({ manageLoading: false });
        }
    }

    handleManageTag(tag) {
        if (!tag || !tag.name || tag.name === 'all') return;
        this.setState(
            {
                showManageModal: true,
                managingTag: tag,
                manageLabelFilter: 'all',
                manageSearchValue: '',
                manageItems: [],
                managePageNumber: 1,
                manageTotalCount: 0,
                managePosTotal: 0,
                manageNegTotal: 0,
                manageHasMore: false,
                manageAddPidsValue: '',
            },
            () =>
                this.fetchManageMembers({
                    managingTag: tag,
                    managePageNumber: 1,
                    manageLabelFilter: 'all',
                    manageSearchValue: '',
                })
        );
    }

    handleManageLabelFilterChange(event) {
        const next = event.target.value;
        this.setState({ manageLabelFilter: next, managePageNumber: 1 }, () => {
            this.fetchManageMembers({ manageLabelFilter: next, managePageNumber: 1 });
        });
    }

    handleManageSearchChange(event) {
        const val = event.target.value;
        this.setState({ manageSearchValue: val });
        // Debounce search
        if (this._searchTimeout) clearTimeout(this._searchTimeout);
        this._searchTimeout = setTimeout(() => {
            this.setState({ managePageNumber: 1 }, () => {
                this.fetchManageMembers({ manageSearchValue: val, managePageNumber: 1 });
            });
        }, 300);
    }

    handleManagePrevPage() {
        if (this.state.manageLoading) return;
        const nextPage = Math.max(1, (this.state.managePageNumber || 1) - 1);
        if (nextPage === this.state.managePageNumber) return;
        this.setState({ managePageNumber: nextPage }, () =>
            this.fetchManageMembers({ managePageNumber: nextPage })
        );
    }

    handleManageNextPage() {
        if (this.state.manageLoading || !this.state.manageHasMore) return;
        const nextPage = (this.state.managePageNumber || 1) + 1;
        this.setState({ managePageNumber: nextPage }, () =>
            this.fetchManageMembers({ managePageNumber: nextPage })
        );
    }

    async handleManageSetPid(pid, label) {
        const managingTag = this.state.managingTag;
        if (!managingTag || !managingTag.name) return;
        try {
            const resp = await csrfFetch('/api/tag_feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pid, tag: managingTag.name, label }),
            });
            const data = await resp.json();
            if (!data || !data.success) {
                throw new Error(data && data.error ? data.error : 'Update failed');
            }
            await this.fetchManageMembers();
        } catch (e) {
            const friendlyMsg = CommonUtils.handleApiError(e, 'Update Tag Feedback');
            console.error('Failed to update tag feedback:', e);
            alert('Failed to update tag: ' + friendlyMsg);
        }
    }

    handleManageCyclePid(pid, currentLabel) {
        const cur = Number(currentLabel || 0);
        const next = cur === 1 ? -1 : cur === -1 ? 0 : 1;
        this.handleManageSetPid(pid, next);
    }

    handleManageAddPidsChange(event) {
        const val = event.target.value;
        this.setState({ manageAddPidsValue: val });

        if (this._pidPreviewTimeout) clearTimeout(this._pidPreviewTimeout);
        this._pidPreviewTimeout = setTimeout(() => {
            const pids = this.parsePidInput(val);
            this.fetchPidPreview(pids);
        }, 250);
    }

    async handleManageAddPids(label) {
        const managingTag = this.state.managingTag;
        if (!managingTag || !managingTag.name) return;
        const raw = (this.state.manageAddPidsValue || '').trim();
        if (!raw) return;
        const pids = this.parsePidInput(raw).slice(0, 200);
        if (pids.length === 0) return;
        this.setState({ manageLoading: true });
        try {
            const tasks = pids.map(pid =>
                csrfFetch('/api/tag_feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pid, tag: managingTag.name, label }),
                })
                    .then(r => r.json())
                    .catch(() => null)
            );
            const results = await Promise.all(tasks);
            const failed = results.filter(r => !r || !r.success).length;
            if (failed) {
                alert(`Added with ${failed} failures (check PIDs).`);
            }
            this.setState({
                manageAddPidsValue: '',
                managePidPreviewItems: [],
                managePidPreviewError: '',
            });
            await this.fetchManageMembers();
        } finally {
            this.setState({ manageLoading: false });
        }
    }

    handleEditingTagNameChange(event) {
        this.setState({ editingTagName: event.target.value });
    }

    handleSaveTagEdit() {
        const { editingTag, editingTagName } = this.state;
        if (!editingTagName.trim()) {
            alert('Tag name cannot be empty');
            return;
        }

        const trimmedName = editingTagName.trim();
        if (this.state.tags.some(tag => tag.name === trimmedName && tag.name !== editingTag.name)) {
            alert('Tag already exists');
            return;
        }

        csrfFetch(
            '/rename/' + encodeURIComponent(editingTag.name) + '/' + encodeURIComponent(trimmedName)
        )
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState(prevState => {
                        const nextTags = normalizeTags(
                            prevState.tags.map(tag =>
                                tag.name === editingTag.name ? { ...tag, name: trimmedName } : tag
                            )
                        );
                        const nextCombinedTags = renameCombinedTagsForTag(
                            combined_tags,
                            editingTag.name,
                            trimmedName
                        );
                        setGlobalCombinedTags(nextCombinedTags, { renderCombined: false });
                        setGlobalTags(nextTags, { renderTags: false });
                        return {
                            tags: nextTags,
                            showEditModal: false,
                            editingTag: null,
                            editingTagName: '',
                        };
                    });
                    console.log('Tag renamed successfully');
                } else {
                    alert('Rename failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error renaming tag:', error);
                alert('Network error, rename failed');
            });
    }

    handleConfirmDelete() {
        const { deletingTag } = this.state;

        csrfFetch('/del/' + encodeURIComponent(deletingTag.name))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState(prevState => {
                        const nextTags = normalizeTags(
                            prevState.tags.filter(tag => tag.name !== deletingTag.name)
                        );
                        const nextCombinedTags = removeCombinedTagsWithTag(
                            combined_tags,
                            deletingTag.name
                        );
                        setGlobalCombinedTags(nextCombinedTags, { renderCombined: false });
                        setGlobalTags(nextTags, { renderTags: false });
                        return {
                            tags: nextTags,
                            showDeleteModal: false,
                            deletingTag: null,
                        };
                    });
                    console.log('Tag deleted successfully');
                } else {
                    alert('Delete failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error deleting tag:', error);
                alert('Network error, delete failed');
            });
    }

    render() {
        const managingTagName =
            this.state.managingTag && this.state.managingTag.name
                ? this.state.managingTag.name
                : '';
        return (
            <TagList
                tags={this.state.tags}
                onEditTag={this.handleEditTag}
                onDeleteTag={this.handleDeleteTag}
                onAddTag={this.handleAddTag}
                onManageTag={this.handleManageTag}
                showEditModal={this.state.showEditModal}
                showDeleteModal={this.state.showDeleteModal}
                showAddModal={this.state.showAddModal}
                editingTagName={this.state.editingTagName}
                newTagName={this.state.newTagName}
                deletingTag={this.state.deletingTag}
                onCloseEditModal={this.handleCloseEditModal}
                onCloseDeleteModal={this.handleCloseDeleteModal}
                onCloseAddModal={this.handleCloseAddModal}
                onEditingTagNameChange={this.handleEditingTagNameChange}
                onNewTagNameChange={this.handleNewTagNameChange}
                onSaveTagEdit={this.handleSaveTagEdit}
                onSaveNewTag={this.handleSaveNewTag}
                onConfirmDelete={this.handleConfirmDelete}
                showManageModal={this.state.showManageModal}
                managingTagName={managingTagName}
                manageLabelFilter={this.state.manageLabelFilter}
                manageSearchValue={this.state.manageSearchValue}
                manageItems={this.state.manageItems}
                manageLoading={this.state.manageLoading}
                managePageNumber={this.state.managePageNumber}
                manageTotalCount={this.state.manageTotalCount}
                managePosTotal={this.state.managePosTotal}
                manageNegTotal={this.state.manageNegTotal}
                manageHasMore={this.state.manageHasMore}
                manageAddPidsValue={this.state.manageAddPidsValue}
                managePidPreviewLoading={this.state.managePidPreviewLoading}
                managePidPreviewItems={this.state.managePidPreviewItems}
                managePidPreviewError={this.state.managePidPreviewError}
                onCloseManageModal={this.handleCloseManageModal}
                onManageLabelFilterChange={this.handleManageLabelFilterChange}
                onManageSearchChange={this.handleManageSearchChange}
                onManagePrevPage={this.handleManagePrevPage}
                onManageNextPage={this.handleManageNextPage}
                onManageCyclePid={this.handleManageCyclePid}
                onManageSetPid={this.handleManageSetPid}
                onManageAddPidsChange={this.handleManageAddPidsChange}
                onManageAddPids={this.handleManageAddPids}
            />
        );
    }
}

const CombinedTag = props => {
    const t = props.comtag;
    const turl = buildTagUrl(t.name, { logic: 'and' });
    const tag_class = 'rel_utag rel_utag_all enhanced-combined-tag';

    return (
        <div class={tag_class}>
            <a href={turl} class="combined-tag-link">
                {t.name}
            </a>
            <div class="combined-tag-actions">
                <span
                    class="combined-tag-edit"
                    onClick={() => props.onEdit(t)}
                    title="Edit combined tag"
                >
                    ✎
                </span>
                <span
                    class="combined-tag-delete"
                    onClick={() => props.onDelete(t)}
                    title="Delete combined tag"
                >
                    ×
                </span>
            </div>
        </div>
    );
};

const CombinedTagList = props => {
    const lst = props.combined_tags;
    const tlst = lst.map((jtag, ix) => (
        <CombinedTag
            key={ix}
            comtag={jtag}
            onEdit={props.onEditCombinedTag}
            onDelete={props.onDeleteCombinedTag}
        />
    ));

    return (
        <div class="enhanced-combined-tag-list">
            <div class="combined-tag-list-actions">
                <span class="tag-stats-inline">({lst.length} combined tags)</span>
                <button
                    class="tag-action-btn add-btn"
                    onClick={props.onAddCombinedTag}
                    title="Add new combined tag"
                >
                    + Add
                </button>
            </div>
            <div id="combinedTagList" class="rel_utags enhanced-combined-tags">
                {tlst}
            </div>

            {/* Add/Edit Combined Tag Modal */}
            {props.showAddEditModal && (
                <div class="modal-overlay" onClick={props.onCloseAddEditModal}>
                    <div class="modal-content wide" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>
                                {props.editingCombinedTag
                                    ? 'Edit Combined Tag'
                                    : 'Add Combined Tag'}
                            </h3>
                            <span class="modal-close" onClick={props.onCloseAddEditModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Select tags to combine:</label>
                                <MultiSelectDropdown
                                    selectedTags={props.selectedTagsForCombination}
                                    negativeTags={[]}
                                    availableTags={props.availableTagsForCombination}
                                    isOpen={props.combinationDropdownOpen}
                                    onToggle={props.onToggleCombinationDropdown}
                                    onTagCycle={props.onCombinationTagToggle}
                                    onClearTag={props.onRemoveCombinationTag}
                                    newTagValue=""
                                    onNewTagChange={() => {}}
                                    onAddNewTag={() => {}}
                                    dropdownId="combination-dropdown"
                                    searchValue={props.combinationSearchValue}
                                    onSearchChange={props.onCombinationSearchChange}
                                    showNewTagInput={false}
                                />
                            </div>
                            {props.selectedTagsForCombination.length > 0 && (
                                <div class="tag-combination-preview">
                                    <h4>Preview Combination:</h4>
                                    <div class="tag-combination-preview-tags">
                                        {props.selectedTagsForCombination.map((tag, ix) => (
                                            <span key={ix} class="tag-combination-preview-tag">
                                                {tag}
                                            </span>
                                        ))}
                                    </div>
                                    <p
                                        style={{
                                            marginTop: '10px',
                                            fontSize: '12px',
                                            color: 'var(--text-color)',
                                            opacity: '0.8',
                                        }}
                                    >
                                        Combination Name:{' '}
                                        {props.selectedTagsForCombination.join(', ')}
                                    </p>
                                </div>
                            )}
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseAddEditModal}>
                                Cancel
                            </button>
                            <button
                                class="btn btn-primary"
                                onClick={props.onSaveCombinedTag}
                                disabled={props.selectedTagsForCombination.length < 2}
                            >
                                {props.editingCombinedTag ? 'Save' : 'Create'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {props.showDeleteModal && (
                <div class="modal-overlay" onClick={props.onCloseDeleteModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Confirm Delete</h3>
                            <span class="modal-close" onClick={props.onCloseDeleteModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <p>
                                Are you sure you want to delete combined tag "
                                <strong>
                                    {props.deletingCombinedTag && props.deletingCombinedTag.name}
                                </strong>
                                "?
                            </p>
                            <p class="warning-text">This action is irreversible.</p>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseDeleteModal}>
                                Cancel
                            </button>
                            <button class="btn btn-danger" onClick={props.onConfirmDelete}>
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

class CombinedTagListComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            combined_tags: props.combined_tags,
            tags: props.tags,
            showAddEditModal: false,
            showDeleteModal: false,
            editingCombinedTag: null,
            deletingCombinedTag: null,
            selectedTagsForCombination: [],
            combinationDropdownOpen: false,
            combinationSearchValue: '',
        };
        this.handleAddCombinedTag = this.handleAddCombinedTag.bind(this);
        this.handleEditCombinedTag = this.handleEditCombinedTag.bind(this);
        this.handleDeleteCombinedTag = this.handleDeleteCombinedTag.bind(this);
        this.handleCloseAddEditModal = this.handleCloseAddEditModal.bind(this);
        this.handleCloseDeleteModal = this.handleCloseDeleteModal.bind(this);
        this.handleSaveCombinedTag = this.handleSaveCombinedTag.bind(this);
        this.handleConfirmDelete = this.handleConfirmDelete.bind(this);
        this.handleToggleCombinationDropdown = this.handleToggleCombinationDropdown.bind(this);
        this.handleCombinationTagToggle = this.handleCombinationTagToggle.bind(this);
        this.handleRemoveCombinationTag = this.handleRemoveCombinationTag.bind(this);
        this.handleCombinationSearchChange = this.handleCombinationSearchChange.bind(this);
        this.handleClickOutside = this.handleClickOutside.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
    }

    componentDidMount() {
        document.addEventListener('mousedown', this.handleClickOutside);
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentDidUpdate(prevProps) {
        const tagsChanged = prevProps.tags !== this.props.tags;
        const combinedChanged = prevProps.combined_tags !== this.props.combined_tags;
        if (tagsChanged || combinedChanged) {
            this.setState({
                tags: this.props.tags,
                combined_tags: this.props.combined_tags,
            });
        }
    }

    componentWillUnmount() {
        document.removeEventListener('mousedown', this.handleClickOutside);
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (this.state.showAddEditModal) {
                this.handleCloseAddEditModal();
            }
            if (this.state.showDeleteModal) {
                this.handleCloseDeleteModal();
            }
            if (this.state.combinationDropdownOpen) {
                this.setState({ combinationDropdownOpen: false });
            }
        }
    }

    handleClickOutside(event) {
        const dropdown = document.getElementById('combination-dropdown');
        if (dropdown && !dropdown.contains(event.target)) {
            this.setState({ combinationDropdownOpen: false });
        }
    }

    handleAddCombinedTag() {
        this.setState({
            showAddEditModal: true,
            editingCombinedTag: null,
            selectedTagsForCombination: [],
            combinationSearchValue: '',
        });
    }

    handleEditCombinedTag(combinedTag) {
        // Parse existing combined tags, compatible with "comma+space" or "comma only" formats
        const existingTags = combinedTag.name
            .split(',')
            .map(tag => tag.trim())
            .filter(tag => tag.length > 0);
        this.setState({
            showAddEditModal: true,
            editingCombinedTag: combinedTag,
            selectedTagsForCombination: existingTags,
            combinationSearchValue: '',
        });
    }

    handleDeleteCombinedTag(combinedTag) {
        this.setState({
            showDeleteModal: true,
            deletingCombinedTag: combinedTag,
        });
    }

    handleCloseAddEditModal() {
        this.setState({
            showAddEditModal: false,
            editingCombinedTag: null,
            selectedTagsForCombination: [],
            combinationDropdownOpen: false,
            combinationSearchValue: '',
        });
    }

    handleCloseDeleteModal() {
        this.setState({
            showDeleteModal: false,
            deletingCombinedTag: null,
        });
    }

    handleSaveCombinedTag() {
        const { editingCombinedTag, selectedTagsForCombination } = this.state;

        if (selectedTagsForCombination.length < 2) {
            alert('Please select at least two tags to combine');
            return;
        }

        const combinationName = selectedTagsForCombination.join(', ');

        if (editingCombinedTag) {
            if (editingCombinedTag.name === combinationName) {
                this.handleCloseAddEditModal();
                return;
            }

            // Edit existing combined tag atomically
            csrfFetch(
                '/rename_ctag/' +
                    encodeURIComponent(editingCombinedTag.name) +
                    '/' +
                    encodeURIComponent(combinationName)
            )
                .then(response => response.text())
                .then(text => {
                    if (text.includes('ok')) {
                        this.setState(prevState => {
                            const nextCombinedTags = renameCombinedTagInList(
                                prevState.combined_tags,
                                editingCombinedTag.name,
                                combinationName
                            );
                            setGlobalCombinedTags(nextCombinedTags, { renderCombined: false });
                            return {
                                combined_tags: nextCombinedTags,
                                showAddEditModal: false,
                                editingCombinedTag: null,
                                selectedTagsForCombination: [],
                                combinationDropdownOpen: false,
                                combinationSearchValue: '',
                            };
                        });
                        console.log('Combined tag edited successfully');
                    } else {
                        throw new Error('Rename failed: ' + text);
                    }
                })
                .catch(error => {
                    console.error('Error editing combined tag:', error);
                    alert('Edit failed: ' + error.message);
                });
        } else {
            // Add new combined tag
            csrfFetch('/add_ctag/' + encodeURIComponent(combinationName))
                .then(response => response.text())
                .then(text => {
                    if (text.includes('ok')) {
                        this.setState(prevState => {
                            const nextCombinedTags = prevState.combined_tags.concat([
                                { name: combinationName },
                            ]);
                            setGlobalCombinedTags(nextCombinedTags, { renderCombined: false });
                            return {
                                combined_tags: nextCombinedTags,
                                showAddEditModal: false,
                                selectedTagsForCombination: [],
                                combinationDropdownOpen: false,
                                combinationSearchValue: '',
                            };
                        });
                        console.log('Combined tag added successfully');
                    } else {
                        alert('Add failed: ' + text);
                    }
                })
                .catch(error => {
                    console.error('Error adding combined tag:', error);
                    alert('Network error, add failed');
                });
        }
    }

    handleConfirmDelete() {
        const { deletingCombinedTag } = this.state;

        csrfFetch('/del_ctag/' + encodeURIComponent(deletingCombinedTag.name))
            .then(response => response.text())
            .then(text => {
                if (text.includes('ok')) {
                    this.setState(prevState => {
                        const nextCombinedTags = prevState.combined_tags.filter(
                            tag => tag.name !== deletingCombinedTag.name
                        );
                        setGlobalCombinedTags(nextCombinedTags, { renderCombined: false });
                        return {
                            combined_tags: nextCombinedTags,
                            showDeleteModal: false,
                            deletingCombinedTag: null,
                        };
                    });
                    console.log('Combined tag deleted successfully');
                } else {
                    alert('Delete failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error deleting combined tag:', error);
                alert('Network error, delete failed');
            });
    }

    handleToggleCombinationDropdown() {
        this.setState(prevState => ({
            combinationDropdownOpen: !prevState.combinationDropdownOpen,
            combinationSearchValue: !prevState.combinationDropdownOpen
                ? ''
                : prevState.combinationSearchValue,
        }));
    }

    handleCombinationTagToggle(tagName) {
        this.setState(prevState => {
            const isSelected = prevState.selectedTagsForCombination.includes(tagName);
            return {
                selectedTagsForCombination: isSelected
                    ? prevState.selectedTagsForCombination.filter(tag => tag !== tagName)
                    : [...prevState.selectedTagsForCombination, tagName],
            };
        });
    }

    handleRemoveCombinationTag(tagName) {
        this.setState(prevState => ({
            selectedTagsForCombination: prevState.selectedTagsForCombination.filter(
                tag => tag !== tagName
            ),
        }));
    }

    handleCombinationSearchChange(event) {
        this.setState({
            combinationSearchValue: event.target.value,
        });
    }

    render() {
        const availableTagsForCombination = this.state.tags
            .filter(tag => tag.name !== 'all' && !tag.neg_only)
            .map(tag => tag.name);

        return (
            <CombinedTagList
                combined_tags={this.state.combined_tags}
                onAddCombinedTag={this.handleAddCombinedTag}
                onEditCombinedTag={this.handleEditCombinedTag}
                onDeleteCombinedTag={this.handleDeleteCombinedTag}
                showAddEditModal={this.state.showAddEditModal}
                showDeleteModal={this.state.showDeleteModal}
                editingCombinedTag={this.state.editingCombinedTag}
                deletingCombinedTag={this.state.deletingCombinedTag}
                onCloseAddEditModal={this.handleCloseAddEditModal}
                onCloseDeleteModal={this.handleCloseDeleteModal}
                onSaveCombinedTag={this.handleSaveCombinedTag}
                onConfirmDelete={this.handleConfirmDelete}
                selectedTagsForCombination={this.state.selectedTagsForCombination}
                availableTagsForCombination={availableTagsForCombination}
                combinationDropdownOpen={this.state.combinationDropdownOpen}
                onToggleCombinationDropdown={this.handleToggleCombinationDropdown}
                onCombinationTagToggle={this.handleCombinationTagToggle}
                onRemoveCombinationTag={this.handleRemoveCombinationTag}
                combinationSearchValue={this.state.combinationSearchValue}
                onCombinationSearchChange={this.handleCombinationSearchChange}
            />
        );
    }
}

const Key = props => {
    const k = props.jkey;
    const kurl = buildKeywordUrl(k.name);
    const key_class =
        'rel_ukey' + (k.name === 'Artificial general intelligence' ? ' rel_ukey_all' : '');
    const isEditable = k.name !== 'Artificial general intelligence';

    return (
        <div class={key_class + ' enhanced-keyword'}>
            <a href={kurl} class="keyword-link">
                {k.name}
            </a>
            {isEditable && (
                <div class="keyword-actions">
                    <span class="keyword-edit" onClick={() => props.onEdit(k)} title="Edit keyword">
                        ✎
                    </span>
                    <span
                        class="keyword-delete"
                        onClick={() => props.onDelete(k)}
                        title="Delete keyword"
                    >
                        ×
                    </span>
                </div>
            )}
        </div>
    );
};

const KeyList = props => {
    const lst = props.keys;
    const klst = lst.map((jkey, ix) => (
        <Key key={ix} jkey={jkey} onEdit={props.onEditKey} onDelete={props.onDeleteKey} />
    ));

    return (
        <div class="enhanced-keyword-list">
            <div class="keyword-list-actions">
                <span class="tag-stats-inline">({lst.length} keywords)</span>
                <button
                    class="tag-action-btn add-btn"
                    onClick={props.onAddKey}
                    title="Add new keyword"
                >
                    + Add
                </button>
            </div>
            <div id="keyList" class="rel_utags enhanced-keywords">
                {klst}
            </div>

            {/* Add Keyword Modal */}
            {props.showAddModal && (
                <div class="modal-overlay" onClick={props.onCloseAddModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Add Keyword</h3>
                            <span class="modal-close" onClick={props.onCloseAddModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Keyword Name:</label>
                                <input
                                    type="text"
                                    value={props.newKeyName}
                                    onChange={props.onNewKeyNameChange}
                                    class="form-input"
                                    placeholder="Enter keyword name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseAddModal}>
                                Cancel
                            </button>
                            <button class="btn btn-primary" onClick={props.onSaveNewKey}>
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Edit Modal */}
            {props.showEditModal && (
                <div class="modal-overlay" onClick={props.onCloseEditModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Edit Keyword</h3>
                            <span class="modal-close" onClick={props.onCloseEditModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <div class="form-group">
                                <label>Keyword Name:</label>
                                <input
                                    type="text"
                                    value={props.editingKeyName}
                                    onChange={props.onEditingKeyNameChange}
                                    class="form-input"
                                    placeholder="Enter new keyword name"
                                />
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseEditModal}>
                                Cancel
                            </button>
                            <button class="btn btn-primary" onClick={props.onSaveKeyEdit}>
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {props.showDeleteModal && (
                <div class="modal-overlay" onClick={props.onCloseDeleteModal}>
                    <div class="modal-content" onClick={e => e.stopPropagation()}>
                        <div class="modal-header">
                            <h3>Confirm Delete</h3>
                            <span class="modal-close" onClick={props.onCloseDeleteModal}>
                                ×
                            </span>
                        </div>
                        <div class="modal-body">
                            <p>
                                Are you sure you want to delete keyword "
                                <strong>{props.deletingKey && props.deletingKey.name}</strong>"?
                            </p>
                            <p class="warning-text">
                                This action is irreversible, all data related to this keyword will
                                be deleted.
                            </p>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-cancel" onClick={props.onCloseDeleteModal}>
                                Cancel
                            </button>
                            <button class="btn btn-danger" onClick={props.onConfirmDelete}>
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

class KeyComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            keys: props.keys,
            showEditModal: false,
            showDeleteModal: false,
            showAddModal: false,
            editingKey: null,
            editingKeyName: '',
            deletingKey: null,
            newKeyName: '',
        };
        this.handleEditKey = this.handleEditKey.bind(this);
        this.handleDeleteKey = this.handleDeleteKey.bind(this);
        this.handleAddKey = this.handleAddKey.bind(this);
        this.handleCloseEditModal = this.handleCloseEditModal.bind(this);
        this.handleCloseDeleteModal = this.handleCloseDeleteModal.bind(this);
        this.handleCloseAddModal = this.handleCloseAddModal.bind(this);
        this.handleEditingKeyNameChange = this.handleEditingKeyNameChange.bind(this);
        this.handleNewKeyNameChange = this.handleNewKeyNameChange.bind(this);
        this.handleSaveKeyEdit = this.handleSaveKeyEdit.bind(this);
        this.handleSaveNewKey = this.handleSaveNewKey.bind(this);
        this.handleConfirmDelete = this.handleConfirmDelete.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
    }

    componentDidMount() {
        document.addEventListener('keydown', this.handleKeyDown);
    }

    componentDidUpdate(prevProps) {
        if (prevProps.keys !== this.props.keys) {
            this.setState({ keys: this.props.keys });
        }
    }

    componentWillUnmount() {
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (this.state.showEditModal) {
                this.handleCloseEditModal();
            }
            if (this.state.showDeleteModal) {
                this.handleCloseDeleteModal();
            }
            if (this.state.showAddModal) {
                this.handleCloseAddModal();
            }
        }
    }

    handleEditKey(key) {
        this.setState({
            showEditModal: true,
            editingKey: key,
            editingKeyName: key.name,
        });
    }

    handleDeleteKey(key) {
        this.setState({
            showDeleteModal: true,
            deletingKey: key,
        });
    }

    handleAddKey() {
        this.setState({
            showAddModal: true,
            newKeyName: '',
        });
    }

    handleCloseAddModal() {
        this.setState({
            showAddModal: false,
            newKeyName: '',
        });
    }

    handleNewKeyNameChange(event) {
        this.setState({ newKeyName: event.target.value });
    }

    handleSaveNewKey() {
        const { newKeyName } = this.state;
        if (!newKeyName.trim()) {
            alert('Keyword name cannot be empty');
            return;
        }

        const trimmedKey = newKeyName.trim();

        // Check if keyword already exists
        if (this.state.keys.some(key => key.name === trimmedKey)) {
            alert('Keyword already exists');
            return;
        }

        csrfFetch('/add_key/' + encodeURIComponent(trimmedKey))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState(prevState => {
                        const nextKeys = [...prevState.keys, { name: trimmedKey, pids: [] }];
                        setGlobalKeys(nextKeys, { renderKeys: false });
                        return {
                            keys: nextKeys,
                            showAddModal: false,
                            newKeyName: '',
                        };
                    });
                    console.log('Keyword added successfully');
                } else {
                    alert('Failed to add keyword: ' + text);
                }
            })
            .catch(error => {
                console.error('Error adding keyword:', error);
                alert('Network error, failed to add keyword');
            });
    }

    handleCloseEditModal() {
        this.setState({
            showEditModal: false,
            editingKey: null,
            editingKeyName: '',
        });
    }

    handleCloseDeleteModal() {
        this.setState({
            showDeleteModal: false,
            deletingKey: null,
        });
    }

    handleEditingKeyNameChange(event) {
        this.setState({ editingKeyName: event.target.value });
    }

    handleSaveKeyEdit() {
        const { editingKey, editingKeyName } = this.state;
        if (!editingKeyName.trim()) {
            alert('Keyword name cannot be empty');
            return;
        }

        const trimmedKeyName = editingKeyName.trim();

        // Check if new name already exists
        if (
            this.state.keys.some(key => key.name === trimmedKeyName && key.name !== editingKey.name)
        ) {
            alert('Keyword already exists');
            return;
        }

        csrfFetch(
            '/rename_key/' +
                encodeURIComponent(editingKey.name) +
                '/' +
                encodeURIComponent(trimmedKeyName)
        )
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState(prevState => {
                        const nextKeys = prevState.keys.map(key =>
                            key.name === editingKey.name ? { ...key, name: trimmedKeyName } : key
                        );
                        setGlobalKeys(nextKeys, { renderKeys: false });
                        return {
                            keys: nextKeys,
                            showEditModal: false,
                            editingKey: null,
                            editingKeyName: '',
                        };
                    });
                    console.log('Keyword renamed successfully');
                } else {
                    alert('Rename failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error renaming keyword:', error);
                alert('Rename failed: ' + error.message);
            });
    }

    handleConfirmDelete() {
        const { deletingKey } = this.state;

        csrfFetch('/del_key/' + encodeURIComponent(deletingKey.name))
            .then(response => response.text())
            .then(text => {
                if (text.startsWith('ok')) {
                    this.setState(prevState => {
                        const nextKeys = prevState.keys.filter(
                            key => key.name !== deletingKey.name
                        );
                        setGlobalKeys(nextKeys, { renderKeys: false });
                        return {
                            keys: nextKeys,
                            showDeleteModal: false,
                            deletingKey: null,
                        };
                    });
                    console.log('Keyword deleted successfully');
                } else {
                    alert('Delete failed: ' + text);
                }
            })
            .catch(error => {
                console.error('Error deleting keyword:', error);
                alert('Network error, delete failed');
            });
    }

    render() {
        return (
            <KeyList
                keys={this.state.keys}
                onEditKey={this.handleEditKey}
                onDeleteKey={this.handleDeleteKey}
                onAddKey={this.handleAddKey}
                showEditModal={this.state.showEditModal}
                showDeleteModal={this.state.showDeleteModal}
                showAddModal={this.state.showAddModal}
                editingKeyName={this.state.editingKeyName}
                newKeyName={this.state.newKeyName}
                deletingKey={this.state.deletingKey}
                onCloseEditModal={this.handleCloseEditModal}
                onCloseDeleteModal={this.handleCloseDeleteModal}
                onCloseAddModal={this.handleCloseAddModal}
                onEditingKeyNameChange={this.handleEditingKeyNameChange}
                onNewKeyNameChange={this.handleNewKeyNameChange}
                onSaveKeyEdit={this.handleSaveKeyEdit}
                onSaveNewKey={this.handleSaveNewKey}
                onConfirmDelete={this.handleConfirmDelete}
            />
        );
    }
}

function normalizeTags(list) {
    const base = (list || []).filter(tag => tag && tag.name && tag.name !== 'all');
    const normalized = base.map(tag => {
        const pos_n = tag.pos_n !== undefined ? Number(tag.pos_n || 0) : undefined;
        const neg_n = tag.neg_n !== undefined ? Number(tag.neg_n || 0) : undefined;
        const n = pos_n !== undefined && neg_n !== undefined ? pos_n + neg_n : Number(tag.n || 0);
        const neg_only =
            tag.neg_only !== undefined
                ? Boolean(tag.neg_only)
                : pos_n !== undefined && neg_n !== undefined
                  ? pos_n === 0 && neg_n > 0
                  : false;
        return {
            name: tag.name,
            n: n,
            pos_n: pos_n !== undefined ? pos_n : tag.pos_n || 0,
            neg_n: neg_n !== undefined ? neg_n : tag.neg_n || 0,
            neg_only: neg_only,
        };
    });
    if (normalized.length > 0) {
        normalized.push({ name: 'all' });
    }
    normalized.sort((a, b) => a.name.localeCompare(b.name));
    return normalized;
}

function normalizeCombinedTags(list) {
    const seen = new Set();
    const normalized = [];
    (list || []).forEach(tag => {
        const name = tag && tag.name ? tag.name : '';
        if (!name || seen.has(name)) return;
        seen.add(name);
        normalized.push({ name });
    });
    normalized.sort((a, b) => a.name.localeCompare(b.name));
    return normalized;
}

function renameCombinedTagsForTag(list, oldTag, newTag) {
    if (!oldTag || !newTag) return normalizeCombinedTags(list);
    const seen = new Set();
    const updated = [];
    (list || []).forEach(tag => {
        const name = tag && tag.name ? tag.name : '';
        if (!name) return;
        const parts = name
            .split(',')
            .map(t => t.trim())
            .filter(Boolean);
        if (parts.length === 0) return;
        const nextParts = parts.includes(oldTag)
            ? parts.map(t => (t === oldTag ? newTag : t))
            : parts;
        const nextName = nextParts.join(', ');
        if (seen.has(nextName)) return;
        seen.add(nextName);
        updated.push({ name: nextName });
    });
    return updated;
}

function removeCombinedTagsWithTag(list, tagName) {
    if (!tagName) return normalizeCombinedTags(list);
    const filtered = (list || []).filter(tag => {
        const name = tag && tag.name ? tag.name : '';
        const parts = name
            .split(',')
            .map(t => t.trim())
            .filter(Boolean);
        return !parts.includes(tagName);
    });
    return normalizeCombinedTags(filtered);
}

function renameCombinedTagInList(list, oldName, newName) {
    if (!oldName || !newName) return normalizeCombinedTags(list);
    const mapped = (list || []).map(tag => {
        if (tag && tag.name === oldName) {
            return { name: newName };
        }
        return tag;
    });
    return normalizeCombinedTags(mapped);
}

function setGlobalTags(nextTags, options) {
    const opts = options || {};
    tags = normalizeTags(nextTags);
    if (opts.renderTags !== false) {
        renderTagList();
    }
    if (opts.renderPaper !== false) {
        renderPaperList();
    }
    if (opts.renderCombined !== false) {
        renderCombinedTagList();
    }
}

function setGlobalCombinedTags(nextCombinedTags, options) {
    const opts = options || {};
    combined_tags = normalizeCombinedTags(nextCombinedTags);
    if (opts.renderCombined !== false) {
        renderCombinedTagList();
    }
}

function setGlobalKeys(nextKeys, options) {
    const opts = options || {};
    keys = Array.isArray(nextKeys) ? nextKeys.slice() : [];
    keys.sort((a, b) => a.name.localeCompare(b.name));
    if (opts.renderKeys !== false) {
        renderKeyList();
    }
}

function adjustTagStats(tagName, posDelta, negDelta) {
    if (!tagName || tagName === 'all') return;
    const posD = posDelta || 0;
    const negD = negDelta || 0;
    if (!posD && !negD) return;
    const baseTags = (tags || []).filter(tag => tag && tag.name && tag.name !== 'all');
    let found = false;
    let tagAdded = false;
    let tagRemoved = false;
    const updated = baseTags.map(tag => {
        if (tag.name !== tagName) return tag;
        found = true;
        const prevPos = tag.pos_n || 0;
        const prevNeg = tag.neg_n || 0;
        const nextPos = Math.max(0, prevPos + posD);
        const nextNeg = Math.max(0, prevNeg + negD);
        const nextCount = nextPos + nextNeg;
        if (prevPos + prevNeg > 0 && nextCount === 0) {
            tagRemoved = true;
        }
        return {
            ...tag,
            n: nextCount,
            pos_n: nextPos,
            neg_n: nextNeg,
            neg_only: nextPos === 0 && nextNeg > 0,
        };
    });
    if (!found && (posD > 0 || negD > 0)) {
        const nextPos = Math.max(0, posD);
        const nextNeg = Math.max(0, negD);
        updated.push({
            name: tagName,
            n: nextPos + nextNeg,
            pos_n: nextPos,
            neg_n: nextNeg,
            neg_only: nextPos === 0 && nextNeg > 0,
        });
        tagAdded = true;
    }
    setGlobalTags(updated, { renderPaper: tagAdded || tagRemoved });
}

// Global reading list state
let readingListPids = new Set();

function fetchReadingList() {
    if (!user) return Promise.resolve();

    return fetch('/api/readinglist/list', { credentials: 'same-origin' })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.items) {
                readingListPids = new Set(data.items.map(item => item.pid));
            }
        })
        .catch(err => {
            console.warn('Failed to fetch reading list:', err);
        });
}

function isInReadingList(pid) {
    return readingListPids.has(pid);
}

function addToReadingListCache(pid) {
    readingListPids.add(pid);
}

function removeFromReadingListCache(pid) {
    readingListPids.delete(pid);
}

// render papers into #wrap
// ReactDOM.render(<PaperList papers={papers} tags={tags} />, document.getElementById('wrap'));
const paperListRoot = document.getElementById('wrap');
const tagwrap_elt = document.getElementById('tagwrap');
const keywrap_elt = document.getElementById('keywrap');
const tagcombwrap_elt = document.getElementById('tagcombwrap');

function renderPaperList() {
    if (paperListRoot) {
        ReactDOM.render(<PaperList papers={papers} tags={tags} />, paperListRoot);
    }
}

function renderTagList() {
    if (tagwrap_elt) {
        ReactDOM.render(<TagListComponent tags={tags} />, tagwrap_elt);
    }
}

function renderKeyList() {
    if (keywrap_elt) {
        ReactDOM.render(<KeyComponent keys={keys} />, keywrap_elt);
    }
}

function renderCombinedTagList() {
    if (tagcombwrap_elt) {
        ReactDOM.render(
            <CombinedTagListComponent combined_tags={combined_tags} tags={tags} />,
            tagcombwrap_elt
        );
    }
}

// Fetch reading list first, then render all components
fetchReadingList().then(() => {
    renderPaperList();

    // render tags into #tagwrap, if it exists
    renderTagList();

    // render keys into #keywrap, if it exists
    renderKeyList();

    renderCombinedTagList();

    setupUserEventStream();

    if (Array.isArray(papers)) {
        papers.forEach(p => {
            if (!p || !p.id) return;
            if (p.summary_status === 'queued' || p.summary_status === 'running') {
                markSummaryPending(p.id);
            }
        });
    }
});
