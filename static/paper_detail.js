'use strict';

// Use shared utilities from common_utils.js
var CommonUtils = window.ArxivSanityCommon;
var renderAbstractMarkdown = CommonUtils.renderAbstractMarkdown;
var renderTldrMarkdown = CommonUtils.renderTldrMarkdown;
var triggerMathJax = CommonUtils.triggerMathJax;

const PaperLite = props => {
    const p = props.paper;

    const formatAuthorsText = (authorsText, options = {}) => {
        if (
            typeof window !== 'undefined' &&
            window.ArxivSanityAuthors &&
            window.ArxivSanityAuthors.format
        ) {
            return window.ArxivSanityAuthors.format(authorsText, options).text;
        }
        return String(authorsText || '');
    };

    // Trigger MathJax after component mounts
    React.useEffect(() => {
        if (p.summary || p.tldr) {
            triggerMathJax(document.getElementById('wrap'));
        }
    }, [p.summary, p.tldr]);

    const has_tldr = Boolean(p.tldr && String(p.tldr).trim());

    const tldr_section = has_tldr ? (
        <div class="rel_tldr">
            <div class="tldr_label">💡 TL;DR</div>
            <div
                class="tldr_text"
                dangerouslySetInnerHTML={{ __html: renderTldrMarkdown(p.tldr) }}
            ></div>
        </div>
    ) : null;

    const abstract_section = has_tldr ? (
        <details
            class="rel_abs_details"
            onToggle={e => {
                try {
                    if (e && e.target && e.target.open) {
                        triggerMathJax(e.target);
                    }
                } catch (err) {}
            }}
        >
            <summary class="rel_abs_summary">Abstract</summary>
            <div
                class="rel_abs"
                dangerouslySetInnerHTML={{ __html: renderAbstractMarkdown(p.summary) }}
            ></div>
        </details>
    ) : (
        <div
            class="rel_abs"
            dangerouslySetInnerHTML={{ __html: renderAbstractMarkdown(p.summary) }}
        ></div>
    );

    return (
        <div class="rel_paper">
            <div class="rel_title">
                <a href={'http://arxiv.org/abs/' + p.id}>{p.title}</a>
            </div>
            <div class="rel_authors" title={p.authors || ''}>
                {formatAuthorsText(p.authors, { maxAuthors: 10, head: 5, tail: 3 })}
            </div>
            <div class="rel_time">{p.time}</div>
            <div class="rel_tags">{p.tags}</div>
            {tldr_section}
            {abstract_section}
        </div>
    );
};

ReactDOM.render(<PaperLite paper={paper} />, document.getElementById('wrap'));
