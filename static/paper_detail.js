'use strict';

// Use shared utilities from common_utils.js
var CommonUtils = window.ArxivSanityCommon;
var renderAbstractMarkdown = CommonUtils.renderAbstractMarkdown;
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
        if (p.summary) {
            triggerMathJax(document.getElementById('wrap'));
        }
    }, [p.summary]);

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
            <div
                class="rel_abs"
                dangerouslySetInnerHTML={{ __html: renderAbstractMarkdown(p.summary) }}
            ></div>
        </div>
    );
};

ReactDOM.render(<PaperLite paper={paper} />, document.getElementById('wrap'));
