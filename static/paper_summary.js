'use strict';

// 保護 script 標籤不被 marked 修改（用於保護 MathJax 生成的公式代碼）
function preserveScripts(text) {
    const scripts = [];
    const placeholder = text.replace(/<script[^>]*>([\s\S]*?)<\/script>/gi, (match) => {
        scripts.push(match);
        return `__SCRIPT_PLACEHOLDER_${scripts.length - 1}__`;
    });
    return { text: placeholder, scripts };
}

// 恢復被保護的 script 標籤
function restoreScripts(text, scripts) {
    return text.replace(/__SCRIPT_PLACEHOLDER_(\d+)__/g, (match, index) => {
        return scripts[parseInt(index)] || match;
    });
}

// 解決 MathJax 與 Marked 衝突的 Markdown 渲染函數
function parseMarkdownWithMath(text) {
    if (!text) return '';

    // 如果 marked 可用，使用完整的 markdown 解析
    if (typeof marked !== 'undefined') {
        const { text: protectedText, scripts } = preserveScripts(text);
        const parsedText = marked.parse(protectedText);
        return restoreScripts(parsedText, scripts);
    } else {
        // 後備方案：使用簡單的文本處理
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // 加粗
            .replace(/\*(.*?)\*/g, '<em>$1</em>')              // 斜體
            .replace(/`(.*?)`/g, '<code>$1</code>')            // 行內代碼
            .replace(/\n/g, '<br>');                           // 換行
    }
}

// 先 Marked 後 MathJax 的渲染函數（完全重寫版）
function renderMarkdownWithMath(text, container) {
    if (!text || !container) return Promise.resolve();

    return new Promise((resolve) => {
        try {
            console.log('Starting markdown rendering, text length:', text.length);

            let htmlContent = '';

            // 檢查 marked 是否可用
            if (typeof marked !== 'undefined') {
                // 配置 marked 選項
                if (marked.setOptions) {
                    marked.setOptions({
                        breaks: true,
                        gfm: true,
                        headerIds: false,
                        mangle: false,
                        sanitize: false,
                        pedantic: false,
                        smartLists: true,
                        smartypants: false,
                        xhtml: false
                    });
                }

                // 保存所有需要保護的內容
                const protectedContent = {
                    mathBlocks: [],
                    mathInlines: [],
                    codeBlocks: []
                };

                // 第一步：保護所有特殊內容
                let processedText = text;

                // 1. 保護代碼塊
                processedText = processedText.replace(/```[\s\S]*?```/g, (match) => {
                    const index = protectedContent.codeBlocks.length;
                    protectedContent.codeBlocks.push(match);
                    return `\n\nCODEBLOCK${index}CODEBLOCK\n\n`;
                });

                // 2. 保護塊級數學公式
                processedText = processedText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
                    const index = protectedContent.mathBlocks.length;
                    protectedContent.mathBlocks.push(match);
                    return `MATHBLOCK${index}MATHBLOCK`;
                });

                // 3. 保護行內數學公式（兼容性更好的版本）
                // 避免使用 lookbehind，因為某些瀏覽器不支持
                processedText = processedText.replace(/\$([^\$\n]+?)\$/g, (match, formula, offset, string) => {
                    // 檢查前後是否是 $（避免匹配 $$）
                    const prevChar = offset > 0 ? string[offset - 1] : '';
                    const nextChar = offset + match.length < string.length ? string[offset + match.length] : '';

                    if (prevChar === '$' || nextChar === '$') {
                        return match; // 這是 $$ 的一部分，不處理
                    }

                    // 檢查是否包含數學符號特徵
                    if (formula.match(/[a-zA-Z_\\{}\^\(\)\[\]]/)) {
                        const index = protectedContent.mathInlines.length;
                        protectedContent.mathInlines.push(match);
                        return `MATHINLINE${index}MATHINLINE`;
                    }
                    return match;
                });

                console.log('Protected content counts:', {
                    codeBlocks: protectedContent.codeBlocks.length,
                    mathBlocks: protectedContent.mathBlocks.length,
                    mathInlines: protectedContent.mathInlines.length
                });

                // 第二步：處理 Markdown
                try {
                    // 嘗試直接解析
                    htmlContent = marked.parse(processedText);
                    console.log('Marked parsing successful, output length:', htmlContent.length);
                } catch (parseError) {
                    console.error('Marked parsing failed:', parseError);
                    // 如果失敗，嘗試分段處理
                    const segments = processedText.split(/\n\n+/);
                    const processedSegments = [];

                    for (let segment of segments) {
                        try {
                            processedSegments.push(marked.parse(segment));
                        } catch (segmentError) {
                            console.warn('Segment parsing failed, using fallback:', segmentError);
                            // 對失敗的段落使用簡單處理
                            processedSegments.push(
                                segment
                                    .replace(/&/g, '&amp;')
                                    .replace(/</g, '&lt;')
                                    .replace(/>/g, '&gt;')
                                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                                    .replace(/`(.*?)`/g, '<code>$1</code>')
                            );
                        }
                    }

                    htmlContent = processedSegments.join('</p><p>');
                    // 確保段落標籤正確
                    if (!htmlContent.startsWith('<p>')) {
                        htmlContent = '<p>' + htmlContent;
                    }
                    if (!htmlContent.endsWith('</p>')) {
                        htmlContent = htmlContent + '</p>';
                    }
                }

                // 第三步：恢復保護的內容
                // 注意：不要全局替換HTML實體，只在占位符周圍處理

                // 恢復代碼塊
                htmlContent = htmlContent.replace(/CODEBLOCK(\d+)CODEBLOCK/g, (match, index) => {
                    const code = protectedContent.codeBlocks[parseInt(index)];
                    if (code) {
                        // 更靈活的代碼塊匹配
                        const codeMatch = code.match(/^```([^\n]*)\n?([\s\S]*?)\n?```$/);
                        if (codeMatch) {
                            const lang = (codeMatch[1] || '').trim();
                            const content = codeMatch[2] || '';
                            const escapedContent = content
                                .replace(/&/g, '&amp;')
                                .replace(/</g, '&lt;')
                                .replace(/>/g, '&gt;');
                            return `<pre><code${lang ? ` class="language-${lang}"` : ''}>${escapedContent}</code></pre>`;
                        } else {
                            // 如果正則匹配失敗，直接返回原始代碼塊（去掉```）
                            const simpleContent = code.replace(/^```[^\n]*\n?/, '').replace(/\n?```$/, '');
                            const escapedContent = simpleContent
                                .replace(/&/g, '&amp;')
                                .replace(/</g, '&lt;')
                                .replace(/>/g, '&gt;');
                            return `<pre><code>${escapedContent}</code></pre>`;
                        }
                    }
                    return match;
                });

                // 恢復數學公式 - 處理可能被包裹在標籤中或被HTML編碼的情況
                // 先處理被<p>標籤包裹的塊級數學公式
                htmlContent = htmlContent.replace(/<p>MATHBLOCK(\d+)MATHBLOCK<\/p>/g, (match, index) => {
                    const math = protectedContent.mathBlocks[parseInt(index)];
                    return math ? `<p>${math}</p>` : match;
                });

                // 處理普通的塊級數學公式
                htmlContent = htmlContent.replace(/MATHBLOCK(\d+)MATHBLOCK/g, (match, index) => {
                    return protectedContent.mathBlocks[parseInt(index)] || match;
                });

                // 處理行內數學公式
                htmlContent = htmlContent.replace(/MATHINLINE(\d+)MATHINLINE/g, (match, index) => {
                    return protectedContent.mathInlines[parseInt(index)] || match;
                });

                // 如果還有被HTML編碼的占位符，嘗試恢復
                if (htmlContent.includes('&lt;') || htmlContent.includes('&gt;') || htmlContent.includes('&amp;')) {
                    // 只在占位符區域進行HTML解碼
                    htmlContent = htmlContent.replace(/(&lt;|&gt;|&amp;)?(CODEBLOCK|MATHBLOCK|MATHINLINE)(\d+)\2(&lt;|&gt;|&amp;)?/g,
                        (match, pre, type, index, post) => {
                            const realType = type;
                            const realIndex = parseInt(index);

                            if (realType === 'CODEBLOCK' && protectedContent.codeBlocks[realIndex]) {
                                const code = protectedContent.codeBlocks[realIndex];
                                const codeMatch = code.match(/^```([^\n]*)\n?([\s\S]*?)\n?```$/);
                                if (codeMatch) {
                                    const lang = (codeMatch[1] || '').trim();
                                    const content = codeMatch[2] || '';
                                    const escapedContent = content
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;');
                                    return `<pre><code${lang ? ` class="language-${lang}"` : ''}>${escapedContent}</code></pre>`;
                                }
                            } else if (realType === 'MATHBLOCK' && protectedContent.mathBlocks[realIndex]) {
                                return protectedContent.mathBlocks[realIndex];
                            } else if (realType === 'MATHINLINE' && protectedContent.mathInlines[realIndex]) {
                                return protectedContent.mathInlines[realIndex];
                            }

                            return match;
                        }
                    );
                }

                // 檢查恢復情況 - 更準確的檢查
                const codeBlocksRemaining = (htmlContent.match(/CODEBLOCK\d+CODEBLOCK/g) || []).length;
                const mathBlocksRemaining = (htmlContent.match(/MATHBLOCK\d+MATHBLOCK/g) || []).length;
                const mathInlinesRemaining = (htmlContent.match(/MATHINLINE\d+MATHINLINE/g) || []).length;
                const totalRemaining = codeBlocksRemaining + mathBlocksRemaining + mathInlinesRemaining;

                if (totalRemaining > 0) {
                    console.warn(`Warning: ${totalRemaining} placeholders were not restored:`, {
                        codeBlocks: codeBlocksRemaining,
                        mathBlocks: mathBlocksRemaining,
                        mathInlines: mathInlinesRemaining
                    });
                }

            } else {
                // marked 不可用時的後備方案
                console.log('Marked not available, using fallback');
                htmlContent = text
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`(.*?)`/g, '<code>$1</code>')
                    .replace(/\n\n+/g, '</p><p>')
                    .replace(/\n/g, '<br>');
                htmlContent = '<p>' + htmlContent + '</p>';
            }

            console.log('Final HTML length:', htmlContent.length);

            // 設置 HTML 內容
            container.innerHTML = htmlContent;

            // 第四步：渲染數學公式
            if (typeof MathJax !== 'undefined' && MathJax.startup) {
                MathJax.startup.promise.then(() => {
                    console.log('Starting MathJax rendering');
                    return MathJax.typesetPromise([container]);
                }).then(() => {
                    console.log('MathJax rendering completed');
                    resolve();
                }).catch((err) => {
                    console.warn('MathJax rendering error:', err);
                    resolve();
                });
            } else {
                console.log('MathJax not available');
                resolve();
            }

        } catch (err) {
            console.error('Critical rendering error:', err);
            console.error('Error stack:', err.stack);

            // 最後的後備方案：顯示原始文本
            const lines = text.split('\n');
            const escapedLines = lines.map(line => {
                return line
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
            });
            container.innerHTML = '<pre>' + escapedLines.join('\n') + '</pre>';
            resolve();
        }
    });
}

// Simplified state management
class SummaryState {
    constructor() {
        this.loading = false;
        this.error = null;
        this.content = null;
        this.paper = null;
        this.pid = null;
    }

    setState(newState) {
        Object.assign(this, newState);
        this.render();
    }

    render() {
        const container = document.getElementById('wrap');
        if (!container) return;

        const htmlContent = this.getHTML();

        // 如果內容包含 summary，使用新的渲染方式處理數學公式
        if (this.content && htmlContent.includes('markdown-content')) {
            // 先設置基本 HTML 結構
            container.innerHTML = htmlContent;

            // 然後對 markdown 內容進行特殊處理
            const markdownContainer = container.querySelector('.markdown-content');
            if (markdownContainer) {
                renderMarkdownWithMath(this.content, markdownContainer);
            }
        } else {
            // 對於不含 markdown 的內容，直接設置
            container.innerHTML = htmlContent;
            this.renderMath();
        }
    }

    renderMath() {
        // 使用 MathJax 渲染數學公式（用於非 markdown 內容）
        if (typeof MathJax !== 'undefined') {
            setTimeout(() => {
                if (MathJax.typeset) {
                    MathJax.typeset();
                } else if (MathJax.typesetPromise) {
                    MathJax.typesetPromise().catch((err) => {
                        console.warn('MathJax rendering error:', err);
                    });
                }
            }, 100);
        }
    }

    getHTML() {
        // Paper header section - styled like paper list with full abstract
        const headerHTML = this.paper ? `
            <div class="paper-header">
                <div class="paper-nav paper-actions-footer">
                    <div class="rel_more"><a href="/?rank=pid&pid=${encodeURIComponent(this.paper.id)}" target="_blank" rel="noopener noreferrer">Similar</a></div>
                    <div class="rel_inspect"><a href="/inspect?pid=${encodeURIComponent(this.paper.id)}" target="_blank" rel="noopener noreferrer">Inspect</a></div>
                    <div class="rel_alphaxiv"><a href="https://www.alphaxiv.org/overview/${encodeURIComponent(this.paper.id)}" target="_blank" rel="noopener noreferrer">alphaXiv</a></div>
                    <div class="rel_cool"><a href="https://papers.cool/arxiv/${encodeURIComponent(this.paper.id)}" target="_blank" rel="noopener noreferrer">Cool</a></div>
                </div>
                <div class="paper-content-section">
                    <div class="paper-main">
                        <h1 class="paper-title">
                            <a href="http://arxiv.org/abs/${this.paper.id}" target="_blank" rel="noopener noreferrer">
                                ${this.paper.title}
                            </a>
                        </h1>
                        <div class="paper-authors-line">
                            ${this.paper.authors}
                        </div>
                        <div class="paper-meta-line">
                            <span class="paper-time">${this.paper.time}</span>
                            ${this.paper.tags ? `<span class="paper-tags">${this.paper.tags}</span>` : ''}
                        </div>
                        <div class="paper-abstract">
                            ${this.paper.summary || 'No abstract available.'}
                        </div>
                    </div>
                </div>
            </div>
        ` : '';

        // Summary content section with logo
        let summaryHTML = '';
        if (this.loading) {
            summaryHTML = `
                <div class="summary-container">
                    <div class="summary-header">
                        <h2>AI Summary</h2>
                    </div>
                    <div class="summary-content">
                        <div class="loading-indicator">
                            <div class="spinner"></div>
                            <p>Generating paper summary, please wait...</p>
                            <p class="loading-note">This may take a few moments</p>
                        </div>
                    </div>
                </div>
            `;
        } else if (this.error) {
            summaryHTML = `
                <div class="summary-container">
                    <div class="summary-header">
                        <h2>AI Summary</h2>
                    </div>
                    <div class="summary-content">
                        <div class="error-message">
                            <p>⚠️ Error generating summary: ${this.error}</p>
                            <button onclick="summaryApp.retry()" class="retry-button">
                                Retry
                            </button>
                        </div>
                    </div>
                </div>
            `;
        } else if (this.content) {
            // 不在這裡處理 markdown，讓 render() 方法中的 renderMarkdownWithMath 函數處理
            summaryHTML = `
                <div class="summary-container">
                    <div class="summary-header">
                        <h2>AI Summary</h2>
                        <div class="summary-meta">
                            <span class="summary-badge">AI Generated</span>
                            <span class="summary-note">This summary is automatically generated by AI</span>
                        </div>
                    </div>
                    <div class="summary-content markdown-content">
                        <!-- 內容將由 renderMarkdownWithMath 函數處理 -->
                    </div>
                </div>
            `;
        } else {
            summaryHTML = `
                <div class="summary-container">
                    <div class="summary-header">
                        <h2>AI Summary</h2>
                    </div>
                    <div class="summary-content">
                        <p>No summary available for this paper at the moment.</p>
                    </div>
                </div>
            `;
        }

        return headerHTML + summaryHTML;
    }
}

// Global app instance
const summaryApp = new SummaryState();

// API call function
async function fetchSummary(pid) {
    try {
        // 創建 AbortController 來控制超時
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5分鐘超時

        const response = await fetch('/api/get_paper_summary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pid: pid }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.success) {
            return data.summary_content;
        } else {
            throw new Error(data.error || 'Unknown error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('請求超時，論文總結過程較長，請稍後重試');
        }
        console.error('Failed to fetch summary:', error);
        throw error;
    }
}

// Retry function
summaryApp.retry = function() {
    if (this.pid) {
        this.loadSummary(this.pid);
    }
};

// Load summary function
summaryApp.loadSummary = async function(pid) {
    this.setState({ loading: true, error: null });

    try {
        const content = await fetchSummary(pid);
        this.setState({ loading: false, content: content });
    } catch (error) {
        this.setState({ loading: false, error: error.message });
    }
};

// Initialize app
function initSummaryApp() {
    console.log('Initializing summary page...');

    // Check required variables
    if (typeof paper === 'undefined') {
        console.error('Paper data is missing!');
        document.getElementById('wrap').innerHTML =
            '<div style="padding: 20px; color: red;">Error: Paper data is missing</div>';
        return;
    }

    if (typeof pid === 'undefined') {
        console.error('Paper ID is missing!');
        document.getElementById('wrap').innerHTML =
            '<div style="padding: 20px; color: red;">Error: Paper ID is missing</div>';
        return;
    }

    // Set initial state
    summaryApp.setState({
        paper: paper,
        pid: pid,
        loading: true
    });

    // Start loading summary
    summaryApp.loadSummary(pid);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initSummaryApp);
