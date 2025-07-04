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

// 先 MathJax 後 Marked 的渲染函數（簡化版）
function renderMarkdownWithMath(text, container) {
    if (!text || !container) return Promise.resolve();

    return new Promise((resolve) => {
        // 創建臨時元素用於渲染
        const tempDiv = document.createElement('div');
        tempDiv.style.display = 'none';
        tempDiv.innerHTML = text;
        document.body.appendChild(tempDiv);

        // 等待 MathJax 完全加載
        if (typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.document) {
            // 第一步：MathJax 渲染數學公式
            MathJax.startup.promise.then(() => {
                return MathJax.typesetPromise([tempDiv]);
            }).then(() => {
                // 第二步：使用 Marked 處理 Markdown
                let htmlContent;
                if (typeof marked !== 'undefined') {
                    htmlContent = marked.parse(tempDiv.innerHTML);
                } else {
                    // 後備方案
                    htmlContent = tempDiv.innerHTML
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.*?)\*/g, '<em>$1</em>')
                        .replace(/`(.*?)`/g, '<code>$1</code>')
                        .replace(/\n/g, '<br>');
                }

                container.innerHTML = htmlContent;

                // 第三步：重新渲染 MathJax
                return MathJax.typesetPromise([container]);
            }).then(() => {
                document.body.removeChild(tempDiv);
                resolve();
            }).catch((err) => {
                console.warn('MathJax rendering error:', err);
                // 發生錯誤時使用簡單渲染
                container.innerHTML = parseMarkdownWithMath(text);
                if (tempDiv.parentNode) {
                    document.body.removeChild(tempDiv);
                }
                resolve();
            });
        } else {
            // MathJax 未加載時直接使用 Markdown 解析
            container.innerHTML = parseMarkdownWithMath(text);
            document.body.removeChild(tempDiv);
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
