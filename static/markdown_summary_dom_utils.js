'use strict';

// Shared summary markdown DOM utilities
// Exposes: window.ArxivSanitySummaryMarkdownDom

(function (global) {
    const NS = 'ArxivSanitySummaryMarkdownDom';
    const CommonUtils = global.ArxivSanityCommon;
    const escapeHtml = CommonUtils.escapeHtml;

    let tocObserver = null;
    let tocCollapsed = null;

    function slugifyHeading(text, slugCounts) {
        const cleaned = String(text || '')
            .trim()
            .toLowerCase()
            .replace(/[!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~]/g, '')
            .replace(/\s+/g, '-');
        const base = cleaned || 'section';
        const count = slugCounts[base] || 0;
        slugCounts[base] = count + 1;
        return count > 0 ? `${base}-${count + 1}` : base;
    }

    function extractHeadingText(token) {
        if (!token) return '';
        if (token.type === 'inline' && Array.isArray(token.children)) {
            return token.children
                .map(child => {
                    if (child.type === 'text' || child.type === 'code_inline') {
                        return child.content;
                    }
                    return '';
                })
                .join('');
        }
        return token.content || '';
    }

    function buildTocHtml(items) {
        if (!items || items.length < 2) return '';
        const list = items
            .map(item => {
                const title = escapeHtml(item.title);
                const slug = escapeHtml(item.slug);
                return `<li class="toc-item toc-level-${item.level}"><a href="#${slug}">${title}</a></li>`;
            })
            .join('');
        return `
        <div class="toc-header">
            <div class="toc-title">Contents</div>
            <div class="toc-actions">
                <span class="toc-count">${items.length}</span>
                <button type="button" class="toc-toggle" aria-expanded="true">Collapse</button>
            </div>
        </div>
        <ul class="toc-list">
            ${list}
        </ul>
        <div class="toc-footer">
            <button type="button" class="toc-back-top" title="Back to top" aria-label="Back to top">↑ Back to top</button>
        </div>
    `;
    }

    function setupTocBackToTop(tocContainer) {
        if (!tocContainer) return;
        const btn = tocContainer.querySelector('.toc-back-top');
        if (!btn) return;

        // Ensure legacy floating back-to-top is removed once TOC exists
        try {
            const existing = document.querySelector('.back-to-top');
            if (existing) existing.remove();
        } catch (e) {}

        const onClick = evt => {
            if (evt) evt.preventDefault();

            // Close mobile drawer if open
            try {
                if (typeof document !== 'undefined') {
                    document.body.classList.remove('toc-drawer-open');
                }
            } catch (e) {}

            try {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } catch (e) {
                // Fallback
                try {
                    window.scrollTo(0, 0);
                } catch (e2) {}
            }
        };

        // Remove old handler if exists (store reference on element to enable proper removal)
        if (btn._tocBackTopClick) {
            btn.removeEventListener('click', btn._tocBackTopClick);
        }
        btn._tocBackTopClick = onClick;
        btn.addEventListener('click', onClick);
    }

    function setActiveTocLink(tocContainer, link) {
        if (!tocContainer) return;
        const active = tocContainer.querySelector('.toc-item a.is-active');
        if (active) {
            active.classList.remove('is-active');
        }
        if (link) {
            link.classList.add('is-active');
        }
    }

    function setupTocObserver(tocContainer, markdownContainer) {
        if (tocObserver) {
            tocObserver.disconnect();
            tocObserver = null;
        }
        if (!tocContainer || !markdownContainer) return;

        const headings = markdownContainer.querySelectorAll('h1, h2, h3, h4');
        if (!headings.length) return;

        const linkMap = new Map();
        const links = tocContainer.querySelectorAll('a[href^="#"]');
        links.forEach(link => {
            const href = link.getAttribute('href') || '';
            const id = href.slice(1);
            if (id) {
                linkMap.set(id, link);
            }
        });

        tocObserver = new IntersectionObserver(
            entries => {
                const visible = entries.filter(entry => entry.isIntersecting);
                if (!visible.length) return;
                visible.sort((a, b) => {
                    if (b.intersectionRatio !== a.intersectionRatio) {
                        return b.intersectionRatio - a.intersectionRatio;
                    }
                    return a.boundingClientRect.top - b.boundingClientRect.top;
                });
                const target = visible[0].target;
                const link = linkMap.get(target.id);
                setActiveTocLink(tocContainer, link);
            },
            {
                rootMargin: '0px 0px -70% 0px',
                threshold: [0, 1],
            }
        );

        headings.forEach(heading => {
            tocObserver.observe(heading);
        });
    }

    function setupTocToggle(tocContainer) {
        if (!tocContainer) return;
        const toggle = tocContainer.querySelector('.toc-toggle');
        if (!toggle) return;

        // Mobile: use sticky TOC instead of drawer (same as desktop but with different styling)
        // No longer using bottom-drawer mode

        if (tocCollapsed === null) {
            // Default to collapsed on mobile/tablet, expanded on desktop
            tocCollapsed = window.matchMedia('(max-width: 960px)').matches;
        }

        const applyState = collapsed => {
            tocContainer.classList.toggle('is-collapsed', collapsed);
            toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            toggle.textContent = collapsed ? 'Expand' : 'Collapse';
        };

        applyState(tocCollapsed);

        toggle.addEventListener('click', () => {
            tocCollapsed = !tocCollapsed;
            applyState(tocCollapsed);
        });
    }

    function wrapMarkdownTables(container) {
        const tables = container.querySelectorAll('table');
        tables.forEach(table => {
            const parent = table.parentElement;
            if (parent && parent.classList.contains('table-wrap')) return;
            const wrapper = document.createElement('div');
            wrapper.className = 'table-wrap';
            if (parent) {
                parent.insertBefore(wrapper, table);
            }
            wrapper.appendChild(table);

            // Add table toolbar
            const toolbar = document.createElement('div');
            toolbar.className = 'table-toolbar';
            toolbar.innerHTML = `
                <button type="button" class="table-btn table-copy-btn" title="Copy Table">
                    <span class="table-btn-icon">📋</span> Copy
                </button>
                <button type="button" class="table-btn table-csv-btn" title="Export CSV">
                    <span class="table-btn-icon">📊</span> CSV
                </button>
                <button type="button" class="table-btn table-sort-btn" title="Enable Sort">
                    <span class="table-btn-icon">↕️</span> Sort
                </button>
            `;
            wrapper.insertBefore(toolbar, table);

            // Setup table actions
            setupTableActions(wrapper, table);
        });
    }

    /**
     * Setup table action buttons (copy, CSV export, sorting)
     */
    function setupTableActions(wrapper, table) {
        const copyBtn = wrapper.querySelector('.table-copy-btn');
        const csvBtn = wrapper.querySelector('.table-csv-btn');
        const sortBtn = wrapper.querySelector('.table-sort-btn');

        // Copy table as Markdown
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const markdown = tableToMarkdown(table);
                const doCopy =
                    CommonUtils && typeof CommonUtils.copyTextToClipboard === 'function'
                        ? CommonUtils.copyTextToClipboard
                        : null;

                if (!doCopy) {
                    showTableToast(wrapper, 'Copy not supported', true);
                    return;
                }

                doCopy(markdown).then(ok => {
                    showTableToast(wrapper, ok ? 'Markdown copied!' : 'Copy failed', !ok);
                });
            });
        }

        // Export as CSV
        if (csvBtn) {
            csvBtn.addEventListener('click', () => {
                const csv = tableToCSV(table);
                downloadFile(csv, 'table.csv', 'text/csv');
                showTableToast(wrapper, 'CSV downloaded!');
            });
        }

        // Enable sorting
        if (sortBtn) {
            sortBtn.addEventListener('click', () => {
                const isEnabled = table.classList.toggle('sortable');
                sortBtn.classList.toggle('active', isEnabled);
                if (isEnabled) {
                    enableTableSorting(table);
                    showTableToast(wrapper, 'Click header to sort');
                } else {
                    disableTableSorting(table);
                }
            });
        }
    }

    /**
     * Convert table to Markdown format
     */
    function tableToMarkdown(table) {
        const rows = [];
        const allRows = table.querySelectorAll('tr');
        let headerProcessed = false;

        allRows.forEach((tr, rowIndex) => {
            const cells = [];
            const isHeader = tr.querySelector('th') !== null;

            tr.querySelectorAll('th, td').forEach(cell => {
                // Escape pipe characters in cell content
                let text = cell.textContent.trim().replace(/\|/g, '\\|');
                cells.push(text);
            });

            if (cells.length > 0) {
                rows.push('| ' + cells.join(' | ') + ' |');

                // Add separator after header row
                if (isHeader && !headerProcessed) {
                    const separator = cells.map(() => '---').join(' | ');
                    rows.push('| ' + separator + ' |');
                    headerProcessed = true;
                }
            }
        });

        // If no header was found, add separator after first row
        if (!headerProcessed && rows.length > 0) {
            const firstRowCells = rows[0].split('|').filter(c => c.trim()).length;
            const separator = Array(firstRowCells).fill('---').join(' | ');
            rows.splice(1, 0, '| ' + separator + ' |');
        }

        return rows.join('\n');
    }

    /**
     * Convert table to plain text (tab-separated)
     */
    function tableToText(table) {
        const rows = [];
        table.querySelectorAll('tr').forEach(tr => {
            const cells = [];
            tr.querySelectorAll('th, td').forEach(cell => {
                cells.push(cell.textContent.trim());
            });
            rows.push(cells.join('\t'));
        });
        return rows.join('\n');
    }

    /**
     * Convert table to CSV format
     * Includes protection against CSV formula injection (OWASP)
     */
    function tableToCSV(table) {
        const rows = [];
        table.querySelectorAll('tr').forEach(tr => {
            const cells = [];
            tr.querySelectorAll('th, td').forEach(cell => {
                let text = cell.textContent.trim();

                // CSV formula injection protection: prefix dangerous characters
                // Characters =, +, -, @, \t, \r can trigger formula execution in Excel/Sheets
                if (/^[=+\-@\t\r]/.test(text)) {
                    text = "'" + text; // Prefix with single quote to prevent formula interpretation
                }

                // Escape quotes and wrap in quotes if contains comma, quote, or newline
                if (text.includes('"') || text.includes(',') || text.includes('\n')) {
                    text = '"' + text.replace(/"/g, '""') + '"';
                }
                cells.push(text);
            });
            rows.push(cells.join(','));
        });
        return rows.join('\n');
    }

    /**
     * Download file helper
     */
    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Show toast message near table
     */
    function showTableToast(wrapper, message, isError = false) {
        const existing = wrapper.querySelector('.table-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'table-toast' + (isError ? ' error' : '');
        toast.textContent = message;
        wrapper.appendChild(toast);

        setTimeout(() => toast.remove(), 2000);
    }

    /**
     * Enable table sorting
     */
    function enableTableSorting(table) {
        const headers = table.querySelectorAll('thead th, tr:first-child th');
        headers.forEach((th, index) => {
            if (th.dataset.sortable === 'false') return;
            th.classList.add('sortable-header');
            th.dataset.sortDir = '';
            th.addEventListener('click', handleHeaderClick);
        });
    }

    /**
     * Disable table sorting
     */
    function disableTableSorting(table) {
        const headers = table.querySelectorAll('.sortable-header');
        headers.forEach(th => {
            th.classList.remove('sortable-header', 'sort-asc', 'sort-desc');
            th.removeEventListener('click', handleHeaderClick);
            delete th.dataset.sortDir;
        });
    }

    /**
     * Handle header click for sorting
     */
    function handleHeaderClick(event) {
        const th = event.currentTarget;
        const table = th.closest('table');
        const tbody = table.querySelector('tbody') || table;
        const headerRow = th.parentElement;
        const columnIndex = Array.from(headerRow.children).indexOf(th);

        // Determine sort direction
        const currentDir = th.dataset.sortDir;
        const newDir = currentDir === 'asc' ? 'desc' : 'asc';

        // Reset other headers
        headerRow.querySelectorAll('.sortable-header').forEach(h => {
            h.classList.remove('sort-asc', 'sort-desc');
            h.dataset.sortDir = '';
        });

        th.dataset.sortDir = newDir;
        th.classList.add('sort-' + newDir);

        // Get rows to sort (skip header row)
        const rows = Array.from(tbody.querySelectorAll('tr')).filter(
            row => row !== headerRow && row.querySelector('td')
        );

        // Sort rows
        rows.sort((a, b) => {
            const aCell = a.children[columnIndex];
            const bCell = b.children[columnIndex];
            if (!aCell || !bCell) return 0;

            let aVal = aCell.textContent.trim();
            let bVal = bCell.textContent.trim();

            // Try numeric comparison
            const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
            const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

            if (!isNaN(aNum) && !isNaN(bNum)) {
                return newDir === 'asc' ? aNum - bNum : bNum - aNum;
            }

            // String comparison
            return newDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        });

        // Reorder rows
        rows.forEach(row => tbody.appendChild(row));
    }

    function setupImageZoom(container) {
        const images = container.querySelectorAll('img');
        images.forEach(img => {
            const newImg = img.cloneNode(true);
            img.parentNode.replaceChild(newImg, img);

            newImg.addEventListener('click', event => {
                event.preventDefault();
                event.stopPropagation();

                // Create overlay using DOM API instead of innerHTML for security
                const overlay = document.createElement('div');
                overlay.className = 'image-zoom-overlay';

                const zoomContainer = document.createElement('div');
                zoomContainer.className = 'image-zoom-container';

                const zoomImg = document.createElement('img');
                zoomImg.src = newImg.src; // Safe: src is already validated
                zoomImg.alt = newImg.alt || '';

                const closeBtn = document.createElement('button');
                closeBtn.className = 'image-zoom-close';
                closeBtn.setAttribute('aria-label', 'Close');
                closeBtn.textContent = '×';

                zoomContainer.appendChild(zoomImg);
                zoomContainer.appendChild(closeBtn);
                overlay.appendChild(zoomContainer);

                document.body.appendChild(overlay);
                document.body.style.overflow = 'hidden';

                overlay.addEventListener('click', evt => {
                    if (
                        evt.target === overlay ||
                        evt.target.classList.contains('image-zoom-close')
                    ) {
                        overlay.remove();
                        document.body.style.overflow = '';
                    }
                });

                const handleEscape = evt => {
                    if (evt.key === 'Escape') {
                        overlay.remove();
                        document.body.style.overflow = '';
                        document.removeEventListener('keydown', handleEscape);
                    }
                };
                document.addEventListener('keydown', handleEscape);
            });

            newImg.addEventListener('error', () => {
                newImg.style.display = 'none';
                const errorMsg = document.createElement('span');
                errorMsg.className = 'image-load-error';
                errorMsg.textContent = '[Image failed to load]';
                newImg.parentNode.insertBefore(errorMsg, newImg.nextSibling);
            });
        });
    }

    global[NS] = {
        slugifyHeading: slugifyHeading,
        extractHeadingText: extractHeadingText,
        buildTocHtml: buildTocHtml,
        setActiveTocLink: setActiveTocLink,
        setupTocObserver: setupTocObserver,
        setupTocToggle: setupTocToggle,
        setupTocBackToTop: setupTocBackToTop,
        wrapMarkdownTables: wrapMarkdownTables,
        setupImageZoom: setupImageZoom,
        setupBackToTop: setupBackToTop,
        downloadFile: downloadFile,
        tableToCSV: tableToCSV,
    };

    /**
     * Setup back to top button
     */
    function setupBackToTop() {
        // Remove existing button if any
        const existing = document.querySelector('.back-to-top');
        if (existing) existing.remove();

        // Prefer "Back to Top" inside TOC when TOC exists
        const toc = document.querySelector('.summary-toc');
        if (toc) return null;

        const btn = document.createElement('button');
        btn.className = 'back-to-top';
        btn.innerHTML = '↑';
        btn.setAttribute('aria-label', 'Back to Top');
        btn.setAttribute('title', 'Back to Top');
        document.body.appendChild(btn);

        // Show/hide based on scroll position
        const toggleVisibility = () => {
            const scrollY = window.scrollY || document.documentElement.scrollTop;
            btn.classList.toggle('visible', scrollY > 300);
        };

        window.addEventListener('scroll', toggleVisibility, { passive: true });
        toggleVisibility();

        // Scroll to top on click
        btn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth',
            });
        });

        return btn;
    }
})(typeof window !== 'undefined' ? window : this);
