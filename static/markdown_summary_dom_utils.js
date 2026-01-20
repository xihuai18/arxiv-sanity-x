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
            return token.children.map((child) => {
                if (child.type === 'text' || child.type === 'code_inline') {
                    return child.content;
                }
                return '';
            }).join('');
        }
        return token.content || '';
    }

    function buildTocHtml(items) {
        if (!items || items.length < 2) return '';
        const list = items.map((item) => {
            const title = escapeHtml(item.title);
            const slug = escapeHtml(item.slug);
            return `<li class="toc-item toc-level-${item.level}"><a href="#${slug}">${title}</a></li>`;
        }).join('');
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
    `;
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
        links.forEach((link) => {
            const href = link.getAttribute('href') || '';
            const id = href.slice(1);
            if (id) {
                linkMap.set(id, link);
            }
        });

        tocObserver = new IntersectionObserver((entries) => {
            const visible = entries.filter((entry) => entry.isIntersecting);
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
        }, {
            rootMargin: '0px 0px -70% 0px',
            threshold: [0, 1]
        });

        headings.forEach((heading) => {
            tocObserver.observe(heading);
        });
    }

    function setupTocToggle(tocContainer) {
        if (!tocContainer) return;
        const toggle = tocContainer.querySelector('.toc-toggle');
        if (!toggle) return;

        if (tocCollapsed === null) {
            tocCollapsed = window.matchMedia('(max-width: 960px)').matches;
        }

        const applyState = (collapsed) => {
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
        tables.forEach((table) => {
            const parent = table.parentElement;
            if (parent && parent.classList.contains('table-wrap')) return;
            const wrapper = document.createElement('div');
            wrapper.className = 'table-wrap';
            if (parent) {
                parent.insertBefore(wrapper, table);
            }
            wrapper.appendChild(table);
        });
    }

    function setupImageZoom(container) {
        const images = container.querySelectorAll('img');
        images.forEach((img) => {
            const newImg = img.cloneNode(true);
            img.parentNode.replaceChild(newImg, img);

            newImg.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();

                const overlay = document.createElement('div');
                overlay.className = 'image-zoom-overlay';
                overlay.innerHTML = `
                <div class="image-zoom-container">
                    <img src="${newImg.src}" alt="${newImg.alt || ''}" />
                    <button class="image-zoom-close" aria-label="Close">&times;</button>
                </div>
            `;
                document.body.appendChild(overlay);
                document.body.style.overflow = 'hidden';

                overlay.addEventListener('click', (evt) => {
                    if (evt.target === overlay || evt.target.classList.contains('image-zoom-close')) {
                        overlay.remove();
                        document.body.style.overflow = '';
                    }
                });

                const handleEscape = (evt) => {
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
        wrapMarkdownTables: wrapMarkdownTables,
        setupImageZoom: setupImageZoom
    };
})(typeof window !== 'undefined' ? window : this);
