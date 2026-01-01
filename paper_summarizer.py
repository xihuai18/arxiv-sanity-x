#!/usr/bin/env python3
"""
Paper Summarizer Module
Features:
1. Download arxiv papers to pdfs directory
2. Parse paper content to markdown using HTML (arXiv/ar5iv) or minerU
3. Summarize papers using LLM models
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import openai
import requests
from loguru import logger

from vars import (
    DATA_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_NAME,
    LLM_SUMMARY_LANG,
    MAIN_CONTENT_MIN_RATIO,
    MINERU_BACKEND,
    MINERU_DEVICE,
    MINERU_MAX_VRAM,
    MINERU_MAX_WORKERS,
    MINERU_PORT,
    SUMMARY_DIR,
    SUMMARY_HTML_SOURCES,
    SUMMARY_MARKDOWN_SOURCE,
    SUMMARY_MIN_CHINESE_RATIO,
)


class PaperSummarizer:
    # Class-level lock to ensure only one minerU process is running
    # Note: This only works within a single process. For multi-process safety,
    # consider using file-based locks or distributed locks
    _mineru_lock = threading.Lock()

    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.mineru_dir = self.data_dir / "mineru"
        self.html_md_dir = self.data_dir / "html_md"

        # Ensure directories exist
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_dir.mkdir(parents=True, exist_ok=True)
        # Cache HTML->Markdown outputs separately from minerU results
        self.html_md_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    @staticmethod
    def _build_summary_result(content: str, meta: Optional[dict] = None) -> dict:
        safe_meta = meta if isinstance(meta, dict) else {}
        return {"content": content, "meta": safe_meta}

    @staticmethod
    def _extract_version_from_url(url: str, raw_pid: str) -> Optional[str]:
        """Extract version number from arXiv URL.

        Args:
            url: The final URL after redirects (e.g., https://arxiv.org/pdf/2301.00001v3)
            raw_pid: The raw paper ID without version

        Returns:
            Version string (e.g., "3") or None if not found
        """
        # Look for version pattern in URL path
        # URL might be like: https://arxiv.org/pdf/2301.00001v3.pdf or https://arxiv.org/pdf/2301.00001v3
        import re

        pattern = re.escape(raw_pid) + r"v(\d+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    def download_arxiv_paper(self, pid: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        Download arXiv paper PDF

        Args:
            pid: Paper ID, e.g. "2301.00001"

        Returns:
            Tuple of (pdf_path, version):
            - pdf_path: Downloaded PDF file path, None if failed
            - version: The actual version downloaded (e.g., "3"), None if unknown
        """
        try:
            raw_pid = self._strip_version_suffix(pid)
            pdf_url = f"https://arxiv.org/pdf/{raw_pid}"
            pdf_path = self.pdfs_dir / f"{raw_pid}.pdf"
            cached_version = None

            # If file already exists, try to determine version
            if pdf_path.exists():
                logger.trace(f"PDF file already exists: {pdf_path}")

                # Strategy 1: Try to get version from database metadata
                db_version = self._get_latest_version_from_meta(pid)
                if db_version:
                    cached_version = str(db_version)
                    logger.trace(f"Using version {cached_version} from database for cached PDF")
                else:
                    # Strategy 2: Check existing MinerU meta.json
                    mineru_meta = self._read_mineru_meta(raw_pid)
                    if mineru_meta.get("cached_version"):
                        cached_version = mineru_meta["cached_version"]
                        logger.trace(f"Using version {cached_version} from MinerU meta")
                    else:
                        # Strategy 3: Make a HEAD request to get current version from redirect
                        try:
                            head_response = requests.head(pdf_url, allow_redirects=True, timeout=10)
                            if head_response.ok:
                                version_from_url = self._extract_version_from_url(head_response.url, raw_pid)
                                if version_from_url:
                                    cached_version = version_from_url
                                    logger.trace(
                                        f"Detected version {cached_version} from HEAD request: {head_response.url}"
                                    )
                        except Exception as e:
                            logger.trace(f"Failed to determine version from HEAD request: {e}")

                return pdf_path, cached_version

            # Use atomic file write with temporary file
            logger.trace(f"Downloading paper {raw_pid} ...")
            with requests.get(pdf_url, stream=True, timeout=30, allow_redirects=True) as response:
                response.raise_for_status()

                # Try to extract version from final URL after redirects
                cached_version = self._extract_version_from_url(response.url, raw_pid)
                if cached_version:
                    logger.trace(f"Detected version {cached_version} from redirect URL: {response.url}")

                # Write to temporary file first, then atomic rename
                temp_fd, temp_path = tempfile.mkstemp(dir=self.pdfs_dir, prefix=f"{raw_pid}_", suffix=".pdf.tmp")
                try:
                    with os.fdopen(temp_fd, "wb") as f:
                        shutil.copyfileobj(response.raw, f)

                    # Atomic rename (on POSIX systems)
                    temp_file = Path(temp_path)
                    try:
                        temp_file.rename(pdf_path)
                    except FileExistsError:
                        # Another process already downloaded the file
                        temp_file.unlink()
                        logger.trace(f"PDF file was downloaded by another process: {pdf_path}")
                        return (pdf_path, cached_version) if pdf_path.exists() else (None, None)

                except Exception:
                    # Clean up temporary file on error
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                    raise

            logger.trace(f"Paper download complete: {pdf_path}")
            return pdf_path, cached_version

        except Exception as e:
            logger.trace(f"Failed to download paper {pid}: {e}")
            return None, None

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Atomically write text to disk to avoid partial reads during concurrent runs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, path)
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        PaperSummarizer._atomic_write_text(path, json.dumps(data, ensure_ascii=False))

    _PID_VERSION_RE = re.compile(r"^(?P<raw>.+)v(?P<ver>\d+)$")

    @classmethod
    def _strip_version_suffix(cls, pid: str) -> str:
        """Remove version suffix from paper ID.

        Args:
            pid: Paper ID, possibly with version suffix (e.g., "2301.00001v2")

        Returns:
            Raw PID without version suffix
        """
        raw_pid, _ = cls._split_pid_version(pid)
        return raw_pid

    @classmethod
    def _split_pid_version(cls, pid: str) -> Tuple[str, Optional[int]]:
        """Split paper ID into raw ID and version number.

        Args:
            pid: Paper ID, possibly with version suffix (e.g., "2301.00001v2")

        Returns:
            Tuple of (raw_pid, version). Version is None if no version suffix.
        """
        pid = (pid or "").strip()
        if not pid:
            return "", None
        match = cls._PID_VERSION_RE.match(pid)
        if not match:
            return pid, None
        raw_pid = match.group("raw")
        try:
            version = int(match.group("ver"))
        except (ValueError, TypeError):
            return pid, None
        return raw_pid, version

    def _get_latest_version_from_meta(self, pid: str) -> Optional[int]:
        raw_pid, explicit_version = self._split_pid_version(pid)
        if explicit_version:
            return explicit_version
        try:
            from aslite.db import get_metas_db
        except Exception as e:
            logger.trace(f"Failed to import metas db: {e}")
            return None

        try:
            with get_metas_db() as metas_db:
                meta = metas_db.get(raw_pid)
        except Exception as e:
            logger.trace(f"Failed to read metas for {raw_pid}: {e}")
            return None

        if not isinstance(meta, dict):
            return None
        version = meta.get("_version")
        if version is None:
            return None
        try:
            return int(version)
        except Exception:
            return None

    def _resolve_cache_pid(self, pid: str) -> str:
        """Resolve PID to cache key format.

        SIMPLIFIED: Always use raw PID (without version) for caching.
        arXiv/ar5iv automatically return the latest version.

        Args:
            pid: Paper ID (may or may not include version)

        Returns:
            Raw PID without version suffix
        """
        raw_pid, _ = self._split_pid_version(pid)
        return raw_pid

    def _normalize_summary_source(self, source: Optional[str]) -> str:
        src = (source or SUMMARY_MARKDOWN_SOURCE or "html").strip().lower()
        if src not in {"html", "mineru"}:
            logger.trace(f"Unknown summary source '{src}', fallback to html")
            return "html"
        return src

    def _normalize_mineru_backend(self, backend: Optional[str] = None) -> str:
        raw = (backend or MINERU_BACKEND or "pipeline").strip().lower()
        aliases = {
            "vlm": "vlm-http-client",
            "http-client": "vlm-http-client",
            "vlm_http_client": "vlm-http-client",
            "vlm-httpclient": "vlm-http-client",
        }
        raw = aliases.get(raw, raw)
        supported = {
            "pipeline",
            "vlm-transformers",
            "vlm-vllm-engine",
            "vlm-lmdeploy-engine",
            "vlm-http-client",
        }
        if raw not in supported:
            logger.trace(f"Unknown minerU backend '{raw}', fallback to pipeline")
            return "pipeline"
        return raw

    def _mineru_md_candidates(self, paper_id: str, backend: Optional[str] = None) -> List[Path]:
        base_dir = self.mineru_dir / paper_id
        auto_md = base_dir / "auto" / f"{paper_id}.md"
        vlm_md = base_dir / "vlm" / f"{paper_id}.md"
        if backend and backend.startswith("vlm"):
            return [vlm_md, auto_md]
        return [auto_md, vlm_md]

    def _find_mineru_markdown(self, paper_id: str, backend: Optional[str] = None) -> Optional[Path]:
        for path in self._mineru_md_candidates(paper_id, backend=backend):
            if path.exists():
                return path
        base_dir = self.mineru_dir / paper_id
        if base_dir.exists():
            for path in base_dir.glob(f"*/{paper_id}.md"):
                if path.is_file():
                    return path
        return None

    def _mineru_meta_path(self, paper_id: str) -> Path:
        """Get MinerU meta.json path for a paper."""
        return self.mineru_dir / paper_id / "meta.json"

    def _read_mineru_meta(self, paper_id: str) -> dict:
        """Read MinerU metadata from file."""
        meta_path = self._mineru_meta_path(paper_id)
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_mineru_meta(self, paper_id: str, backend: str, cached_version: Optional[str] = None) -> None:
        """Write MinerU metadata to file.

        Args:
            paper_id: Raw paper ID (without version suffix)
            backend: MinerU backend used (pipeline, vlm-http-client, etc.)
            cached_version: Version of the PDF that was actually downloaded
        """
        meta_path = self._mineru_meta_path(paper_id)
        meta = {
            "updated_at": time.time(),
            "backend": backend,
        }
        if cached_version:
            meta["cached_version"] = cached_version
        self._atomic_write_json(meta_path, meta)

    def _parse_html_sources(self) -> List[str]:
        raw = (SUMMARY_HTML_SOURCES or "").strip()
        if not raw:
            raw = "ar5iv,arxiv"
        sources: List[str] = []
        for item in raw.split(","):
            src = item.strip().lower()
            if not src:
                continue
            if src not in {"ar5iv", "arxiv"}:
                logger.trace(f"Unsupported HTML source '{src}', skipping")
                continue
            if src not in sources:
                sources.append(src)
        if not sources:
            sources = ["ar5iv", "arxiv"]
        return sources

    def _html_cache_paths(self, cache_pid: str) -> Tuple[Path, Path, Path]:
        """Get HTML cache paths in new folder structure.

        Returns:
            Tuple of (md_path, meta_path, images_dir)
        """
        paper_dir = self.html_md_dir / cache_pid
        md_path = paper_dir / f"{cache_pid}.md"
        meta_path = paper_dir / "meta.json"
        images_dir = paper_dir / "images"
        return md_path, meta_path, images_dir

    def _read_html_meta(self, cache_pid: str) -> dict:
        _, meta_path, _ = self._html_cache_paths(cache_pid)
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _read_html_markdown_cache(self, cache_pid: str) -> Optional[str]:
        md_path, _, _ = self._html_cache_paths(cache_pid)
        if not md_path.exists():
            return None
        try:
            with open(md_path, encoding="utf-8") as f:
                content = f.read()
            return content if content.strip() else None
        except Exception as e:
            logger.trace(f"Failed to read HTML markdown cache for {cache_pid}: {e}")
            return None

    def _write_html_markdown_cache(self, cache_pid: str, markdown: str, source: str, url: str) -> None:
        """Write HTML markdown cache with metadata.

        Args:
            cache_pid: Raw paper ID (without version suffix)
            markdown: Markdown content
            source: HTML source (ar5iv, arxiv)
            url: Source URL (may contain version info)
        """
        md_path, meta_path, _ = self._html_cache_paths(cache_pid)
        self._atomic_write_text(md_path, markdown)

        # Try to extract cached version from URL (e.g., .../html/2512.21789v3)
        cached_version = None
        if url:
            # Extract the paper ID part from URL and check for version
            import re

            match = re.search(r"/(\d+\.\d+)(v(\d+))?", url)
            if match and match.group(3):
                cached_version = match.group(3)

        meta = {
            "updated_at": time.time(),
            "source": source,  # ar5iv or arxiv
            "url": url,
        }
        if cached_version:
            meta["cached_version"] = cached_version
        self._atomic_write_json(meta_path, meta)

    def _html_tools(self):
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify as html_to_markdown
        except Exception as e:
            logger.trace(f"HTML parsing dependencies missing: {e}")
            return None, None
        return BeautifulSoup, html_to_markdown

    def _download_image(self, img_url: str, save_path: Path) -> bool:
        """Download image from URL and save to local path.

        Args:
            img_url: Image URL
            save_path: Local path to save image

        Returns:
            True if download successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            headers = {"User-Agent": "arxiv-sanity-x/summary"}
            response = requests.get(img_url, headers=headers, timeout=30, stream=True)
            if response.status_code != 200:
                logger.trace(f"Failed to download image {img_url}: {response.status_code}")
                return False

            # Write to temporary file first, then atomic rename
            temp_fd, temp_path = tempfile.mkstemp(dir=save_path.parent, suffix=save_path.suffix)
            try:
                with os.fdopen(temp_fd, "wb") as f:
                    shutil.copyfileobj(response.raw, f)

                # Atomic rename
                temp_file = Path(temp_path)
                try:
                    temp_file.rename(save_path)
                except FileExistsError:
                    temp_file.unlink()
                    logger.trace(f"Image already exists: {save_path}")
                    return True

            except Exception:
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                raise

            logger.trace(f"Downloaded image: {save_path.name}")
            return True

        except Exception as e:
            logger.trace(f"Error downloading image {img_url}: {e}")
            return False

    def _replace_math_nodes(self, soup) -> None:
        # Prefer MathML LaTeX annotations when present (common in ar5iv/arXiv HTML).
        for span in soup.find_all("span"):
            classes = span.get("class") or []
            if "ltx_Math" in classes:
                latex = span.get("data-latex") or span.get("data-tex")
                if not latex:
                    annotation = span.find("annotation", attrs={"encoding": "application/x-tex"})
                    if annotation and annotation.string:
                        latex = annotation.string
                if latex:
                    span.replace_with(soup.new_string(f"${latex.strip()}$"))

        for math in soup.find_all("math"):
            annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
            latex = annotation.string if annotation and annotation.string else ""
            if latex:
                math.replace_with(soup.new_string(f"$$ {latex.strip()} $$"))
                continue

            fallback_text = math.get_text(" ", strip=True)
            if fallback_text:
                math.replace_with(soup.new_string(fallback_text))
            else:
                math.decompose()

    def _html_to_markdown(self, html: str, cache_pid: str, base_url: str) -> Optional[str]:
        """Convert HTML to Markdown and download images.

        Args:
            html: HTML content
            cache_pid: Cache paper ID for organizing images
            base_url: Base URL for resolving relative image paths

        Returns:
            Markdown content with updated image references
        """
        BeautifulSoup, html_to_markdown = self._html_tools()
        if not BeautifulSoup or not html_to_markdown:
            return None

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        self._replace_math_nodes(soup)

        main = soup.find("article") or soup.find("main") or soup.body or soup
        if main:
            for tag in main.find_all(["nav", "aside"]):
                tag.decompose()

        # Download images and update references
        _, _, images_dir = self._html_cache_paths(cache_pid)
        img_counter = {}

        for img_tag in (main or soup).find_all("img"):
            src = img_tag.get("src")
            if not src:
                # Remove img tags without src
                img_tag.decompose()
                continue

            # Resolve relative URLs using urljoin (handles both ar5iv and arxiv)
            # Ensure base_url ends with / so urljoin treats it as a directory
            from urllib.parse import urljoin

            base_with_slash = base_url if base_url.endswith("/") else f"{base_url}/"
            img_url = urljoin(base_with_slash, src)

            # Generate consistent image filename
            # Use original extension if available
            ext = Path(img_url.split("?")[0]).suffix or ".png"

            # Create a descriptive filename from alt text or use counter
            alt_text = img_tag.get("alt", "")
            if alt_text:
                # Sanitize alt text for filename
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", alt_text)[:50]
                safe_name = re.sub(r"_+", "_", safe_name).strip("_")
            else:
                safe_name = f"image_{len(img_counter) + 1}"

            # Ensure unique filename
            base_name = safe_name
            counter = img_counter.get(base_name, 0)
            if counter > 0:
                safe_name = f"{base_name}_{counter}"
            img_counter[base_name] = counter + 1

            img_filename = f"{safe_name}{ext}"
            img_path = images_dir / img_filename

            # Download image
            if self._download_image(img_url, img_path):
                # Update img tag src to point to local file
                img_tag["src"] = f"images/{img_filename}"
            else:
                # Remove img tag if download fails to prevent LLM from referencing broken images
                logger.trace(f"Removing image tag due to failed download: {img_url}")
                img_tag.decompose()

        markdown = html_to_markdown(str(main), heading_style="ATX")
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        return markdown if markdown else None

    def _fetch_html_from_source(self, pid: str, source: str) -> Tuple[Optional[str], str]:
        if source == "ar5iv":
            url = f"https://ar5iv.labs.arxiv.org/html/{pid}"
        else:
            url = f"https://arxiv.org/html/{pid}"

        try:
            headers = {"User-Agent": "arxiv-sanity-x/summary"}
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                logger.trace(f"HTML fetch failed ({source}) {response.status_code}: {url}")
                return None, url

            # Check if redirected to abstract page (ar5iv redirects to arxiv/abs when HTML not available)
            final_url = response.url
            if "/abs/" in final_url:
                logger.trace(f"HTML not available ({source}), redirected to abstract page: {final_url}")
                return None, url

            return response.text, url
        except Exception as e:
            logger.trace(f"HTML fetch error ({source}) {pid}: {e}")
            return None, url

    def _get_markdown_from_html(self, pid: str) -> Tuple[Optional[str], Optional[str], str]:
        """Get markdown from HTML source with versioned caching.

        Args:
            pid: Paper ID (may or may not include version)

        Returns:
            Tuple of (markdown_content, source, cache_pid)
            - markdown_content: The parsed markdown or None
            - source: The HTML source used (ar5iv, arxiv) or None
            - cache_pid: Raw PID used for storage (always unversioned)
        """
        # Always use raw PID for storage - arXiv returns latest version automatically
        cache_pid = self._resolve_cache_pid(pid)

        # Check cache first
        cached = self._read_html_markdown_cache(cache_pid)
        if cached:
            meta = self._read_html_meta(cache_pid)
            return cached, meta.get("source"), cache_pid

        for source in self._parse_html_sources():
            html, url = self._fetch_html_from_source(cache_pid, source)
            if not html:
                continue
            markdown = self._html_to_markdown(html, cache_pid, url)
            if not markdown:
                logger.trace(f"HTML parse produced empty markdown for {pid} from {source}")
                continue
            self._write_html_markdown_cache(cache_pid, markdown, source, url)
            return markdown, source, cache_pid

        return None, None, cache_pid

    def _acquire_file_lock(self, lock_path: Path, timeout: int = 60) -> Optional[int]:
        """
        Acquire a file-based lock for multi-process synchronization

        Args:
            lock_path: Path to the lock file
            timeout: Maximum wait time in seconds

        Returns:
            File descriptor if lock acquired, None if timeout
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        try:
            stale_s = float(os.environ.get("ARXIV_SANITY_MINERU_LOCK_STALE_SEC", "3600"))
        except Exception:
            stale_s = 3600.0

        while time.time() - start_time < timeout:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                return fd
            except FileExistsError:
                if stale_s > 0:
                    try:
                        age = time.time() - lock_path.stat().st_mtime
                        if age > stale_s:
                            lock_path.unlink(missing_ok=True)
                            logger.trace(f"Removed stale minerU lock: {lock_path}")
                            continue
                    except Exception:
                        pass
                # Lock file exists, wait and retry
                time.sleep(0.5)

        logger.trace(f"Failed to acquire lock after {timeout} seconds: {lock_path}")
        return None

    def _release_file_lock(self, fd: int, lock_path: Path):
        """Release a file-based lock"""
        try:
            os.close(fd)
            lock_path.unlink(missing_ok=True)
        except Exception as e:
            logger.trace(f"Error releasing lock: {e}")

    def _acquire_gpu_slot(self, timeout: int = 600) -> Optional[int]:
        """
        Acquire a GPU slot for MinerU process (implements semaphore with file-based locks)

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Slot number (0 to MINERU_MAX_WORKERS-1) if acquired, None if timeout
        """
        slots_dir = self.mineru_dir / ".gpu_slots"
        slots_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        max_workers = max(1, MINERU_MAX_WORKERS)

        try:
            stale_s = float(os.environ.get("ARXIV_SANITY_MINERU_LOCK_STALE_SEC", "3600"))
        except Exception:
            stale_s = 3600.0

        while time.time() - start_time < timeout:
            # Try to acquire any available slot
            for slot in range(max_workers):
                slot_lock_path = slots_dir / f"slot_{slot}.lock"
                try:
                    fd = os.open(str(slot_lock_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                    logger.trace(f"Acquired GPU slot {slot}/{max_workers}")
                    # Write process info to lock file
                    try:
                        os.write(fd, f"{os.getpid()}\\n{time.time()}\\n".encode())
                    except Exception:
                        pass
                    return slot
                except FileExistsError:
                    # Check if lock is stale
                    if stale_s > 0:
                        try:
                            age = time.time() - slot_lock_path.stat().st_mtime
                            if age > stale_s:
                                slot_lock_path.unlink(missing_ok=True)
                                logger.trace(f"Removed stale GPU slot lock: {slot_lock_path}")
                                continue
                        except Exception:
                            pass

            # No slot available, wait and retry
            time.sleep(1.0)

        logger.trace(f"Failed to acquire GPU slot after {timeout} seconds")
        return None

    def _release_gpu_slot(self, slot: int):
        """Release a GPU slot"""
        try:
            slots_dir = self.mineru_dir / ".gpu_slots"
            slot_lock_path = slots_dir / f"slot_{slot}.lock"
            slot_lock_path.unlink(missing_ok=True)
            logger.trace(f"Released GPU slot {slot}")
        except Exception as e:
            logger.trace(f"Error releasing GPU slot {slot}: {e}")

    def parse_pdf_with_mineru(
        self, pdf_path: Path, cache_pid: Optional[str] = None, cached_version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Parse PDF to Markdown using minerU (with multi-process lock protection)

        Args:
            pdf_path: PDF file path
            cache_pid: Optional paper ID to use for output directory (should be raw PID).
                       If not provided, uses the PDF filename (stem).
            cached_version: Version of the PDF that was actually downloaded (e.g., "3")

        Returns:
            Generated Markdown file path, None if failed
        """
        try:
            # Use cache_pid if provided, otherwise use PDF filename
            pdf_name = cache_pid or pdf_path.stem
            output_dir = self.mineru_dir
            backend = self._normalize_mineru_backend()

            # Check if already parsed
            existing_md_path = self._find_mineru_markdown(pdf_name, backend=backend)
            if existing_md_path:
                logger.trace(f"Markdown file already exists: {existing_md_path}")
                return existing_md_path
            else:
                logger.trace(f"Markdown file not found for {pdf_name} in minerU outputs")

            # Use file-based lock for multi-process synchronization
            lock_path = self.mineru_dir / f".{pdf_name}.lock"
            lock_fd = None
            gpu_slot = None

            try:
                # Acquire file lock (works across processes)
                logger.trace(f"Waiting for file lock to parse PDF: {pdf_path}")
                lock_fd = self._acquire_file_lock(lock_path, timeout=300)

                if lock_fd is None:
                    logger.trace(f"Failed to acquire lock for {pdf_name}, skipping")
                    return None

                logger.trace(f"Acquired file lock, start parsing PDF: {pdf_path}")

                # Check again if already parsed (avoid duplicate work during lock wait)
                existing_md_path = self._find_mineru_markdown(pdf_name, backend=backend)
                if existing_md_path:
                    logger.trace(f"File generated during lock wait: {existing_md_path}")
                    return existing_md_path

                # Normalize device setting
                device = (MINERU_DEVICE or "cuda").strip().lower()
                if device not in ["cuda", "cpu"]:
                    logger.trace(f"Invalid device '{device}', fallback to cuda")
                    device = "cuda"

                # Acquire GPU slot for pipeline backend with GPU (limit concurrent GPU processes)
                if backend == "pipeline" and device == "cuda":
                    logger.trace(f"Waiting for GPU slot (max workers: {MINERU_MAX_WORKERS})...")
                    gpu_slot = self._acquire_gpu_slot(timeout=600)
                    if gpu_slot is None:
                        logger.trace(f"Failed to acquire GPU slot for {pdf_name}, skipping")
                        return None
                    logger.trace(f"Acquired GPU slot {gpu_slot} for {pdf_name}")

                # Build minerU command (backend selection per MinerU CLI docs)
                cmd = [
                    "mineru",
                    "-p",
                    str(pdf_path),
                    "-o",
                    str(output_dir),
                    "-l",
                    "en",
                    "-b",
                    backend,
                ]

                # Configure backend-specific parameters
                if backend == "pipeline":
                    # Use configured device (cuda or cpu)
                    cmd.extend(["-d", device])
                    # Only set VRAM limit for GPU mode
                    if device == "cuda" and MINERU_MAX_VRAM > 0:
                        cmd.extend(["--vram", str(MINERU_MAX_VRAM)])
                elif backend == "vlm-http-client":
                    cmd.extend(["-u", f"http://127.0.0.1:{MINERU_PORT}"])

                logger.trace(f"Executing command: {' '.join(cmd)}")
                start_time = time.time()

                # Execute command
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                elapsed_time = time.time() - start_time
                logger.trace(f"minerU execution complete, elapsed {elapsed_time:.2f} seconds")

                if result.returncode != 0:
                    logger.trace(f"minerU execution failed: {result.stderr}")
                    # Delete PDF file when minerU parsing fails
                    try:
                        pdf_path.unlink(missing_ok=True)
                        logger.trace(f"Parse failed, deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")
                    return None

                # Post-execution migration logic: check if new vlm directory was generated, rename to auto
                # Check generated Markdown file
                existing_md_path = self._find_mineru_markdown(pdf_name, backend=backend)
                if existing_md_path:
                    logger.trace(f"PDF parse complete: {existing_md_path}")

                    # Write meta.json for MinerU output with version info
                    self._write_mineru_meta(pdf_name, backend, cached_version=cached_version)

                    # Delete PDF file after successful parsing to save space
                    try:
                        pdf_path.unlink(missing_ok=True)
                        logger.trace(f"Deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")

                    # Clean up files other than images and markdown
                    self._cleanup_mineru_output(output_dir / pdf_name)

                    return existing_md_path
                else:
                    logger.trace(f"Generated Markdown file not found for {pdf_name} in minerU outputs")
                    # Delete PDF file when parsing fails
                    try:
                        pdf_path.unlink(missing_ok=True)
                        logger.trace(f"Parse failed, deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")
                    return None

            finally:
                # Always release the GPU slot first (if acquired)
                if gpu_slot is not None:
                    self._release_gpu_slot(gpu_slot)

                # Always release the file lock
                if lock_fd is not None:
                    self._release_file_lock(lock_fd, lock_path)
                    logger.trace(f"Released file lock for {pdf_name}")

        except subprocess.TimeoutExpired:
            logger.trace("minerU execution timeout")
            # Delete PDF file when parsing times out
            try:
                pdf_path.unlink(missing_ok=True)
                logger.trace(f"Parse timeout, deleted PDF source file: {pdf_path}")
            except Exception as e:
                logger.trace(f"Failed to delete PDF file: {e}")
            return None
        except Exception as e:
            logger.trace(f"PDF parse failed: {e}")
            # Delete PDF file when parsing exception occurs
            try:
                pdf_path.unlink(missing_ok=True)
                logger.trace(f"Parse exception, deleted PDF source file: {pdf_path}")
            except Exception as e:
                logger.trace(f"Failed to delete PDF file: {e}")
            return None

    def _cleanup_mineru_output(self, output_path: Path):
        """
        Clean up minerU output directory, keep only images and markdown files

        Args:
            output_path: minerU output paper directory path (e.g. data/mineru/2507.01679)
        """
        try:
            output_dirs = []
            for dir_name in ("auto", "vlm"):
                dir_path = output_path / dir_name
                if dir_path.exists():
                    output_dirs.append(dir_path)
            if not output_dirs:
                return

            # File patterns to keep
            paper_id = output_path.name
            keep_files = {f"{paper_id}.md"}  # Keep markdown files
            keep_dirs = {"images"}  # Keep images directory

            for target_dir in output_dirs:
                items_to_remove = []

                for item in target_dir.iterdir():
                    if item.is_dir():
                        if item.name not in keep_dirs:
                            # Mark unnecessary directories for deletion
                            items_to_remove.append(item)
                    elif item.is_file():
                        if item.name not in keep_files:
                            # Mark unnecessary files for deletion (JSON, PDF and other intermediate files)
                            items_to_remove.append(item)

                # Execute deletion
                for item in items_to_remove:
                    try:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                            logger.trace(f"Deleted directory: {item}")
                        else:
                            item.unlink(missing_ok=True)
                            logger.trace(f"Deleted file: {item}")
                    except Exception as e:
                        logger.trace(f"Failed to delete {item}: {e}")

        except Exception as e:
            logger.trace(f"Failed to clean minerU output: {e}")

    def extract_main_content(self, markdown_content: str) -> str:
        """
        Extract main paper content, remove references, acknowledgements etc.

        Uses a prioritized approach:
        1. First try to find References/Bibliography (most reliable end marker)
        2. If not found, try Acknowledgements
        3. If not found, try Appendix/Supplementary Material
        4. Take the earliest match found among the highest priority group

        Args:
            markdown_content: Complete Markdown content

        Returns:
            Extracted main content
        """
        try:
            lines = markdown_content.split("\n")
            total_lines = len(lines)

            # Priority groups of ending markers (higher priority = more reliable indicator)
            # Each group is a list of patterns; we search groups in order and stop at first match
            priority_groups = [
                # Priority 1: References/Bibliography - most reliable end marker
                [
                    # Markdown headings: ## References, ### Bibliography
                    r"(?i)^#{1,6}\s*references?\s*$",
                    r"(?i)^#{1,6}\s*bibliography\s*$",
                    # Numbered headings: 7. References, 7 References, [7] References
                    r"(?i)^\[?\d+\]?\.?\s*references?\s*$",
                    r"(?i)^\[?\d+\]?\.?\s*bibliography\s*$",
                    # Bold format: **References**
                    r"(?i)^\*{1,2}references?\*{1,2}\s*$",
                    r"(?i)^\*{1,2}bibliography\*{1,2}\s*$",
                    # Plain text
                    r"(?i)^references?\s*$",
                    r"(?i)^bibliography\s*$",
                ],
                # Priority 2: Acknowledgements - usually before references but after main content
                [
                    r"(?i)^#{1,6}\s*acknowledge?ments?\s*$",
                    r"(?i)^\[?\d+\]?\.?\s*acknowledge?ments?\s*$",
                    r"(?i)^\*{1,2}acknowledge?ments?\*{1,2}\s*$",
                    r"(?i)^acknowledge?ments?\s*$",
                ],
                # Priority 3: Appendix/Supplementary - fallback if no references found
                [
                    # Markdown headings
                    r"(?i)^#{1,6}\s*appendix\s*",
                    r"(?i)^#{1,6}\s*appendices\s*",
                    r"(?i)^#{1,6}\s*supplementary\s*(materials?)?\s*$",
                    r"(?i)^#{1,6}\s*supplemental\s*(materials?)?\s*$",
                    # Numbered/lettered headings: A. Appendix, Appendix A
                    r"(?i)^[A-Z]\.?\s*appendix\s*",
                    r"(?i)^appendix\s+[A-Z]\s*",
                    # Bold format
                    r"(?i)^\*{1,2}appendix\*{1,2}\s*",
                    r"(?i)^\*{1,2}appendices\*{1,2}\s*",
                    r"(?i)^\*{1,2}supplementary\s*(materials?)?\*{1,2}\s*$",
                    # Plain text
                    r"(?i)^appendix\s*$",
                    r"(?i)^appendices\s*$",
                ],
            ]

            def find_earliest_match(patterns: list) -> int:
                """Find the earliest line index matching any pattern in the list."""
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    for pattern in patterns:
                        if re.match(pattern, stripped):
                            # Found the earliest match, return immediately
                            return i
                return total_lines

            # Search through priority groups until we find a match
            end_index = total_lines
            for group_idx, patterns in enumerate(priority_groups):
                match_idx = find_earliest_match(patterns)
                if match_idx < total_lines:
                    end_index = match_idx
                    logger.trace(
                        f"Found end marker at line {match_idx + 1} (priority group {group_idx + 1}): {lines[match_idx].strip()[:50]}"
                    )
                    break

            if end_index == total_lines:
                logger.trace("No end markers found, using full content")
                return markdown_content

            # Extract main content
            main_content = "\n".join(lines[:end_index])

            # If extracted content is too short, return original content
            if len(main_content.strip()) < len(markdown_content.strip()) * MAIN_CONTENT_MIN_RATIO:
                logger.trace(
                    f"Extracted main content too short ({len(main_content.strip())} < {len(markdown_content.strip()) * MAIN_CONTENT_MIN_RATIO:.0f}), using original content"
                )
                return markdown_content

            logger.trace(f"Paper content truncated from {len(markdown_content)} to {len(main_content)} characters")
            return main_content

        except Exception as e:
            logger.trace(f"Failed to extract main content: {e}")
            return markdown_content

    def summarize_with_llm(self, markdown_content: str, model: Optional[str] = None) -> dict:
        """
        Summarize paper content using LLM

        Args:
            markdown_content: Paper content in Markdown format

        Returns:
            Paper summary
        """
        summary_meta = {}

        try:
            # Choose language prompt based on configuration
            if LLM_SUMMARY_LANG == "en":
                # English prompt - Technical blog style
                prompt = f"""
You are an experienced technical blogger who excels at interpreting academic papers.
Please transform the following paper into a technical blog post that allows readers to
understand the key techniques, core results, and important details without reading the original paper.

<Target Audience>
Readers with foundational knowledge in the relevant field

<Content Requirements>
1. **Must Include**: Research motivation and background, core contributions (1-3 points), detailed method/algorithm explanation, key experimental results, limitations or future directions
2. **Focus on Expansion**: The method/algorithm section needs detailed introduction, with intuitive explanations, necessary formulas and mathematical derivations, or analogies to describe complex concepts
3. **Information Sourcing**: All statements must be based on the original text, do not speculate on content not explicitly stated by the authors
4. **Image Usage**: When a certain image significantly helps understanding the paper, please reference the image in the blog
   - Only select the most critical images for understanding (typically 1-3)
   - Generally, a paper will have one or two images that can illustrate the core method and unique contributions; if such images exist, please reference them
   - Use numbered format: `![Figure 1: Figure title. Brief description](images/filename.png)`
   - Reference images in the text by number, such as "As shown in Figure 1,..." , "Figure 2 shows..."
   - Image numbers must be consecutive (Figure 1, Figure 2, Figure 3...)
   - **Do not use blockquotes (>) to wrap images**
   - **Do not modify image paths or filenames**
5. **Appropriate Detail Level**: Without significantly increasing the reading burden, add as comprehensive important paper details as possible in the blog

<Style Requirements>
- Clear structure with hierarchical headings
- Avoid empty platitudes; every paragraph should have substantive content
- Briefly explain technical terms when they first appear
- Use `$…$` (inline) or `$$…$$` (block) for all formulas. Use basic LaTeX formulas when possible
- Ensure LaTeX and Markdown syntax compatibility and correctness

<Paper Content>
{markdown_content}
</Paper Content>

<Output Format>
## Technical Blog
The detailed technical blog content (with key figures where appropriate)

## TL;DR
A concise 2-3 sentence summary of the paper's core contribution and significance
</Output Format>

<Important Notes>
1. Ensure the blog accurately reflects the paper content; **do not add information not in the paper**
2. Do not guess or fabricate data; skip quantitative analysis rather than summarize incorrect data
3. Please write in **English**
4. Make the content accessible yet technically rigorous
5. Only include figures that exist in the paper content above; do not invent figure references
</Important Notes>
"""
            else:
                # Chinese prompt (default) - Technical blog style
                prompt = f"""
你是一位经验丰富的技术博主，擅长解读学术论文。请将以下论文解读为一篇技术博客，目标是让读者"无需阅读原文也能理解关键技术、核心结果与重要细节"。

<目标受众>
有一定本领域基础的读者

<内容要求>
1. **必须包含**：研究动机与背景、核心贡献（1-3点）、方法/算法详解、关键实验结果、局限性或未来方向
2. **重点展开**：方法/算法部分需要详细介绍，需配合直观解释，必要时配合公式和数学推导或者用类比描述复杂概念
3. **信息溯源**：所有陈述需基于原文，不臆测作者未明确表述的内容
4. **图片使用**：当某张图片对理解论文有重要帮助时，请在博客中引用该图片
   - 只选择对理解最关键的图片（通常1-3张）
   - 一般而言一篇论文会有一到两张能诠释核心方法以及独特贡献的图片，如果这样的图片存在，请引用它们
   - 使用编号格式：`![图1： 图标题。简要说明](images/filename.png)`
   - 正文中通过编号引用图片，如"如图1所示，..."、"图2展示了..."
   - 图片编号必须连续（图1、图2、图3...）
   - **不要使用引用块（>）包裹图片**
   - **不要修改图片路径或文件名**
5. **详略得当**：在不显著增加阅读负担的前提下，在博客中增加尽可能全面的重要论文细节

<风格要求>
- 结构清晰，使用标题分层
- 避免空洞套话，每段都有实质内容
- 专业术语首次出现时简要解释
- 所有公式用 `$…$`（行内）或 `$$…$$`（独立），要尽量使用最基本的 LaTeX 公式
- 注意所使用的 LaTeX 和 Markdown 语法兼容性和正确性

<论文内容>
{markdown_content}
</论文内容>

<输出格式>
## 技术博客
具体的技术博客内容（在适当位置包含关键图片）

## TL;DR
用2-3句话概括论文的核心贡献和意义

</输出格式>

<注意事项>
1. 请确保博客准确反映论文内容，**不要添加论文中没有的信息**
2. 不要猜测更不要臆造数据，宁可跳过定量分析，也不要总结错误数据
3. 请一定用**中文**撰写
4. 保持内容易读的同时确保技术严谨性
5. 只引用论文内容中存在的图片，不要编造图片引用
</注意事项>
"""

            modelid = (model or LLM_NAME or "").strip() or LLM_NAME
            summary_meta = {
                "model": modelid,
                "prompt": prompt,
                "generated_at": time.time(),
            }
            logger.trace(f"Calling {modelid} to generate paper summary...")

            response = self.client.chat.completions.create(
                model=modelid,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32768,
                extra_body={"reasoning": {"effort": "low"}},  # Use low reasoning effort
            )

            # Validate response structure
            if not response.choices:
                logger.trace("LLM returned empty choices")
                return self._build_summary_result("# Error\n\nLLM returned no response", summary_meta)

            message = response.choices[0].message
            summary = message.content if message else None

            if not summary:
                logger.trace("LLM returned empty content")
                return self._build_summary_result("# Error\n\nLLM returned empty content", summary_meta)

            # Log reasoning content if available
            if hasattr(message, "reasoning") and message.reasoning:
                logger.trace(f"Original summary Thinking:\n{message.reasoning}")
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                logger.trace(f"Original summary Thinking:\n{message.reasoning_content}")
            else:
                logger.trace(f"Original summary content:\n{summary[:500]}...")

            # Extract content after </think> tag if present
            if "</think>" in summary:
                summary = summary.split("</think>", 1)[1].strip()
            else:
                summary = summary.strip()

            # Parse detailed summary and TL;DR sections
            parsed_summary = self._parse_summary_sections(summary)
            logger.trace("Paper summary generation complete")
            return self._build_summary_result(parsed_summary, summary_meta)

        except Exception as e:
            logger.trace(f"LLM summary failed: {e}")
            return self._build_summary_result(f"# Error\n\nSummary generation failed: {str(e)}", summary_meta)

    def _parse_summary_sections(self, summary: str) -> str:
        """
        Parse summary content from LLM output and reorganize with TL;DR first.
        Strategy: Extract TL;DR section, treat everything else as blog content.
        Output format: TL;DR first, then blog content.

        Args:
            summary: Original summary content

        Returns:
            Reorganized summary content with TL;DR first
        """
        # Extract TL;DR section (including heading, match any heading level #, ##, ### etc.)
        tldr_match = re.search(r"(#{1,6}\s*TL;DR\s*\n.*?)(?=\n#{1,6}\s|$)", summary, re.DOTALL | re.IGNORECASE)

        if tldr_match:
            tldr_section = tldr_match.group(1).strip()
            # Remove TL;DR section from summary to get blog content
            blog_content = summary[: tldr_match.start()] + summary[tldr_match.end() :]
            blog_content = blog_content.strip()

            if tldr_section and blog_content:
                # Reorganize: TL;DR first, then blog
                result = f"{tldr_section}\n\n{blog_content}"
                result = re.sub(r"\n{3,}", "\n\n", result).strip()
                return result

        # Fallback: just clean up the original content
        cleaned = re.sub(r"\n{3,}", "\n\n", summary).strip()
        return cleaned

    def _postprocess_image_paths(self, summary: str, pid: str, source: str = "html") -> str:
        """
        Post-process image paths in summary to use correct API URLs.

        Converts relative paths like `images/xxx.png` to API paths for web rendering.
        - HTML source: `/api/paper_image/{pid}/{filename}`
        - MinerU source: `/api/mineru_image/{pid}/{filename}`

        Args:
            summary: Summary content with image references
            pid: Paper ID for constructing image URLs
            source: Image source type ("html" or "mineru")

        Returns:
            Summary with corrected image paths
        """
        # Pattern to match markdown image syntax: ![alt](images/filename.ext)
        pattern = r"!\[([^\]]*)\]\(images/([^)]+)\)"

        # Choose API endpoint based on source
        api_prefix = "/api/paper_image" if source == "html" else "/api/mineru_image"

        def replace_path(match):
            alt_text = match.group(1)
            filename = match.group(2)
            # Convert to API endpoint path
            return f"![{alt_text}]({api_prefix}/{pid}/{filename})"

        return re.sub(pattern, replace_path, summary)

    def generate_summary(self, pid: str, source: Optional[str] = None, model: Optional[str] = None) -> dict:
        """
        Main entry function for generating paper summary

        Args:
            pid: Paper ID (may or may not include version)
            source: Markdown source override ("html" or "mineru")
            model: LLM model name

        Returns:
            Dict containing paper summary markdown content and metadata
        """
        try:
            # Validate input
            pid = (pid or "").strip()
            if not pid:
                return self._build_summary_result("# Error\n\nPaper ID is empty or invalid")

            summary_source = self._normalize_summary_source(source)

            if summary_source == "html":
                markdown_content, html_source, cache_pid = self._get_markdown_from_html(pid)
                if markdown_content:
                    main_content = self.extract_main_content(markdown_content)
                    logger.info(f"Summarizing {pid} (html:{html_source or 'unknown'}, cache:{cache_pid}) ...")
                    summary = self.summarize_with_llm(main_content, model=model)
                    # Post-process image paths using the actual cache_pid
                    summary["content"] = self._postprocess_image_paths(summary["content"], cache_pid)
                    return summary
                logger.info(f"HTML fetch/parse failed for {pid}, fallback to minerU.")

            # Use raw PID for MinerU storage - arXiv returns latest version automatically
            cache_pid = self._resolve_cache_pid(pid)
            backend = self._normalize_mineru_backend()

            # Pre-check if parsing result already exists
            existing_md_path = self._find_mineru_markdown(cache_pid, backend=backend)
            if existing_md_path:
                logger.trace(f"Found existing parse result: {existing_md_path}")
                # Read Markdown content directly
                with open(existing_md_path, encoding="utf-8") as f:
                    markdown_content = f.read()

                if markdown_content.strip():
                    # Step 3: Extract main paper content
                    main_content = self.extract_main_content(markdown_content)
                    # Step 4: Generate summary using LLM
                    logger.info(f"Summarizing {cache_pid} (mineru) ...")
                    summary = self.summarize_with_llm(main_content, model=model)
                    # Post-process image paths for MinerU source
                    summary["content"] = self._postprocess_image_paths(summary["content"], cache_pid, source="mineru")
                    return summary
                else:
                    logger.trace("Existing Markdown file content is empty, re-parsing")

            # Step 1: Download paper PDF (use cache_pid for filename)
            logger.info(f"Downloading {cache_pid}.pdf ...")
            pdf_path, cached_version = self.download_arxiv_paper(cache_pid)
            if not pdf_path:
                return self._build_summary_result("# Error\n\nUnable to download paper PDF")

            # Step 2: Parse PDF to Markdown using minerU
            logger.info(f"Parsing {cache_pid}.pdf ...")
            md_path = self.parse_pdf_with_mineru(pdf_path, cache_pid=cache_pid, cached_version=cached_version)
            if not md_path:
                return self._build_summary_result("# Error\n\nUnable to parse PDF to Markdown")

            # Step 3: Read Markdown content
            with open(md_path, encoding="utf-8") as f:
                markdown_content = f.read()

            if not markdown_content.strip():
                return self._build_summary_result("# Error\n\nParsed Markdown content is empty")

            # Step 4: Extract main paper content
            main_content = self.extract_main_content(markdown_content)

            # Step 5: Generate summary using LLM
            logger.info(f"Summarizing {cache_pid} (mineru) ...")
            summary = self.summarize_with_llm(main_content, model=model)
            # Post-process image paths for MinerU source
            summary["content"] = self._postprocess_image_paths(summary["content"], cache_pid, source="mineru")

            return summary

        except Exception as e:
            logger.error(f"Error occurred while generating paper summary: {e}")
            return self._build_summary_result(f"# Error\n\nFailed to generate summary: {str(e)}")


# Global instance with thread-safe initialization
_summarizer = None
_summarizer_lock = threading.Lock()


def get_summarizer() -> PaperSummarizer:
    """Get global PaperSummarizer instance (thread-safe singleton)"""
    global _summarizer
    if _summarizer is None:
        with _summarizer_lock:
            # Double-check locking pattern
            if _summarizer is None:
                _summarizer = PaperSummarizer()
    return _summarizer


def generate_paper_summary(pid: str, source: Optional[str] = None, model: Optional[str] = None) -> dict:
    """
    External interface function for generating paper summary

    Args:
        pid: Paper ID
        source: Markdown source override ("html" or "mineru")

    Returns:
        Dict containing paper summary markdown content and metadata
    """
    summarizer = get_summarizer()
    return summarizer.generate_summary(pid, source=source, model=model)


# =============================================================================
# Reusable Summary Caching Utilities
# =============================================================================

_PID_VERSION_RE = re.compile(r"^(?P<raw>.+)v(?P<ver>\d+)$")


def split_pid_version(pid: str) -> Tuple[str, Optional[int]]:
    """
    Split paper ID into raw ID and version number.

    Args:
        pid: Paper ID, possibly with version suffix (e.g., "2301.00001v2")

    Returns:
        Tuple of (raw_pid, version). Version is None if no version suffix.
    """
    pid = (pid or "").strip()
    if not pid:
        return "", None
    match = _PID_VERSION_RE.match(pid)
    if not match:
        return pid, None
    raw_pid = match.group("raw")
    try:
        version = int(match.group("ver"))
    except (ValueError, TypeError):
        return pid, None
    return raw_pid, version


def resolve_cache_pid(pid: str, meta: Optional[dict] = None) -> Tuple[str, str, bool]:
    """
    Resolve cache PID from paper ID.

    SIMPLIFIED: Always use raw PID (without version) for caching.
    arXiv/ar5iv automatically return the latest version when accessed without version.
    This simplifies cache management - we only keep the latest version.

    Args:
        pid: Paper ID (may include version suffix)
        meta: Optional paper metadata dict (unused, kept for API compatibility)

    Returns:
        Tuple of (cache_pid, raw_pid, has_explicit_version)
        - cache_pid: Always the raw PID (used for storage)
        - raw_pid: The raw PID without version
        - has_explicit_version: Whether the input had a version suffix
    """
    raw_pid, explicit_version = split_pid_version(pid)
    # Always use raw_pid for caching - we only keep the latest version
    return raw_pid, raw_pid, explicit_version is not None


def normalize_to_versioned_pid(pid: str, meta: Optional[dict] = None, base_dir: Optional[Path] = None) -> str:
    """
    Normalize PID to always include a version number.

    This ensures all cache directories follow the versioned format (e.g., "2511.08653v3").

    Strategy:
    1. If PID already has version → return as-is
    2. If meta has _version → use that version
    3. If base_dir provided → scan for highest existing version
    4. Default to v1

    Args:
        pid: Paper ID (may or may not include version)
        meta: Optional paper metadata dict
        base_dir: Optional base directory to scan for existing versions

    Returns:
        Versioned PID string (always has vN suffix)
    """
    raw_pid, explicit_version = split_pid_version(pid)

    # Strategy 1: Already has version
    if explicit_version:
        return pid

    # Strategy 2: Get version from metadata
    if isinstance(meta, dict):
        idv = meta.get("_idv")
        if isinstance(idv, str) and idv.strip():
            # Validate it has version
            _, v = split_pid_version(idv.strip())
            if v:
                return idv.strip()
        version = meta.get("_version")
        if version is not None:
            try:
                return f"{raw_pid}v{int(version)}"
            except Exception:
                pass

    # Strategy 3: Scan directory for existing versions
    if base_dir and base_dir.exists():
        highest_version = 0
        try:
            for entry in base_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(raw_pid):
                    _, v = split_pid_version(entry.name)
                    if v and v > highest_version:
                        highest_version = v
        except Exception:
            pass
        if highest_version > 0:
            return f"{raw_pid}v{highest_version}"

    # Strategy 4: Default to v1
    return f"{raw_pid}v1"


def find_cache_dir(base_dir: Path, pid: str, create: bool = False) -> Tuple[Optional[Path], str]:
    """
    Find or create a cache directory using raw PID (no version suffix).

    Simplified approach: arXiv/ar5iv automatically return latest version,
    so we always store with raw PID only.

    Args:
        base_dir: Base directory (e.g., Path("data/html_md"))
        pid: Paper ID (version suffix will be stripped if present)
        create: Whether to create the directory if it doesn't exist

    Returns:
        Tuple of (cache_dir_path or None, raw_pid)
    """
    raw_pid, _ = split_pid_version(pid)
    cache_dir = base_dir / raw_pid

    if cache_dir.exists():
        return cache_dir, raw_pid

    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir, raw_pid

    return None, raw_pid


# Keep for backward compatibility during migration
def find_versioned_cache_dir(
    base_dir: Path, pid: str, meta: Optional[dict] = None, create: bool = False
) -> Tuple[Optional[Path], str]:
    """
    DEPRECATED: Use find_cache_dir instead.
    This function now delegates to find_cache_dir (ignores meta parameter).
    """
    return find_cache_dir(base_dir, pid, create)


def model_cache_key(model: Optional[str]) -> str:
    """
    Generate cache key from model name.

    Args:
        model: Model name

    Returns:
        Sanitized model key for cache filenames

    Raises:
        ValueError: If model name is empty or invalid
    """
    model = (model or "").strip()
    if not model:
        raise ValueError("Model name is required for summary caching")
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    if not cleaned:
        raise ValueError("Model name is invalid for summary caching")
    return cleaned


def summary_cache_paths(cache_pid: str, model: Optional[str]) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Get cache file paths for summary.

    Args:
        cache_pid: Cache key (raw pid or pidvN)
        model: Model name

    Returns:
        Tuple of (cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock)
    """
    base_dir = Path(SUMMARY_DIR) / cache_pid
    model_key = model_cache_key(model)
    cache_file = base_dir / f"{model_key}.md"
    meta_file = base_dir / f"{model_key}.meta.json"
    lock_file = base_dir / f".{model_key}.lock"

    legacy_cache = Path(SUMMARY_DIR) / f"{cache_pid}.md"
    legacy_meta = Path(SUMMARY_DIR) / f"{cache_pid}.meta.json"
    legacy_lock = Path(SUMMARY_DIR) / f".{cache_pid}.lock"
    return cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock


def normalize_summary_source(source: Optional[str]) -> str:
    """
    Normalize summary markdown source.

    Args:
        source: Source string ("html" or "mineru")

    Returns:
        Normalized source string
    """
    src = (source or SUMMARY_MARKDOWN_SOURCE or "html").strip().lower()
    if src not in {"html", "mineru"}:
        return "html"
    return src


def summary_source_matches(meta: dict, summary_source: str) -> bool:
    """
    Check if cached summary source matches requested source.

    Args:
        meta: Summary metadata dict
        summary_source: Requested source

    Returns:
        True if sources match
    """
    cached_source = (meta.get("source") or "mineru").strip().lower()
    if cached_source not in {"html", "mineru"}:
        cached_source = "mineru"
    return cached_source == summary_source


def read_summary_meta(meta_path: Path) -> dict:
    """
    Read summary metadata from file.

    Args:
        meta_path: Path to metadata JSON file

    Returns:
        Metadata dict, empty dict on error
    """
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_summary_result(result) -> Tuple[str, dict]:
    """
    Normalize summary result from various formats.

    Args:
        result: Summary result (dict with content/meta, or string)

    Returns:
        Tuple of (content, meta)
    """
    if isinstance(result, dict):
        content = result.get("content") or ""
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    else:
        content = result if isinstance(result, str) else str(result or "")
        meta = {}
    return content, meta


def calculate_chinese_ratio(text: str) -> float:
    """
    Calculate the ratio of Chinese characters in text.

    Args:
        text: Input text

    Returns:
        float: Chinese character ratio (0.0 to 1.0)
    """
    if not text or not text.strip():
        return 0.0

    # Text after removing whitespace characters
    clean_text = re.sub(r"\s+", "", text)
    if not clean_text:
        return 0.0

    # Count Chinese characters (including Chinese punctuation)
    chinese_chars = re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", clean_text)
    chinese_count = len(chinese_chars)

    # Calculate ratio
    total_chars = len(clean_text)
    ratio = chinese_count / total_chars if total_chars > 0 else 0.0

    return ratio


def summary_quality(summary_content: str) -> Tuple[str, Optional[float]]:
    """
    Evaluate summary quality based on language settings.

    Args:
        summary_content: Summary text content

    Returns:
        Tuple of (quality, chinese_ratio). Quality is "ok" or "low_chinese".
        chinese_ratio is None for non-Chinese languages.
    """
    lang = (LLM_SUMMARY_LANG or "").strip().lower()
    if lang.startswith("zh"):
        ratio = calculate_chinese_ratio(summary_content)
        quality = "ok" if ratio >= SUMMARY_MIN_CHINESE_RATIO else "low_chinese"
        return quality, ratio
    return "ok", None


def _summary_lock_stale_seconds() -> float:
    """Get stale lock timeout from environment."""
    try:
        seconds = float(os.environ.get("ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC", "3600"))
    except Exception:
        seconds = 3600.0
    return max(0.0, seconds)


def _is_lock_stale(lock_path: Path, stale_s: float) -> bool:
    """Check if lock file is stale."""
    if stale_s <= 0:
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    except Exception:
        return False
    return age > stale_s


def acquire_summary_lock(lock_path: Path, timeout_s: int = 300) -> Optional[int]:
    """
    Acquire a file-based lock for summary caching.

    Args:
        lock_path: Path to lock file
        timeout_s: Timeout in seconds

    Returns:
        File descriptor if lock acquired, None if timeout
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    stale_s = _summary_lock_stale_seconds()
    while time.time() - start_time < timeout_s:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
            return fd
        except FileExistsError:
            if _is_lock_stale(lock_path, stale_s):
                try:
                    lock_path.unlink(missing_ok=True)
                    logger.warning(f"Removed stale summary lock: {lock_path}")
                    continue
                except Exception:
                    pass
            time.sleep(0.2)
    return None


def release_summary_lock(fd: int, lock_path: Path) -> None:
    """
    Release a file-based lock.

    Args:
        fd: File descriptor from acquire_summary_lock
        lock_path: Path to lock file
    """
    try:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to release summary lock: {e}")


def atomic_write_text(path: Path, content: str) -> None:
    """
    Atomically write text to disk to avoid partial reads.

    Args:
        path: Target file path
        content: Text content to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass


def atomic_write_json(path: Path, data: dict) -> None:
    """
    Atomically write JSON to disk.

    Args:
        path: Target file path
        data: Dict to write as JSON
    """
    atomic_write_text(path, json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    # Test code
    import argparse
    import sys

    logger.remove()
    logger.add(sys.stderr, level="TRACE")

    parser = argparse.ArgumentParser(description="Test paper summarizer")
    parser.add_argument("pid", help="Paper ID")
    parser.add_argument("--model", help="Override default model")
    args = parser.parse_args()

    logger.trace(f"Test paper ID: {args.pid}")
    if args.model:
        logger.trace(f"Using model: {args.model}")
    summary = generate_paper_summary(args.pid, model=args.model)
    logger.trace("\n" + "=" * 50)
    logger.trace("Paper summary:")
    logger.trace("=" * 50)
    logger.trace(summary.get("content"))
