#!/usr/bin/env python3
"""
Paper Summarizer Module
Features:
1. Download arxiv papers to pdfs directory
2. Parse paper content to markdown using HTML (arXiv/ar5iv) or minerU
3. Summarize papers using LLM models
"""

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
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
    MINERU_API_KEY,
    MINERU_API_POLL_INTERVAL,
    MINERU_API_TIMEOUT,
    MINERU_BACKEND,
    MINERU_DEVICE,
    MINERU_ENABLED,
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
            "api",
        }
        if raw not in supported:
            logger.trace(f"Unknown minerU backend '{raw}', fallback to pipeline")
            return "pipeline"
        return raw

    def _mineru_md_candidates(self, paper_id: str, backend: Optional[str] = None) -> List[Path]:
        base_dir = self.mineru_dir / paper_id
        auto_md = base_dir / "auto" / f"{paper_id}.md"
        vlm_md = base_dir / "vlm" / f"{paper_id}.md"
        api_md = base_dir / "api" / f"{paper_id}.md"
        if backend == "api":
            return [api_md, vlm_md, auto_md]
        elif backend and backend.startswith("vlm"):
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
            stale_s = float(os.environ.get("ARXIV_SANITY_MINERU_LOCK_STALE_SEC", "600"))
        except Exception:
            stale_s = 600.0

        while time.time() - start_time < timeout:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                # Write PID to lock file for stale detection
                try:
                    os.write(fd, f"{os.getpid()}\n{time.time()}\n".encode())
                except Exception:
                    pass
                return fd
            except FileExistsError:
                # Check if lock is stale (by time or dead process)
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    # Strategy 1: Time-based stale detection
                    if stale_s > 0 and age > stale_s:
                        lock_path.unlink(missing_ok=True)
                        logger.trace(f"Removed stale minerU lock (age={age:.0f}s): {lock_path}")
                        continue
                    # Strategy 2: PID-based stale detection (check if owner process is dead)
                    try:
                        lock_content = lock_path.read_text().strip()
                        lines = lock_content.split("\n")
                        if lines and lines[0]:
                            owner_pid = int(lines[0])
                            # Check if process is still running
                            try:
                                os.kill(owner_pid, 0)  # Signal 0 just checks if process exists
                            except OSError:
                                # Process doesn't exist, lock is stale
                                lock_path.unlink(missing_ok=True)
                                logger.trace(f"Removed orphan minerU lock (dead PID {owner_pid}): {lock_path}")
                                continue
                    except (ValueError, IndexError, OSError):
                        pass  # Can't read PID, fall back to time-based only
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
            stale_s = float(os.environ.get("ARXIV_SANITY_MINERU_LOCK_STALE_SEC", "600"))
        except Exception:
            stale_s = 600.0

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
                    # Check if lock is stale (by time or dead process)
                    try:
                        age = time.time() - slot_lock_path.stat().st_mtime
                        # Strategy 1: Time-based stale detection
                        if stale_s > 0 and age > stale_s:
                            slot_lock_path.unlink(missing_ok=True)
                            logger.trace(f"Removed stale GPU slot lock (age={age:.0f}s): {slot_lock_path}")
                            continue
                        # Strategy 2: PID-based stale detection (check if owner process is dead)
                        try:
                            lock_content = slot_lock_path.read_text().strip()
                            lines = lock_content.split("\\n")
                            if lines:
                                owner_pid = int(lines[0])
                                # Check if process is still running
                                try:
                                    os.kill(owner_pid, 0)  # Signal 0 just checks if process exists
                                except OSError:
                                    # Process doesn't exist, lock is stale
                                    slot_lock_path.unlink(missing_ok=True)
                                    logger.trace(
                                        f"Removed orphan GPU slot lock (dead PID {owner_pid}): {slot_lock_path}"
                                    )
                                    continue
                        except (ValueError, IndexError, OSError):
                            pass  # Can't read PID, fall back to time-based only
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
        self, pdf_path: Optional[Path], cache_pid: Optional[str] = None, cached_version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Parse PDF to Markdown using minerU (with multi-process lock protection)

        Args:
            pdf_path: PDF file path (can be None for API backend)
            cache_pid: Optional paper ID to use for output directory (should be raw PID).
                       If not provided, uses the PDF filename (stem). Required when pdf_path is None.
            cached_version: Version of the PDF that was actually downloaded (e.g., "3")

        Returns:
            Generated Markdown file path, None if failed
        """
        try:
            # Use cache_pid if provided, otherwise use PDF filename
            pdf_name = cache_pid or (pdf_path.stem if pdf_path else None)
            if not pdf_name:
                logger.error("cache_pid is required when pdf_path is None")
                return None
            output_dir = self.mineru_dir
            backend = self._normalize_mineru_backend()

            # Handle API backend separately
            if backend == "api":
                return self._parse_pdf_with_mineru_api(pdf_path, pdf_name, cached_version)

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

                    # Normalize image filenames and references
                    output_path = existing_md_path.parent
                    self._normalize_mineru_images(output_path, existing_md_path)

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
            if pdf_path:
                try:
                    pdf_path.unlink(missing_ok=True)
                    logger.trace(f"Parse exception, deleted PDF source file: {pdf_path}")
                except Exception as e:
                    logger.trace(f"Failed to delete PDF file: {e}")
            return None

    def _parse_pdf_with_mineru_api(
        self, pdf_path: Optional[Path], pdf_name: str, cached_version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Parse PDF using MinerU API service (following reference implementation)
        """
        # Check API key
        if not MINERU_API_KEY or not MINERU_API_KEY.strip():
            logger.error("MinerU API key not configured")
            raise ValueError("MINERU_API_KEY_MISSING")

        # Check if already parsed
        existing_md_path = self._find_mineru_markdown(pdf_name, backend="api")
        if existing_md_path:
            logger.trace(f"Markdown file already exists: {existing_md_path}")
            return existing_md_path

        # Acquire file lock
        lock_path = self.mineru_dir / f".{pdf_name}.lock"
        lock_fd = self._acquire_file_lock(lock_path, timeout=300)
        if lock_fd is None:
            logger.trace(f"Failed to acquire lock for {pdf_name}, skipping")
            return None

        try:
            # Check again after acquiring lock
            existing_md_path = self._find_mineru_markdown(pdf_name, backend="api")
            if existing_md_path:
                logger.trace(f"File generated during lock wait: {existing_md_path}")
                return existing_md_path

            # === Step 1: Submit task (exactly like reference code) ===
            url = "https://mineru.net/api/v4/extract/task"
            header = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MINERU_API_KEY.strip()}",
            }

            # Build arXiv PDF URL
            arxiv_url = f"https://arxiv.org/pdf/{pdf_name}.pdf"
            if cached_version:
                arxiv_url = f"https://arxiv.org/pdf/{pdf_name}v{cached_version}.pdf"

            data = {
                "url": arxiv_url,
                "model_version": "vlm",
                "data_id": f"arxiv_{pdf_name.replace('.', '_')}",
                "enable_formula": True,
                "enable_table": True,
            }

            logger.trace(f"Submitting task for: {arxiv_url}")
            res = requests.post(url, headers=header, json=data, timeout=30)

            # Check HTTP status
            if res.status_code == 401:
                logger.error("MinerU API: login required (401)")
                raise ValueError("MINERU_API_EXPIRED")

            result = res.json()
            logger.trace(f"Submit response: {result}")

            # Check API response code (0 = success)
            if result.get("code") != 0:
                error_msg = result.get("msg", "unknown error")
                logger.error(f"MinerU API error: code={result.get('code')}, msg={error_msg}")
                raise ValueError(f"MINERU_API_ERROR: {error_msg}")

            # Get task_id directly (like reference code)
            try:
                task_id = result["data"]["task_id"]
            except (KeyError, TypeError):
                logger.error(f"Failed to get task_id from response: {result}")
                raise ValueError("MINERU_API_INVALID_RESPONSE")

            logger.info(f"MinerU task created: {task_id}")

            # === Step 2: Poll for completion (with timeout) ===
            start_time = time.time()
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > MINERU_API_TIMEOUT:
                    logger.error(f"Task {task_id} timed out after {MINERU_API_TIMEOUT}s")
                    raise ValueError("MINERU_API_TIMEOUT")

                time.sleep(MINERU_API_POLL_INTERVAL)

                query_url = f"https://mineru.net/api/v4/extract/task/{task_id}"
                res = requests.get(query_url, headers=header, timeout=30)
                result = res.json()

                # Check API response code
                if result.get("code") != 0:
                    error_msg = result.get("msg", "unknown error")
                    logger.error(f"MinerU query error: code={result.get('code')}, msg={error_msg}")
                    raise ValueError(f"MINERU_API_QUERY_ERROR: {error_msg}")

                # Get state directly (like reference code)
                try:
                    state = result["data"]["state"]
                except (KeyError, TypeError):
                    logger.error(f"Failed to get state from response: {result}")
                    raise ValueError("MINERU_API_QUERY_FAILED")

                logger.trace(f"Task {task_id} state: {state} (elapsed: {elapsed:.0f}s)")

                if state == "done":
                    download_url = result["data"]["full_zip_url"]
                    logger.info(f"Task completed: {download_url}")
                    break
                elif state in ["running", "pending", "converting"]:
                    # converting: 格式转换中，也需要继续等待
                    continue
                elif state == "failed":
                    err_msg = result["data"].get("err_msg", "unknown error")
                    logger.error(f"Task failed: {err_msg}")
                    raise ValueError(f"MINERU_API_TASK_FAILED: {err_msg}")
                else:
                    logger.error(f"Unknown task state: {state}, response: {result}")
                    raise ValueError(f"MINERU_API_UNKNOWN_STATE: {state}")

            # === Step 3: Download and extract (exactly like reference code) ===
            output_path = self.mineru_dir / pdf_name / "api"
            output_path.mkdir(parents=True, exist_ok=True)

            logger.trace("Downloading ZIP...")
            response = requests.get(download_url, timeout=120)

            logger.trace(f"Extracting to {output_path}")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(output_path)

            # Find and rename markdown file
            md_files = list(output_path.rglob("*.md"))
            if not md_files:
                logger.error("No markdown file found in result")
                raise ValueError("MINERU_API_NO_MARKDOWN")

            source_md = md_files[0]
            target_md = output_path / f"{pdf_name}.md"

            if source_md != target_md:
                import shutil

                shutil.copy2(source_md, target_md)
                logger.trace(f"Copied: {source_md} -> {target_md}")

            # Write metadata and cleanup
            self._write_mineru_meta(pdf_name, "api", cached_version=cached_version)
            self._cleanup_mineru_output(self.mineru_dir / pdf_name)
            self._normalize_mineru_images(output_path, target_md)

            logger.info(f"Successfully parsed: {target_md}")
            return target_md

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"MinerU API parse failed: {e}")
            return None
        finally:
            if lock_fd is not None:
                self._release_file_lock(lock_fd, lock_path)
                logger.trace(f"Released file lock for {pdf_name}")

    def _cleanup_mineru_output(self, output_path: Path):
        """
        Clean up minerU output directory, keep only images and markdown files

        Args:
            output_path: minerU output paper directory path (e.g. data/mineru/2507.01679)
        """
        try:
            output_dirs = []
            for dir_name in ("auto", "vlm", "api"):
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

    def _normalize_mineru_images(self, output_path: Path, markdown_path: Path):
        """
        Normalize image filenames and references in markdown to sequential format (image-1, image-2, ...)

        Args:
            output_path: minerU output paper directory path (e.g. data/mineru/2507.01679/auto)
            markdown_path: Path to the markdown file
        """
        try:
            if not markdown_path.exists():
                logger.trace(f"Markdown file not found: {markdown_path}")
                return

            # Read markdown content
            with open(markdown_path, encoding="utf-8") as f:
                markdown_content = f.read()

            images_dir = output_path / "images"
            if not images_dir.exists() or not images_dir.is_dir():
                logger.trace(f"Images directory not found: {images_dir}")
                return

            # Find all image references in markdown (pattern: ![alt](images/filename.ext))
            # Use ordered dict to preserve order and avoid duplicates
            image_pattern = re.compile(r"!\[([^\]]*)\]\(images/([^)]+)\)")
            matches = image_pattern.findall(markdown_content)

            if not matches:
                logger.trace("No image references found in markdown")
                return

            # Track unique images in order of first appearance
            seen = set()
            ordered_images = []
            for alt_text, img_filename in matches:
                if img_filename not in seen:
                    seen.add(img_filename)
                    ordered_images.append(img_filename)

            # Create rename mapping: old_filename -> new_filename
            rename_map = {}
            for idx, old_filename in enumerate(ordered_images, start=1):
                old_path = images_dir / old_filename
                if not old_path.exists():
                    logger.trace(f"Image file not found: {old_path}")
                    continue

                # Preserve file extension
                ext = old_path.suffix
                new_filename = f"image-{idx}{ext}"
                rename_map[old_filename] = new_filename

            if not rename_map:
                logger.trace("No images to rename")
                return

            # Rename image files
            for old_filename, new_filename in rename_map.items():
                old_path = images_dir / old_filename
                new_path = images_dir / new_filename

                try:
                    # Handle case where target already exists
                    if new_path.exists():
                        if old_path.samefile(new_path):
                            continue
                        # Use temporary name to avoid conflicts
                        temp_path = images_dir / f".tmp_{new_filename}"
                        old_path.rename(temp_path)
                        old_path = temp_path

                    old_path.rename(new_path)
                    logger.trace(f"Renamed image: {old_filename} -> {new_filename}")
                except Exception as e:
                    logger.trace(f"Failed to rename {old_filename}: {e}")

            # Update markdown references
            updated_content = markdown_content
            for old_filename, new_filename in rename_map.items():
                # Replace all occurrences of the old filename
                old_ref = f"images/{old_filename}"
                new_ref = f"images/{new_filename}"
                updated_content = updated_content.replace(old_ref, new_ref)

            # Write updated markdown
            self._atomic_write_text(markdown_path, updated_content)
            logger.trace(f"Updated {len(rename_map)} image references in markdown")

            # Clean up unreferenced image files (e.g., hash-named files from MinerU API)
            referenced_images = set(rename_map.values())  # New filenames that are referenced
            for img_file in images_dir.iterdir():
                if img_file.is_file() and img_file.name not in referenced_images:
                    try:
                        img_file.unlink()
                        logger.trace(f"Deleted unreferenced image: {img_file.name}")
                    except Exception as e:
                        logger.trace(f"Failed to delete unreferenced image {img_file.name}: {e}")

        except Exception as e:
            logger.trace(f"Failed to normalize minerU images: {e}")

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
                # English prompt - Academic technical blog style
                prompt = rf"""
You are an experienced academic technical blogger who excels at transforming research papers into rigorous yet accessible technical blog posts with proper academic conventions.

## Task Objective
Transform the paper below into an academic-style technical blog post that enables readers to understand the key techniques, core results, and important details without reading the original paper, while maintaining scholarly rigor.

## Target Audience
Technical professionals and researchers with foundational knowledge in the relevant field

## Blog Content Requirements

### Required Sections
1. **Research Background & Motivation**: Explain what problem the research addresses and why it matters. Cite relevant context from the paper.
2. **Core Contributions**: Distill 1-3 key innovations with precise technical claims
3. **Method/Algorithm Details**: This is the focus section—provide detailed technical explanations with:
   - Formal problem definitions with proper mathematical notation
   - Key algorithmic steps or theoretical derivations
   - Intuitive interpretations alongside formal descriptions
4. **Theoretical Analysis** (if applicable): Present key theorems, lemmas, or complexity analysis
5. **Experimental Results**: Summarize main findings with quantitative comparisons (only use data explicitly provided in the paper)
6. **Limitations & Future Directions**: Discuss shortcomings or future research opportunities

### Figure Citation Guidelines
For figures in the original paper, first infer what the figure depicts from the surrounding context. Include figures that significantly aid understanding (typically 2-4 figures).

**Figure Insertion Format**:
```
![Figure N: Descriptive caption explaining the figure content and its significance](images/original-filename.png)
```

**Cross-Reference Rules**:
- Number figures consecutively starting from 1 (Figure 1, Figure 2, Figure 3...)
- **Always cross-reference figures in the text before or near their appearance**, using academic phrasing:
  - "As illustrated in Figure 1, the proposed architecture consists of..."
  - "Figure 2 presents the comparative results, showing that..."
  - "The attention mechanism (Figure 3) enables..."
- Keep original file paths and filenames unchanged
- Do not wrap figures in Markdown blockquotes (>)
- Only include figures that actually exist in the paper content
- Place figures near their first textual reference

### Mathematical Notation Guidelines
**Inline vs. Display Math**:
- Use `$...$` for inline math: variables ($x$, $\theta$), short expressions ($O(n \log n)$), or references to equations
- Use `$$...$$` for display math: important equations, definitions, or derivations that deserve emphasis

**Equation Formatting**:
- For key equations, add descriptive context before and after:
  ```
  The loss function is defined as:
  $$\mathcal{{L}} = \mathbb{{E}}_{{x \sim p_{{data}}}} \left[ \log D(x) \right] + \mathbb{{E}}_{{z \sim p_z}} \left[ \log(1 - D(G(z))) \right]$$
  where $D$ denotes the discriminator and $G$ denotes the generator.
  ```
- Use aligned environments for multi-line derivations:
  ```
  $$\begin{{aligned}}
  \nabla_\theta J(\theta) &= \mathbb{{E}}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right] \\
  &\approx \frac{{1}}{{N}} \sum_{{i=1}}^N \sum_t \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) R(\tau^i)
  \end{{aligned}}$$
  ```

**Theorem/Definition Formatting**:
Use blockquotes (`>`) to visually distinguish theorems, definitions, and lemmas from regular text:

```
> **Definition 1** (Markov Decision Process). *A Markov Decision Process (MDP) is a tuple $(\mathcal{{S}}, \mathcal{{A}}, P, R, \gamma)$ where $\mathcal{{S}}$ is the state space, $\mathcal{{A}}$ is the action space, $P$ is the transition probability, $R$ is the reward function, and $\gamma \in [0,1)$ is the discount factor.*
```

```
> **Theorem 1** (Convergence Guarantee). *Under Assumptions 1-3, Algorithm 1 converges to a stationary point at rate $O(1/\sqrt{{T}})$.*
>
> *Proof sketch.* The key insight is that... (brief outline of the proof approach)
```

- Use consistent formatting: **bold** for theorem/definition label, *italics* for the statement
- Include proof sketches when they provide insight, indented within the same blockquote
- Number theorems, definitions, and lemmas consecutively within each category

**LaTeX Best Practices**:
- Use `\mathbb` for number sets ($\mathbb{{R}}$, $\mathbb{{E}}$), `\mathcal` for calligraphic letters ($\mathcal{{L}}$, $\mathcal{{D}}$)
- Use `\text` for text within math: $P(\text{{success}})$
- Prefer `\cdot` over `*` for multiplication, `\times` for cross product
- Convert complex custom macros to standard LaTeX for compatibility

### Table Usage Guidelines
Use Markdown tables to present structured information clearly. Tables are especially effective for:

**When to Use Tables**:
- Comparing experimental results across methods/datasets
- Summarizing hyperparameters or model configurations
- Listing notation definitions
- Comparing related work or method characteristics

**Table Formatting**:
```
| Method | Accuracy | F1 Score | Latency (ms) |
|--------|----------|----------|-------------|
| Baseline | 85.2 | 83.1 | 12.3 |
| Proposed | **91.4** | **89.7** | 15.1 |
```

**Best Practices**:
- Add a brief caption or description before the table
- Use **bold** to highlight best results
- Keep tables concise; avoid overly wide tables
- Align numerical data for easy comparison
- Only include tables when they genuinely improve clarity over prose

### Writing Style
- Use hierarchical headings to organize content (#, ##, ###, etc.)
- Every paragraph should have substantive content; avoid empty descriptions
- Define technical terms and notation precisely when first introduced
- Maintain formal academic tone while remaining accessible
- Use precise language: "achieves", "demonstrates", "outperforms" rather than "is good at"
- Connect sections with logical transitions

## Paper Content
{markdown_content}

## Output Format Requirements

Please output strictly according to the following structure:

```markdown
# [Blog Title: Concise and impactful, reflecting the paper's core contribution]

[Main content: Organized by the sections above, with figures and equations properly integrated]

## TL;DR

[2-3 sentences summarizing: What problem does the paper solve, what method is proposed, and what are the key results]
```

## Critical Constraints
1. **Academic Rigor**: All content must be based on the paper; maintain scholarly precision
2. **Data Accuracy**: Quote exact numbers from the paper; never guess or fabricate data
3. **Language**: Write entirely in **English** with formal academic style
4. **Figure Integrity**: Only include figures that exist in the paper; always cross-reference them in text
5. **Mathematical Precision**: Ensure all equations are syntactically correct and properly contextualized
"""
            else:
                # Chinese prompt (default) - Academic technical blog style
                prompt = rf"""
你是一位经验丰富的学术技术博主，擅长将研究论文解读为既严谨又通俗易懂的技术博客，同时保持学术规范，专业但不晦涩。

## 任务目标
将下方论文转化为一篇学术风格的技术博客，使读者无需阅读原文即可理解关键技术、核心结果与重要细节，同时保持学术严谨性。

## 目标受众
具有本领域基础知识的技术人员和研究者

## 博客内容要求

### 必须包含的章节
1. **研究背景与动机**：解释该研究要解决什么问题、为什么重要，引用论文中的相关背景
2. **核心贡献**：提炼 1-3 个核心创新点，使用精确的技术表述
3. **方法/算法详解**：这是重点章节，需详细介绍技术方案，包括：
   - 使用规范数学符号的形式化问题定义
   - 关键算法步骤或理论推导
   - 在形式化描述旁配合直观解释
4. **理论分析**（如适用）：呈现关键定理、引理或复杂度分析
5. **实验结果**：总结主要发现并进行定量比较（仅使用论文中明确给出的数据）
6. **局限性与展望**：讨论方法的不足或未来研究方向

### 图片引用规范
对于论文原文中的图片，请先根据上下文判断图片内容。选择对理解内容有重要帮助的图片纳入博客（通常 2-4 张）。

**图片插入格式**：
```
![图N：描述性标题，说明图片内容及其意义](images/原始文件名.png)
```

**交叉引用规则**：
- 图片编号从 1 开始，必须连续递增（图1、图2、图3...）
- **必须在图片出现前或附近的正文中引用该图片**，使用学术化表述：
  - "如图1所示，所提出的架构由...组成"
  - "图2展示了对比实验结果，表明..."
  - "注意力机制（图3）使得模型能够..."
- 保持原始文件路径和文件名不变
- 不要用 Markdown 引用块（>）包裹图片
- 只引用论文内容中实际存在的图片
- 图片应放置在其首次文字引用的附近

### 数学公式规范
**行内公式 vs 独立公式**：
- 使用 `$...$` 表示行内公式：变量（$x$、$\theta$）、简短表达式（$O(n \log n)$）或对公式的引用
- 使用 `$$...$$` 表示独立公式：重要方程式、定义或需要强调的推导

**公式格式化**：
- 对于关键公式，在前后添加描述性上下文：
  ```
  损失函数定义为：
  $$\mathcal{{L}} = \mathbb{{E}}_{{x \sim p_{{data}}}} \left[ \log D(x) \right] + \mathbb{{E}}_{{z \sim p_z}} \left[ \log(1 - D(G(z))) \right]$$
  其中 $D$ 表示判别器，$G$ 表示生成器。
  ```
- 使用 aligned 环境处理多行推导：
  ```
  $$\begin{{aligned}}
  \nabla_\theta J(\theta) &= \mathbb{{E}}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right] \\
  &\approx \frac{{1}}{{N}} \sum_{{i=1}}^N \sum_t \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) R(\tau^i)
  \end{{aligned}}$$
  ```

**定理/定义格式**：
使用引用块（`>`）将定理、定义和引理与正文在视觉上区分开来：

```
> **定义 1**（马尔可夫决策过程）。*马尔可夫决策过程（MDP）是一个五元组 $(\mathcal{{S}}, \mathcal{{A}}, P, R, \gamma)$，其中 $\mathcal{{S}}$ 是状态空间，$\mathcal{{A}}$ 是动作空间，$P$ 是转移概率，$R$ 是奖励函数，$\gamma \in [0,1)$ 是折扣因子。*
```

```
> **定理 1**（收敛性保证）。*在假设 1-3 成立的条件下，算法 1 以 $O(1/\sqrt{{T}})$ 的速率收敛到驻点。*
>
> *证明概要。* 核心思路是...（简要说明证明方法）
```

- 使用统一格式：**粗体**标注定理/定义标签，*斜体*表示陈述内容
- 当证明思路有助于理解时，在同一引用块内缩进添加证明概要
- 定理、定义和引理在各自类别内连续编号

**LaTeX 最佳实践**：
- 使用 `\mathbb` 表示数集（$\mathbb{{R}}$、$\mathbb{{E}}$），`\mathcal` 表示花体字母（$\mathcal{{L}}$、$\mathcal{{D}}$）
- 在数学环境中使用 `\text` 插入文本：$P(\text{{成功}})$
- 乘法优先使用 `\cdot` 而非 `*`，叉积使用 `\times`
- 将复杂的自定义宏转换为标准 LaTeX 以确保兼容性

### 表格使用规范
适当使用 Markdown 表格来清晰呈现结构化信息，提高可读性。

**适用场景**：
- 对比不同方法/数据集上的实验结果
- 汇总超参数或模型配置
- 列出符号定义说明
- 比较相关工作或方法特性

**表格格式**：
```
| 方法 | 准确率 | F1 分数 | 延迟 (ms) |
|------|--------|---------|----------|
| 基线方法 | 85.2 | 83.1 | 12.3 |
| 本文方法 | **91.4** | **89.7** | 15.1 |
```

**最佳实践**：
- 在表格前添加简要说明或标题
- 使用**粗体**突出最佳结果
- 保持表格简洁，避免过宽的表格
- 数值数据对齐以便于比较
- 仅在表格确实比文字描述更清晰时使用

### 写作风格
- 使用层级标题组织内容（#、##、### 等）
- 每段都应有实质性内容，避免空洞描述
- 专业术语和符号在首次出现时给出精确定义
- 保持正式的学术语调，同时确保可读性
- 使用精确的表述："实现了"、"证明了"、"优于"，而非"表现不错"
- 各章节之间使用逻辑性过渡衔接

## 论文原文
{markdown_content}

## 输出格式要求

请严格按以下结构输出：

```markdown
# [博客标题：简洁有力，体现论文核心贡献]

[正文内容：按上述章节组织，图片和公式应恰当融入行文]

## TL;DR

[2-3句话概括：论文解决了什么问题、提出了什么方法、取得了哪些关键成果]
```

## 重要约束
1. **学术严谨**：所有内容必须基于论文，保持学术精确性
2. **数据准确**：引用论文中的确切数字，不得猜测或编造数据
3. **语言要求**：全文使用中文撰写，保持正式学术风格
4. **图片规范**：只引用论文中实际存在的图片，必须在正文中交叉引用
5. **公式精确**：确保所有公式语法正确，并有恰当的上下文说明
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
                temperature=0.6,
                top_p=0.95,
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
        Parse summary content from LLM output and reorganize with TL;DR after title.

        Handles:
        - Code block wrappers (```markdown ... ```)
        - Various TL;DR formats (headings, bold, blockquote, plain text)

        Output format: Title (if exists), then TL;DR, then remaining content.

        Args:
            summary: Original summary content

        Returns:
            Reorganized summary content with TL;DR after title
        """
        summary = summary.strip()
        if not summary:
            return summary

        # Remove code block markers if present at start or end
        lines = summary.split("\n")
        # Remove opening ``` (with optional language tag) if at start
        if lines and re.match(r"^```\s*(?:markdown|md|text)?\s*$", lines[0].strip(), re.IGNORECASE):
            lines = lines[1:]
        # Remove closing ``` if at end
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        summary = "\n".join(lines).strip()

        # Extract TL;DR section using a unified approach
        # First, find any TL;DR marker and its content
        tldr_content = None
        tldr_start = None
        tldr_end = None
        is_blockquote = False

        # Unified pattern to find TL;DR marker (handles heading, bold, blockquote, plain)
        # Group 1: optional blockquote prefix, Group 2: the TL;DR marker line
        tldr_marker_pattern = r"^(>?\s*)(#{1,6}\s*TL;DR|\*{1,2}TL;DR\*{1,2}|TL;DR):?\s*$"

        for i, line in enumerate(summary.split("\n")):
            marker_match = re.match(tldr_marker_pattern, line.strip(), re.IGNORECASE)
            if marker_match:
                is_blockquote = line.strip().startswith(">")
                tldr_start = summary.find(line)

                # Find content after marker until next heading or end
                remaining_lines = summary.split("\n")[i + 1 :]
                content_lines = []

                for remaining_line in remaining_lines:
                    # Stop at next heading (not in blockquote context)
                    if re.match(r"^#{1,6}\s+\S", remaining_line):
                        break
                    # For blockquote TL;DR, stop when blockquote ends
                    if is_blockquote and remaining_line.strip() and not remaining_line.strip().startswith(">"):
                        break
                    content_lines.append(remaining_line)

                # Calculate end position
                content_text = "\n".join(content_lines)
                tldr_end = tldr_start + len(line) + 1 + len(content_text)

                # Clean content
                if is_blockquote:
                    # Remove > prefix from each line
                    cleaned_lines = [re.sub(r"^>\s?", "", l) for l in content_lines]
                    tldr_content = "\n".join(cleaned_lines).strip()
                else:
                    tldr_content = content_text.strip()

                break

        if tldr_content:
            # Remove TL;DR section from original position
            blog_content = summary[:tldr_start].strip() + "\n\n" + summary[tldr_end:].strip()
            blog_content = re.sub(r"\n{3,}", "\n\n", blog_content).strip()

            if blog_content:
                # Check if blog content starts with a title (# Title)
                title_match = re.match(r"^(#\s+[^\n]+)\n*", blog_content)
                if title_match:
                    title = title_match.group(1).strip()
                    remaining = blog_content[title_match.end() :].strip()
                    normalized_tldr = f"## TL;DR\n\n{tldr_content}"
                    if remaining:
                        return f"{title}\n\n{normalized_tldr}\n\n{remaining}"
                    else:
                        return f"{title}\n\n{normalized_tldr}"
                else:
                    normalized_tldr = f"## TL;DR\n\n{tldr_content}"
                    return f"{normalized_tldr}\n\n{blog_content}"
            else:
                return f"## TL;DR\n\n{tldr_content}"

        # Fallback: no TL;DR found, just clean up
        return re.sub(r"\n{3,}", "\n\n", summary).strip()

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
                # Check if MinerU is enabled before fallback
                if not MINERU_ENABLED:
                    return self._build_summary_result(
                        "# PDF Parsing Service Unavailable\n\nThe PDF parsing service is currently disabled. Unable to generate paper summary. Please contact the administrator to enable the MinerU service or use HTML parsing."
                    )

            # MinerU is explicitly requested, check if enabled
            if not MINERU_ENABLED:
                return self._build_summary_result(
                    "# PDF Parsing Service Unavailable\n\nThe PDF parsing service is currently disabled. Unable to generate paper summary. Please contact the administrator to enable the MinerU service or use HTML parsing."
                )

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
            # API backend doesn't need to download PDF, it uses arXiv URL directly
            backend = self._normalize_mineru_backend()
            if backend == "api":
                logger.info(f"Using MinerU API backend for {cache_pid} (skipping PDF download)")
                pdf_path = None
                cached_version = None
            else:
                logger.info(f"Downloading {cache_pid}.pdf ...")
                pdf_path, cached_version = self.download_arxiv_paper(cache_pid)
                if not pdf_path:
                    return self._build_summary_result("# Error\n\nUnable to download paper PDF")

            # Step 2: Parse PDF to Markdown using minerU
            if backend != "api":
                logger.info(f"Parsing {cache_pid}.pdf ...")
            try:
                md_path = self.parse_pdf_with_mineru(pdf_path, cache_pid=cache_pid, cached_version=cached_version)
            except ValueError as e:
                error_msg = str(e)
                if "MINERU_API_EXPIRED" in error_msg:
                    return self._build_summary_result(
                        "# MinerU API Service Unavailable\n\n"
                        "The MinerU API key has expired or authentication failed. "
                        "Please contact the administrator to update the API credentials."
                    )
                elif "MINERU_API_KEY_MISSING" in error_msg:
                    return self._build_summary_result(
                        "# MinerU API Configuration Error\n\n"
                        "MinerU API key is not configured. Please contact the administrator."
                    )
                else:
                    return self._build_summary_result(
                        f"# MinerU API Error\n\n"
                        f"Failed to parse PDF using MinerU API. Please try again later or contact the administrator.\n\n"
                        f"Error details: {error_msg}"
                    )

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
        seconds = float(os.environ.get("ARXIV_SANITY_SUMMARY_LOCK_STALE_SEC", "600"))
    except Exception:
        seconds = 600.0
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
            # Write PID to lock file for stale detection
            try:
                os.write(fd, f"{os.getpid()}\n{time.time()}\n".encode())
            except Exception:
                pass
            return fd
        except FileExistsError:
            # Strategy 1: Time-based stale detection
            if _is_lock_stale(lock_path, stale_s):
                try:
                    lock_path.unlink(missing_ok=True)
                    logger.warning(f"Removed stale summary lock (time): {lock_path}")
                    continue
                except Exception:
                    pass
            # Strategy 2: PID-based stale detection (check if owner process is dead)
            try:
                lock_content = lock_path.read_text().strip()
                lines = lock_content.split("\n")
                if lines and lines[0]:
                    owner_pid = int(lines[0])
                    try:
                        os.kill(owner_pid, 0)  # Signal 0 just checks if process exists
                    except OSError:
                        # Process doesn't exist, lock is stale
                        lock_path.unlink(missing_ok=True)
                        logger.warning(f"Removed orphan summary lock (dead PID {owner_pid}): {lock_path}")
                        continue
            except (ValueError, IndexError, OSError):
                pass  # Can't read PID, fall back to time-based only
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
