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
    SUMMARY_HTML_SOURCES,
    SUMMARY_MARKDOWN_SOURCE,
    VLLM_MINERU_PORT,
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

    def download_arxiv_paper(self, pid: str) -> Optional[Path]:
        """
        Download arXiv paper PDF

        Args:
            pid: Paper ID, e.g. "2301.00001"

        Returns:
            Downloaded PDF file path, None if failed
        """
        try:
            # arXiv PDF URL format
            pdf_url = f"https://arxiv.org/pdf/{pid}"
            pdf_path = self.pdfs_dir / f"{pid}.pdf"

            # If file already exists, return directly
            if pdf_path.exists():
                logger.trace(f"PDF file already exists: {pdf_path}")
                return pdf_path

            # Use atomic file write with temporary file
            logger.trace(f"Downloading paper {pid} ...")
            with requests.get(pdf_url, stream=True, timeout=30) as response:
                response.raise_for_status()

                # Write to temporary file first, then atomic rename
                temp_fd, temp_path = tempfile.mkstemp(dir=self.pdfs_dir, prefix=f"{pid}_", suffix=".pdf.tmp")
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
                        return pdf_path if pdf_path.exists() else None

                except Exception:
                    # Clean up temporary file on error
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                    raise

            logger.trace(f"Paper download complete: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.trace(f"Failed to download paper {pid}: {e}")
            return None

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
        PaperSummarizer._atomic_write_text(path, json.dumps(data))

    _PID_VERSION_RE = re.compile(r"^(?P<raw>.+)v(?P<ver>\d+)$")

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

    def _resolve_versioned_pid(self, pid: str) -> str:
        raw_pid, explicit_version = self._split_pid_version(pid)
        if explicit_version:
            return pid
        version = self._get_latest_version_from_meta(raw_pid)
        if not version:
            return raw_pid
        return f"{raw_pid}v{version}"

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

    def _html_cache_paths(self, cache_pid: str) -> Tuple[Path, Path]:
        md_path = self.html_md_dir / f"{cache_pid}.md"
        meta_path = self.html_md_dir / f"{cache_pid}.meta.json"
        return md_path, meta_path

    def _read_html_meta(self, cache_pid: str) -> dict:
        _, meta_path = self._html_cache_paths(cache_pid)
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _read_html_markdown_cache(self, cache_pid: str) -> Optional[str]:
        md_path, _ = self._html_cache_paths(cache_pid)
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
        md_path, meta_path = self._html_cache_paths(cache_pid)
        self._atomic_write_text(md_path, markdown)
        meta = {"updated_at": time.time(), "source": source, "url": url}
        self._atomic_write_json(meta_path, meta)

    def _html_tools(self):
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify as html_to_markdown
        except Exception as e:
            logger.trace(f"HTML parsing dependencies missing: {e}")
            return None, None
        return BeautifulSoup, html_to_markdown

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

    def _html_to_markdown(self, html: str) -> Optional[str]:
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

    def _get_markdown_from_html(self, pid: str) -> Tuple[Optional[str], Optional[str]]:
        cache_pid = self._resolve_versioned_pid(pid)
        cached = self._read_html_markdown_cache(cache_pid)
        if cached:
            meta = self._read_html_meta(cache_pid)
            return cached, meta.get("source")

        for source in self._parse_html_sources():
            html, url = self._fetch_html_from_source(cache_pid, source)
            if not html:
                continue
            markdown = self._html_to_markdown(html)
            if not markdown:
                logger.trace(f"HTML parse produced empty markdown for {pid} from {source}")
                continue
            self._write_html_markdown_cache(cache_pid, markdown, source, url)
            return markdown, source

        return None, None

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

    def parse_pdf_with_mineru(self, pdf_path: Path) -> Optional[Path]:
        """
        Parse PDF to Markdown using minerU (with multi-process lock protection)

        Args:
            pdf_path: PDF file path

        Returns:
            Generated Markdown file path, None if failed
        """
        try:
            pdf_name = pdf_path.stem
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

                # Build minerU command (backend selection per MinerU CLI docs)
                # cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir), "-l", "en", "-d", "cuda", "--vram", "2"]
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
                if backend == "vlm-http-client":
                    cmd.extend(["-u", f"http://127.0.0.1:{VLLM_MINERU_PORT}"])

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
1. **Must Include**: Research motivation & background, core contributions (1-3 points),
   detailed method/algorithm explanation, key experimental results, limitations or future directions
2. **Focus on Methods**: The method/algorithm section should be explained in detail with
   intuitive explanations, formulas with mathematical derivations where necessary,
   or analogies for complex concepts
3. **Source Fidelity**: All statements must be based on the original paper;
   do not speculate on content the authors did not explicitly state
4. **Figure/Table Handling**: Clearly describe the meaning of key figures and tables in text
5. **Balanced Detail**: Include as many important paper details as possible
   without significantly increasing reading burden

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
The detailed technical blog content

## TL;DR
A concise 2-3 sentence summary of the paper's core contribution and significance
</Output Format>

<Important Notes>
1. Ensure the blog accurately reflects the paper content; **do not add information not in the paper**
2. Do not guess or fabricate data; skip quantitative analysis rather than summarize incorrect data
3. Please write in **English**
4. Make the content accessible yet technically rigorous
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
4. **图表处理**：对论文核心图表用文字清晰描述其含义
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
具体的技术博客内容

## TL;DR
用2-3句话概括论文的核心贡献和意义

</输出格式>

<注意事项>
1. 请确保博客准确反映论文内容，**不要添加论文中没有的信息**
2. 不要猜测更不要臆造数据，宁可跳过定量分析，也不要总结错误数据
3. 请一定用**中文**撰写
4. 保持内容易读的同时确保技术严谨性
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
        Expected input format: Technical Blog section followed by TL;DR section.
        Output format: TL;DR first, then Technical Blog.

        Args:
            summary: Original summary content

        Returns:
            Reorganized summary content with TL;DR first
        """
        # Extract TL;DR section (match until end of string or next ## heading)
        tldr_match = re.search(r"##\s*TL;DR\s*\n(.*?)(?=\n##[^#]|$)", summary, re.DOTALL | re.IGNORECASE)
        tldr_content = tldr_match.group(1).strip() if tldr_match else ""

        # Extract Technical Blog section (Chinese or English, match until TL;DR or end)
        blog_match = re.search(
            r"##\s*(技术博客|Technical Blog)\s*\n(.*?)(?=\n##\s*TL;DR|$)",
            summary,
            re.DOTALL | re.IGNORECASE,
        )
        blog_content = blog_match.group(2).strip() if blog_match else ""

        if tldr_content and blog_content:
            # Reorganize: TL;DR first, then Technical Blog
            formatted = f"""## TL;DR

{tldr_content}

## Technical Blog

{blog_content}"""
            return formatted

        # Fallback: just clean up the original content
        cleaned = re.sub(r"\n{3,}", "\n\n", summary).strip()
        return cleaned

    def generate_summary(self, pid: str, source: Optional[str] = None, model: Optional[str] = None) -> dict:
        """
        Main entry function for generating paper summary

        Args:
            pid: Paper ID
            source: Markdown source override ("html" or "mineru")

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
                markdown_content, html_source = self._get_markdown_from_html(pid)
                if markdown_content:
                    main_content = self.extract_main_content(markdown_content)
                    logger.info(f"Summarizing {pid} (html:{html_source or 'unknown'}) ...")
                    return self.summarize_with_llm(main_content, model=model)
                logger.info(f"HTML fetch/parse failed for {pid}, fallback to minerU.")

            # Pre-check if parsing result already exists
            existing_md_path = self._find_mineru_markdown(pid, backend=self._normalize_mineru_backend())
            if existing_md_path:
                logger.trace(f"Found existing parse result: {existing_md_path}")
                # Read Markdown content directly
                with open(existing_md_path, encoding="utf-8") as f:
                    markdown_content = f.read()

                if markdown_content.strip():
                    # Step 3: Extract main paper content
                    main_content = self.extract_main_content(markdown_content)
                    # Step 4: Generate summary using LLM
                    logger.info(f"Summarizing {pid}.pdf ...")
                    summary = self.summarize_with_llm(main_content, model=model)
                    return summary
                else:
                    logger.trace("Existing Markdown file content is empty, re-parsing")

            # Step 1: Download paper PDF
            logger.info(f"Downloading {pid}.pdf ...")
            pdf_path = self.download_arxiv_paper(pid)
            if not pdf_path:
                return self._build_summary_result("# Error\n\nUnable to download paper PDF")

            # Step 2: Parse PDF to Markdown using minerU
            logger.info(f"Parsing {pid}.pdf ...")
            md_path = self.parse_pdf_with_mineru(pdf_path)
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
            logger.info(f"Summarizing {pid}.pdf ...")
            summary = self.summarize_with_llm(main_content, model=model)

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


if __name__ == "__main__":
    # Test code
    import sys

    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    if len(sys.argv) > 1:
        test_pid = sys.argv[1]
        logger.trace(f"Test paper ID: {test_pid}")
        summary = generate_paper_summary(test_pid)
        logger.trace("\n" + "=" * 50)
        logger.trace("Paper summary:")
        logger.trace("=" * 50)
        logger.trace(summary.get("content"))
    else:
        logger.trace("Usage: python paper_summarizer.py <paper_id>")
        logger.trace("Example: python paper_summarizer.py 2301.00001")
