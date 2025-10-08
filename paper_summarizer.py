#!/usr/bin/env python3
"""
Paper Summarizer Module
Features:
1. Download arxiv papers to pdfs directory
2. Parse PDFs to markdown using minerU
3. Summarize papers using LLM models
"""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

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

        # Ensure directories exist
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

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
            response = requests.get(pdf_url, stream=True, timeout=30)
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

        while time.time() - start_time < timeout:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                return fd
            except FileExistsError:
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

            # Migration logic: if vlm directory exists, rename to auto
            vlm_dir = output_dir / pdf_name / "vlm"
            auto_dir = output_dir / pdf_name / "auto"
            if vlm_dir.exists() and not auto_dir.exists():
                try:
                    vlm_dir.rename(auto_dir)
                    logger.trace(f"Migrated vlm directory to auto: {vlm_dir} -> {auto_dir}")
                except (FileExistsError, OSError) as e:
                    # Another process may have already renamed it
                    logger.trace(f"Failed to migrate vlm to auto (may already exist): {e}")

            # Check if already parsed
            expected_md_path = auto_dir / f"{pdf_name}.md"
            if expected_md_path.exists():
                logger.trace(f"Markdown file already exists: {expected_md_path}")
                return expected_md_path
            else:
                logger.trace(f"{expected_md_path} not found")

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
                if expected_md_path.exists():
                    logger.trace(f"File generated during lock wait: {expected_md_path}")
                    return expected_md_path

                # Build minerU command
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
                    "vlm-http-client",
                    "-u",
                    f"http://127.0.0.1:{VLLM_MINERU_PORT}",
                ]

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
                vlm_dir_new = output_dir / pdf_name / "vlm"
                auto_dir_new = output_dir / pdf_name / "auto"
                if vlm_dir_new.exists() and not auto_dir_new.exists():
                    try:
                        vlm_dir_new.rename(auto_dir_new)
                        logger.trace(f"Migrated newly generated vlm to auto: {vlm_dir_new} -> {auto_dir_new}")
                    except (FileExistsError, OSError) as e:
                        # Another process may have already renamed it
                        logger.trace(f"Failed to migrate newly generated vlm to auto (may already exist): {e}")

                # Check generated Markdown file
                if expected_md_path.exists():
                    logger.trace(f"PDF parse complete: {expected_md_path}")

                    # Delete PDF file after successful parsing to save space
                    try:
                        pdf_path.unlink(missing_ok=True)
                        logger.trace(f"Deleted PDF source file: {pdf_path}")
                    except Exception as e:
                        logger.trace(f"Failed to delete PDF file: {e}")

                    # Clean up files other than images and markdown
                    self._cleanup_mineru_output(output_dir / pdf_name)

                    return expected_md_path
                else:
                    logger.trace(f"Generated Markdown file not found: {expected_md_path}")
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
            auto_dir = output_path / "auto"
            if not auto_dir.exists():
                return

            # File patterns to keep
            paper_id = output_path.name
            keep_files = {f"{paper_id}.md"}  # Keep markdown files
            keep_dirs = {"images"}  # Keep images directory

            items_to_remove = []

            for item in auto_dir.iterdir():
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

        Args:
            markdown_content: Complete Markdown content

        Returns:
            Extracted main content
        """
        try:
            # Common ending markers (case insensitive)
            end_patterns = [
                r"(?i)#+\s*references?\s*",
                r"(?i)#+\s*acknowledgements?\s*",
                r"(?i)#+\s*appendix\s*",
                r"(?i)#+\s*appendices\s*",
                r"(?i)^\s*references?\s*",
                r"(?i)^\s*acknowledgements?\s*",
                r"(?i)^\s*appendix\s*",
                r"(?i)^\s*appendices\s*",
            ]

            lines = markdown_content.split("\n")
            end_index = len(lines)

            # Find the earliest occurrence of ending markers
            for i, line in enumerate(lines):
                line = line.strip()
                for pattern in end_patterns:
                    if re.match(pattern, line):
                        end_index = min(end_index, i)
                        break

            # Extract main content
            main_content = "\n".join(lines[:end_index])

            # If extracted content is too short, return original content
            if len(main_content.strip()) < len(markdown_content.strip()) * MAIN_CONTENT_MIN_RATIO:
                logger.trace("Extracted main content too short, using original content")
                return markdown_content

            logger.trace(f"Paper content truncated from {len(markdown_content)} to {len(main_content)} characters")
            return main_content

        except Exception as e:
            logger.trace(f"Failed to extract main content: {e}")
            return markdown_content

    def summarize_with_llm(self, markdown_content: str) -> str:
        """
        Summarize paper content using LLM

        Args:
            markdown_content: Paper content in Markdown format

        Returns:
            Paper summary
        """
        try:
            # Choose language prompt based on configuration
            if LLM_SUMMARY_LANG == "en":
                # English prompt
                prompt = f"""
You are a senior researcher in deep learning and computer science, skilled at analyzing and summarizing academic papers. Please carefully read the following paper content and provide a professional, accurate, and structured **English** summary.

<Summary Requirements>
1. Use academic and rigorous language, formulas are allowed
2. Use `$…$` (inline) or `$$…$$` (block) for all formulas. Use basic Latex formulas, if possible.
3. Each section should have specific content, avoid vague descriptions
4. Use appropriate subheadings, lists, and emphasis to organize content
</Summary Requirements>

<Paper Content>
{markdown_content}
</Paper Content>

Output should contain two parts, detailed summary and TL;DR, in the following format:
<Output Format>
## Detailed Paper Summary
Your detailed summary of this paper

## TL;DR
Your brief summary of this paper
</Output Format>

<Notes>
1. Ensure the summary accurately reflects the paper content, **do not add information not in the paper** except for personal insights
2. Do not guess or fabricate data, rather skip quantitative analysis than summarize incorrect data
3. Please summarize in **English**
</Notes>
"""
            else:
                # Chinese prompt (default)
                prompt = f"""
你是一位深度学习和计算机科学领域的资深研究员，擅长分析和总结学术论文。请仔细阅读以下论文内容，并提供一个专业、准确、结构化的**中文**总结。

<总结要求>
1. 要是用学术且严谨的语言，可以适当使用公式
2. 所有公式用 `$…$`（行内）或 `$$…$$`（独立），要尽量使用最基本的 LaTex 公式
3. 每个部分都要有具体内容，避免空泛的描述
4. 适当使用子标题、列表和重点标记来组织内容
5. 注意所使用的 Latex 和 Markdown 语法兼容性和正确性
</总结要求>

<论文内容>
{markdown_content}
</论文内容>

输出包含两部分，分别是详细总结和 TL;DR，按照如下格式：
<输出格式>
## 论文详细总结
你对这篇论文的详细总结

## TL;DR
你对这篇论文的简短总结
</输出格式>

<注意事项>
1. 请确保总结的时候准确反映论文内容，除了个人见解部分**不要添加论文中没有的信息**
2. 不要猜测更不要臆造数据，宁可不总结论文中的定量分析，也不要总结错误数据
3. 请一定用**中文**进行总结
</注意事项>
"""

            modelid = LLM_NAME
            logger.trace(f"Calling {modelid} to generate paper summary...")

            response = self.client.chat.completions.create(
                model=modelid,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32768,
                extra_body={"reasoning": {"effort": "low"}},  # Use low reasoning effort
            )

            summary = response.choices[0].message.content

            # Extract content after </think>
            if hasattr(response.choices[0].message, "reasoning"):
                logger.trace(f"Original summary Thinking:\n{response.choices[0].message.reasoning}")
            elif hasattr(response.choices[0].message, "reasoning_content"):
                logger.trace(f"Original summary Thinking:\n{response.choices[0].message.reasoning_content}")
            else:
                logger.trace(f"Original summary content:\n{summary}")

            if "</think>" in summary:
                summary = summary.split("</think>", 1)[1].strip()
            else:
                summary = summary.strip()

            # Parse detailed summary and TL;DR sections
            parsed_summary = self._parse_summary_sections(summary)
            logger.trace("Paper summary generation complete")
            return parsed_summary

        except Exception as e:
            logger.trace(f"LLM summary failed: {e}")
            return f"# Error\n\nSummary generation failed: {str(e)}"

    def _parse_summary_sections(self, summary: str) -> str:
        """
        Parse summary content, extract detailed summary and TL;DR sections, reorganize format (TL;DR first)

        Args:
            summary: Original summary content

        Returns:
            Reorganized summary content
        """
        try:
            import re

            # New parsing logic: based on Markdown heading format
            # Extract detailed summary section (support both Chinese and English)
            # Try Chinese format first
            detailed_match = re.search(r"##\s*论文详细总结\s*\n(.*?)(?=##\s*TL;DR|$)", summary, re.DOTALL)
            if not detailed_match:
                # Try English format
                detailed_match = re.search(r"##\s*Detailed Paper Summary\s*\n(.*?)(?=##\s*TL;DR|$)", summary, re.DOTALL)
            detailed_summary = detailed_match.group(1).strip() if detailed_match else ""

            # Extract TL;DR section (content after ## TL;DR)
            tldr_match = re.search(r"##\s*TL;DR\s*\n(.*?)(?=##|$)", summary, re.DOTALL)
            tldr_summary = tldr_match.group(1).strip() if tldr_match else ""

            # Reorganize format: TL;DR first, detailed summary second
            if tldr_summary and detailed_summary:
                formatted_summary = f"""## TL;DR

{tldr_summary}

## Detailed Summary

{detailed_summary}"""
                return formatted_summary
            elif detailed_summary:
                # If only detailed summary exists, add title and return
                return f"## Detailed Summary\n\n{detailed_summary}"
            elif tldr_summary:
                # If only TL;DR exists, return it
                return f"## TL;DR\n\n{tldr_summary}"
            else:
                # If neither extracted, return original content
                logger.trace("Failed to parse detailed summary and TL;DR, returning original content")
                return summary

        except Exception as e:
            logger.trace(f"Failed to parse summary sections: {e}")
            return summary

    def generate_summary(self, pid: str) -> str:
        """
        Main entry function for generating paper summary

        Args:
            pid: Paper ID

        Returns:
            Paper summary in Markdown format
        """
        try:
            # Step 0: Pre-check and migrate vlm directory (if exists)
            vlm_dir = self.mineru_dir / pid / "vlm"
            auto_dir = self.mineru_dir / pid / "auto"
            if vlm_dir.exists() and not auto_dir.exists():
                try:
                    vlm_dir.rename(auto_dir)
                    logger.trace(f"Migrated vlm directory to auto: {vlm_dir} -> {auto_dir}")
                except (FileExistsError, OSError) as e:
                    # Another process may have already renamed it
                    logger.trace(f"Failed to migrate vlm to auto (may already exist): {e}")

            # Pre-check if parsing result already exists
            expected_md_path = self.mineru_dir / pid / "auto" / f"{pid}.md"
            if expected_md_path.exists():
                logger.trace(f"Found existing parse result: {expected_md_path}")
                # Read Markdown content directly
                with open(expected_md_path, encoding="utf-8") as f:
                    markdown_content = f.read()

                if markdown_content.strip():
                    # Step 3: Extract main paper content
                    main_content = self.extract_main_content(markdown_content)
                    # Step 4: Generate summary using LLM
                    logger.info(f"Summarizing {pid}.pdf ...")
                    summary = self.summarize_with_llm(main_content)
                    return summary
                else:
                    logger.trace("Existing Markdown file content is empty, re-parsing")

            # Step 1: Download paper PDF
            logger.info(f"Downloading {pid}.pdf ...")
            pdf_path = self.download_arxiv_paper(pid)
            if not pdf_path:
                return "# Error\n\nUnable to download paper PDF"

            # Step 2: Parse PDF to Markdown using minerU
            logger.info(f"Parsing {pid}.pdf ...")
            md_path = self.parse_pdf_with_mineru(pdf_path)
            if not md_path:
                return "# Error\n\nUnable to parse PDF to Markdown"

            # Step 3: Read Markdown content
            with open(md_path, encoding="utf-8") as f:
                markdown_content = f.read()

            if not markdown_content.strip():
                return "# Error\n\nParsed Markdown content is empty"

            # Step 4: Extract main paper content
            main_content = self.extract_main_content(markdown_content)

            # Step 5: Generate summary using LLM
            logger.info(f"Summarizing {pid}.pdf ...")
            summary = self.summarize_with_llm(main_content)

            return summary

        except Exception as e:
            logger.error(f"Error occurred while generating paper summary: {e}")
            return f"# Error\n\nFailed to generate summary: {str(e)}"


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


def generate_paper_summary(pid: str) -> str:
    """
    External interface function for generating paper summary

    Args:
        pid: Paper ID

    Returns:
        Paper summary in Markdown format
    """
    summarizer = get_summarizer()
    return summarizer.generate_summary(pid)


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
        logger.trace(summary)
    else:
        logger.trace("Usage: python paper_summarizer.py <paper_id>")
        logger.trace("Example: python paper_summarizer.py 2301.00001")
