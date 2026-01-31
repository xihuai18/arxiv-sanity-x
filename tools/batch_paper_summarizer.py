"""
Batch Paper Summarizer

Features:
1. Get the latest n papers from the database
2. Built-in non-singleton paper summarizer for HTML/minerU parsing and LLM summarization
3. Automatically cache summaries to the directory used by serve.py
4. Support multi-threaded concurrent processing to improve efficiency
5. Provide detailed progress tracking and error handling
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

# Ensure repository root is importable when executing this file directly.
# e.g. `python tools/batch_paper_summarizer.py` would otherwise miss sibling packages like `aslite/`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Repository layer for cleaner data access
from aslite.repositories import MetaRepository, PaperRepository, TagRepository
from config import settings
from tools.paper_summarizer import (
    PaperSummarizer,
    acquire_summary_lock,
    atomic_write_json,
    atomic_write_text,
    normalize_summary_result,
    normalize_summary_source,
    read_summary_meta,
    release_summary_lock,
    resolve_cache_pid,
    summary_cache_paths,
    summary_source_matches,
)

LLM_NAME = settings.llm.name
SUMMARY_DIR = str(settings.summary_dir)
SUMMARY_MARKDOWN_SOURCE = settings.summary.markdown_source


class BatchPaperSummarizer(PaperSummarizer):
    """
    Batch Paper Summarizer Class
    Inherits from PaperSummarizer, reuses core logic, supports true concurrent processing
    """

    def __init__(self, processor=None):
        """
        Initialize paper summarizer

        Args:
            processor: BatchProcessor instance for recording error details
        """
        super().__init__()  # Call parent class initialization
        self.cache_dir = Path(SUMMARY_DIR)
        self.processor = processor  # For recording error details

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _record_failure_detail(self, pid: str, reason: str, message: str, exception: Exception = None):
        """Record failure details to processor"""
        if self.processor:
            self.processor._record_failure_detail(pid, reason, message, exception)

    def download_arxiv_paper(self, pid: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        Download arXiv paper PDF, reuse parent class method and add error recording

        Args:
            pid: Paper ID, e.g. "2301.00001"

        Returns:
            Tuple of (pdf_path, version):
            - pdf_path: Downloaded PDF file path, None if failed
            - version: The actual version downloaded (e.g., "3"), None if unknown
        """
        try:
            return super().download_arxiv_paper(pid)
        except Exception as e:
            error_msg = f"Failed to download paper {pid}: {str(e)}"
            logger.error(error_msg)
            # Record detailed error information
            self._record_failure_detail(pid, "download_failed", error_msg, e)
            return None, None

    def parse_pdf_with_mineru(
        self, pdf_path: Path, cache_pid: Optional[str] = None, cached_version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Parse PDF to Markdown using minerU
        Now completely relies on parent class implementation which already has proper file locking

        Args:
            pdf_path: PDF file path (required for all backends, including API backend)
            cache_pid: Optional paper ID to use for output directory (should be raw PID)
            cached_version: Version of the PDF that was actually downloaded (e.g., "3")

        Returns:
            Generated Markdown file path, None if failed
        """
        try:
            # Directly use parent class method - it already has file-based locking
            result = super().parse_pdf_with_mineru(pdf_path, cache_pid=cache_pid, cached_version=cached_version)
            if result is None:
                # Record failure details
                pid_for_error = cache_pid or pdf_path.stem
                self._record_failure_detail(
                    pid_for_error,
                    "parse_failed",
                    "Parent class parse_pdf_with_mineru returned None",
                    Exception("minerU parsing failed"),
                )
            return result

        except Exception as e:
            error_msg = f"PDF parse failed: {e}"
            logger.trace(error_msg)
            # Record failure details
            pid_for_error = cache_pid or pdf_path.stem
            self._record_failure_detail(pid_for_error, "parse_failed", error_msg, e)
            return None

    def generate_summary(self, pid: str, source: Optional[str] = None, model: Optional[str] = None) -> dict:
        """
        Main entry function for generating paper summary, reuse parent class logic

        Args:
            pid: Paper ID
            source: Markdown source override ("html" or "mineru")
            model: LLM model name

        Returns:
            Dict containing paper summary markdown content and metadata
        """
        # Directly call parent class generate_summary method
        # Parent class method already includes vlm to auto migration logic
        return super().generate_summary(pid, source=source, model=model)


class BatchProcessor:
    """Batch Processor Class"""

    def __init__(self, max_workers: int = 2, model: Optional[str] = None):
        """
        Initialize batch processor

        Args:
            max_workers: Maximum number of worker threads, now supports true concurrent processing
            model: LLM model name for summary generation
        """
        self.max_workers = max_workers
        self.cache_dir = Path(SUMMARY_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Use provided model or default from config
        self.model = (model or LLM_NAME or "").strip()
        if not self.model:
            raise ValueError("Model name is required for batch processing")

        # Statistics - add detailed failure reason statistics
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "cached": 0,
            "skipped": 0,
            # Detailed failure reason statistics
            "failure_reasons": {
                "download_failed": 0,  # PDF download failed
                "parse_failed": 0,  # minerU parsing failed
                "parse_timeout": 0,  # minerU parsing timeout
                "empty_content": 0,  # Parsed content is empty
                "llm_failed": 0,  # LLM generation failed
                "cache_failed": 0,  # Cache failed
                "other_error": 0,  # Other unknown errors
            },
        }
        self.stats_lock = threading.Lock()

        # Failure details record - record specific information for each failed paper
        self.failure_details = {}
        self.failure_lock = threading.Lock()

    def _record_failure_detail(self, pid: str, reason: str, message: str, exception: Exception = None):
        """
        Record failure details

        Args:
            pid: Paper ID
            reason: Failure reason category
            message: Detailed error message
            exception: Exception object (optional)
        """
        with self.failure_lock:
            self.failure_details[pid] = {
                "reason": reason,
                "message": message,
                "exception_type": type(exception).__name__ if exception else None,
                "exception_str": str(exception) if exception else None,
                "timestamp": time.time(),
            }

        # Update statistics
        with self.stats_lock:
            if reason in self.stats["failure_reasons"]:
                self.stats["failure_reasons"][reason] += 1
            else:
                self.stats["failure_reasons"]["other_error"] += 1

    def get_latest_papers(self, n: int) -> List[Tuple[str, Dict]]:
        """
        Get the latest n papers from the database

        Args:
            n: Number of papers to fetch

        Returns:
            List[Tuple[str, Dict]]: List of paper IDs and metadata, sorted by time in descending order
        """
        logger.debug(f"Fetching the latest {n} papers...")

        if n <= 0:
            return []

        # Use the optimized MetaRepository.get_latest_n() method
        # which uses the time index table for efficient queries
        latest_papers = MetaRepository.get_latest_n(n, use_index=True)

        logger.debug(f"Successfully fetched {len(latest_papers)} latest papers")

        return latest_papers

    def get_priority_papers(self, time_delta_days: float = 7.0, limit: int = 50) -> List[Tuple[str, Dict]]:
        """
        Get high-priority papers that will be recommended in emails but don't have summaries yet.

        Priority papers are those that:
        1. Will appear in email recommendations (SVM recommended for each user's tags)
        2. Are within the time_delta_days window
        3. Don't have a cached summary yet

        Args:
            time_delta_days: Only consider papers from the last N days (matches email time_delta)
            limit: Maximum number of priority papers to return

        Returns:
            List[Tuple[str, Dict]]: List of paper IDs and metadata, sorted by recommendation frequency
        """
        import requests

        from config import settings

        API_BASE_URL = f"http://localhost:{settings.serve_port}"
        API_TIMEOUT = 60

        logger.debug(f"Finding priority papers (email recommendations, no summary, last {time_delta_days} days)...")

        # Collect all user tags
        user_tags_map = {}  # user -> {tag: [pids]}
        for user, user_tags in TagRepository.get_all_tags().items():
            if isinstance(user_tags, dict) and user_tags:
                user_tags_map[user] = user_tags

        if not user_tags_map:
            logger.debug("No user tags found")
            return []

        logger.debug(f"Found {len(user_tags_map)} users with tags")

        # Get SVM recommendations for each user's each tag via API
        # Track recommendation frequency (papers recommended more often are higher priority)
        recommendation_count = {}  # pid -> count

        for user, tags in user_tags_map.items():
            for tag_name, tag_pids in tags.items():
                if not tag_pids:
                    continue

                try:
                    api_key = str(getattr(settings.reco, "api_key", "") or "").strip()
                    headers = {"X-ARXIV-SANITY-API-KEY": api_key} if api_key else None
                    payload = {
                        "tag_name": tag_name,
                        "user": user,
                        "time_delta": time_delta_days,
                        "limit": 50,  # Top 50 recommendations per tag
                        "C": 0.1,
                    }
                    resp = requests.post(
                        f"{API_BASE_URL}/api/tag_search",
                        json=payload,
                        headers=headers,
                        timeout=API_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for pid in data.get("pids", []):
                            recommendation_count[pid] = recommendation_count.get(pid, 0) + 1
                except Exception as e:
                    logger.debug(f"Failed to get recommendations for {user}/{tag_name}: {e}")
                    continue

        if not recommendation_count:
            logger.debug("No recommendations found from API")
            return []

        logger.debug(f"Found {len(recommendation_count)} unique recommended papers")

        # Filter: no summary yet
        summary_source = normalize_summary_source(SUMMARY_MARKDOWN_SOURCE)
        priority_papers = []

        for pid, count in recommendation_count.items():
            # Check if summary already cached
            raw_pid = pid.split("v")[0] if "v" in pid else pid
            if self.is_summary_cached(raw_pid, summary_source):
                continue

            meta = MetaRepository.get_by_id(pid)
            if not meta:
                continue

            priority_papers.append((pid, meta, count))

        # Sort by recommendation count (higher = more users will see it), then by time
        priority_papers.sort(key=lambda x: (-x[2], -x[1].get("_time", 0)))
        priority_papers = [(pid, meta) for pid, meta, _count in priority_papers[:limit]]

        logger.debug(f"Found {len(priority_papers)} priority papers needing summaries")

        return priority_papers

    def get_papers_with_priority(
        self, num_papers: int, priority_time_delta: float = 7.0, priority_limit: int = 50
    ) -> List[Tuple[str, Dict]]:
        """
        Get papers for processing with priority papers first.

        Order:
        1. Priority papers (tagged by users, no summary, recent)
        2. Latest papers (fill remaining slots)

        Args:
            num_papers: Total number of papers to return
            priority_time_delta: Time window for priority papers (days)
            priority_limit: Max number of priority papers

        Returns:
            List of (pid, meta) tuples with priority papers first
        """
        # Get priority papers first
        priority_papers = self.get_priority_papers(
            time_delta_days=priority_time_delta, limit=min(priority_limit, num_papers)
        )

        priority_count = len(priority_papers)
        remaining_slots = num_papers - priority_count

        if remaining_slots <= 0:
            logger.debug(f"Using {priority_count} priority papers (limit reached)")
            return priority_papers[:num_papers]

        # Get latest papers to fill remaining slots
        latest_papers = self.get_latest_papers(remaining_slots + priority_count)

        # Remove duplicates (priority papers already included)
        priority_pids = {pid for pid, _ in priority_papers}
        latest_papers = [(pid, meta) for pid, meta in latest_papers if pid not in priority_pids]
        latest_papers = latest_papers[:remaining_slots]

        # Combine: priority first, then latest
        combined = priority_papers + latest_papers

        logger.debug(f"Combined queue: {priority_count} priority + {len(latest_papers)} latest = {len(combined)} total")

        return combined

    def is_summary_cached(self, cache_pid: str, summary_source: str) -> bool:
        """
        Check if summary is already cached (supports both new layered and legacy flat structure)

        Args:
            cache_pid: Cache key (raw pid or pidvN)
            summary_source: Current markdown source

        Returns:
            bool: Whether it is cached
        """
        # Try new layered structure first: SUMMARY_DIR/{cache_pid}/{model}.md
        cache_file, meta_file, _, legacy_cache, legacy_meta, _ = summary_cache_paths(cache_pid, self.model)

        # Check new structure
        if cache_file.exists() and cache_file.stat().st_size > 0:
            meta = read_summary_meta(meta_file)
            if summary_source_matches(meta, summary_source):
                return True

        # Check legacy flat structure: SUMMARY_DIR/{cache_pid}.md
        if legacy_cache.exists() and legacy_cache.stat().st_size > 0:
            meta = read_summary_meta(legacy_meta)
            # For legacy cache, also check model match
            cached_model = (meta.get("model") or "").strip()
            if summary_source_matches(meta, summary_source):
                if not cached_model or cached_model == self.model:
                    return True

        return False

    def cache_summary(
        self,
        cache_pid: str,
        summary_content: str,
        source: Optional[str] = None,
        summary_meta: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Cache summary to the directory used by serve.py (uses layered structure)

        Args:
            cache_pid: Cache key (raw pid or pidvN)
            summary_content: Summary content
            source: Markdown source override ("html" or "mineru")
            summary_meta: Optional metadata from summarizer

        Returns:
            Tuple[bool, Optional[str]]: (cached, None)
        """
        try:
            # Use layered cache structure: SUMMARY_DIR/{cache_pid}/{model}.md
            cache_file, meta_file, _, _, _, _ = summary_cache_paths(cache_pid, self.model)
            summary_source = normalize_summary_source(source)

            # Only cache successful summaries (not error messages)
            if not summary_content.startswith("# Error"):
                atomic_write_text(cache_file, summary_content)
                meta = {}
                if isinstance(summary_meta, dict):
                    meta.update(summary_meta)
                # Ensure source exists for cache checks; keep detailed value if provided
                meta.setdefault("source", summary_source)
                meta.setdefault("generated_at", time.time())
                atomic_write_json(meta_file, meta)

                logger.debug(f"Summary cached to: {cache_file}")
                return True, None
            else:
                logger.warning(f"Summary generation failed, not caching: {cache_pid}")
                return False, None

        except Exception as e:
            logger.error(f"Failed to cache summary {cache_pid}: {e}")
            return False, None

    def process_single_paper(self, pid: str, paper_info: Dict, skip_cached: bool = True) -> Tuple[str, bool, str]:
        """
        Process single paper, each thread uses independent BatchPaperSummarizer instance

        Args:
            pid: Paper ID
            paper_info: Paper information
            skip_cached: Whether to skip cached papers

        Returns:
            Tuple[str, bool, str]: (Paper ID, success status, summary content or error message)
        """
        try:
            summary_source = normalize_summary_source(None)
            cache_pid, raw_pid, has_explicit_version = resolve_cache_pid(pid, paper_info)

            # Check if already cached
            if skip_cached and self.is_summary_cached(cache_pid, summary_source):
                logger.trace(f"Skipped cached paper: {pid}")
                with self.stats_lock:
                    self.stats["cached"] += 1
                return pid, True, "Cached"

            # Get basic paper information
            title = paper_info.get("title", "Unknown Title")
            authors = ", ".join(a.get("name", "") for a in paper_info.get("authors", []))

            logger.trace(f"Start processing paper: {pid}")
            logger.trace(f"Title: {title}")
            logger.debug(f"Authors: {authors}")

            # Create independent BatchPaperSummarizer instance for each thread, pass processor for error recording
            summarizer = BatchPaperSummarizer(processor=self)

            # Use layered lock structure
            _, _, lock_file, _, _, legacy_lock = summary_cache_paths(cache_pid, self.model)
            # Use legacy lock if legacy cache exists
            legacy_cache = self.cache_dir / f"{cache_pid}.md"
            if legacy_cache.exists() and not lock_file.parent.exists():
                lock_file = legacy_lock
            lock_fd = None

            try:
                lock_fd = acquire_summary_lock(lock_file, timeout_s=300)
                if lock_fd is None:
                    if self.is_summary_cached(cache_pid, summary_source):
                        with self.stats_lock:
                            self.stats["cached"] += 1
                        return pid, True, "Cached"
                    return pid, False, "Summary busy"

                if skip_cached and self.is_summary_cached(cache_pid, summary_source):
                    logger.trace(f"Skipped cached paper after lock: {pid}")
                    with self.stats_lock:
                        self.stats["cached"] += 1
                    return pid, True, "Cached"

                # Call built-in paper summarizer to generate summary
                start_time = time.time()
                if summary_source == "html":
                    pid_for_summary = pid if has_explicit_version else cache_pid
                else:
                    pid_for_summary = pid if has_explicit_version else raw_pid

                summary_result = summarizer.generate_summary(pid_for_summary, source=summary_source, model=self.model)
                summary_content, summary_meta = normalize_summary_result(summary_result)
                end_time = time.time()

                # Check if summary generation was successful
                if summary_content.startswith("# Error"):
                    logger.error(f"Summary generation failed: {pid}")
                    with self.stats_lock:
                        self.stats["failed"] += 1
                    return pid, False, summary_content

                # Cache summary
                cache_success, _ = self.cache_summary(
                    cache_pid, summary_content, source=summary_source, summary_meta=summary_meta
                )

                if cache_success:
                    logger.success(f"Paper processed successfully: {pid} (elapsed: {end_time - start_time:.2f}s)")
                    with self.stats_lock:
                        self.stats["success"] += 1
                    return pid, True, summary_content
                else:
                    logger.error(f"Summary cache failed: {pid}")
                    with self.stats_lock:
                        self.stats["failed"] += 1
                    return pid, False, "Cache failed"
            finally:
                if lock_fd is not None:
                    release_summary_lock(lock_fd, lock_file)

        except Exception as e:
            logger.error(f"Error occurred while processing paper {pid}: {e}")
            with self.stats_lock:
                self.stats["failed"] += 1
            return pid, False, str(e)

    def format_time_str(self, timestamp: float) -> str:
        """Format timestamp"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

    def batch_process(
        self, papers: List[Tuple[str, Dict]], skip_cached: bool = True, dry_run: bool = False, max_retries: int = 3
    ) -> Dict:
        """
        Batch process papers with failure retry mechanism

        Args:
            papers: List of papers
            skip_cached: Whether to skip cached papers
            dry_run: Whether to run in dry-run mode
            max_retries: Maximum retry attempts

        Returns:
            Dict: Processing result statistics
        """
        # Get detailed paper information using Repository layer for batch efficiency
        logger.trace("Loading paper details from database...")
        pids_list = [pid for pid, _meta in papers]
        papers_data = PaperRepository.get_by_ids(pids_list)
        # Fill missing with empty dict
        papers_data = {pid: papers_data.get(pid, {}) for pid in pids_list}

        self.stats["total"] = len(papers)

        if dry_run:
            logger.debug("=== Dry run mode - only displaying paper info ===")

            # Use progress bar to display paper information
            with tqdm(papers, desc="Checking papers", unit="paper", leave=True, ncols=100, file=sys.stderr) as pbar:
                for pid, meta in pbar:
                    paper_info = papers_data.get(pid, {})
                    title = paper_info.get("title", "Unknown Title")
                    authors = ", ".join(a.get("name", "") for a in paper_info.get("authors", []))
                    time_str = self.format_time_str(meta["_time"])
                    summary_source = normalize_summary_source(None)
                    cache_pid, _, _ = resolve_cache_pid(pid, meta)
                    cached_status = "Cached" if self.is_summary_cached(cache_pid, summary_source) else "Not Cached"

                    pbar.set_postfix_str(f"{pid} ({cached_status})")

                    if logger.level("TRACE").no <= logger._core.min_level:
                        logger.trace(f"Title: {title}")
                        logger.trace(f"Authors: {authors}")
                        logger.trace(f"Time: {time_str}")
                        logger.trace(f"Status: {cached_status}")
                        logger.trace("-" * 80)

            return self.stats

        # Actual processing mode
        logger.trace(f"=== Start batch processing {len(papers)} papers ===")
        logger.trace(f"Max worker threads: {self.max_workers}")
        logger.trace(f"Skip cached: {'yes' if skip_cached else 'no'}")
        logger.trace(f"Max retries: {max_retries}")

        start_time = time.time()

        # Initialize processing queue and retry count
        processing_queue = [(pid, meta, 0) for pid, meta in papers]  # (pid, meta, retry_count)

        round_num = 1
        while processing_queue:
            current_round_papers = processing_queue
            processing_queue = []

            if round_num > 1:
                logger.debug(f"=== Round {round_num} processing (retry) ===")
                logger.debug(f"Number of papers to retry: {len(current_round_papers)}")
            else:
                logger.debug(f"=== Starting round {round_num} processing ===")

            # Create progress bar
            desc = f"Round {round_num} processing" if round_num > 1 else "Processing papers"
            pbar = tqdm(total=len(current_round_papers), desc=desc, unit="paper", leave=True, ncols=120)

            # Initialize current round counters
            round_success = 0
            round_failed = 0

            # Use thread pool for actual concurrent processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks in batch for efficiency
                future_to_paper = {
                    executor.submit(self.process_single_paper, pid, papers_data.get(pid, {}), skip_cached): (
                        pid,
                        meta,
                        retry_count,
                    )
                    for pid, meta, retry_count in current_round_papers
                }

                # Process completed tasks
                for future in as_completed(future_to_paper):
                    pid, meta, retry_count = future_to_paper[future]
                    try:
                        result_pid, success, message = future.result()

                        # Update counters and progress bar
                        if success:
                            round_success += 1
                            if message == "Cached":
                                pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | {result_pid} (Cached)")
                            else:
                                pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | {result_pid}")
                        else:
                            round_failed += 1
                            pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | ✗ {result_pid}")

                        pbar.update(1)

                        if not success and message != "Cached":
                            # Processing failed, check if retry is needed
                            if retry_count < max_retries:
                                logger.warning(
                                    f"Processing failed, will retry ({retry_count + 1}/{max_retries}): {result_pid}"
                                )
                                processing_queue.append((pid, meta, retry_count + 1))
                            else:
                                logger.error(f"Processing failed, reached max retries: {result_pid} - {message}")
                                # Final failure statistics already updated in process_single_paper

                    except Exception as e:
                        logger.error(f"Task execution exception {pid}: {e}")
                        # Update counters and progress bar
                        round_failed += 1
                        pbar.set_postfix_str(f"✓{round_success} ✗{round_failed} | ✗ {pid} (Exception)")
                        pbar.update(1)

                        # Exceptions also need retry
                        if retry_count < max_retries:
                            logger.warning(f"Task exception, will retry ({retry_count + 1}/{max_retries}): {pid}")
                            processing_queue.append((pid, meta, retry_count + 1))
                        else:
                            logger.error(f"Task exception, reached max retries: {pid}")
                            with self.stats_lock:
                                self.stats["failed"] += 1

            # Close progress bar
            pbar.close()

            round_num += 1

            # Avoid infinite retries
            if round_num > max_retries + 1:
                logger.warning("Reached max retry rounds, stopping processing")
                break

        end_time = time.time()
        total_time = end_time - start_time

        # Display final statistics
        self.print_final_stats(total_time)

        return self.stats

    def print_final_stats(self, total_time: float):
        """Print final statistics"""
        logger.success("=" * 60)
        logger.success("Processing complete! Statistics:")
        logger.success(f"Total papers: {self.stats['total']}")
        logger.success(f"Successfully processed: {self.stats['success']}")
        logger.success(f"Failed: {self.stats['failed']}")
        logger.success(f"Skipped cached: {self.stats['cached']}")
        logger.success(f"Total time: {total_time:.2f} s")

        processed_count = self.stats["success"] + self.stats["failed"]
        if processed_count > 0:
            avg_time = total_time / processed_count
            logger.success(f"Average processing time: {avg_time:.2f} s/paper")

        if self.stats["total"] > 0:
            success_rate = (self.stats["success"] / self.stats["total"]) * 100
            logger.success(f"Success rate: {success_rate:.1f}%")

        logger.success(f"Summaries saved to: {self.cache_dir}")
        logger.success("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch process latest papers and generate summaries")
    parser.add_argument(
        "-n", "--num-papers", type=int, default=10, help="Number of latest papers to process (default: 10)"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        help="Maximum number of worker threads (default: 2, recommended not to exceed 4)",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, help=f"LLM model name for summary generation (default: {LLM_NAME})"
    )
    parser.add_argument("--no-skip-cached", action="store_true", help="Do not skip cached papers, reprocess all papers")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode, only display paper information without processing"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry count for failed papers (default: 3)")

    # Priority queue arguments
    parser.add_argument(
        "--priority",
        action="store_true",
        help="Enable priority queue: process papers that will appear in email recommendations first",
    )
    parser.add_argument(
        "--priority-days",
        type=float,
        default=2.0,
        help="Time window for priority papers in days (default: 2, matches email time_delta)",
    )
    parser.add_argument(
        "--priority-limit",
        type=int,
        default=50,
        help="Maximum number of priority papers to process (default: 50)",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Enqueue summaries to Huey with low priority instead of processing locally",
    )

    args = parser.parse_args()

    # Set log level
    logger.remove()
    base_level = settings.log_level.upper()
    level = "DEBUG" if args.verbose else base_level
    logger.add(sys.stdout, level=level, format="\n{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")

    # Parameter validation
    if args.workers > 8:
        logger.warning(
            "Too many worker threads may cause GPU memory shortage or resource contention, recommended not to exceed 8"
        )
    elif args.workers < 1:
        logger.error("Number of worker threads must be at least 1")
        sys.exit(1)

    if args.num_papers <= 0:
        logger.error("Number of papers must be greater than 0")
        sys.exit(1)

    if args.max_retries < 0:
        logger.error("Retry count cannot be negative")
        sys.exit(1)

    try:
        # Create batch processor with model
        processor = BatchProcessor(max_workers=args.workers, model=args.model)
        logger.debug(f"Using model: {processor.model}")

        # Get papers - with priority queue if enabled
        if args.priority:
            logger.debug("Priority queue enabled: processing tagged papers first")
            papers_to_process = processor.get_papers_with_priority(
                num_papers=args.num_papers,
                priority_time_delta=args.priority_days,
                priority_limit=args.priority_limit,
            )
        else:
            papers_to_process = processor.get_latest_papers(args.num_papers)

        if not papers_to_process:
            logger.warning("No papers found")
            return

        skip_cached = not args.no_skip_cached

        if args.queue:
            try:
                from tasks import SUMMARY_PRIORITY_LOW, enqueue_summary_task
            except Exception as e:
                logger.error(f"Huey task queue unavailable: {e}")
                sys.exit(2)

            summary_source = normalize_summary_source(None)
            enqueued = 0
            skipped = 0
            failed = 0

            for pid, meta in papers_to_process:
                try:
                    cache_pid, _raw_pid, _ = resolve_cache_pid(pid, meta)
                    if skip_cached and processor.is_summary_cached(cache_pid, summary_source):
                        skipped += 1
                        continue
                    enqueue_summary_task(pid, model=processor.model, priority=SUMMARY_PRIORITY_LOW)
                    enqueued += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to enqueue summary for {pid}: {e}")

            logger.success(f"Enqueued {enqueued} summaries (skipped {skipped}, failed {failed})")
            if failed > 0:
                sys.exit(1)
            return

        # Batch processing (local)
        results = processor.batch_process(
            papers_to_process, skip_cached=skip_cached, dry_run=args.dry_run, max_retries=args.max_retries
        )

        if not args.dry_run and results["failed"] > 0:
            logger.warning("Some papers failed to process")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("User interrupted the program")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error occurred during program execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
