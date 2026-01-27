"""
Upload service for handling uploaded PDF papers.

This module provides business logic for:
- Uploading and storing PDF files
- Triggering MinerU parsing
- Extracting metadata via LLM
- Managing uploaded paper lifecycle
"""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from aslite.repositories import (
    NegativeTagRepository,
    TagRepository,
    UploadedPaperRepository,
)
from backend.utils.upload_utils import (
    compute_bytes_sha256,
    generate_upload_pid,
    get_upload_dir,
    get_upload_pdf_path,
    sanitize_filename,
    validate_upload_pid,
)
from config import settings

DATA_DIR = str(settings.data_dir)
# Main LLM settings (for fallback)
LLM_NAME = settings.llm.name
LLM_BASE_URL = settings.llm.base_url
LLM_API_KEY = settings.llm.api_key

# Reasonable limits to prevent abuse / DB bloat.
# Keep them generous to avoid surprising users, but bounded.
MAX_META_TITLE_CHARS = 512
MAX_META_ABSTRACT_CHARS = 20000
MAX_META_AUTHOR_COUNT = 200
MAX_META_AUTHOR_CHARS = 200


def _infer_meta_extracted_ok(record: Dict[str, Any], title: str, abstract: str, authors: List) -> bool:
    """Infer meta_extracted_ok status from available metadata.

    For backward compatibility with older records that don't have the meta_extracted_ok field,
    we infer it from the presence of title, abstract, or authors.

    Args:
        record: The uploaded paper record
        title: Paper title (may be empty)
        abstract: Paper abstract (may be empty)
        authors: List of authors (may be empty)

    Returns:
        bool: True if metadata was extracted (or can be inferred), False otherwise
    """
    if "meta_extracted_ok" in record:
        return bool(record.get("meta_extracted_ok"))

    # Infer from available metadata
    inferred_title = str(title or "").strip()
    inferred_abs = str(abstract or "").strip()
    inferred_authors = authors if isinstance(authors, list) else []
    inferred_authors = [str(a).strip() for a in inferred_authors if str(a).strip()]

    return bool(inferred_title or inferred_abs or inferred_authors)


def _validate_meta_override_inputs(
    *,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    abstract: Optional[str] = None,
) -> tuple[Optional[str], Optional[list[str]], Optional[int], Optional[str]]:
    """Validate and normalize user-supplied meta override inputs.

    Raises:
        ValueError: for invalid types/values.
    """
    norm_title: Optional[str] = None
    if title is not None:
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        norm_title = title.strip()
        if len(norm_title) > MAX_META_TITLE_CHARS:
            raise ValueError(f"title too long (max {MAX_META_TITLE_CHARS} chars)")

    norm_authors: Optional[list[str]] = None
    if authors is not None:
        if not isinstance(authors, list):
            raise ValueError("authors must be a list")
        if len(authors) > MAX_META_AUTHOR_COUNT:
            raise ValueError(f"authors too many (max {MAX_META_AUTHOR_COUNT})")
        out: list[str] = []
        for a in authors:
            if a is None:
                continue
            if not isinstance(a, str):
                raise ValueError("author must be a string")
            s = a.strip()
            if not s:
                continue
            if len(s) > MAX_META_AUTHOR_CHARS:
                raise ValueError(f"author name too long (max {MAX_META_AUTHOR_CHARS} chars)")
            out.append(s)
        norm_authors = out

    norm_year: Optional[int] = None
    if year is not None:
        if not isinstance(year, int):
            raise ValueError("year must be an integer")
        # Keep wide bounds; year is not displayed for uploads, but can be stored.
        if year < 1900 or year > 2100:
            raise ValueError("year out of range")
        norm_year = int(year)

    norm_abs: Optional[str] = None
    if abstract is not None:
        if not isinstance(abstract, str):
            raise ValueError("abstract must be a string")
        norm_abs = abstract.strip()
        if len(norm_abs) > MAX_META_ABSTRACT_CHARS:
            raise ValueError(f"abstract too long (max {MAX_META_ABSTRACT_CHARS} chars)")

    return norm_title, norm_authors, norm_year, norm_abs


def _invalidate_upload_features(pid: str) -> None:
    """Invalidate cached upload feature file (best effort)."""
    try:
        from backend.services.upload_similarity_service import get_upload_features_path

        feat_path = get_upload_features_path(pid)
        if feat_path.exists():
            feat_path.unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass


def _get_extract_info_client():
    """Get OpenAI client for Extract Info task.

    Uses settings.extract_info configuration, falling back to main LLM settings.
    This is lazy-loaded to avoid import issues at module load time.
    """
    import openai

    base_url = settings.extract_info.base_url or LLM_BASE_URL
    api_key = settings.extract_info.api_key or LLM_API_KEY

    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _normalize_extracted_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM metadata output to expected types."""
    if not isinstance(meta, dict):
        return {"title": "", "authors": [], "year": None, "abstract": None}

    title = meta.get("title")
    title = title.strip() if isinstance(title, str) else ""

    abstract = meta.get("abstract")
    if abstract is not None and not isinstance(abstract, str):
        abstract = str(abstract)
    abstract = abstract.strip() if isinstance(abstract, str) else None

    authors_raw = meta.get("authors")
    authors_list: list[str] = []
    if isinstance(authors_raw, (list, tuple)):
        for a in authors_raw:
            if not isinstance(a, str):
                a = str(a)
            s = a.strip()
            if s:
                authors_list.append(s)
    elif isinstance(authors_raw, str):
        s = authors_raw.strip()
        if s:
            if "," in s or ";" in s:
                parts = re.split(r"[;,]", s)
            elif " and " in s:
                parts = s.split(" and ")
            else:
                parts = [s]
            authors_list = [p.strip() for p in parts if p.strip()]

    return {
        "title": title,
        "authors": authors_list,
        "year": None,
        "abstract": abstract,
    }


# Patterns to detect Introduction section (stop extracting front matter here)
INTRO_PATTERNS = [
    r"^#{1,3}\s*(?:\d+\.?\s*)?Introduction\b",
    r"^(?:\d+\.?\s*)?Introduction\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?引言\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Background\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Related\s+Work\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Preliminaries\b",
    r"^#{1,3}\s*(?:\d+\.?\s*)?Problem\s+Statement\b",
]

METADATA_EXTRACTION_PROMPT = """Extract metadata from this academic paper's front matter.

Input:
---
{content}
---

Return JSON only:
{{"title": "...", "authors": ["..."], "abstract": "..."}}

Rules:
- Use null for missing fields, [] for missing authors
- No affiliations/emails in author names
- JSON only, no explanation"""


def _redact_error_message(msg: str, max_len: int = 300) -> str:
    """Redact likely filesystem paths from user-visible error messages."""
    if not msg:
        return ""
    # Collapse whitespace/newlines to keep the UI compact.
    msg = " ".join(str(msg).split())
    try:
        msg = msg.replace(DATA_DIR, "<DATA_DIR>")
    except Exception:
        pass

    # Redact deep absolute paths (avoid replacing short URL paths like /api).
    msg = re.sub(r"/(?:[^\s/]+/){2,}[^\s/]+", "<PATH>", msg)
    msg = re.sub(r"[A-Za-z]:\\\\(?:[^\s\\\\]+\\\\){2,}[^\s\\\\]+", "<PATH>", msg)
    return msg[:max_len]


def extract_front_matter(md_content: str, max_chars: int = 12000) -> str:
    """Extract front matter from markdown content (before Introduction).

    Args:
        md_content: Full markdown content
        max_chars: Maximum characters to return

    Returns:
        Front matter text
    """
    lines = md_content.split("\n")
    for i, line in enumerate(lines):
        for pattern in INTRO_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                front = "\n".join(lines[:i])
                return front[:max_chars] if len(front) > max_chars else front
    return md_content[:max_chars]


def extract_metadata_with_llm(front_matter: str) -> Dict[str, Any]:
    """Extract metadata from front matter using LLM.

    Uses settings.extract_info configuration (default model: glm-4.7).
    Falls back to main LLM if extract_info model fails.

    Uses OpenAI client library for consistency with main LLM logic in paper_summarizer.

    Args:
        front_matter: Text content before Introduction

    Returns:
        Dictionary with title, authors, year (always None), abstract
    """
    extract_base_url = settings.extract_info.base_url or LLM_BASE_URL
    if not extract_base_url or not front_matter.strip():
        return {"title": "", "authors": [], "year": None, "abstract": None}

    prompt = METADATA_EXTRACTION_PROMPT.format(content=front_matter[:8000])

    # Try extract_info model first, then fallback to main LLM
    models_to_try = [
        (settings.extract_info.model_name, _get_extract_info_client()),
    ]
    # Add main LLM as fallback if different from extract_info model
    if settings.extract_info.model_name != LLM_NAME:
        import openai

        main_client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        models_to_try.append((LLM_NAME, main_client))

    for model_name, client in models_to_try:
        try:
            temperature = settings.extract_info.temperature
            max_tokens = settings.extract_info.max_tokens
            timeout = settings.extract_info.timeout

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            if not response.choices:
                logger.warning(f"LLM {model_name} returned empty choices for metadata extraction")
                continue

            choice = response.choices[0]

            # Check for truncated response (token limit reached)
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                logger.warning(f"LLM {model_name} response truncated (finish_reason=length), may have incomplete JSON")

            content = choice.message.content or ""
            reasoning_content = getattr(choice.message, "reasoning_content", None) or ""

            # Merge content and reasoning_content for reasoning models
            combined_content = f"{reasoning_content}\n{content}".strip()

            if not combined_content:
                logger.warning(f"LLM {model_name} returned empty content for metadata extraction")
                continue

            # Find all JSON objects and use the last one (final output from reasoning)
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            json_matches = list(re.finditer(json_pattern, combined_content, re.DOTALL))

            # Try JSON matches from last to first (last one is usually the final answer)
            for match in reversed(json_matches):
                try:
                    meta = json.loads(match.group())
                    if meta.get("title") or meta.get("authors"):
                        logger.info(f"Successfully extracted metadata using {model_name}")
                        return _normalize_extracted_metadata(meta)
                except json.JSONDecodeError:
                    continue

            # Fallback: try to parse the entire content as JSON
            try:
                meta = json.loads(content.strip())
                if meta.get("title") or meta.get("authors"):
                    logger.info(f"Successfully extracted metadata using {model_name}")
                    return _normalize_extracted_metadata(meta)
            except json.JSONDecodeError:
                pass

            # Don't log response content to avoid leaking sensitive paper information
            logger.warning(f"Failed to parse JSON from {model_name} response (length={len(combined_content)})")
        except Exception as e:
            logger.warning(f"Failed to extract metadata with {model_name}: {e}")

    return {"title": "", "authors": [], "year": None, "abstract": None}


def create_uploaded_paper(
    user: str,
    file_content: bytes,
    original_filename: str,
    max_uploads_per_user: int = 100,
) -> Tuple[str, Dict[str, Any], bool]:
    """Create a new uploaded paper record.

    Args:
        user: Username (owner)
        file_content: PDF file content
        original_filename: Original filename
        max_uploads_per_user: Maximum uploads allowed per user (for quota enforcement)

    Returns:
        Tuple of (pid, paper_data, is_new)

    Raises:
        QuotaExceededError: If user has reached upload limit
    """
    sha256 = compute_bytes_sha256(file_content)

    # Fast path: return existing record if present.
    existing = UploadedPaperRepository.get_by_sha256(user, sha256)
    if existing:
        pid, data = existing
        logger.info(f"Duplicate upload detected for user {user}, returning existing pid {pid}")
        return pid, data, False

    now = time.time()
    safe_name = sanitize_filename(original_filename)

    # Create record + sha mapping atomically to avoid duplicate PIDs under concurrent uploads.
    # Quota check is also done inside the transaction to prevent race conditions.
    # File write happens after commit; on failure we best-effort clean up DB entries.
    from aslite.db import get_uploaded_papers_db, get_uploaded_papers_index_db

    sha_key = UploadedPaperRepository.sha256_mapping_key(user, sha256)
    pid: str | None = None
    paper_data: dict[str, Any] | None = None

    try:
        with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                mapped_pid = updb.get(sha_key)
                if isinstance(mapped_pid, str) and mapped_pid:
                    record = UploadedPaperRepository.get(mapped_pid)
                    if isinstance(record, dict) and record.get("owner") == user and record.get("sha256") == sha256:
                        return mapped_pid, record, False
                    # Stale mapping - overwrite below.

                # Atomic quota check inside transaction
                # Count user's existing uploads by scanning the index
                with get_uploaded_papers_index_db(flag="c", autocommit=False) as idx_db:
                    indexed_pids = idx_db.get(user, [])
                    # Verify each indexed PID actually exists and belongs to user
                    valid_count = 0
                    for indexed_pid in indexed_pids:
                        rec = updb.get(indexed_pid)
                        if isinstance(rec, dict) and rec.get("owner") == user:
                            valid_count += 1

                    if valid_count >= max_uploads_per_user:
                        raise QuotaExceededError(f"Upload limit reached (max {max_uploads_per_user} papers)")

                # Generate a PID that doesn't collide with an existing record.
                for _ in range(5):
                    candidate = generate_upload_pid()
                    if not updb.get(candidate):
                        pid = candidate
                        break
                if not pid:
                    raise RuntimeError("Failed to generate unique upload pid")

                paper_data = {
                    "pid": pid,
                    "owner": user,
                    "created_time": now,
                    "updated_time": now,
                    "original_filename": safe_name,
                    "size_bytes": len(file_content),
                    "sha256": sha256,
                    "parse_status": "queued",
                    "parse_error": None,
                    "meta_extracted": {
                        "title": "",
                        "authors": [],
                        "year": None,
                        "abstract": None,
                    },
                    "meta_extracted_ok": False,
                    "meta_override": {},
                    "summary_task_id": None,
                }

                updb[pid] = paper_data
                updb[sha_key] = pid

        # Write PDF to disk (best effort).
        upload_dir = get_upload_dir(pid, DATA_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = get_upload_pdf_path(pid, DATA_DIR)
        with open(pdf_path, "wb") as f:
            f.write(file_content)

        UploadedPaperRepository.add_to_index(user, pid)
        logger.info(f"Created uploaded paper {pid} for user {user}")
        return pid, paper_data or {}, True

    except QuotaExceededError:
        raise
    except Exception:
        # Best-effort cleanup to avoid stale mapping/record.
        try:
            if pid:
                UploadedPaperRepository.delete(pid)
                UploadedPaperRepository.remove_from_index(user, pid)
            UploadedPaperRepository.remove_sha256_mapping(user, sha256, pid=pid)
        except Exception:
            pass
        raise


class QuotaExceededError(Exception):
    """Raised when user has exceeded their upload quota."""


def process_uploaded_pdf(pid: str, user: str, model: str | None = None):
    """Process an uploaded PDF: parse with MinerU and extract metadata.

    This is called by the Huey task.

    Args:
        pid: Upload PID
        user: Username
        model: LLM model for summary (optional)
    """
    # Validate PID format for security
    if not validate_upload_pid(pid):
        logger.error(f"Invalid upload PID format: {pid}")
        return

    from tools.paper_summarizer import PaperSummarizer

    record = UploadedPaperRepository.get(pid)
    if not record:
        logger.error(f"Uploaded paper {pid} not found")
        return

    if record.get("owner") != user:
        logger.error(f"User {user} does not own uploaded paper {pid}")
        return

    # Update status to running
    UploadedPaperRepository.update(pid, {"parse_status": "running", "parse_error": None})

    try:
        # Get PDF path
        pdf_path = get_upload_pdf_path(pid, DATA_DIR)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Parse with MinerU (use pid as cache_pid, keep_pdf=True to preserve uploaded file)
        summarizer = PaperSummarizer()
        md_path = summarizer.parse_pdf_with_mineru(pdf_path, cache_pid=pid, keep_pdf=True)

        if not md_path or not md_path.exists():
            raise RuntimeError("MinerU parsing returned empty content")

        # Read the markdown content
        md_content = md_path.read_text(encoding="utf-8")

        # Update parse status first (parsing succeeded)
        UploadedPaperRepository.update(
            pid,
            {
                "parse_status": "ok",
                "parse_error": None,
            },
        )

        # Extract metadata from front matter (separate step, can fail independently)
        meta_extracted_ok = False
        try:
            front_matter = extract_front_matter(md_content)
            meta_extracted = extract_metadata_with_llm(front_matter)
            # Check if we got meaningful data
            if meta_extracted.get("title") or meta_extracted.get("authors"):
                meta_extracted_ok = True
                UploadedPaperRepository.update(
                    pid,
                    {
                        "meta_extracted": meta_extracted,
                        "meta_extracted_ok": True,
                    },
                )
            else:
                logger.warning(f"Metadata extraction returned empty for {pid}")
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {pid}: {e}")

        logger.info(f"Successfully processed uploaded paper {pid} (meta_ok={meta_extracted_ok})")

        # Trigger summary generation
        try:
            from tasks import enqueue_summary_task

            task_id = enqueue_summary_task(pid, model=model, user=user)
            if task_id:
                UploadedPaperRepository.update(pid, {"summary_task_id": task_id})
        except Exception as e:
            logger.warning(f"Failed to enqueue summary for {pid}: {e}")

    except Exception as e:
        logger.error(f"Failed to process uploaded paper {pid}: {e}")
        parse_error = _redact_error_message(f"{type(e).__name__}: {e}") or type(e).__name__
        UploadedPaperRepository.update(
            pid,
            {
                "parse_status": "failed",
                "parse_error": parse_error,
            },
        )
        raise


def get_uploaded_papers_list(user: str) -> List[Dict[str, Any]]:
    """Get list of uploaded papers for a user in display format.

    Args:
        user: Username

    Returns:
        List of paper items for frontend display
    """
    from backend.services.summary_service import get_summary_status

    papers = UploadedPaperRepository.get_by_owner(user)

    # Get user tags for these papers
    user_tags = TagRepository.get_user_tags(user)
    user_neg_tags = NegativeTagRepository.get_user_neg_tags(user)

    result = []
    for pid, data in papers.items():
        meta = data.get("meta_extracted", {})
        override = data.get("meta_override", {})

        # Merge meta with overrides
        title = override.get("title") or meta.get("title") or data.get("original_filename", pid)
        authors_list = override.get("authors") or meta.get("authors") or []
        # Do not expose year/time for uploaded papers.
        abstract = override.get("abstract") or meta.get("abstract")

        # Backward compatibility: older records may not have `meta_extracted_ok`.
        # Infer it from available meta fields so frontend gating stays consistent with API behavior.
        meta_extracted_ok = _infer_meta_extracted_ok(data, title, abstract, authors_list)

        # Get tags for this paper
        utags = []
        ntags = []
        for tag, pids in user_tags.items():
            if pid in pids:
                utags.append(tag)
        for tag, pids in user_neg_tags.items():
            if pid in pids:
                ntags.append(tag)

        # Get summary status and TL;DR
        summary_status, summary_last_error = get_summary_status(pid)

        # Extract TL;DR from summary if available
        tldr = ""
        if summary_status == "ok":
            from backend.services.summary_service import extract_tldr_from_summary

            tldr = extract_tldr_from_summary(pid) or ""

        result.append(
            {
                "id": pid,
                "kind": "upload",
                "title": title,
                "authors": ", ".join(authors_list) if authors_list else "",
                "time": "",
                "summary": abstract,
                "tldr": tldr,
                "utags": utags,
                "ntags": ntags,
                "parse_status": data.get("parse_status", ""),
                "parse_error": data.get("parse_error") or "",
                "meta_extracted_ok": meta_extracted_ok,
                "summary_status": summary_status or "",
                "summary_last_error": summary_last_error or "",
                "summary_task_id": data.get("summary_task_id") or "",
                "created_time": data.get("created_time", 0),
                "original_filename": data.get("original_filename", ""),
            }
        )

    # Sort by created_time descending
    result.sort(key=lambda x: x.get("created_time", 0), reverse=True)
    return result


def update_uploaded_paper_meta(
    pid: str,
    user: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    abstract: Optional[str] = None,
) -> bool:
    """Update metadata override for an uploaded paper.

    Args:
        pid: Upload PID
        user: Username (must be owner)
        title: New title (optional)
        authors: New authors list (optional)
        year: New year (optional)
        abstract: New abstract (optional)

    Returns:
        True if updated successfully
    """
    record = UploadedPaperRepository.get(pid)
    if not record:
        return False

    if record.get("owner") != user:
        return False

    # Validate inputs early.
    title, authors, year, abstract = _validate_meta_override_inputs(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
    )

    override = record.get("meta_override", {})

    if title is not None:
        override["title"] = title
    if authors is not None:
        override["authors"] = authors
    if year is not None:
        override["year"] = year
    if abstract is not None:
        override["abstract"] = abstract

    UploadedPaperRepository.update(pid, {"meta_override": override})
    _invalidate_upload_features(pid)
    return True


def delete_uploaded_paper(pid: str, user: str) -> tuple[bool, str]:
    """Delete an uploaded paper and all associated data.

    Uses two-phase delete: first delete files, then delete DB records.
    This ensures we don't leave orphaned files on disk if DB delete succeeds
    but file delete fails.

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        Tuple of (success, message).
        - success=True, message="deleted": successfully deleted
        - success=False, message="...": error reason
    """
    # Validate PID format for security
    if not validate_upload_pid(pid):
        logger.error(f"Invalid upload PID format: {pid}")
        return False, "invalid_pid"

    record = UploadedPaperRepository.get(pid)
    if not record:
        return False, "not_found"

    if record.get("owner") != user:
        return False, "not_owner"

    sha256 = record.get("sha256")

    # Phase 1: Delete files first (critical paths)
    # If this fails, we abort and keep DB intact so user can retry
    critical_errors = []

    upload_dir = get_upload_dir(pid, DATA_DIR)
    if upload_dir.exists():
        try:
            shutil.rmtree(upload_dir)
        except Exception as e:
            critical_errors.append(f"upload_dir: {e}")
            logger.error(f"Failed to delete upload directory for {pid}: {e}")

    # If critical file deletion failed, abort
    if critical_errors:
        return False, f"file_delete_failed: {'; '.join(critical_errors)}"

    # Phase 2: Delete DB records (now safe since files are gone)
    try:
        UploadedPaperRepository.delete(pid)
        UploadedPaperRepository.remove_from_index(user, pid)

        # Remove sha256 mapping so future uploads can re-create a clean mapping.
        if sha256:
            try:
                UploadedPaperRepository.remove_sha256_mapping(user, str(sha256), pid=pid)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to delete DB records for {pid}: {e}")
        # Files are already deleted, so we should still try to clean up
        # but report partial failure
        return False, f"db_delete_failed: {e}"

    # Phase 3: Clean up non-critical caches (best effort, don't fail on errors)
    # Delete MinerU cache
    mineru_dir = Path(DATA_DIR) / "mineru" / pid
    if mineru_dir.exists():
        try:
            shutil.rmtree(mineru_dir)
        except Exception as e:
            logger.warning(f"Failed to delete MinerU cache for {pid}: {e}")

    # Delete summary cache
    summary_dir = Path(DATA_DIR) / "summary" / pid
    if summary_dir.exists():
        try:
            shutil.rmtree(summary_dir)
        except Exception as e:
            logger.warning(f"Failed to delete summary cache for {pid}: {e}")

    # Clean up tags (best effort)
    try:
        from aslite.db import get_neg_tags_db, get_tags_db

        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {})
            changed = False
            for tag in list(tags.keys()):
                if pid in tags[tag]:
                    tags[tag].discard(pid)
                    if not tags[tag]:
                        del tags[tag]
                    changed = True
            if changed:
                tdb[user] = tags

        with get_neg_tags_db(flag="c") as ntdb:
            neg_tags = ntdb.get(user, {})
            changed = False
            for tag in list(neg_tags.keys()):
                if pid in neg_tags[tag]:
                    neg_tags[tag].discard(pid)
                    if not neg_tags[tag]:
                        del neg_tags[tag]
                    changed = True
            if changed:
                ntdb[user] = neg_tags
    except Exception as e:
        logger.warning(f"Failed to clean up tags for {pid}: {e}")

    logger.info(f"Deleted uploaded paper {pid} for user {user}")
    return True, "deleted"


def retry_parse_uploaded_paper(pid: str, user: str) -> tuple[bool, str]:
    """Retry parsing for a failed uploaded paper.

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        Tuple of (success, message).
        - success=True, message="queued": newly enqueued
        - success=True, message="already_in_progress": idempotent, already queued/running
        - success=False, message="...": error reason
    """
    # Validate PID format for security
    if not validate_upload_pid(pid):
        logger.error(f"Invalid upload PID format: {pid}")
        return False, "invalid_pid"

    record = UploadedPaperRepository.get(pid)
    if not record:
        return False, "not_found"

    if record.get("owner") != user:
        return False, "not_owner"

    current_status = record.get("parse_status", "")

    # Idempotent: if already queued or running, return success
    if current_status in ("queued", "running"):
        logger.info(f"Parse already in progress for {pid} (status={current_status})")
        return True, "already_in_progress"

    # Only allow retry for failed status
    if current_status != "failed":
        return False, "not_failed"

    # Atomically check-and-set status
    from aslite.db import get_uploaded_papers_db

    try:
        with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                current_record = updb.get(pid)
                if not isinstance(current_record, dict):
                    return False, "not_found"

                actual_status = current_record.get("parse_status", "")
                if actual_status in ("queued", "running"):
                    return True, "already_in_progress"
                if actual_status != "failed":
                    return False, "not_failed"

                current_record["parse_status"] = "queued"
                current_record["parse_error"] = None
                updb[pid] = current_record
    except Exception as e:
        logger.error(f"Failed to update parse status for {pid}: {e}")
        return False, "db_error"

    try:
        from tasks import huey, process_uploaded_pdf_task

        # Use huey.enqueue() to run task asynchronously
        task = process_uploaded_pdf_task.s(pid, user)
        huey.enqueue(task)
    except Exception as e:
        logger.error(f"Failed to enqueue retry for {pid}: {e}")
        try:
            UploadedPaperRepository.update(pid, {"parse_status": "failed", "parse_error": "enqueue_failed"})
        except Exception:
            pass
        return False, "enqueue_failed"

    return True, "queued"


def trigger_parse_only(pid: str, user: str) -> tuple[bool, str]:
    """Trigger MinerU parsing only (without metadata extraction).

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        Tuple of (success, message). Message explains why if not successful.
        - success=True, message="queued": newly enqueued
        - success=True, message="already_in_progress": idempotent, already queued/running
        - success=False, message="...": error reason
    """
    if not validate_upload_pid(pid):
        logger.error(f"Invalid upload PID format: {pid}")
        return False, "invalid_pid"

    record = UploadedPaperRepository.get(pid)
    if not record:
        return False, "not_found"

    if record.get("owner") != user:
        return False, "not_owner"

    current_status = record.get("parse_status", "")

    # Idempotent: if already queued or running, return success without re-enqueue
    if current_status in ("queued", "running"):
        logger.info(f"Parse already in progress for {pid} (status={current_status})")
        return True, "already_in_progress"

    # Already parsed successfully - no need to re-parse
    if current_status == "ok":
        return False, "already_parsed"

    # Atomically check-and-set status to avoid race conditions
    # Use compare-and-swap pattern: only update if status hasn't changed
    from aslite.db import get_uploaded_papers_db

    try:
        with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
            with updb.transaction(mode="IMMEDIATE"):
                current_record = updb.get(pid)
                if not isinstance(current_record, dict):
                    return False, "not_found"

                # Re-check status inside transaction
                actual_status = current_record.get("parse_status", "")
                if actual_status in ("queued", "running"):
                    return True, "already_in_progress"
                if actual_status == "ok":
                    return False, "already_parsed"

                # Update status atomically
                current_record["parse_status"] = "queued"
                current_record["parse_error"] = None
                updb[pid] = current_record
    except Exception as e:
        logger.error(f"Failed to update parse status for {pid}: {e}")
        return False, "db_error"

    try:
        from tasks import huey, parse_uploaded_pdf_task

        task = parse_uploaded_pdf_task.s(pid, user)
        huey.enqueue(task)
    except Exception as e:
        logger.error(f"Failed to enqueue parse for {pid}: {e}")
        # Rollback status on enqueue failure
        try:
            UploadedPaperRepository.update(pid, {"parse_status": "failed", "parse_error": "enqueue_failed"})
        except Exception:
            pass
        return False, "enqueue_failed"

    return True, "queued"


def trigger_extract_info(pid: str, user: str) -> bool:
    """Trigger metadata extraction only (requires parsing to be done).

    Args:
        pid: Upload PID
        user: Username (must be owner)

    Returns:
        True if extraction was triggered
    """
    if not validate_upload_pid(pid):
        logger.error(f"Invalid upload PID format: {pid}")
        return False

    record = UploadedPaperRepository.get(pid)
    if not record:
        logger.warning(f"Record not found for {pid}")
        return False

    if record.get("owner") != user:
        logger.warning(f"User {user} does not own {pid}")
        return False

    # Only allow if already parsed successfully
    if record.get("parse_status") != "ok":
        logger.warning(f"Paper {pid} not parsed yet (status: {record.get('parse_status')})")
        return False

    # Only allow if not already extracted successfully
    # Use explicit True check to allow re-extraction for old records without this field
    if record.get("meta_extracted_ok") is True:
        logger.warning(f"Paper {pid} already has extracted metadata")
        return False

    try:
        from tasks import extract_info_task, huey

        task = extract_info_task.s(pid, user)
        huey.enqueue(task)
    except Exception as e:
        logger.error(f"Failed to enqueue extract info for {pid}: {e}")
        return False

    return True


def do_extract_metadata(pid: str, user: str) -> bool:
    """Actually perform metadata extraction (called by task).

    Args:
        pid: Upload PID
        user: Username

    Returns:
        True if extraction succeeded
    """
    if not validate_upload_pid(pid):
        return False

    record = UploadedPaperRepository.get(pid)
    if not record or record.get("owner") != user:
        return False

    if record.get("parse_status") != "ok":
        return False

    try:
        from tools.paper_summarizer import PaperSummarizer

        summarizer = PaperSummarizer()
        backend = summarizer._normalize_mineru_backend()
        md_path = summarizer._find_mineru_markdown(pid, backend=backend)

        if not md_path or not md_path.exists():
            logger.error(f"MinerU markdown not found for {pid}")
            return False

        md_content = md_path.read_text(encoding="utf-8")
        front_matter = extract_front_matter(md_content)
        meta_extracted = extract_metadata_with_llm(front_matter)

        if meta_extracted.get("title") or meta_extracted.get("authors"):
            UploadedPaperRepository.update(
                pid,
                {
                    "meta_extracted": meta_extracted,
                    "meta_extracted_ok": True,
                },
            )
            logger.info(f"Successfully extracted metadata for {pid}")
            return True
        else:
            logger.warning(f"Metadata extraction returned empty for {pid}")
            return False

    except Exception as e:
        logger.error(f"Failed to extract metadata for {pid}: {e}")
        return False


def do_parse_only(pid: str, user: str) -> bool:
    """Actually perform MinerU parsing only (called by task).

    Args:
        pid: Upload PID
        user: Username

    Returns:
        True if parsing succeeded
    """
    if not validate_upload_pid(pid):
        return False

    record = UploadedPaperRepository.get(pid)
    if not record or record.get("owner") != user:
        return False

    UploadedPaperRepository.update(pid, {"parse_status": "running", "parse_error": None})

    try:
        from tools.paper_summarizer import PaperSummarizer

        pdf_path = get_upload_pdf_path(pid, DATA_DIR)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        summarizer = PaperSummarizer()
        md_path = summarizer.parse_pdf_with_mineru(pdf_path, cache_pid=pid, keep_pdf=True)

        if not md_path or not md_path.exists():
            raise RuntimeError("MinerU parsing returned empty content")

        UploadedPaperRepository.update(
            pid,
            {
                "parse_status": "ok",
                "parse_error": None,
            },
        )
        logger.info(f"Successfully parsed uploaded paper {pid}")
        return True

    except Exception as e:
        logger.error(f"Failed to parse uploaded paper {pid}: {e}")
        parse_error = _redact_error_message(f"{type(e).__name__}: {e}") or type(e).__name__
        UploadedPaperRepository.update(
            pid,
            {
                "parse_status": "failed",
                "parse_error": parse_error,
            },
        )
        return False


def get_upload_summary_context(pid: str, user: str) -> Optional[Dict[str, Any]]:
    """Get context for rendering summary page for an uploaded paper.

    Args:
        pid: Upload PID
        user: Username

    Returns:
        Context dictionary for template, or None if not found/unauthorized
    """
    record = UploadedPaperRepository.get(pid)
    if not record:
        return None

    if record.get("owner") != user:
        return None

    meta = record.get("meta_extracted", {})
    override = record.get("meta_override", {})

    title = override.get("title") or meta.get("title") or record.get("original_filename", pid)
    authors_list = override.get("authors") or meta.get("authors") or []
    abstract = override.get("abstract") or meta.get("abstract")
    meta_extracted_ok = _infer_meta_extracted_ok(record, title, abstract, authors_list)

    # Format upload time for display
    created_time = record.get("created_time", 0)
    if created_time:
        from datetime import datetime

        dt = datetime.fromtimestamp(created_time)
        time_str = f"Uploaded: {dt.strftime('%Y-%m-%d %H:%M')}"
    else:
        time_str = ""

    # Build paper-like structure for template
    paper = {
        "id": pid,
        "kind": "upload",
        "title": title,
        "authors": ", ".join(authors_list) if authors_list else "",
        "time": time_str,
        "summary": abstract or "",
        "tags": "",
        "utags": [],
        "ntags": [],
        "parse_status": record.get("parse_status", ""),
        "meta_extracted_ok": meta_extracted_ok,
        "created_time": created_time,
    }

    return {
        "paper": paper,
        "pid": pid,
    }
