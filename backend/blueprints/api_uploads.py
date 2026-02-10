"""
API Blueprint for uploaded papers functionality.

All endpoints require authentication.
"""

import time

from flask import Blueprint, abort, g, jsonify, request, send_file
from loguru import logger

from backend.utils.upload_utils import get_upload_pdf_path, validate_upload_pid
from backend.utils.validation import csrf_protect
from config import settings

bp = Blueprint("api_uploads", __name__, url_prefix="/api")

DATA_DIR = str(settings.data_dir)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
MAX_UPLOADS_PER_USER = 100  # Maximum uploads per user to prevent storage abuse


def _looks_like_pdf(content: bytes, max_scan: int = 1024) -> bool:
    """Best-effort PDF header check.

    Some generators may prepend UTF-8 BOM or whitespace before '%PDF'. We allow
    a small scan window to reduce false negatives.
    """
    if not content:
        return False
    head = content[: int(max_scan)]
    # Strip UTF-8 BOM if present.
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:]
    # Skip leading whitespace/newlines.
    head = head.lstrip(b" \t\r\n\f\v")
    return head.startswith(b"%PDF")


def _api_error(error: str, status: int = 400, **extra):
    """Return a standardized JSON error response."""
    resp = {"success": False, "error": error}
    resp.update(extra)
    return jsonify(resp), status


def _api_success(**data):
    """Return a standardized JSON success response."""
    resp = {"success": True}
    resp.update(data)
    return jsonify(resp)


@bp.route("/upload_pdf", methods=["POST"])
def api_upload_pdf():
    """Upload a PDF file.

    Expects multipart/form-data with 'file' field.
    Returns the new paper's PID and triggers async processing.

    Quota enforcement is done atomically inside create_uploaded_paper to prevent
    race conditions from concurrent uploads.
    """
    if g.user is None:
        return _api_error("Not logged in", 401)

    csrf_protect()

    if "file" not in request.files:
        return _api_error("No file provided", 400)

    file = request.files["file"]
    if not file.filename:
        return _api_error("No file selected", 400)

    if not file.filename.lower().endswith(".pdf"):
        return _api_error("Only PDF files are supported", 400)

    # Check file size before reading entire content (best effort: handle non-seekable streams).
    file_content: bytes | None = None
    try:
        file.seek(0, 2)  # Seek to end
        file_size = int(file.tell() or 0)
        file.seek(0)  # Seek back to start
    except Exception:
        file_content = file.read()
        file_size = len(file_content or b"")

    if file_size > MAX_UPLOAD_SIZE:
        return _api_error(f"File too large (max {MAX_UPLOAD_SIZE // 1024 // 1024}MB)", 413)

    if file_size < 100:
        return _api_error("File too small to be a valid PDF", 400)

    # Read file content (reuse if already read above).
    if file_content is None:
        file_content = file.read()

    # Check PDF magic bytes (with small preamble tolerance)
    if not _looks_like_pdf(file_content):
        return _api_error("Invalid PDF file", 400)

    try:
        from backend.services.upload_service import (
            QuotaExceededError,
            create_uploaded_paper,
        )

        pid, paper_data, is_new = create_uploaded_paper(
            user=g.user,
            file_content=file_content,
            original_filename=file.filename,
            max_uploads_per_user=MAX_UPLOADS_PER_USER,
        )

        # Defensive: if the record already exists but the PDF was deleted, restore it from this upload.
        try:
            pdf_path = get_upload_pdf_path(pid, DATA_DIR)
            if not pdf_path.exists():
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pdf_path, "wb") as f:
                    f.write(file_content)
        except Exception as e:
            logger.warning(f"Failed to restore missing PDF for {pid}: {e}")

        # Only trigger async processing if not already processed/processing.
        # For duplicate uploads (same SHA256), we must transition parse_status to
        # "queued" atomically to avoid duplicate enqueues before the worker flips it.
        parse_status = paper_data.get("parse_status", "") or ""
        should_enqueue = False

        if is_new:
            should_enqueue = True
            parse_status = "queued"
        else:
            if parse_status not in ("ok", "running", "queued"):
                try:
                    from aslite.db import get_uploaded_papers_db

                    with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
                        with updb.transaction(mode="IMMEDIATE"):
                            record = updb.get(pid)
                            if isinstance(record, dict) and record.get("owner") == g.user:
                                current = (record.get("parse_status") or "").strip()
                                if current not in ("ok", "running", "queued"):
                                    record["parse_status"] = "queued"
                                    record["parse_error"] = None
                                    record["updated_time"] = time.time()
                                    updb[pid] = record
                                    paper_data = record
                                    parse_status = "queued"
                                    should_enqueue = True
                                else:
                                    paper_data = record
                                    parse_status = current
                except Exception as e:
                    logger.warning(f"Failed to update parse_status for re-upload {pid}: {e}")
                    should_enqueue = True
                    parse_status = "queued"

        if should_enqueue:
            try:
                from backend.services.upload_service import register_upload_task_enqueue
                from tasks import huey, process_uploaded_pdf_task

                task = process_uploaded_pdf_task.s(pid, g.user)
                enqueue_result = huey.enqueue(task)
                task_id = register_upload_task_enqueue(
                    task_type="process",
                    pid=pid,
                    user=g.user,
                    task=task,
                    enqueue_result=enqueue_result,
                )
            except Exception as e:
                logger.warning(f"Failed to enqueue processing for {pid}: {e}")
                try:
                    from aslite.repositories import UploadedPaperRepository

                    UploadedPaperRepository.update(
                        pid,
                        {"parse_status": "failed", "parse_error": "enqueue_failed"},
                    )
                except Exception as update_err:
                    logger.warning(f"Failed to mark enqueue failure for {pid}: {update_err}")
                return _api_error("Failed to enqueue processing. Please retry.", 500)
        else:
            task_id = str(paper_data.get("parse_task_id") or "")

        return _api_success(
            pid=pid,
            parse_status=parse_status or "queued",
            original_filename=paper_data.get("original_filename", ""),
            task_id=task_id,
        )

    except QuotaExceededError as e:
        return _api_error(str(e), 429)

    except Exception as e:
        logger.error(f"Failed to upload PDF: {e}")
        return _api_error("Upload failed. Please try again.", 500)


@bp.route("/uploaded_papers/list", methods=["GET"])
def api_uploaded_papers_list():
    """Get list of uploaded papers for current user."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    try:
        from backend.services.upload_service import get_uploaded_papers_list

        papers = get_uploaded_papers_list(g.user)
        return _api_success(papers=papers)

    except Exception as e:
        logger.error(f"Failed to list uploaded papers: {e}")
        return _api_error("Failed to list papers. Please try again.", 500)


@bp.route("/uploaded_papers/update_meta", methods=["POST"])
def api_uploaded_papers_update_meta():
    """Update metadata for an uploaded paper."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    csrf_protect()

    data = request.get_json(silent=True)
    if not data:
        return _api_error("No JSON data provided", 400)

    pid = data.get("pid", "").strip()
    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from backend.services.upload_service import update_uploaded_paper_meta

        success = update_uploaded_paper_meta(
            pid=pid,
            user=g.user,
            title=data.get("title"),
            authors=data.get("authors"),
            year=data.get("year"),
            abstract=data.get("abstract"),
        )

        if not success:
            return _api_error("Paper not found or access denied", 404)

        return _api_success(pid=pid)

    except ValueError as e:
        # Validation errors from upload_service.
        return _api_error(str(e), 400)

    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")
        return _api_error("Failed to update metadata. Please try again.", 500)


@bp.route("/uploaded_papers/delete", methods=["POST"])
def api_uploaded_papers_delete():
    """Delete an uploaded paper.

    Uses two-phase delete: files first, then DB records.
    Returns error if file deletion fails so user can retry.
    """
    if g.user is None:
        return _api_error("Not logged in", 401)

    csrf_protect()

    data = request.get_json(silent=True)
    if not data:
        return _api_error("No JSON data provided", 400)

    pid = data.get("pid", "").strip()
    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from backend.services.upload_service import delete_uploaded_paper

        success, message = delete_uploaded_paper(pid, g.user)

        if success:
            return _api_success(pid=pid)

        if message == "not_found":
            return _api_error("Paper not found", 404)
        elif message == "not_owner":
            return _api_error("Paper not found", 404)
        elif message == "invalid_pid":
            return _api_error("Invalid paper ID", 400)
        elif message.startswith("file_delete_failed"):
            return _api_error("Failed to delete files. Please try again.", 500)
        elif message.startswith("db_delete_failed"):
            return _api_error("Partial deletion occurred. Please contact support.", 500)
        else:
            return _api_error(f"Delete failed: {message}", 500)

    except Exception as e:
        logger.error(f"Failed to delete uploaded paper: {e}")
        return _api_error("Failed to delete paper. Please try again.", 500)


@bp.route("/uploaded_papers/retry_parse", methods=["POST"])
def api_uploaded_papers_retry_parse():
    """Retry parsing for a failed uploaded paper.

    Returns:
        - 200 with parse_status="queued" if newly enqueued
        - 200 with parse_status="already_in_progress" if already queued/running (idempotent)
        - 400 if not in failed state
        - 404 if not found or not owner
    """
    if g.user is None:
        return _api_error("Not logged in", 401)

    csrf_protect()

    data = request.get_json(silent=True)
    if not data:
        return _api_error("No JSON data provided", 400)

    pid = data.get("pid", "").strip()
    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from backend.services.upload_service import retry_parse_uploaded_paper

        success, message, task_id = retry_parse_uploaded_paper(pid, g.user)

        if success:
            return _api_success(pid=pid, parse_status=message, task_id=task_id)

        if message == "not_found":
            return _api_error("Paper not found", 404)
        elif message == "not_owner":
            return _api_error("Paper not found", 404)
        elif message == "not_failed":
            return _api_error("Paper is not in failed state, cannot retry", 400)
        elif message == "invalid_pid":
            return _api_error("Invalid paper ID", 400)
        elif message == "db_error":
            return _api_error("Database error, please retry", 500)
        elif message == "enqueue_failed":
            return _api_error("Failed to enqueue task, please retry", 500)
        else:
            return _api_error(f"Retry failed: {message}", 400)

    except Exception as e:
        logger.error(f"Failed to retry parse: {e}")
        return _api_error("Failed to retry parsing. Please try again.", 500)


@bp.route("/uploaded_papers/parse", methods=["POST"])
def api_uploaded_papers_parse():
    """Trigger MinerU parsing for an uploaded paper.

    Returns:
        - 200 with parse_status="queued" if newly enqueued
        - 200 with parse_status="already_in_progress" if already queued/running (idempotent)
        - 400 if invalid input
        - 404 if not found or not owner
        - 409 if already parsed successfully
    """
    if g.user is None:
        return _api_error("Not logged in", 401)

    csrf_protect()

    data = request.get_json(silent=True)
    if not data:
        return _api_error("No JSON data provided", 400)

    pid = data.get("pid", "").strip()
    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from backend.services.upload_service import trigger_parse_only

        success, message, task_id = trigger_parse_only(pid, g.user)

        if success:
            # Both "queued" and "already_in_progress" are success cases
            return _api_success(pid=pid, parse_status=message, task_id=task_id)

        # Handle specific error cases with appropriate HTTP status codes
        if message == "not_found":
            return _api_error("Paper not found", 404)
        elif message == "not_owner":
            return _api_error("Paper not found", 404)  # Don't leak existence
        elif message == "already_parsed":
            return _api_error("Paper already parsed successfully", 409)
        elif message == "invalid_pid":
            return _api_error("Invalid paper ID", 400)
        elif message == "db_error":
            return _api_error("Database error, please retry", 500)
        elif message == "enqueue_failed":
            return _api_error("Failed to enqueue task, please retry", 500)
        else:
            return _api_error(f"Parse failed: {message}", 400)

    except Exception as e:
        logger.error(f"Failed to trigger parse: {e}")
        return _api_error("Failed to trigger parsing. Please try again.", 500)


@bp.route("/uploaded_papers/extract_info", methods=["POST"])
def api_uploaded_papers_extract_info():
    """Trigger metadata extraction for an uploaded paper."""
    if g.user is None:
        return _api_error("Not logged in", 401)

    csrf_protect()

    data = request.get_json(silent=True)
    if not data:
        return _api_error("No JSON data provided", 400)

    pid = data.get("pid", "").strip()
    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from aslite.repositories import UploadedPaperRepository
        from backend.services.upload_service import trigger_extract_info

        # Get record to provide better error message
        record = UploadedPaperRepository.get(pid)
        if not record:
            return _api_error("Paper not found", 404)

        if record.get("owner") != g.user:
            # Do not leak existence of private upload PIDs to other authenticated users.
            return _api_error("Paper not found", 404)

        from backend.services.upload_service import _normalize_upload_parse_status

        parse_status, _parse_error = _normalize_upload_parse_status(pid, record)
        if parse_status != "ok":
            return _api_error(f"Paper not parsed yet (status: {parse_status})", 400)

        if record.get("meta_extracted_ok") is True:
            return _api_error("Metadata already extracted", 400)

        success, task_id = trigger_extract_info(pid, g.user)

        if not success:
            return _api_error("Failed to enqueue extraction task", 500)

        return _api_success(pid=pid, task_id=task_id)

    except Exception as e:
        logger.error(f"Failed to trigger extract info: {e}")
        return _api_error("Failed to trigger extraction. Please try again.", 500)


@bp.route("/uploaded_papers/pdf/<pid>", methods=["GET"])
def api_uploaded_papers_pdf(pid: str):
    """Download the original PDF for an uploaded paper."""
    if g.user is None:
        abort(401)

    if not pid or not validate_upload_pid(pid):
        abort(404)

    try:
        from aslite.repositories import UploadedPaperRepository

        record = UploadedPaperRepository.get(pid)
        if not record:
            abort(404)

        if record.get("owner") != g.user:
            abort(404)

        pdf_path = get_upload_pdf_path(pid, DATA_DIR)
        if not pdf_path.exists():
            abort(404)

        filename = record.get("original_filename", f"{pid}.pdf")

        # Flask<2.0 does not support download_name; fallback to attachment_filename.
        try:
            return send_file(
                pdf_path,
                mimetype="application/pdf",
                as_attachment=True,
                download_name=filename,
            )
        except TypeError:
            return send_file(
                pdf_path,
                mimetype="application/pdf",
                as_attachment=True,
                attachment_filename=filename,
            )

    except Exception as e:
        if hasattr(e, "code"):
            raise
        logger.error(f"Failed to serve PDF: {e}")
        abort(500)


@bp.route("/uploaded_papers/similar/<pid>", methods=["GET"])
def api_uploaded_papers_similar(pid: str):
    """Find similar arXiv papers for an uploaded paper.

    Query params:
        limit: Maximum number of results (default 20, max 100)

    Returns:
        List of similar papers with scores
    """
    if g.user is None:
        return _api_error("Not logged in", 401)

    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from aslite.repositories import UploadedPaperRepository

        record = UploadedPaperRepository.get(pid)
        if not record:
            return _api_error("Paper not found", 404)

        if record.get("owner") != g.user:
            return _api_error("Paper not found", 404)

        # Check if paper has been parsed
        from backend.services.upload_service import _normalize_upload_parse_status

        parse_status, _parse_error = _normalize_upload_parse_status(pid, record)
        if parse_status != "ok":
            return _api_error(f"Paper must be parsed first (status: {parse_status})", 400)

        # Similarity features rely on extracted metadata (title/authors/abstract).
        # Keep this aligned with frontend gating (readinglist.js disables Similar until meta_extracted_ok).
        #
        # Backward compatibility: older records/tests may not have `meta_extracted_ok`.
        if "meta_extracted_ok" in record and record.get("meta_extracted_ok") is not True:
            return _api_error("Extract metadata first", 400)

        # Get limit from query params
        try:
            limit = int(request.args.get("limit", 20))
        except (ValueError, TypeError):
            limit = 20
        limit = max(1, min(limit, 100))

        from backend.services.upload_similarity_service import find_similar_papers

        results = find_similar_papers(pid, limit=limit)

        return _api_success(papers=results, count=len(results))

    except Exception as e:
        logger.error(f"Failed to find similar papers for {pid}: {e}")
        return _api_error("Failed to find similar papers. Please try again.", 500)


@bp.route("/uploaded_papers/tldr/<pid>", methods=["GET"])
def api_uploaded_paper_tldr(pid):
    """Get TL;DR for an uploaded paper.

    Returns the TL;DR extracted from the summary if available.
    Used for real-time updates after summary generation completes.
    """
    if g.user is None:
        return _api_error("Not logged in", 401)

    if not pid or not validate_upload_pid(pid):
        return _api_error("Invalid paper ID", 400)

    try:
        from aslite.repositories import UploadedPaperRepository
        from backend.services.summary_service import (
            extract_tldr_from_summary,
            get_summary_status,
        )

        # Verify ownership
        data = UploadedPaperRepository.get(pid)
        if not data or data.get("owner") != g.user:
            return _api_error("Paper not found or access denied", 404)

        # Get summary status and TL;DR
        summary_status, _ = get_summary_status(pid)
        tldr = ""
        if summary_status == "ok":
            tldr = extract_tldr_from_summary(pid) or ""

        return _api_success(pid=pid, tldr=tldr, summary_status=summary_status)

    except Exception as e:
        logger.error(f"Failed to get TL;DR for {pid}: {e}")
        return _api_error("Failed to get TL;DR. Please try again.", 500)
