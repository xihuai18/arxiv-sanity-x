"""Summary and task API routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("summary", __name__)


@bp.route("/api/get_paper_summary", methods=["POST"])
def api_get_paper_summary():
    """Get paper summary
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pid
          properties:
            pid:
              type: string
              description: Paper ID (e.g., "2301.00001")
            model:
              type: string
              description: LLM model name (optional)
            force:
              type: boolean
              description: Force regenerate summary
            cache_only:
              type: boolean
              description: Only return cached summary
    responses:
      200:
        description: Summary content
        schema:
          type: object
          properties:
            success:
              type: boolean
            summary_content:
              type: string
            summary_meta:
              type: object
      403:
        description: CSRF token missing/invalid
      404:
        description: Paper not found or cache miss
    """
    return legacy.api_get_paper_summary()


@bp.route("/api/get_paper_tldr", methods=["POST"])
def api_get_paper_tldr():
    """Get paper TL;DR (best-effort, from cached summary).
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pid
          properties:
            pid:
              type: string
              description: Paper ID (e.g., "2301.00001")
    responses:
      200:
        description: TL;DR content (empty if unavailable)
        schema:
          type: object
          properties:
            success:
              type: boolean
            pid:
              type: string
            tldr:
              type: string
            summary_status:
              type: string
      403:
        description: CSRF token missing/invalid
      404:
        description: Paper not found
    """
    return legacy.api_get_paper_tldr()


@bp.route("/api/trigger_paper_summary", methods=["POST"])
def api_trigger_paper_summary():
    """Trigger summary generation
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pid
          properties:
            pid:
              type: string
            model:
              type: string
            priority:
              type: integer
            force:
              type: boolean
            force_regenerate:
              type: boolean
    responses:
      200:
        description: Task queued
        schema:
          type: object
          properties:
            success:
              type: boolean
            status:
              type: string
            task_id:
              type: string
    """
    return legacy.api_trigger_paper_summary()


@bp.route("/api/trigger_paper_summary_bulk", methods=["POST"])
def api_trigger_paper_summary_bulk():
    """Trigger summary generation in batch
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - items
          properties:
            items:
              type: array
              items:
                type: object
                required: [pid]
                properties:
                  pid:
                    type: string
                  model:
                    type: string
                  priority:
                    type: integer
                  force:
                    type: boolean
                  force_regenerate:
                    type: boolean
    responses:
      200:
        description: Tasks queued
    """
    return legacy.api_trigger_paper_summary_bulk()


@bp.route("/api/task_status/<task_id>", methods=["GET"])
def api_task_status(task_id):
    """Get task status
    ---
    tags:
      - Summary
    parameters:
      - in: path
        name: task_id
        type: string
        required: true
    responses:
      200:
        description: Task status
      404:
        description: Task not found
    """
    return legacy.api_task_status(task_id)


@bp.route("/api/queue_stats", methods=["GET"])
def api_queue_stats():
    """Get queue statistics
    ---
    tags:
      - Summary
    responses:
      200:
        description: Queue stats
        schema:
          type: object
          properties:
            success:
              type: boolean
            queued:
              type: integer
            running:
              type: integer
    """
    return legacy.api_queue_stats()


@bp.route("/api/summary_status", methods=["POST"])
def api_summary_status():
    """Get summary status for papers
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            pids:
              type: array
              items:
                type: string
            model:
              type: string
    responses:
      200:
        description: Status for each paper
    """
    return legacy.api_summary_status()


@bp.route("/api/clear_model_summary", methods=["POST"])
def api_clear_model_summary():
    """Clear summary for specific model
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pid
            - model
          properties:
            pid:
              type: string
            model:
              type: string
    responses:
      200:
        description: Summary cleared
    """
    return legacy.api_clear_model_summary()


@bp.route("/api/clear_paper_cache", methods=["POST"])
def api_clear_paper_cache():
    """Clear all caches for a paper
    ---
    tags:
      - Summary
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pid
          properties:
            pid:
              type: string
    responses:
      200:
        description: Cache cleared
    """
    return legacy.api_clear_paper_cache()


@bp.route("/api/check_paper_summaries", methods=["GET"])
def api_check_paper_summaries():
    """Check paper summaries status
    ---
    tags:
      - Summary
    responses:
      200:
        description: Summary check results
    """
    return legacy.api_check_paper_summaries()
