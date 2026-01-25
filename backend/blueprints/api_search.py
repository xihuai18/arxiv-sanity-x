"""Search and recommendation API routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("search", __name__)


@bp.route("/api/keyword_search", methods=["POST"])
def api_keyword_search():
    """Search papers by keyword
    ---
    tags:
      - Search
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            q:
              type: string
              description: Search query
            time_filter:
              type: string
              enum: [day, week, month, year, all]
              description: Time filter
            skip_num:
              type: integer
              description: Number of results to skip
            search_query:
              type: string
              description: Additional search query
    responses:
      200:
        description: Search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            papers:
              type: array
            has_more:
              type: boolean
    """
    return legacy.api_keyword_search()


@bp.route("/api/tag_search", methods=["POST"])
def api_tag_search():
    """Search papers by tag
    ---
    tags:
      - Search
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            tag:
              type: string
              description: Tag name to search
            time_filter:
              type: string
              enum: [day, week, month, year, all]
            skip_num:
              type: integer
    responses:
      200:
        description: Search results
      401:
        description: Not logged in
    """
    return legacy.api_tag_search()


@bp.route("/api/tags_search", methods=["POST"])
def api_tags_search():
    """Search papers by multiple tags (SVM ranking)
    ---
    tags:
      - Search
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            time_filter:
              type: string
              enum: [day, week, month, year, all]
            skip_num:
              type: integer
    responses:
      200:
        description: Search results ranked by SVM
      401:
        description: Not logged in
    """
    return legacy.api_tags_search()


@bp.route("/cache_status")
def cache_status():
    """Get cache status
    ---
    tags:
      - System
    responses:
      200:
        description: Cache status information
        schema:
          type: object
          properties:
            features_file_mtime:
              type: number
            papers_db_file_mtime:
              type: number
    """
    return legacy.cache_status()
