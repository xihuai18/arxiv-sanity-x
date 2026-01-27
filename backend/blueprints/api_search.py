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
            keyword:
              type: string
              description: Search keyword (alias: q, search_query)
            q:
              type: string
              description: Alias of keyword
            search_query:
              type: string
              description: Alias of keyword
            time_delta:
              type: number
              description: Time window in days (alias: time_filter)
            time_filter:
              type: string
              enum: [day, week, month, year, all]
              description: Alias of time_delta (named window)
            limit:
              type: integer
              description: Maximum number of results to return
            skip_num:
              type: integer
              description: Number of results to skip
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
            tag_name:
              type: string
              description: Tag name to search (alias: tag)
            tag:
              type: string
              description: Alias of tag_name
            user:
              type: string
              description: Backward compatible user field (must match session user if provided)
            time_delta:
              type: number
              description: Time window in days (alias: time_filter)
            time_filter:
              type: string
              enum: [day, week, month, year, all]
            skip_num:
              type: integer
            limit:
              type: integer
            C:
              type: number
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
            tags:
              type: array
              items:
                type: string
              description: Tag list (comma-separated string also accepted)
            user:
              type: string
              description: Backward compatible user field (must match session user if provided)
            logic:
              type: string
              enum: [and, or]
            time_filter:
              type: string
              enum: [day, week, month, year, all]
            time_delta:
              type: number
            skip_num:
              type: integer
            limit:
              type: integer
            C:
              type: number
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
