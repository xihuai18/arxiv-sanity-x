"""Reading list API routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("readinglist", __name__)


@bp.route("/api/readinglist/add", methods=["POST"])
def api_readinglist_add():
    """Add paper to reading list
    ---
    tags:
      - Reading List
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
              description: Paper ID
    responses:
      200:
        description: Paper added
        schema:
          type: object
          properties:
            success:
              type: boolean
            top_tags:
              type: array
              items:
                type: string
            task_id:
              type: string
      401:
        description: Not logged in
    """
    return legacy.api_readinglist_add()


@bp.route("/api/readinglist/remove", methods=["POST"])
def api_readinglist_remove():
    """Remove paper from reading list
    ---
    tags:
      - Reading List
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
        description: Paper removed
      401:
        description: Not logged in
      404:
        description: Paper not in reading list
    """
    return legacy.api_readinglist_remove()


@bp.route("/api/readinglist/list", methods=["GET"])
def api_readinglist_list():
    """Get reading list
    ---
    tags:
      - Reading List
    responses:
      200:
        description: Reading list items
        schema:
          type: object
          properties:
            success:
              type: boolean
            items:
              type: array
      401:
        description: Not logged in
    """
    return legacy.api_readinglist_list()
