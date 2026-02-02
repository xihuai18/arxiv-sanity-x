"""Server-sent events endpoints."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("sse", __name__)


@bp.route("/api/user_stream", methods=["GET"])
def api_user_stream():
    """Server-sent events stream for user updates
    ---
    tags:
      - SSE
    description: |
      Real-time event stream for user-specific updates.
      Events include: tag updates, summary status changes, reading list updates.

      Note: This endpoint returns a text/event-stream response.
    responses:
      200:
        description: SSE stream
        content:
          text/event-stream:
            schema:
              type: string
      401:
        description: Not logged in
    """
    return legacy.api_user_stream()


@bp.route("/api/sse_stats", methods=["GET"])
def api_sse_stats():
    """Get process-local SSE stats
    ---
    tags:
      - SSE
    responses:
      200:
        description: SSE stats
    """
    return legacy.api_sse_stats()
