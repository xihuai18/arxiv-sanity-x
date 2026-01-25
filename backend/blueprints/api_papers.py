"""Paper-related API routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("papers", __name__)


@bp.route("/api/paper_image/<pid>/<filename>", methods=["GET"])
def api_paper_image(pid, filename):
    """Get paper image from HTML/MD cache
    ---
    tags:
      - Papers
    parameters:
      - in: path
        name: pid
        type: string
        required: true
        description: Paper ID
      - in: path
        name: filename
        type: string
        required: true
        description: Image filename
    responses:
      200:
        description: Image file
      404:
        description: Image not found
    """
    return legacy.api_paper_image(pid, filename)


@bp.route("/api/mineru_image/<pid>/<filename>", methods=["GET"])
def api_mineru_image(pid, filename):
    """Get paper image from MinerU cache
    ---
    tags:
      - Papers
    parameters:
      - in: path
        name: pid
        type: string
        required: true
        description: Paper ID
      - in: path
        name: filename
        type: string
        required: true
        description: Image filename
    responses:
      200:
        description: Image file
      404:
        description: Image not found
    """
    return legacy.api_mineru_image(pid, filename)


@bp.route("/api/llm_models", methods=["GET"])
def api_llm_models():
    """Get available LLM models
    ---
    tags:
      - System
    responses:
      200:
        description: List of available LLM models
        schema:
          type: object
          properties:
            success:
              type: boolean
            models:
              type: array
              items:
                type: string
            default:
              type: string
    """
    return legacy.api_llm_models()
