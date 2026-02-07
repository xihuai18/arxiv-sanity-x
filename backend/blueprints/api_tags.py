"""Tag and keyword API routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("tags", __name__)


@bp.route("/api/tag_feedback", methods=["POST"])
def api_tag_feedback():
    """Submit tag feedback for a paper
    ---
    tags:
      - Tags
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pid
            - tag
            - label
          properties:
            pid:
              type: string
              description: Paper ID
            tag:
              type: string
              description: Tag name
            label:
              type: integer
              enum: [-1, 0, 1]
              description: "1=positive, -1=negative, 0=remove"
    responses:
      200:
        description: Feedback recorded
      401:
        description: Not logged in
    """
    return legacy.api_tag_feedback()


@bp.route("/api/tag_members", methods=["GET"])
def api_tag_members():
    """List papers under a tag
    ---
    tags:
      - Tags
    parameters:
      - in: query
        name: tag
        type: string
        required: true
      - in: query
        name: label
        type: string
        enum: [all, pos, neg]
        default: all
      - in: query
        name: search
        type: string
      - in: query
        name: page_number
        type: integer
        default: 1
      - in: query
        name: page_size
        type: integer
        default: 20
    responses:
      200:
        description: List of papers
      401:
        description: Not logged in
    """
    return legacy.api_tag_members()


@bp.route("/api/paper_titles", methods=["POST"])
def api_paper_titles():
    """Resolve paper titles for PIDs
    ---
    tags:
      - Tags
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pids
          properties:
            pids:
              type: array
              items:
                type: string
    responses:
      200:
        description: Paper titles
        schema:
          type: object
          properties:
            success:
              type: boolean
            items:
              type: array
              items:
                type: object
                properties:
                  pid:
                    type: string
                  title:
                    type: string
                  exists:
                    type: boolean
    """
    return legacy.api_paper_titles()


@bp.route("/add_tag/<tag>", methods=["POST"])
def add_tag(tag):
    return legacy.add_tag(tag)


@bp.route("/add/<pid>/<tag>", methods=["POST"])
def add(pid, tag):
    return legacy.add(pid, tag)


@bp.route("/sub/<pid>/<tag>", methods=["POST"])
def sub(pid, tag):
    return legacy.sub(pid, tag)


@bp.route("/del/<tag>", methods=["POST"])
def delete_tag(tag):
    return legacy.delete_tag(tag)


@bp.route("/rename/<otag>/<ntag>", methods=["POST"])
def rename_tag(otag, ntag):
    return legacy.rename_tag(otag, ntag)


@bp.route("/add_ctag/<ctag>", methods=["POST"])
def add_ctag(ctag):
    return legacy.add_ctag(ctag)


@bp.route("/del_ctag/<ctag>", methods=["POST"])
def delete_ctag(ctag):
    return legacy.delete_ctag(ctag)


@bp.route("/rename_ctag/<otag>/<ntag>", methods=["POST"])
def rename_ctag(otag, ntag):
    return legacy.rename_ctag(otag, ntag)


@bp.route("/add_key/<keyword>", methods=["POST"])
def add_key(keyword):
    return legacy.add_key(keyword)


@bp.route("/del_key/<keyword>", methods=["POST"])
def delete_key(keyword):
    return legacy.delete_key(keyword)


@bp.route("/rename_key/<okey>/<nkey>", methods=["POST"])
def rename_key(okey, nkey):
    return legacy.rename_key(okey, nkey)
