"""User and auth API routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("user", __name__)


@bp.route("/api/user_state", methods=["GET"])
def api_user_state():
    """Get current user state
    ---
    tags:
      - User
    responses:
      200:
        description: User state information
        schema:
          type: object
          properties:
            user:
              type: string
              description: Username or null if not logged in
            tags:
              type: array
              items:
                type: object
            keys:
              type: array
              items:
                type: object
    """
    return legacy.api_user_state()


@bp.route("/login", methods=["POST"])
def login():
    """User login
    ---
    tags:
      - User
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - username
          properties:
            username:
              type: string
              description: Username to login as
    responses:
      200:
        description: Login successful
      400:
        description: Invalid username
    """
    return legacy.login()


@bp.route("/logout", methods=["GET", "POST"])
def logout():
    """User logout
    ---
    tags:
      - User
    responses:
      302:
        description: Redirect to home page
    """
    return legacy.logout()


@bp.route("/register_email", methods=["POST"])
def register_email():
    """Register email for notifications
    ---
    tags:
      - User
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - email
          properties:
            email:
              type: string
              format: email
    responses:
      200:
        description: Email registered
      400:
        description: Invalid email
      401:
        description: Not logged in
    """
    return legacy.register_email()
