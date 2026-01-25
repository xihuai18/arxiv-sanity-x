"""Web page routes."""

from __future__ import annotations

from flask import Blueprint

from .. import legacy

bp = Blueprint("web", __name__)


@bp.route("/", methods=["GET"])
def main():
    return legacy.main()


@bp.route("/inspect", methods=["GET"])
def inspect():
    return legacy.inspect()


@bp.route("/summary", methods=["GET"])
def summary():
    return legacy.summary()


@bp.route("/profile")
def profile():
    return legacy.profile()


@bp.route("/stats")
def stats():
    return legacy.stats()


@bp.route("/about")
def about():
    return legacy.about()


@bp.route("/readinglist", methods=["GET"])
def readinglist_page():
    return legacy.readinglist_page()
