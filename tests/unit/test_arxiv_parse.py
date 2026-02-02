"""Unit tests for arXiv URL parsing helpers."""

from __future__ import annotations

import pytest


def test_parse_arxiv_url_with_version():
    from aslite.arxiv import parse_arxiv_url

    idv, rawid, version = parse_arxiv_url("http://arxiv.org/abs/1512.08756v2")
    assert idv == "1512.08756v2"
    assert rawid == "1512.08756"
    assert version == 2


def test_parse_arxiv_url_without_version_defaults_to_v1():
    from aslite.arxiv import parse_arxiv_url

    idv, rawid, version = parse_arxiv_url("http://arxiv.org/abs/1512.08756")
    assert idv == "1512.08756v1"
    assert rawid == "1512.08756"
    assert version == 1


def test_parse_arxiv_url_invalid_raises():
    from aslite.arxiv import parse_arxiv_url

    with pytest.raises(ValueError):
        parse_arxiv_url("")


def test_filter_latest_version_tolerates_missing_or_invalid_versions():
    from aslite.arxiv import filter_latest_version

    out = filter_latest_version(
        [
            "1512.08756v2",
            "1512.08756",
            "1512.08756v10",
            "1512.08756vX",
            "bad",
            "",
            None,
        ]
    )
    assert "1512.08756v10" in out
    assert "badv1" not in out
