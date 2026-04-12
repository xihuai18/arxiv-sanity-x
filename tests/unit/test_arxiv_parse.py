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


def test_get_entries_by_ids_sets_max_results(monkeypatch):
    import aslite.arxiv as arxiv

    captured = {}

    monkeypatch.setattr(arxiv, "parse_response", lambda _response: [])

    def _fake_open(url):
        captured["url"] = url
        return b""

    monkeypatch.setattr(arxiv, "_open_arxiv_api_url", _fake_open)

    arxiv.get_entries_by_ids([f"2301.{idx:05d}" for idx in range(11)])

    assert "max_results=11" in captured["url"]


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


def test_is_withdrawn_entry_detects_withdrawn_comment():
    from aslite.arxiv import is_withdrawn_entry

    assert (
        is_withdrawn_entry(
            {
                "title": "Example",
                "summary": "Original abstract.",
                "arxiv_comment": "This paper has been withdrawn by the authors.",
            }
        )
        is True
    )


@pytest.mark.parametrize(
    "comment",
    [
        "This article is withdrawn due to a technical error identified after submission.",
        "The paper is withdrawn pending completion of the approval process.",
        "We are withdrawing this preprint because it contains initial experimental results only.",
        "We have decided to withdraw this manuscript because it needs substantial revision.",
    ],
)
def test_is_withdrawn_entry_detects_additional_withdrawn_phrasings(comment):
    from aslite.arxiv import is_withdrawn_entry

    assert is_withdrawn_entry({"title": "Example", "arxiv_comment": comment}) is True


def test_is_withdrawn_abs_page_detects_withdrawn_banner():
    from aslite.arxiv import is_withdrawn_abs_page

    html = """
    <html>
      <body>
        <div class="announcement">This paper has been withdrawn by the author(s).</div>
      </body>
    </html>
    """

    assert is_withdrawn_abs_page(html) is True


def test_resolve_latest_nonwithdrawn_version_uses_previous_visible_version():
    from aslite.arxiv import resolve_latest_nonwithdrawn_version

    latest_entry = {
        "_id": "0812.0848",
        "_idv": "0812.0848v2",
        "_version": 2,
        "title": "Example",
        "summary": "Original abstract.",
        "arxiv_comment": "This paper has been withdrawn by the author.",
    }
    previous_entry = {
        "_id": "0812.0848",
        "_idv": "0812.0848v1",
        "_version": 1,
        "title": "Example",
        "summary": "Original abstract.",
    }

    resolved = resolve_latest_nonwithdrawn_version(
        latest_entry,
        entries_getter=lambda ids: [previous_entry] if ids == ["0812.0848v1"] else [],
    )

    assert resolved == previous_entry


def test_resolve_latest_nonwithdrawn_version_returns_none_when_all_versions_withdrawn():
    from aslite.arxiv import resolve_latest_nonwithdrawn_version

    latest_entry = {
        "_id": "0812.0848",
        "_idv": "0812.0848v3",
        "_version": 3,
        "title": "Example",
        "summary": "Original abstract.",
        "arxiv_comment": "This paper has been withdrawn by the author.",
    }

    def _all_withdrawn(ids):
        return [
            {
                "_id": "0812.0848",
                "_idv": pid,
                "_version": int(pid.rsplit("v", 1)[1]),
                "title": "Example",
                "summary": "Original abstract.",
                "arxiv_comment": "This paper has been withdrawn by the author.",
            }
            for pid in ids
        ]

    assert resolve_latest_nonwithdrawn_version(latest_entry, entries_getter=_all_withdrawn) is None


def test_resolve_latest_nonwithdrawn_version_can_force_latest_as_withdrawn():
    from aslite.arxiv import resolve_latest_nonwithdrawn_version

    latest_entry = {
        "_id": "2510.06170",
        "_idv": "2510.06170v3",
        "_version": 3,
        "title": "Example",
        "summary": "Original abstract.",
        "arxiv_comment": "The new version is available at arXiv:2512.15548",
    }
    previous_entry = {
        "_id": "2510.06170",
        "_idv": "2510.06170v2",
        "_version": 2,
        "title": "Example",
        "summary": "Original abstract.",
    }

    resolved = resolve_latest_nonwithdrawn_version(
        latest_entry,
        entries_getter=lambda ids: [previous_entry] if "2510.06170v2" in ids else [],
        treat_latest_as_withdrawn=True,
    )

    assert resolved == previous_entry
