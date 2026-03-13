from __future__ import annotations

import html


def _body_text(response) -> str:
    return html.unescape(response.get_data(as_text=True))


def test_homepage_query_in_tags_mode_switches_back_to_search(client):
    response = client.get("/?rank=tags&q=graph+learning")

    assert response.status_code == 200
    body = _body_text(response)
    assert "Search query only works with rank 'search' or 'time'; switched to 'search'." in body


def test_homepage_search_rank_requires_query(client):
    response = client.get("/?rank=search")

    assert response.status_code == 200
    body = _body_text(response)
    assert "Rank 'search' requires a query; using 'time' instead." in body


def test_homepage_removes_inline_click_handlers(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'onclick="window.location.href' not in body
    assert 'onclick="move_page(' not in body
    assert 'oninput="document.getElementById(' not in body
    assert 'onclick="(function(){var w=document.getElementById(' not in body
    assert 'id="search-form"' in body
    assert 'data-page-offset="-1"' in body
    assert 'id="clear-filters-link"' in body
