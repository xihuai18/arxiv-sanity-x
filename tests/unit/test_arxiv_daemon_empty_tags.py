"""Unit tests for tools.arxiv_daemon edge cases."""

from __future__ import annotations

import argparse


def test_run_with_empty_all_tags_does_not_query(monkeypatch):
    import tools.arxiv_daemon as d

    # If called, this would indicate we tried to query with empty tags.
    monkeypatch.setattr(d, "get_response", lambda *_a, **_k: (_ for _ in ()).throw(BaseException("should not call")))

    args = argparse.Namespace(
        num=10,
        num_total=-1,
        start=0,
        break_after=1,
        init=False,
        max_r=10,
    )

    rc = d.run(args, all_tags=[], empty_response_fallback=3)
    assert rc == 1
