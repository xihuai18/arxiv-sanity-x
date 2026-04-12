"""Unit tests for withdrawn handling in tools.arxiv_daemon."""

from __future__ import annotations

import argparse


def test_run_tombstones_latest_withdrawn_paper(monkeypatch):
    import tools.arxiv_daemon as tool

    pid = "2604.12345"
    latest = {
        "_id": pid,
        "_idv": f"{pid}v3",
        "_version": 3,
        "_time": 300.0,
        "_time_str": "Apr 12 2026",
        "title": "Withdrawn Example",
        "arxiv_comment": "This paper has been withdrawn by the authors.",
    }

    monkeypatch.setattr(tool, "get_response", lambda **_kwargs: b"")
    monkeypatch.setattr(tool, "parse_response", lambda _resp: [latest])
    monkeypatch.setattr(
        tool.PaperRepository,
        "get_by_ids",
        lambda _pids: {pid: {"_id": pid, "title": "Old"}},
    )
    monkeypatch.setattr(tool.PaperRepository, "count", lambda: 1)

    captured = {}

    def _capture_apply(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        tool.PaperCorpusRepository, "apply_daemon_batch", _capture_apply
    )

    args = argparse.Namespace(
        num=1,
        num_total=-1,
        start=0,
        break_after=1,
        init=False,
        max_r=10,
    )

    rc = tool.run(args, all_tags=["cs.AI"], empty_response_fallback=3)

    assert rc == 0
    assert captured.get("papers") == {}
    assert captured.get("metas") == {}
    tombstones = captured.get("tombstones") or {}
    assert pid in tombstones
    assert tombstones[pid].get("reason") == "withdrawn_only"
