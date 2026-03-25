"""Unit tests for tools.repair_paper_history."""

from __future__ import annotations

from aslite.repositories import (
    MetaRepository,
    PaperRepository,
    PaperTombstoneRepository,
)


def test_build_parser_has_expected_flags():
    from tools.repair_paper_history import build_parser

    parser = build_parser()
    args = parser.parse_args([])

    assert hasattr(args, "pid")
    assert hasattr(args, "scan_tombstones")
    assert hasattr(args, "scan_withdrawn")
    assert hasattr(args, "apply")
    assert hasattr(args, "output")


def test_run_dry_run_does_not_write(monkeypatch):
    import tools.repair_paper_history as tool

    called = []
    monkeypatch.setattr(
        tool,
        "get_entries_by_ids",
        lambda _ids: [
            {
                "_id": "2301.00001",
                "_idv": "2301.00001v1",
                "_version": 1,
                "_time": 1.0,
                "title": "Paper",
            }
        ],
    )
    monkeypatch.setattr(
        tool.PaperCorpusRepository,
        "apply_daemon_batch",
        lambda **kwargs: called.append(kwargs),
    )

    rc = tool.main(["--pid", "2301.00001"])

    assert rc == 0
    assert called == []


def test_apply_restores_visible_paper_and_clears_tombstone(monkeypatch):
    import tools.repair_paper_history as tool

    pid = "2602.21910"
    PaperTombstoneRepository.save(
        pid,
        {
            "pid": pid,
            "reason": "withdrawn_only",
            "latest_idv": f"{pid}v1",
            "deleted_at": 1.0,
        },
    )

    monkeypatch.setattr(
        tool,
        "get_entries_by_ids",
        lambda _ids: [
            {
                "_id": pid,
                "_idv": f"{pid}v2",
                "_version": 2,
                "_time": 2.0,
                "title": "Recovered Paper",
            }
        ],
    )

    rc = tool.main(["--pid", pid, "--apply"])

    assert rc == 0
    paper = PaperRepository.get_by_id(pid)
    meta = MetaRepository.get_by_id(pid)
    tombstone = PaperTombstoneRepository.get_by_id(pid)
    assert isinstance(paper, dict)
    assert paper.get("_effective_idv") == f"{pid}v2"
    assert isinstance(meta, dict)
    assert meta.get("_effective_idv") == f"{pid}v2"
    assert tombstone is None
    PaperRepository.delete(pid)
    MetaRepository.delete(pid)


def test_apply_tombstones_withdrawn_only_paper(monkeypatch):
    import tools.repair_paper_history as tool

    pid = "2506.21583"
    PaperRepository.save(
        pid,
        {
            "_id": pid,
            "_idv": f"{pid}v1",
            "_version": 1,
            "_time": 1.0,
            "title": "Old Visible Paper",
            "_effective_idv": f"{pid}v1",
            "_effective_version": 1,
        },
    )
    MetaRepository.save_many({pid: {"_time": 1.0, "_id": pid, "_idv": f"{pid}v1", "_version": 1}})

    latest_withdrawn = {
        "_id": pid,
        "_idv": f"{pid}v2",
        "_version": 2,
        "_time": 2.0,
        "title": "Paper",
        "arxiv_comment": "This paper has been withdrawn by the authors.",
    }

    monkeypatch.setattr(tool, "get_entries_by_ids", lambda _ids: [latest_withdrawn])
    monkeypatch.setattr(tool, "resolve_latest_nonwithdrawn_version", lambda _entry: None)

    rc = tool.main(["--pid", pid, "--apply"])

    assert rc == 0
    assert PaperRepository.get_by_id(pid) is None
    assert MetaRepository.get_by_id(pid) is None
    tombstone = PaperTombstoneRepository.get_by_id(pid)
    assert isinstance(tombstone, dict)
    assert tombstone.get("reason") == "withdrawn_only"
    PaperTombstoneRepository.delete(pid)


def test_collect_candidates_defaults_to_tombstones():
    import tools.repair_paper_history as tool

    pid = "2402.03627"
    PaperTombstoneRepository.save(pid, {"pid": pid, "reason": "withdrawn_only"})

    parser = tool.build_parser()
    args = parser.parse_args([])
    candidates = tool._collect_candidate_pids(args)

    assert (pid, "tombstone") in candidates
    PaperTombstoneRepository.delete(pid)


def test_plan_pid_refreshes_stale_tombstone(monkeypatch):
    import tools.repair_paper_history as tool

    pid = "2501.99999"
    PaperTombstoneRepository.save(
        pid,
        {
            "pid": pid,
            "reason": "withdrawn_only",
            "latest_idv": f"{pid}v1",
            "latest_version": 1,
            "latest_comment": "old comment",
            "deleted_at": 1.0,
        },
    )

    latest_withdrawn = {
        "_id": pid,
        "_idv": f"{pid}v2",
        "_version": 2,
        "_time": 2.0,
        "title": "Paper",
        "arxiv_comment": "This paper has been withdrawn by the authors.",
    }
    monkeypatch.setattr(tool, "get_entries_by_ids", lambda _ids: [latest_withdrawn])
    monkeypatch.setattr(tool, "resolve_latest_nonwithdrawn_version", lambda _entry: None)

    plan = tool.plan_pid(pid, "manual")

    assert plan["action"] == "tombstone"
    PaperTombstoneRepository.delete(pid)


def test_apply_failure_returns_nonzero(monkeypatch):
    import tools.repair_paper_history as tool

    pid = "2603.00001"
    monkeypatch.setattr(
        tool,
        "get_entries_by_ids",
        lambda _ids: [
            {
                "_id": pid,
                "_idv": f"{pid}v1",
                "_version": 1,
                "_time": 1.0,
                "title": "Paper",
            }
        ],
    )
    monkeypatch.setattr(
        tool.PaperCorpusRepository,
        "apply_daemon_batch",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    rc = tool.main(["--pid", pid, "--apply"])

    assert rc == 1
