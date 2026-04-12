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
    assert hasattr(args, "scan_public")
    assert hasattr(args, "scan_public_newest")
    assert hasattr(args, "scan_recent_public")
    assert hasattr(args, "withdrawn_only")
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


def test_collect_candidates_can_scan_current_public_papers():
    import tools.repair_paper_history as tool

    pid = "2402.03628"
    PaperRepository.save(pid, {"_id": pid, "_idv": f"{pid}v1", "_version": 1, "title": "Paper"})

    parser = tool.build_parser()
    args = parser.parse_args(["--scan-public"])
    candidates = tool._collect_candidate_pids(args)

    assert (pid, "public") in candidates
    PaperRepository.delete(pid)


def test_collect_candidates_scan_public_respects_limit_early(monkeypatch):
    import tools.repair_paper_history as tool

    seen = []

    def _iter_all_papers():
        for idx in range(10):
            pid = f"2402.{idx:05d}"
            seen.append(pid)
            yield pid, {"_id": pid, "title": f"Paper {idx}"}

    monkeypatch.setattr(tool.PaperRepository, "iter_all_papers", _iter_all_papers)

    parser = tool.build_parser()
    args = parser.parse_args(["--scan-public", "--limit", "2"])
    candidates = tool._collect_candidate_pids(args)

    assert candidates == [("2402.00000", "public"), ("2402.00001", "public")]
    assert seen == ["2402.00000", "2402.00001"]


def test_collect_candidates_scan_public_newest_preserves_order(monkeypatch):
    import tools.repair_paper_history as tool

    monkeypatch.setattr(
        tool.MetaRepository,
        "iter_latest_all",
        lambda batch_size=500: iter(
            [
                ("2402.00003", {"_time": 3.0}),
                ("2402.00002", {"_time": 2.0}),
                ("2402.00001", {"_time": 1.0}),
            ]
        ),
    )

    parser = tool.build_parser()
    args = parser.parse_args(["--scan-public-newest", "--limit", "2"])
    candidates = tool._collect_candidate_pids(args)

    assert candidates == [
        ("2402.00003", "public_newest"),
        ("2402.00002", "public_newest"),
    ]


def test_scan_public_apply_tombstones_current_db_withdrawn_paper(monkeypatch):
    import tools.repair_paper_history as tool

    pid = "2604.00001"
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

    rc = tool.main(["--scan-public", "--apply"])

    assert rc == 0
    assert PaperRepository.get_by_id(pid) is None
    assert MetaRepository.get_by_id(pid) is None
    tombstone = PaperTombstoneRepository.get_by_id(pid)
    assert isinstance(tombstone, dict)
    assert tombstone.get("reason") == "withdrawn_only"
    PaperTombstoneRepository.delete(pid)


def test_run_aborts_after_consecutive_api_failures(monkeypatch):
    import tools.repair_paper_history as tool

    pids = [f"2604.{idx:05d}" for idx in range(4)]
    for pid in pids:
        PaperRepository.save(pid, {"_id": pid, "_idv": f"{pid}v1", "_version": 1, "title": "Paper"})

    calls = []
    monkeypatch.setattr(
        tool,
        "get_entries_by_ids",
        lambda _ids: (
            calls.append(list(_ids)),
            (_ for _ in ()).throw(RuntimeError("429")),
        )[1],
    )

    apply_calls = []
    monkeypatch.setattr(
        tool.PaperCorpusRepository,
        "apply_daemon_batch",
        lambda **kwargs: apply_calls.append(kwargs),
    )

    rc = tool.main(
        [
            "--scan-public",
            "--apply",
            "--process-batch-size",
            "2",
            "--api-batch-size",
            "1",
            "--api-delay",
            "0",
            "--api-stop-after-failures",
            "2",
        ]
    )

    assert rc == 1
    assert len(calls) == 2
    assert apply_calls == []
    for pid in pids:
        PaperRepository.delete(pid)


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
