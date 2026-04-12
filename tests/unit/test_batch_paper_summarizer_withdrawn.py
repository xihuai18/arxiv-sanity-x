from __future__ import annotations

from pathlib import Path

from aslite.repositories import (
    MetaRepository,
    PaperRepository,
    PaperTombstoneRepository,
)
from tools.batch_paper_summarizer import BatchPaperSummarizer, BatchProcessor


def test_refresh_public_visibility_from_arxiv_tombstones_withdrawn_only(monkeypatch):
    import tools.batch_paper_summarizer as tool

    pid = "2603.19298"
    PaperRepository.save(
        pid,
        {
            "_id": pid,
            "_idv": f"{pid}v2",
            "_version": 2,
            "title": "Withdrawn Example",
        },
    )
    MetaRepository.save_many(
        {
            pid: {
                "_id": pid,
                "_idv": f"{pid}v2",
                "_version": 2,
                "_time": 123.0,
            }
        }
    )

    latest = {
        "_id": pid,
        "_idv": f"{pid}v2",
        "_version": 2,
        "_time": 456.0,
        "title": "Withdrawn Example",
        "arxiv_comment": "This article is withdrawn due to a technical error identified after submission.",
    }

    monkeypatch.setattr(tool, "get_entries_by_ids", lambda _ids: [latest])
    PaperTombstoneRepository.save("seed", {"pid": "seed", "reason": "seed"})
    PaperTombstoneRepository.delete("seed")

    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    result = processor._refresh_public_visibility_from_arxiv(pid)

    assert result == {"action": "tombstone", "pid": pid, "latest_idv": f"{pid}v2"}
    assert PaperRepository.get_by_id(pid) is None
    assert MetaRepository.get_by_id(pid) is None
    tombstone = PaperTombstoneRepository.get_by_id(pid)
    assert isinstance(tombstone, dict)
    assert tombstone.get("reason") == "withdrawn_only"


def test_refresh_public_visibility_from_arxiv_tombstones_even_with_previous_visible_version(
    monkeypatch,
):
    import tools.batch_paper_summarizer as tool

    pid = "2510.19139"
    PaperRepository.save(
        pid,
        {
            "_id": pid,
            "_idv": f"{pid}v2",
            "_version": 2,
            "_effective_idv": f"{pid}v2",
            "_effective_version": 2,
            "title": "Visible Previous Version",
        },
    )
    MetaRepository.save_many(
        {
            pid: {
                "_id": pid,
                "_idv": f"{pid}v2",
                "_version": 2,
                "_time": 200.0,
            }
        }
    )
    latest = {
        "_id": pid,
        "_idv": f"{pid}v3",
        "_version": 3,
        "_time": 300.0,
        "title": "Withdrawn Latest Version",
        "arxiv_comment": "We have decided to withdraw this manuscript because it needs substantial revision.",
    }

    monkeypatch.setattr(tool, "get_entries_by_ids", lambda _ids: [latest])
    PaperTombstoneRepository.save("seed", {"pid": "seed", "reason": "seed"})
    PaperTombstoneRepository.delete("seed")

    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    result = processor._refresh_public_visibility_from_arxiv(pid)

    assert result == {"action": "tombstone", "pid": pid, "latest_idv": f"{pid}v3"}
    assert MetaRepository.get_by_id(pid) is None
    assert PaperRepository.get_by_id(pid) is None
    tombstone = PaperTombstoneRepository.get_by_id(pid)
    assert isinstance(tombstone, dict)
    assert tombstone.get("reason") == "withdrawn_only"


def test_process_single_paper_marks_withdrawn_after_abs_probe(monkeypatch, tmp_path):
    import tools.batch_paper_summarizer as tool

    pid = "2510.06170"
    paper_info = {
        "_id": pid,
        "_idv": f"{pid}v3",
        "_version": 3,
        "title": "Withdrawn Latest Version",
        "authors": [],
        "link": f"https://arxiv.org/abs/{pid}v3",
    }

    class _FakeSummarizer(BatchPaperSummarizer):
        def __init__(self, processor=None):
            self.processor = processor
            self.calls = 0

        def generate_summary(self, _pid, source=None, model=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "content": "# Error\n\nUnable to download paper PDF",
                    "meta": {},
                }
            return {
                "content": "# Title\n\n## TL;DR\n\nok",
                "meta": {"source": source or "mineru"},
            }

    cache_root = Path(tmp_path) / "summary"
    cache_root.mkdir(parents=True, exist_ok=True)

    def _fake_summary_cache_paths(cache_pid, model):
        paper_dir = cache_root / cache_pid
        return (
            paper_dir / f"{model}.md",
            paper_dir / f"{model}.meta.json",
            paper_dir / f"{model}.lock",
            cache_root / f"{cache_pid}.md",
            cache_root / f"{cache_pid}.meta.json",
            cache_root / f"{cache_pid}.lock",
        )

    monkeypatch.setattr(tool, "BatchPaperSummarizer", _FakeSummarizer)
    monkeypatch.setattr(tool, "summary_cache_paths", _fake_summary_cache_paths)
    monkeypatch.setattr(tool, "acquire_summary_lock", lambda _path, timeout_s=300: object())
    monkeypatch.setattr(tool, "release_summary_lock", lambda _fd, _path: None)
    monkeypatch.setattr(tool, "check_withdrawn_via_abs_page", lambda pid_arg, abs_url=None: True)
    monkeypatch.setattr(
        BatchProcessor,
        "_refresh_public_visibility_from_arxiv",
        lambda self, raw_pid, force_latest_withdrawn=False: {
            "action": "tombstone",
            "pid": raw_pid,
            "latest_idv": f"{raw_pid}v3",
        },
    )

    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    monkeypatch.setattr(processor, "is_summary_cached", lambda cache_pid, summary_source: False)
    monkeypatch.setattr(
        processor,
        "cache_summary",
        lambda cache_pid, summary_content, source=None, summary_meta=None: (True, None),
    )

    result_pid, success, message = processor.process_single_paper(pid, paper_info, skip_cached=True)

    assert (result_pid, success, message) == (pid, True, "Withdrawn")


def test_process_single_paper_repairs_withdrawn_even_when_cached(monkeypatch):
    pid = "2603.19298"
    paper_info = {
        "_id": pid,
        "_idv": f"{pid}v2",
        "_version": 2,
        "title": "Withdrawn Example",
        "authors": [],
        "arxiv_comment": "This article is withdrawn due to a technical error identified after submission.",
    }

    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    monkeypatch.setattr(processor, "is_summary_cached", lambda cache_pid, summary_source: True)
    monkeypatch.setattr(
        processor,
        "_refresh_public_visibility_from_arxiv",
        lambda raw_pid, force_latest_withdrawn=False: {
            "action": "tombstone",
            "pid": raw_pid,
            "latest_idv": f"{raw_pid}v2",
        },
    )

    result_pid, success, message = processor.process_single_paper(pid, paper_info, skip_cached=True)

    assert (result_pid, success, message) == (pid, True, "Withdrawn")
    assert processor.stats["skipped"] == 1


def test_process_single_paper_withdrawn_skips_before_cache_check(monkeypatch, tmp_path):
    pass

    pid = "2510.19139"
    paper_info = {
        "_id": pid,
        "_idv": f"{pid}v3",
        "_version": 3,
        "title": "Withdrawn Latest Version",
        "authors": [],
        "arxiv_comment": "We have decided to withdraw this manuscript because it needs substantial revision.",
    }

    cache_root = Path(tmp_path) / "summary"
    cache_root.mkdir(parents=True, exist_ok=True)

    def _fake_summary_cache_paths(cache_pid, model):
        paper_dir = cache_root / cache_pid
        return (
            paper_dir / f"{model}.md",
            paper_dir / f"{model}.meta.json",
            paper_dir / f"{model}.lock",
            cache_root / f"{cache_pid}.md",
            cache_root / f"{cache_pid}.meta.json",
            cache_root / f"{cache_pid}.lock",
        )

    cache_checks = []

    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    monkeypatch.setattr(
        processor,
        "is_summary_cached",
        lambda cache_pid, summary_source: (cache_checks.append((cache_pid, summary_source)) or True),
    )
    monkeypatch.setattr(
        processor,
        "_refresh_public_visibility_from_arxiv",
        lambda raw_pid, force_latest_withdrawn=False: {
            "action": "tombstone",
            "pid": raw_pid,
            "latest_idv": f"{raw_pid}v3",
        },
    )

    result_pid, success, message = processor.process_single_paper(pid, paper_info, skip_cached=True)

    assert (result_pid, success, message) == (pid, True, "Withdrawn")
    assert cache_checks == []


def test_process_single_paper_withdrawn_skips_before_lock(monkeypatch, tmp_path):
    import tools.batch_paper_summarizer as tool

    pid = "2510.19139"
    paper_info = {
        "_id": pid,
        "_idv": f"{pid}v3",
        "_version": 3,
        "title": "Withdrawn Latest Version",
        "authors": [],
        "arxiv_comment": "We have decided to withdraw this manuscript because it needs substantial revision.",
    }

    cache_root = Path(tmp_path) / "summary"
    cache_root.mkdir(parents=True, exist_ok=True)

    def _fake_summary_cache_paths(cache_pid, model):
        paper_dir = cache_root / cache_pid
        return (
            paper_dir / f"{model}.md",
            paper_dir / f"{model}.meta.json",
            paper_dir / f"{model}.lock",
            cache_root / f"{cache_pid}.md",
            cache_root / f"{cache_pid}.meta.json",
            cache_root / f"{cache_pid}.lock",
        )

    monkeypatch.setattr(tool, "summary_cache_paths", _fake_summary_cache_paths)
    monkeypatch.setattr(
        tool,
        "acquire_summary_lock",
        lambda _path, timeout_s=300: (_ for _ in ()).throw(AssertionError("lock should not be acquired")),
    )

    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    monkeypatch.setattr(processor, "is_summary_cached", lambda cache_pid, summary_source: True)
    monkeypatch.setattr(
        processor,
        "_refresh_public_visibility_from_arxiv",
        lambda raw_pid, force_latest_withdrawn=False: {
            "action": "tombstone",
            "pid": raw_pid,
            "latest_idv": f"{raw_pid}v3",
        },
    )

    result_pid, success, message = processor.process_single_paper(pid, paper_info, skip_cached=True)

    assert (result_pid, success, message) == (pid, True, "Withdrawn")
