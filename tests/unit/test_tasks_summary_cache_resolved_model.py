"""Unit tests for task-side summary cache lookup with resolved_model fallback."""

from __future__ import annotations

import json


def test_read_cached_summary_uses_resolved_model_cache(monkeypatch, tmp_path):
    import tasks

    pid = "2301.00001"
    request_model = "requested-model"
    resolved_model = "fallback-model"
    cache_dir = tmp_path / pid
    cache_dir.mkdir(parents=True, exist_ok=True)

    body = cache_dir / f"{resolved_model}.md"
    meta = cache_dir / f"{resolved_model}.meta.json"
    long_body = " ".join(["detail"] * 80)
    body.write_text(f"# Title\n\n## TL;DR\n\nhello\n\n## Body\n\n{long_body}", encoding="utf-8")
    meta.write_text(json.dumps({"source": "html", "model": resolved_model}), encoding="utf-8")

    def _paths(_pid, _model):
        cache_file = cache_dir / f"{_model}.md"
        meta_file = cache_dir / f"{_model}.meta.json"
        lock_file = cache_dir / f".{_model}.lock"
        legacy_cache = tmp_path / f"{_pid}.md"
        legacy_meta = tmp_path / f"{_pid}.meta.json"
        legacy_lock = tmp_path / f".{_pid}.lock"
        return cache_file, meta_file, lock_file, legacy_cache, legacy_meta, legacy_lock

    monkeypatch.setattr(tasks, "summary_cache_paths", _paths)
    monkeypatch.setattr(
        tasks.SummaryStatusRepository,
        "get_status",
        lambda _pid, _model: {"status": "ok", "resolved_model": resolved_model},
    )

    cache_file, meta_file, _lock_file, legacy_cache, legacy_meta, _legacy_lock = _paths(pid, request_model)
    summary, out_meta = tasks._read_cached_summary(
        cache_file,
        meta_file,
        legacy_cache,
        legacy_meta,
        request_model,
        status_pid=pid,
    )
    assert "## TL;DR" in (summary or "")
    assert out_meta.get("model") == resolved_model
