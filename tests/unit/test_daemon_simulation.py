"""Unit tests for tools.daemon job logic.

We don't run APScheduler here (it blocks). Instead we test the job functions
(fetch_compute/gen_summary) directly by monkeypatching the command runner.
"""

from __future__ import annotations


def test_fetch_compute_skips_on_fetch_failure(monkeypatch):
    import tools.daemon as d

    calls: list[tuple[str, list[str]]] = []

    def fake_run_cmd(cmd, name: str) -> bool:
        calls.append((name, list(cmd)))
        return False if name == "fetch" else True

    monkeypatch.setattr(d, "_run_cmd", fake_run_cmd)

    # Should only attempt fetch; compute/summary should be skipped.
    d.fetch_compute()

    assert [name for name, _cmd in calls] == ["fetch"]


def test_fetch_compute_runs_compute_and_optional_embeddings(monkeypatch):
    import tools.daemon as d

    calls: list[tuple[str, list[str]]] = []

    def fake_run_cmd(cmd, name: str) -> bool:
        calls.append((name, list(map(str, cmd))))
        return True

    monkeypatch.setattr(d, "_run_cmd", fake_run_cmd)

    # Keep test deterministic.
    monkeypatch.setattr(d, "ENABLE_SUMMARY", False)

    # Case 1: embeddings enabled -> compute should include --use_embeddings.
    monkeypatch.setattr(d, "ENABLE_EMBEDDINGS", True)
    calls.clear()
    d.fetch_compute()
    compute_cmd = next(cmd for name, cmd in calls if name == "compute")
    assert "--use_embeddings" in compute_cmd

    # Case 2: embeddings disabled -> compute should NOT include --use_embeddings.
    monkeypatch.setattr(d, "ENABLE_EMBEDDINGS", False)
    calls.clear()
    d.fetch_compute()
    compute_cmd = next(cmd for name, cmd in calls if name == "compute")
    assert "--use_embeddings" not in compute_cmd


def test_gen_summary_respects_enable_summary(monkeypatch):
    import tools.daemon as d

    calls: list[str] = []

    def fake_run_cmd(_cmd, name: str) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(d, "_run_cmd", fake_run_cmd)

    monkeypatch.setattr(d, "ENABLE_SUMMARY", False)
    ok = d.gen_summary()
    assert ok is True
    assert calls == []

    monkeypatch.setattr(d, "ENABLE_SUMMARY", True)
    ok = d.gen_summary()
    assert ok is True
    assert calls == ["generate_summary"]
