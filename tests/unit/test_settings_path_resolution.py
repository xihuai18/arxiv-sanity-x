"""Unit tests for settings path resolution."""

from __future__ import annotations

from config.settings import PROJECT_ROOT, Settings


def test_relative_data_dir_resolves_to_project_root(monkeypatch):
    monkeypatch.setenv("ARXIV_SANITY_DATA_DIR", "data")
    s = Settings()
    assert s.data_dir == (PROJECT_ROOT / "data").resolve()


def test_relative_summary_dir_resolves_to_project_root(monkeypatch):
    monkeypatch.setenv("ARXIV_SANITY_SUMMARY_DIR", "data/summary")
    s = Settings()
    assert s.summary_dir == (PROJECT_ROOT / "data" / "summary").resolve()
