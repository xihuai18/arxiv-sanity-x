"""Fast single-paper lookup contracts for data_service."""

from __future__ import annotations

from pathlib import Path


def test_data_service_exposes_single_meta_fast_path():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "backend" / "services" / "data_service.py").read_text(encoding="utf-8")

    assert "def get_meta(pid: str)" in text
    assert "MetaRepository.get_by_id(pid)" in text


def test_data_service_paper_exists_uses_single_meta_lookup():
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "backend" / "services" / "data_service.py").read_text(encoding="utf-8")

    assert "return get_meta(pid) is not None" in text
