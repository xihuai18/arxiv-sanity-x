"""Unit tests for PID version handling in summary_utils."""

from __future__ import annotations


def test_get_summary_file_does_not_split_on_non_version_v(tmp_path, monkeypatch):
    import backend.utils.summary_utils as su

    monkeypatch.setattr(su, "_summary_dir", lambda: str(tmp_path))

    pid = "up_vpaper"
    d = tmp_path / pid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "m.md"
    p.write_text("x", encoding="utf-8")

    got = su.get_summary_file(pid, preferred_model="m")
    assert got is not None
    assert got.name == "m.md"


def test_get_summary_file_uses_model_cache_key_for_preferred_model(tmp_path, monkeypatch):
    import backend.utils.summary_utils as su

    monkeypatch.setattr(su, "_summary_dir", lambda: str(tmp_path))

    pid = "2301.00001"
    d = tmp_path / pid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "openai_gpt-5.4.md"
    p.write_text("x", encoding="utf-8")

    got = su.get_summary_file(pid, preferred_model="openai/gpt-5.4")
    assert got is not None
    assert got.name == "openai_gpt-5.4.md"
