from __future__ import annotations

import json
from types import SimpleNamespace


def test_run_normalizes_layered_alias_cache(tmp_path):
    import tools.normalize_summary_cache_format as tool

    summary_dir = tmp_path / "summary"
    cache_dir = summary_dir / "2301.00001"
    cache_dir.mkdir(parents=True)
    (cache_dir / "openai_gpt-5.4.md").write_text("# ok", encoding="utf-8")
    (cache_dir / "openai_gpt-5.4.meta.json").write_text(
        json.dumps({"llm_model": "openai/gpt-5.4", "source": "html"}),
        encoding="utf-8",
    )

    rc = tool.run(
        SimpleNamespace(
            summary_dir=str(summary_dir),
            limit=0,
            apply=True,
            refresh_stats=False,
        )
    )

    assert rc == 0
    assert not (cache_dir / "openai_gpt-5.4.md").exists()
    assert (cache_dir / "gpt-5.4.md").exists()
    meta = json.loads((cache_dir / "gpt-5.4.meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "gpt-5.4"
    assert "llm_model" not in meta
    assert meta["resolved_model"] == "openai/gpt-5.4"
    assert meta["source"] == "html"


def test_run_moves_legacy_root_cache_into_layered_layout(tmp_path):
    import tools.normalize_summary_cache_format as tool

    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    (summary_dir / "2301.00002.md").write_text("# ok", encoding="utf-8")
    (summary_dir / "2301.00002.meta.json").write_text(
        json.dumps({"llm_model": "gpt-5.4", "source": "html"}),
        encoding="utf-8",
    )

    rc = tool.run(
        SimpleNamespace(
            summary_dir=str(summary_dir),
            limit=0,
            apply=True,
            refresh_stats=False,
        )
    )

    assert rc == 0
    assert not (summary_dir / "2301.00002.md").exists()
    assert not (summary_dir / "2301.00002.meta.json").exists()
    assert (summary_dir / "2301.00002" / "gpt-5.4.md").exists()
    meta = json.loads((summary_dir / "2301.00002" / "gpt-5.4.meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "gpt-5.4"
    assert "llm_model" not in meta
