"""Tests for static asset manifest fallback resolution."""

from __future__ import annotations

import json


def test_manifest_fallback_resolves_hashed_file_when_manifest_missing(tmp_path, monkeypatch):
    from backend.utils import manifest as m

    dist_dir = tmp_path / "static" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "common_utils-ABC12345.js").write_text("// ok", encoding="utf-8")

    # Point manifest path to a non-existent manifest.json under tmp.
    monkeypatch.setattr(m, "_MANIFEST_PATH", str(dist_dir / "manifest.json"))
    m.clear_manifest_cache()

    assert m.static_url("dist/common_utils.js") == "dist/common_utils-ABC12345.js"


def test_manifest_fallback_resolves_hashed_file_when_entry_missing(tmp_path, monkeypatch):
    from backend.utils import manifest as m

    dist_dir = tmp_path / "static" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "paper_list-XYZ99999.js").write_text("// ok", encoding="utf-8")

    # Create a manifest that does not include paper_list.js.
    manifest_path = dist_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"other.js": "other-00000000.js"}), encoding="utf-8")

    monkeypatch.setattr(m, "_MANIFEST_PATH", str(manifest_path))
    m.clear_manifest_cache()

    assert m.static_url("dist/paper_list.js") == "dist/paper_list-XYZ99999.js"
