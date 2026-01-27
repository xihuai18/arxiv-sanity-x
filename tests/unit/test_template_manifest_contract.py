"""Template-manifest contract tests.

Ensures that all `hashed_static('dist/...')` references in Jinja templates have
corresponding entries in `static/dist/manifest.json` when it exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_HASHED_STATIC_RE = re.compile(r"""hashed_static\(\s*['"]dist/(?P<name>[^'"]+)['"]\s*\)""")


def test_templates_reference_existing_manifest_entries():
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "static" / "dist" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("static/dist/manifest.json not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)

    templates_dir = repo_root / "templates"
    missing: list[str] = []

    for tpl in sorted(templates_dir.glob("*.html")):
        text = tpl.read_text(encoding="utf-8", errors="ignore")
        for m in _HASHED_STATIC_RE.finditer(text):
            name = m.group("name").strip()
            if name not in manifest:
                missing.append(f"{tpl.name}: dist/{name}")

    assert not missing, "Missing manifest entries for templates:\n" + "\n".join(missing)
