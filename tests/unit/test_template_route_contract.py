"""Template route contract tests.

Ensures that literal route paths referenced in templates via href/action exist
in the Flask app with compatible HTTP methods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteRef:
    method: str
    path: str
    source: str


_HREF_RE = re.compile(r"""href\s*=\s*['"](?P<url>/[^'"]+)['"]""", re.IGNORECASE)
_FORM_RE = re.compile(
    r"""<form\b(?P<attrs>[^>]*?)>""",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_RE = re.compile(r"""action\s*=\s*['"](?P<url>/[^'"]+)['"]""", re.IGNORECASE)
_METHOD_ATTR_RE = re.compile(r"""method\s*=\s*['"](?P<m>[A-Za-z]+)['"]""", re.IGNORECASE)


def _strip_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    # Drop query/hash
    url = url.split("#", 1)[0].split("?", 1)[0]
    return url


def _iter_template_route_refs(templates_dir: Path) -> list[RouteRef]:
    refs: list[RouteRef] = []

    for path in sorted(templates_dir.glob("*.html")):
        text = path.read_text(encoding="utf-8", errors="ignore")

        for m in _HREF_RE.finditer(text):
            url = m.group("url")
            if "{{" in url or "}}" in url:
                continue
            if url.startswith("/static/"):
                continue
            if url.startswith("//"):
                continue
            refs.append(RouteRef(method="GET", path=_strip_url(url), source=f"{path.name}:{m.start()}"))

        for m in _FORM_RE.finditer(text):
            attrs = m.group("attrs") or ""
            action_m = _ACTION_RE.search(attrs)
            if not action_m:
                continue
            url = action_m.group("url")
            if "{{" in url or "}}" in url:
                continue
            if url.startswith("/static/"):
                continue
            method_m = _METHOD_ATTR_RE.search(attrs)
            method = (method_m.group("m") if method_m else "GET").upper()
            refs.append(RouteRef(method=method, path=_strip_url(url), source=f"{path.name}:{m.start()}"))

    # Deduplicate.
    return sorted(set(refs), key=lambda r: (r.method, r.path, r.source))


def test_template_routes_exist(app):
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "templates"
    refs = _iter_template_route_refs(templates_dir)

    routes = [(rule.rule, set(rule.methods or [])) for rule in app.url_map.iter_rules()]

    missing: list[str] = []
    for ref in refs:
        ok = False
        for rule, methods in routes:
            if rule == ref.path and ref.method in methods:
                ok = True
                break
        if not ok:
            missing.append(f"{ref.method} {ref.path} from {ref.source}")

    assert not missing, "Missing routes referenced by templates:\n" + "\n".join(missing)
