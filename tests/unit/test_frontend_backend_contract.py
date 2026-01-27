"""Frontend-backend API contract tests.

These tests ensure that API endpoints referenced by frontend JS sources exist
in the Flask app routing table with compatible HTTP methods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrontendCall:
    method: str
    path: str
    is_prefix: bool
    source: str


_FETCH_RE = re.compile(r"""(?P<fn>csrfFetch|fetch)\(\s*(?P<q>['"`])(?P<url>/api/[^'"`]+)(?P=q)""")
_EVENTSOURCE_RE = re.compile(r"""EventSource\(\s*(?P<q>['"])(?P<url>/api/[^'"]+)(?P=q)""")
_XHR_OPEN_RE = re.compile(
    r"""\.open\(\s*(?P<q1>['"])(?P<m>[A-Za-z]+)(?P=q1)\s*,\s*(?P<q2>['"])(?P<url>/api/[^'"]+)(?P=q2)"""
)
_METHOD_RE = re.compile(r"""method\s*:\s*(?P<q>['"])(?P<m>[A-Za-z]+)(?P=q)""")


def _iter_frontend_calls(static_dir: Path) -> list[FrontendCall]:
    calls: list[FrontendCall] = []

    for path in sorted(static_dir.rglob("*.js")):
        # Only source files are treated as API contract. dist/ is derived and noisy.
        if "dist" in path.parts:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")

        for m in _FETCH_RE.finditer(text):
            fn = m.group("fn")
            url = m.group("url")
            url_no_query = url.split("?", 1)[0]

            # Template literals: treat `${...}` suffix as variable path segment.
            is_prefix = "${" in url_no_query
            if is_prefix:
                url_no_query = url_no_query.split("${", 1)[0]

            # String concatenation often appends a path segment:
            # e.g. fetch('/api/x/' + id). Only treat it as a prefix when the
            # literal clearly ends at a path boundary to avoid false positives
            # for query concatenation (e.g. '/api/x?pid=' + pid).
            if not is_prefix:
                tail = text[m.end() : m.end() + 40].lstrip()
                if tail.startswith("+") and url_no_query.endswith("/"):
                    is_prefix = True

            # Approximate method inference from the call-site.
            method_tail = text[m.end() : m.end() + 300]
            mm = _METHOD_RE.search(method_tail)
            if mm:
                method = mm.group("m").upper()
            else:
                method = "POST" if fn == "csrfFetch" else "GET"

            calls.append(
                FrontendCall(
                    method=method,
                    path=url_no_query,
                    is_prefix=is_prefix,
                    source=f"{path.name}:{m.start()}",
                )
            )

        for m in _EVENTSOURCE_RE.finditer(text):
            url = m.group("url").split("?", 1)[0]
            calls.append(FrontendCall(method="GET", path=url, is_prefix=False, source=f"{path.name}:{m.start()}"))

        for m in _XHR_OPEN_RE.finditer(text):
            method = m.group("m").upper()
            url = m.group("url").split("?", 1)[0]
            calls.append(FrontendCall(method=method, path=url, is_prefix=False, source=f"{path.name}:{m.start()}"))

    # Deduplicate (same endpoint can appear in multiple places).
    return sorted(set(calls), key=lambda c: (c.method, c.path, c.is_prefix, c.source))


def _route_index(app) -> list[tuple[str, set[str]]]:
    return [(rule.rule, set(rule.methods or [])) for rule in app.url_map.iter_rules()]


def _matches_route(call: FrontendCall, rule: str) -> bool:
    if call.is_prefix:
        return rule.startswith(call.path) and "<" in rule[len(call.path) :]
    return rule == call.path


def test_frontend_api_paths_exist(app):
    repo_root = Path(__file__).resolve().parents[2]
    static_dir = repo_root / "static"
    assert static_dir.is_dir()

    calls = _iter_frontend_calls(static_dir)
    routes = _route_index(app)

    missing: list[str] = []

    for call in calls:
        matched = False
        for rule, methods in routes:
            if not _matches_route(call, rule):
                continue
            if call.method in methods:
                matched = True
                break
        if not matched:
            missing.append(f"{call.method} {call.path} (prefix={call.is_prefix}) from {call.source}")

    assert not missing, "Missing backend routes referenced by frontend:\n" + "\n".join(missing)
