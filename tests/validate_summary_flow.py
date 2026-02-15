#!/usr/bin/env python3
"""
Summary Flow Validation Script (non-test utility)

This script helps reproduce/validate the summary page backend flow without a browser:
- Fetches `/summary?pid=...` to obtain CSRF token + cookies (works for anonymous sessions)
- Calls:
  - POST `/api/get_paper_summary` (cache-only)
  - POST `/api/trigger_paper_summary` (optional)
  - POST `/api/summary_status` (optional polling)
  - GET  `/api/queue_stats` (optional, per poll tick)
  - GET  `/api/task_status/<task_id>` (optional, when task_id is returned)

It is intentionally NOT part of pytest; run it manually when diagnosing production issues
like "loading takes long", "task not tracked", or intermittent `ERR_EMPTY_RESPONSE`.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import cookiejar
from typing import Any

_META_CSRF_1 = re.compile(
    r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)["\']',
    flags=re.IGNORECASE,
)
_META_CSRF_2 = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']csrf-token["\']',
    flags=re.IGNORECASE,
)


def _join(base_url: str, path: str) -> str:
    base = (base_url or "").rstrip("/")
    p = path if path.startswith("/") else f"/{path}"
    return f"{base}{p}"


def _read_json(resp) -> dict[str, Any]:
    raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _extract_csrf_token(html: str) -> str:
    m = _META_CSRF_1.search(html) or _META_CSRF_2.search(html)
    return (m.group(1) if m else "").strip()


@dataclass
class Session:
    opener: urllib.request.OpenerDirector
    csrf_token: str


def create_session(base_url: str, pid: str, timeout_s: float) -> Session:
    cj = cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    url = _join(base_url, "/summary") + "?pid=" + urllib.parse.quote(pid)
    with opener.open(url, timeout=timeout_s) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    tok = _extract_csrf_token(html)
    if not tok:
        raise RuntimeError("Failed to extract CSRF token from /summary HTML")
    return Session(opener=opener, csrf_token=tok)


def get_json(session: Session, url: str, *, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with session.opener.open(req, timeout=timeout_s) as resp:
        return _read_json(resp)


def post_json(session: Session, url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-CSRF-Token": session.csrf_token,
        },
    )
    with session.opener.open(req, timeout=timeout_s) as resp:
        return _read_json(resp)


def safe_call(fn):
    t0 = time.time()
    try:
        data = fn()
        dt = (time.time() - t0) * 1000.0
        return True, dt, data, None
    except urllib.error.HTTPError as e:
        dt = (time.time() - t0) * 1000.0
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return False, dt, None, f"HTTPError {e.code}: {body[:300]}"
    except urllib.error.URLError as e:
        dt = (time.time() - t0) * 1000.0
        return False, dt, None, f"URLError: {e}"
    except Exception as e:
        dt = (time.time() - t0) * 1000.0
        return False, dt, None, f"Error: {e}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate summary API flow (manual script)")
    parser.add_argument("--base-url", default="", help="Base URL, e.g. http://localhost:55555")
    parser.add_argument("--pid", required=True, help="Paper ID, e.g. 2301.00001")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model ids. Empty means server default. Example: gpt-4o-mini,glm-4.7",
    )
    parser.add_argument("--trigger", action="store_true", help="Trigger summary generation for each model")
    parser.add_argument("--force", action="store_true", help="Force regenerate (only when --trigger)")
    parser.add_argument("--poll", action="store_true", help="Poll /api/summary_status until done")
    parser.add_argument("--timeout-s", type=float, default=8.0, help="HTTP request timeout seconds")
    parser.add_argument("--poll-interval-s", type=float, default=2.0, help="Poll interval seconds")
    parser.add_argument("--poll-deadline-s", type=float, default=120.0, help="Overall poll deadline seconds")
    args = parser.parse_args(argv)

    base_url = args.base_url.strip() or "http://localhost:55555"
    pid = args.pid.strip()
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not models:
        models = [""]  # default model

    print(f"[validate] base={base_url} pid={pid} models={models}")

    ok, dt, sess, err = safe_call(lambda: create_session(base_url, pid, args.timeout_s))
    if not ok:
        print(f"[session] FAIL {dt:.1f}ms {err}")
        return 2
    session = sess
    print(f"[session] OK {dt:.1f}ms csrf_token_len={len(session.csrf_token)}")

    task_ids: dict[str, str] = {}

    # Initial cache check (cache-only)
    for model in models:
        payload: dict[str, Any] = {"pid": pid, "cache_only": True}
        if model:
            payload["model"] = model
        ok, dt, data, err = safe_call(
            lambda p=payload: post_json(session, _join(base_url, "/api/get_paper_summary"), p, timeout_s=args.timeout_s)
        )
        label = model or "<default>"
        if not ok:
            print(f"[get_paper_summary:{label}] FAIL {dt:.1f}ms {err}")
            continue
        cached = bool(data.get("cached", True))
        has_content = bool(data.get("summary_content"))
        print(f"[get_paper_summary:{label}] OK {dt:.1f}ms cached={cached} has_content={has_content}")

    if args.trigger:
        for model in models:
            payload: dict[str, Any] = {"pid": pid}
            if model:
                payload["model"] = model
            if args.force:
                payload["force_regenerate"] = True
            ok, dt, data, err = safe_call(
                lambda p=payload: post_json(
                    session, _join(base_url, "/api/trigger_paper_summary"), p, timeout_s=args.timeout_s
                )
            )
            label = model or "<default>"
            if not ok:
                print(f"[trigger:{label}] FAIL {dt:.1f}ms {err}")
                continue
            print(f"[trigger:{label}] OK {dt:.1f}ms status={data.get('status')} task_id={data.get('task_id')}")
            tid = str(data.get("task_id") or "").strip()
            if tid:
                task_ids[label] = tid

    if task_ids:
        for label, tid in task_ids.items():
            ok, dt, data, err = safe_call(
                lambda u=_join(base_url, f"/api/task_status/{urllib.parse.quote(tid)}"): get_json(
                    session, u, timeout_s=args.timeout_s
                )
            )
            if not ok:
                print(f"[task_status:{label}] FAIL {dt:.1f}ms {err}")
                continue
            print(
                f"[task_status:{label}] OK {dt:.1f}ms status={data.get('status')} queue_rank={data.get('queue_rank')} queue_total={data.get('queue_total')}"
            )

    if args.poll:
        deadline = time.time() + float(args.poll_deadline_s)
        finished: dict[str, bool] = {m or "<default>": False for m in models}

        while time.time() < deadline and not all(finished.values()):
            for model in models:
                label = model or "<default>"
                payload: dict[str, Any] = {"pids": [pid]}
                if model:
                    payload["model"] = model
                ok, dt, data, err = safe_call(
                    lambda p=payload: post_json(
                        session, _join(base_url, "/api/summary_status"), p, timeout_s=args.timeout_s
                    )
                )
                if not ok:
                    print(f"[summary_status:{label}] FAIL {dt:.1f}ms {err}")
                    finished[label] = False
                    continue

                statuses = data.get("statuses") or {}
                s = statuses.get(pid) or statuses.get(pid.split("v")[0]) or {}
                st = str(s.get("status") or "").strip()
                last_err = s.get("last_error")
                print(f"[summary_status:{label}] OK {dt:.1f}ms status={st} last_error={last_err}")
                finished[label] = st in {"ok", "failed", "canceled", "not_found"}

            ok, dt, qdata, qerr = safe_call(
                lambda: get_json(session, _join(base_url, "/api/queue_stats"), timeout_s=args.timeout_s)
            )
            if ok:
                print(f"[queue_stats] OK {dt:.1f}ms queued={qdata.get('queued')} running={qdata.get('running')}")
            else:
                print(f"[queue_stats] FAIL {dt:.1f}ms {qerr}")

            time.sleep(float(args.poll_interval_s))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
