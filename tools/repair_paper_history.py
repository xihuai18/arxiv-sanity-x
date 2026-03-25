"""Repair historical arXiv paper visibility/version records.

Usage:
    python -m tools repair_paper_history --pid 2506.21583 --apply
    python -m tools repair_paper_history --scan-tombstones --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from aslite.arxiv import (
    get_entries_by_ids,
    is_withdrawn_entry,
    resolve_latest_nonwithdrawn_version,
)
from aslite.repositories import (
    PaperCorpusRepository,
    PaperRepository,
    PaperTombstoneRepository,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair arXiv paper history visibility records")
    parser.add_argument(
        "--pid",
        action="append",
        default=[],
        help="Raw arXiv pid to repair (repeatable)",
    )
    parser.add_argument(
        "--scan-tombstones",
        action="store_true",
        help="Scan existing tombstones and try to restore visible papers",
    )
    parser.add_argument(
        "--scan-withdrawn",
        action="store_true",
        help="Scan current public papers flagged with _latest_withdrawn",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of pids to process")
    parser.add_argument("--apply", action="store_true", help="Write changes to the database")
    parser.add_argument(
        "--output",
        choices=("table", "json"),
        default="table",
        help="Output format for repair plan",
    )
    return parser


def _normalize_manual_pids(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values or []:
        pid = str(value or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _collect_candidate_pids(args: argparse.Namespace) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen = set()

    def _add(pid: str, source: str) -> None:
        raw_pid = str(pid or "").strip()
        if not raw_pid or raw_pid in seen:
            return
        seen.add(raw_pid)
        candidates.append((raw_pid, source))

    for pid in _normalize_manual_pids(args.pid):
        _add(pid, "manual")

    include_tombstones = bool(args.scan_tombstones or (not args.pid and not args.scan_withdrawn))
    if include_tombstones:
        try:
            for pid, _data in PaperTombstoneRepository.iter_all():
                _add(pid, "tombstone")
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan tombstones: {exc}\n")

    if args.scan_withdrawn:
        try:
            for pid, paper in PaperRepository.iter_all_papers():
                if isinstance(paper, dict) and bool(paper.get("_latest_withdrawn", False)):
                    _add(pid, "withdrawn_flag")
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan withdrawn records: {exc}\n")

    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


def _build_public_records(latest_paper: dict, selected_paper: dict) -> tuple[dict, dict]:
    effective_paper = dict(selected_paper)
    effective_paper["_effective_idv"] = effective_paper.get("_idv")
    effective_paper["_effective_version"] = effective_paper.get("_version")
    effective_paper["_latest_idv"] = latest_paper.get("_idv")
    effective_paper["_latest_version"] = latest_paper.get("_version")
    effective_paper["_latest_withdrawn"] = bool(is_withdrawn_entry(latest_paper))

    meta = {
        "_time": effective_paper.get("_time", 0),
        "_id": effective_paper.get("_id"),
        "_idv": effective_paper.get("_idv"),
        "_version": effective_paper.get("_version"),
        "_effective_idv": effective_paper.get("_effective_idv"),
        "_effective_version": effective_paper.get("_effective_version"),
        "_latest_idv": effective_paper.get("_latest_idv"),
        "_latest_version": effective_paper.get("_latest_version"),
        "_latest_withdrawn": bool(effective_paper.get("_latest_withdrawn", False)),
    }
    return effective_paper, meta


def _build_tombstone(latest_paper: dict, existing: dict | None, previous_paper: dict | None) -> dict:
    raw_pid = str(latest_paper.get("_id") or "").strip()
    now = time.time()
    tombstone = dict(existing or {})
    tombstone.update(
        {
            "pid": raw_pid,
            "reason": "withdrawn_only",
            "deleted_at": float(tombstone.get("deleted_at") or now),
            "seen_at": now,
            "latest_idv": latest_paper.get("_idv"),
            "latest_version": latest_paper.get("_version"),
            "latest_comment": latest_paper.get("arxiv_comment") or latest_paper.get("comment") or "",
        }
    )
    if isinstance(previous_paper, dict):
        tombstone.setdefault(
            "previous_effective_idv",
            previous_paper.get("_effective_idv") or previous_paper.get("_idv") or "",
        )
        tombstone.setdefault(
            "previous_effective_version",
            previous_paper.get("_effective_version") or previous_paper.get("_version"),
        )
        tombstone.setdefault("previous_title", previous_paper.get("title") or "")
    return tombstone


def _same_public_state(current: dict | None, desired: dict) -> bool:
    if not isinstance(current, dict):
        return False
    for key in (
        "_id",
        "_idv",
        "_version",
        "_effective_idv",
        "_effective_version",
        "_latest_idv",
        "_latest_version",
        "_latest_withdrawn",
    ):
        if current.get(key) != desired.get(key):
            return False
    return True


def _same_tombstone_state(current: dict | None, desired: dict) -> bool:
    if not isinstance(current, dict):
        return False
    for key in (
        "reason",
        "latest_idv",
        "latest_version",
        "latest_comment",
        "previous_effective_idv",
        "previous_effective_version",
        "previous_title",
    ):
        if current.get(key) != desired.get(key):
            return False
    return True


def plan_pid(pid: str, source: str) -> dict:
    raw_pid = str(pid or "").strip()
    try:
        current_paper = PaperRepository.get_by_id(raw_pid)
    except Exception as exc:
        sys.stderr.write(f"Warning: failed to read current paper {raw_pid}: {exc}\n")
        current_paper = None
    try:
        current_tombstone = PaperTombstoneRepository.get_by_id(raw_pid)
    except Exception as exc:
        sys.stderr.write(f"Warning: failed to read tombstone {raw_pid}: {exc}\n")
        current_tombstone = None
    entries = get_entries_by_ids([raw_pid])

    plan = {
        "pid": raw_pid,
        "source": source,
        "action": "no_op",
        "reason": "",
        "paper_exists": bool(current_paper),
        "tombstone_exists": bool(current_tombstone),
        "latest_idv": "",
        "effective_idv": "",
        "papers": {},
        "metas": {},
        "tombstones": {},
    }

    if not entries:
        plan["reason"] = "not_found_in_api"
        return plan

    latest_paper = entries[0]
    plan["latest_idv"] = latest_paper.get("_idv") or ""
    latest_is_withdrawn = bool(is_withdrawn_entry(latest_paper))

    if latest_is_withdrawn:
        selected_paper = resolve_latest_nonwithdrawn_version(latest_paper)
        if selected_paper is None:
            desired_tombstone = _build_tombstone(latest_paper, current_tombstone, current_paper)
            plan["action"] = "tombstone"
            plan["reason"] = "latest_withdrawn_and_no_visible_version"
            plan["tombstones"] = {raw_pid: desired_tombstone}
            if (
                bool(current_tombstone)
                and not bool(current_paper)
                and _same_tombstone_state(current_tombstone, desired_tombstone)
            ):
                plan["action"] = "no_op"
            return plan
    else:
        selected_paper = latest_paper

    desired_paper, desired_meta = _build_public_records(latest_paper, selected_paper)
    plan["effective_idv"] = desired_paper.get("_effective_idv") or ""
    plan["papers"] = {raw_pid: desired_paper}
    plan["metas"] = {raw_pid: desired_meta}

    if current_tombstone:
        plan["action"] = "restore"
        plan["reason"] = "visible_version_available"
        return plan
    if not _same_public_state(current_paper, desired_paper):
        plan["action"] = "refresh"
        plan["reason"] = "public_record_out_of_date"
        return plan

    plan["action"] = "no_op"
    plan["reason"] = "already_consistent"
    return plan


def _render_table(plans: list[dict]) -> str:
    if not plans:
        return "No candidates found."
    header = f"{'PID':<16} {'Source':<14} {'Action':<10} {'Latest':<18} {'Effective':<18} Reason"
    lines = [header, "-" * len(header)]
    for plan in plans:
        lines.append(
            f"{plan['pid']:<16} {plan['source']:<14} {plan['action']:<10} "
            f"{str(plan.get('latest_idv') or ''):<18} {str(plan.get('effective_idv') or ''):<18} {plan.get('reason') or ''}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    candidates = _collect_candidate_pids(args)
    plans = [plan_pid(pid, source) for pid, source in candidates]

    if getattr(args, "output", "table") == "json":
        sys.stdout.write(json.dumps(plans, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(_render_table(plans) + "\n")

    if not getattr(args, "apply", False):
        sys.stdout.write("Dry-run only. Use --apply to persist changes.\n")
        return 0

    papers_updates: dict[str, dict] = {}
    metas_updates: dict[str, dict] = {}
    tombstone_updates: dict[str, dict] = {}
    changed = 0
    for plan in plans:
        action = plan.get("action")
        if action == "no_op":
            continue
        changed += 1
        papers_updates.update(plan.get("papers") or {})
        metas_updates.update(plan.get("metas") or {})
        tombstone_updates.update(plan.get("tombstones") or {})

    if changed == 0:
        sys.stdout.write("No changes to apply.\n")
        return 0

    try:
        PaperCorpusRepository.apply_daemon_batch(
            papers=papers_updates,
            metas=metas_updates,
            tombstones=tombstone_updates,
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: failed to apply repair batch: {exc}\n")
        return 1
    sys.stdout.write(f"Applied {changed} repair action(s).\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
