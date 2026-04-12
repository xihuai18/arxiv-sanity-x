"""Repair historical arXiv paper visibility/version records.

Usage:
    python -m tools repair_paper_history --pid 2506.21583 --apply
    python -m tools repair_paper_history --scan-tombstones --limit 100
    python -m tools repair_paper_history --scan-public --apply
    python -m tools repair_paper_history --scan-recent-public 250 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TypeVar

from aslite.arxiv import get_entries_by_ids, is_withdrawn_entry
from aslite.repositories import (
    MetaRepository,
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
    parser.add_argument(
        "--scan-public",
        action="store_true",
        help="Scan all current public papers in the database",
    )
    parser.add_argument(
        "--scan-public-newest",
        action="store_true",
        help="Scan all current public papers from newest to oldest",
    )
    parser.add_argument(
        "--scan-recent-public",
        type=int,
        default=0,
        help="Scan the latest N public papers from the current database",
    )
    parser.add_argument(
        "--withdrawn-only",
        action="store_true",
        help="Only plan tombstones for currently withdrawn papers; skip generic refreshes",
    )
    parser.add_argument(
        "--changes-only",
        action="store_true",
        help="Only print non-no_op plans in table/json output",
    )
    parser.add_argument(
        "--process-batch-size",
        type=int,
        default=500,
        help="How many candidate pids to plan in each local batch",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=50,
        help="How many arXiv ids to fetch in each id_list request",
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between arXiv batch requests",
    )
    parser.add_argument(
        "--api-stop-after-failures",
        type=int,
        default=3,
        help="Abort after this many consecutive failed arXiv batch requests",
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


_T = TypeVar("_T")


def _iter_chunks(values: list[_T], chunk_size: int) -> list[list[_T]]:
    size = max(1, int(chunk_size or 1))
    return [values[start : start + size] for start in range(0, len(values), size)]


def _prefetch_latest_entries(
    pids: list[str],
    *,
    batch_size: int = 50,
    delay_s: float = 1.0,
    stop_after_failures: int = 3,
) -> tuple[dict[str, dict], set[str], bool]:
    entries_by_pid: dict[str, dict] = {}
    completed_pids: set[str] = set()
    normalized = _normalize_manual_pids(pids)
    if not normalized:
        return entries_by_pid, completed_pids, False

    api_chunks = _iter_chunks(normalized, batch_size)
    consecutive_failures = 0
    max_failures = max(1, int(stop_after_failures or 1))
    sleep_s = max(0.0, float(delay_s or 0.0))

    for index, chunk in enumerate(api_chunks):
        try:
            entries = get_entries_by_ids(chunk)
            consecutive_failures = 0
            completed_pids.update(chunk)
        except Exception as exc:
            consecutive_failures += 1
            sys.stderr.write(f"Warning: failed to fetch latest arXiv entries for chunk of {len(chunk)} ids: {exc}\n")
            if consecutive_failures >= max_failures:
                sys.stderr.write(f"ERROR: stopping after {consecutive_failures} consecutive arXiv API failures.\n")
                return entries_by_pid, completed_pids, True
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            raw_pid = str(entry.get("_id") or "").strip()
            if raw_pid:
                entries_by_pid[raw_pid] = entry
        if sleep_s > 0 and index < len(api_chunks) - 1:
            time.sleep(sleep_s)
    return entries_by_pid, completed_pids, False


def _collect_candidate_pids(args: argparse.Namespace) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen = set()
    limit = max(0, int(getattr(args, "limit", 0) or 0))

    def _limit_reached() -> bool:
        return limit > 0 and len(candidates) >= limit

    def _add(pid: str, source: str) -> None:
        if _limit_reached():
            return
        raw_pid = str(pid or "").strip()
        if not raw_pid or raw_pid in seen:
            return
        seen.add(raw_pid)
        candidates.append((raw_pid, source))

    for pid in _normalize_manual_pids(args.pid):
        _add(pid, "manual")

    include_tombstones = bool(
        args.scan_tombstones
        or (
            not args.pid
            and not args.scan_withdrawn
            and not bool(getattr(args, "scan_public", False))
            and not bool(getattr(args, "scan_recent_public", 0) or 0)
        )
    )
    if include_tombstones:
        try:
            for pid, _data in PaperTombstoneRepository.iter_all():
                _add(pid, "tombstone")
                if _limit_reached():
                    return candidates
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan tombstones: {exc}\n")

    if args.scan_withdrawn:
        try:
            for pid, paper in PaperRepository.iter_all_papers():
                if isinstance(paper, dict) and bool(paper.get("_latest_withdrawn", False)):
                    _add(pid, "withdrawn_flag")
                    if _limit_reached():
                        return candidates
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan withdrawn records: {exc}\n")

    if bool(getattr(args, "scan_recent_public", 0) or 0):
        try:
            recent_n = max(0, int(args.scan_recent_public or 0))
            for pid, _meta in MetaRepository.get_latest_n(recent_n, use_index=True):
                _add(pid, "recent_public")
                if _limit_reached():
                    return candidates
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan recent public papers: {exc}\n")

    if bool(getattr(args, "scan_public_newest", False)):
        try:
            batch_size = max(1, int(getattr(args, "process_batch_size", 500) or 500))
            for pid, _meta in MetaRepository.iter_latest_all(batch_size=batch_size):
                _add(pid, "public_newest")
                if _limit_reached():
                    return candidates
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan current public papers newest-first: {exc}\n")

    if bool(getattr(args, "scan_public", False)):
        try:
            for pid, _paper in PaperRepository.iter_all_papers():
                _add(pid, "public")
                if _limit_reached():
                    return candidates
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to scan current public papers: {exc}\n")
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


def plan_pid(
    pid: str,
    source: str,
    *,
    latest_entry: dict | None = None,
    allow_api_lookup: bool = True,
    withdrawn_only: bool = False,
) -> dict:
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
    if isinstance(latest_entry, dict):
        entries = [latest_entry]
    elif allow_api_lookup:
        entries = get_entries_by_ids([raw_pid])
    else:
        entries = []

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
        desired_tombstone = _build_tombstone(latest_paper, current_tombstone, current_paper)
        plan["action"] = "tombstone"
        plan["reason"] = "latest_withdrawn"
        plan["tombstones"] = {raw_pid: desired_tombstone}
        if (
            bool(current_tombstone)
            and not bool(current_paper)
            and _same_tombstone_state(current_tombstone, desired_tombstone)
        ):
            plan["action"] = "no_op"
        return plan

    if withdrawn_only:
        plan["action"] = "no_op"
        plan["reason"] = "not_withdrawn"
        return plan

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


def _render_table(plans: list[dict], *, include_header: bool = True) -> str:
    if not plans:
        return "No candidates found."
    header = f"{'PID':<16} {'Source':<14} {'Action':<10} {'Latest':<18} {'Effective':<18} Reason"
    lines = [header, "-" * len(header)] if include_header else []
    for plan in plans:
        lines.append(
            f"{plan['pid']:<16} {plan['source']:<14} {plan['action']:<10} "
            f"{str(plan.get('latest_idv') or ''):<18} {str(plan.get('effective_idv') or ''):<18} {plan.get('reason') or ''}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    candidates = _collect_candidate_pids(args)
    papers_updates: dict[str, dict] = {}
    metas_updates: dict[str, dict] = {}
    tombstone_updates: dict[str, dict] = {}
    plans: list[dict] = []
    process_batch_size = max(1, int(getattr(args, "process_batch_size", 500) or 500))
    api_batch_size = max(1, int(getattr(args, "api_batch_size", 50) or 50))
    api_delay = max(0.0, float(getattr(args, "api_delay", 1.0) or 0.0))
    api_stop_after_failures = max(1, int(getattr(args, "api_stop_after_failures", 3) or 1))
    output_mode = getattr(args, "output", "table")
    printed_header = False
    changed = 0
    aborted = False
    processed = 0
    total_candidates = len(candidates)

    if candidates:
        sys.stderr.write(
            f"Planning {len(candidates)} candidate(s) in batches of {process_batch_size} "
            f"(api_batch={api_batch_size}, api_delay={api_delay:.1f}s).\n"
        )

    for candidate_chunk in _iter_chunks(candidates, process_batch_size):
        chunk_size = len(candidate_chunk)
        chunk_start = processed + 1 if total_candidates else 0
        chunk_end = processed + chunk_size
        if total_candidates:
            pct = (processed / total_candidates) * 100.0
            sys.stderr.write(
                f"Progress: {processed}/{total_candidates} ({pct:.1f}%) processed, {changed} change(s) planned. "
                f"Fetching candidates {chunk_start}-{chunk_end}.\n"
            )
        latest_entries, completed_pids, chunk_aborted = _prefetch_latest_entries(
            [pid for pid, _source in candidate_chunk],
            batch_size=api_batch_size,
            delay_s=api_delay,
            stop_after_failures=api_stop_after_failures,
        )
        chunk_plans = [
            plan_pid(
                pid,
                source,
                latest_entry=latest_entries.get(pid),
                allow_api_lookup=False,
                withdrawn_only=bool(getattr(args, "withdrawn_only", False)),
            )
            for pid, source in candidate_chunk
            if pid in completed_pids
        ]

        if output_mode == "json":
            if bool(getattr(args, "changes_only", False)):
                plans.extend([plan for plan in chunk_plans if plan.get("action") != "no_op"])
            else:
                plans.extend(chunk_plans)
        elif chunk_plans:
            display_plans = chunk_plans
            if bool(getattr(args, "changes_only", False)):
                display_plans = [plan for plan in chunk_plans if plan.get("action") != "no_op"]
            if display_plans:
                rendered = _render_table(display_plans, include_header=not printed_header)
                sys.stdout.write(rendered + "\n")
                printed_header = True

        for plan in chunk_plans:
            action = plan.get("action")
            if action == "no_op":
                continue
            changed += 1
            papers_updates.update(plan.get("papers") or {})
            metas_updates.update(plan.get("metas") or {})
            tombstone_updates.update(plan.get("tombstones") or {})

        processed += len(completed_pids)

        if chunk_aborted:
            aborted = True
            break

    if output_mode == "json":
        sys.stdout.write(json.dumps(plans, ensure_ascii=False, indent=2) + "\n")
    elif not candidates:
        sys.stdout.write("No candidates found.\n")

    if aborted:
        sys.stderr.write("ERROR: aborted due to repeated arXiv API failures; no changes were applied.\n")
        return 1

    if total_candidates:
        sys.stderr.write(f"Progress: {processed}/{total_candidates} (100.0%) processed, {changed} change(s) planned.\n")

    sys.stdout.write(f"Planned {changed} change(s) across {len(candidates)} candidate(s).\n")

    if not getattr(args, "apply", False):
        sys.stdout.write("Dry-run only. Use --apply to persist changes.\n")
        return 0

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
