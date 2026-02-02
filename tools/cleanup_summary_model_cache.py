#!/usr/bin/env python3
"""Cleanup summary cache files for a given model by scanning meta.json files.

Default is dry-run (no deletion). Use --apply to actually delete files.

This script targets the on-disk summary cache under settings.summary_dir, i.e.:
- {SUMMARY_DIR}/{pid}/{model_key}.md
- {SUMMARY_DIR}/{pid}/{model_key}.meta.json
- {SUMMARY_DIR}/{pid}/.{model_key}.lock

It also handles legacy root files:
- {SUMMARY_DIR}/{pid}.md
- {SUMMARY_DIR}/{pid}.meta.json
- {SUMMARY_DIR}/.{pid}.lock

Match policy:
- If meta declares model via `llm_model` or `model`, match it.
- Otherwise, fall back to matching the filename model key (stem before .meta.json),
  compared against model_cache_key(target_model).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

# Ensure repo root is importable when executing directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, _REPO_ROOT)

from config import settings  # noqa: E402
from tools.paper_summarizer import model_cache_key  # noqa: E402


@dataclass(frozen=True)
class CacheTriplet:
    """Represents a (body, meta, lock) set for a cached summary."""

    body_path: Path
    meta_path: Path
    lock_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _declared_model(meta: dict[str, Any]) -> str:
    """Extract declared model from meta if present."""
    for key in ("llm_model", "model"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    llm = meta.get("llm")
    if isinstance(llm, dict):
        for key in ("model", "name", "id"):
            val = llm.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _iter_cache_triplets(summary_dir: Path) -> Iterable[CacheTriplet]:
    """Yield cache triplets (body, meta, lock) for all meta.json files."""
    if not summary_dir.exists():
        return

    for entry in summary_dir.iterdir():
        if entry.is_dir():
            for meta_path in entry.glob("*.meta.json"):
                if not meta_path.is_file() or meta_path.name.startswith("."):
                    continue
                name = meta_path.name
                stem = name[: -len(".meta.json")] if name.endswith(".meta.json") else meta_path.stem
                body_path = entry / f"{stem}.md"
                lock_path = entry / f".{stem}.lock"
                yield CacheTriplet(body_path=body_path, meta_path=meta_path, lock_path=lock_path)
        elif entry.is_file() and entry.name.endswith(".meta.json") and not entry.name.startswith("."):
            # Legacy root meta json: {pid}.meta.json
            pid = entry.name[: -len(".meta.json")]
            body_path = summary_dir / f"{pid}.md"
            lock_path = summary_dir / f".{pid}.lock"
            yield CacheTriplet(body_path=body_path, meta_path=entry, lock_path=lock_path)


def _matches_target(*, triplet: CacheTriplet, target_model: str, target_model_key: str) -> bool:
    meta = _read_json(triplet.meta_path)
    declared = _declared_model(meta)
    if declared:
        return declared == target_model

    # No declared model: fall back to filename model key.
    name = triplet.meta_path.name
    stem = name[: -len(".meta.json")] if name.endswith(".meta.json") else triplet.meta_path.stem
    return stem == target_model_key


def _safe_unlink(path: Path, *, apply: bool) -> bool:
    if not path.exists():
        return False
    if not apply:
        return True
    try:
        path.unlink()
        return True
    except Exception as e:
        logger.warning(f"Failed to delete {path}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup summary cache for a given model by scanning meta.json files.")
    parser.add_argument("--model", default="mimo-v2-flash", help="Target model name (exact match).")
    parser.add_argument(
        "--summary-dir", default=str(settings.summary_dir), help="Override summary dir (default from config)."
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default is dry-run).")
    parser.add_argument(
        "--refresh-stats",
        action="store_true",
        help="After deletion, rebuild persisted stats snapshot (requires backend import).",
    )
    args = parser.parse_args()

    target_model = str(args.model or "").strip()
    if not target_model:
        logger.error("--model is required")
        return 2

    summary_dir = Path(args.summary_dir).expanduser().resolve()
    if not summary_dir.exists():
        logger.error(f"Summary dir not found: {summary_dir}")
        return 2

    try:
        target_model_key = model_cache_key(target_model)
    except Exception:
        # If model name is invalid for cache keying, filename fallback matching cannot work.
        target_model_key = ""

    logger.info(f"summary_dir={summary_dir}")
    logger.info(f"target_model={target_model} target_model_key={target_model_key or '(n/a)'} apply={bool(args.apply)}")

    matched: list[CacheTriplet] = []
    scanned = 0
    for triplet in _iter_cache_triplets(summary_dir):
        scanned += 1
        if _matches_target(triplet=triplet, target_model=target_model, target_model_key=target_model_key):
            matched.append(triplet)

    # Deduplicate paths across triplets.
    to_delete: list[Path] = []
    for t in matched:
        to_delete.extend([t.meta_path, t.body_path, t.lock_path])
    uniq_paths: list[Path] = []
    seen: set[str] = set()
    for p in to_delete:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq_paths.append(p)

    logger.info(f"scanned_meta_files={scanned} matched_meta_files={len(matched)} delete_paths={len(uniq_paths)}")
    if not uniq_paths:
        return 0

    deleted_ok = 0
    missing = 0
    for p in uniq_paths:
        if not p.exists():
            missing += 1
            logger.info(f"[MISSING] {p}")
            continue

        ok = _safe_unlink(p, apply=bool(args.apply))
        if args.apply:
            status = "DELETED" if ok else "FAIL"
        else:
            status = "WOULD_DELETE" if ok else "FAIL"
        logger.info(f"[{status}] {p}")
        if ok:
            deleted_ok += 1

    logger.info(
        f"done ok={deleted_ok} missing={missing} total_paths={len(uniq_paths)} (dry_run={not bool(args.apply)})"
    )

    if args.apply and args.refresh_stats:
        try:
            from backend.services.summary_service import (
                refresh_summary_cache_stats_full,
            )

            refresh_summary_cache_stats_full()
            logger.info("refreshed summary cache stats snapshot")
        except Exception as e:
            logger.warning(f"Failed to refresh summary cache stats snapshot: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
