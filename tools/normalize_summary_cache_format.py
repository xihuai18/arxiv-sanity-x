from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from config.model_aliases import display_model_id
from tools.paper_summarizer import model_cache_key


@dataclass(frozen=True)
class CacheEntry:
    pid: str
    stem: str
    body_path: Path
    meta_path: Path
    lock_path: Path
    is_legacy_root: bool


@dataclass(frozen=True)
class NormalizationPlan:
    entry: CacheEntry
    target_body_path: Path
    target_meta_path: Path
    target_lock_path: Path
    normalized_meta: dict[str, Any]
    changed_meta: bool
    moved_paths: bool
    skipped_reason: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize cached summary files and meta.json to the latest format.")
    parser.add_argument(
        "--summary-dir",
        default=str(settings.summary_dir),
        help="Summary cache directory (default from config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of meta files to process (0 = no limit)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write normalized meta files and move caches into the latest layout.",
    )
    parser.add_argument(
        "--refresh-stats",
        action="store_true",
        help="Refresh persisted summary cache stats after apply.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_cache_entries(summary_dir: Path):
    if not summary_dir.exists():
        return

    for entry in sorted(summary_dir.iterdir()):
        if entry.is_dir():
            pid = entry.name
            for meta_path in sorted(entry.glob("*.meta.json")):
                if not meta_path.is_file() or meta_path.name.startswith("."):
                    continue
                stem = meta_path.name[: -len(".meta.json")]
                yield CacheEntry(
                    pid=pid,
                    stem=stem,
                    body_path=entry / f"{stem}.md",
                    meta_path=meta_path,
                    lock_path=entry / f".{stem}.lock",
                    is_legacy_root=False,
                )
        elif entry.is_file() and entry.name.endswith(".meta.json") and not entry.name.startswith("."):
            pid = entry.name[: -len(".meta.json")]
            yield CacheEntry(
                pid=pid,
                stem=pid,
                body_path=summary_dir / f"{pid}.md",
                meta_path=entry,
                lock_path=summary_dir / f".{pid}.lock",
                is_legacy_root=True,
            )


def _infer_from_stem(stem: str) -> tuple[str, str]:
    text = str(stem or "").strip()
    if not text:
        return "", ""
    if "_" not in text:
        return text, ""

    for provider in ("openai", "rightcode-openai"):
        prefix = provider + "_"
        if text.startswith(prefix):
            model_id = text[len(prefix) :]
            if model_id.startswith("gpt-"):
                return model_id, f"{provider}/{model_id}"
    return "", ""


def _declared_model(meta: dict[str, Any]) -> str:
    for key in ("model", "llm_model", "resolved_model"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    llm = meta.get("llm")
    if isinstance(llm, dict):
        provider = str(llm.get("provider") or "").strip()
        model = str(llm.get("model") or llm.get("name") or llm.get("id") or "").strip()
        if provider and model:
            return f"{provider}/{model}"
    return ""


def _normalize_meta(entry: CacheEntry, meta: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    normalized = dict(meta)

    if "generated_at" not in normalized or normalized.get("generated_at") in (None, ""):
        generated_at = normalized.get("updated_at")
        if generated_at in (None, ""):
            try:
                generated_at = entry.meta_path.stat().st_mtime
            except Exception:
                generated_at = None
        if generated_at not in (None, ""):
            normalized["generated_at"] = generated_at

    declared = _declared_model(normalized)
    alias_model = display_model_id(declared) if declared else ""
    resolved_model = str(normalized.get("resolved_model") or "").strip()

    if not alias_model:
        stem_alias, stem_resolved = _infer_from_stem(entry.stem)
        alias_model = stem_alias
        if not resolved_model:
            resolved_model = stem_resolved

    if not resolved_model and declared and "/" in declared:
        resolved_model = declared

    if alias_model:
        normalized["model"] = alias_model
        normalized.pop("llm_model", None)
    if resolved_model:
        normalized["resolved_model"] = resolved_model

    return normalized, alias_model, resolved_model


def build_plan(entry: CacheEntry, summary_dir: Path) -> NormalizationPlan:
    meta = _read_json(entry.meta_path)
    normalized_meta, alias_model, _resolved_model = _normalize_meta(entry, meta)

    if not alias_model:
        return NormalizationPlan(
            entry=entry,
            target_body_path=entry.body_path,
            target_meta_path=entry.meta_path,
            target_lock_path=entry.lock_path,
            normalized_meta=normalized_meta,
            changed_meta=normalized_meta != meta,
            moved_paths=False,
            skipped_reason="cannot_infer_model",
        )

    target_stem = model_cache_key(alias_model)
    target_dir = summary_dir / entry.pid
    target_body_path = target_dir / f"{target_stem}.md"
    target_meta_path = target_dir / f"{target_stem}.meta.json"
    target_lock_path = target_dir / f".{target_stem}.lock"

    moved_paths = (
        entry.body_path != target_body_path
        or entry.meta_path != target_meta_path
        or entry.lock_path != target_lock_path
    )

    return NormalizationPlan(
        entry=entry,
        target_body_path=target_body_path,
        target_meta_path=target_meta_path,
        target_lock_path=target_lock_path,
        normalized_meta=normalized_meta,
        changed_meta=normalized_meta != meta,
        moved_paths=moved_paths,
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _move_if_needed(src: Path, dst: Path) -> None:
    if src == dst or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    src.replace(dst)


def run(args: argparse.Namespace) -> int:
    summary_dir = Path(args.summary_dir).expanduser().resolve()
    if not summary_dir.exists():
        logger.error(f"Summary dir not found: {summary_dir}")
        return 2

    entries = list(_iter_cache_entries(summary_dir))
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    plans = [build_plan(entry, summary_dir) for entry in entries]
    changed_meta = sum(1 for plan in plans if plan.changed_meta)
    moved_paths = sum(1 for plan in plans if plan.moved_paths and not plan.skipped_reason)
    skipped = [plan for plan in plans if plan.skipped_reason]

    print(f"summary_dir={summary_dir}")
    print(f"entries={len(plans)} changed_meta={changed_meta} moved_paths={moved_paths} skipped={len(skipped)}")

    if not args.apply:
        for plan in plans[:20]:
            marker = "SKIP" if plan.skipped_reason else "PLAN"
            print(
                f"[{marker}] pid={plan.entry.pid} stem={plan.entry.stem} target={plan.target_meta_path.name} reason={plan.skipped_reason or 'normalize'}"
            )
        if len(plans) > 20:
            print(f"... truncated ({len(plans) - 20} more)")
        return 0

    applied = 0
    for plan in plans:
        if plan.skipped_reason:
            continue
        try:
            if plan.moved_paths:
                _move_if_needed(plan.entry.body_path, plan.target_body_path)
                _move_if_needed(plan.entry.meta_path, plan.target_meta_path)
                _move_if_needed(plan.entry.lock_path, plan.target_lock_path)
            elif plan.changed_meta:
                plan.target_meta_path.parent.mkdir(parents=True, exist_ok=True)

            if plan.changed_meta:
                _write_json(plan.target_meta_path, plan.normalized_meta)
            applied += 1
        except Exception as exc:
            logger.warning(f"Failed to normalize {plan.entry.meta_path}: {exc}")

    print(f"applied={applied}")

    if args.refresh_stats:
        try:
            from backend.services.summary_service import (
                refresh_summary_cache_stats_full,
            )

            refresh_summary_cache_stats_full()
            print("refreshed summary cache stats")
        except Exception as exc:
            logger.warning(f"Failed to refresh summary cache stats: {exc}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
