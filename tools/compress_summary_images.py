from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import tools.paper_summarizer as paper_summarizer_module
from config import settings
from tools.paper_summarizer import PaperSummarizer, atomic_write_text

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CompressionResult:
    scanned: int = 0
    changed: int = 0
    markdown_updated: int = 0
    bytes_before: int = 0
    bytes_after: int = 0

    def merge(self, other: CompressionResult) -> None:
        self.scanned += other.scanned
        self.changed += other.changed
        self.markdown_updated += other.markdown_updated
        self.bytes_before += other.bytes_before
        self.bytes_after += other.bytes_after


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _iter_html_tasks(base_dir: Path):
    for paper_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        markdown_paths = sorted(p for p in paper_dir.glob("*.md") if p.is_file())
        images_dir = paper_dir / "images"
        if not markdown_paths or not images_dir.is_dir():
            continue
        yield images_dir, markdown_paths


def _iter_mineru_tasks(base_dir: Path):
    for paper_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        for backend_dir in sorted(path for path in paper_dir.iterdir() if path.is_dir()):
            markdown_paths = sorted(p for p in backend_dir.glob("*.md") if p.is_file())
            images_dir = backend_dir / "images"
            if not markdown_paths or not images_dir.is_dir():
                continue
            yield images_dir, markdown_paths


def _compress_task(summarizer: PaperSummarizer, images_dir: Path, markdown_paths: list[Path]) -> CompressionResult:
    result = CompressionResult()
    rename_map: dict[str, str] = {}
    manifest = summarizer._load_compression_manifest(images_dir)

    for image_path in sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        try:
            before_size = image_path.stat().st_size
        except OSError:
            continue
        final_path = summarizer._compress_cached_image(
            image_path,
            manifest_state=manifest,
            persist_manifest=False,
        )
        try:
            after_size = final_path.stat().st_size
        except OSError:
            after_size = before_size

        result.scanned += 1
        result.bytes_before += before_size
        result.bytes_after += after_size
        if final_path.name != image_path.name or after_size != before_size:
            result.changed += 1
        if final_path.name != image_path.name:
            rename_map[image_path.name] = final_path.name

    if rename_map:
        for markdown_path in markdown_paths:
            content = markdown_path.read_text(encoding="utf-8")
            updated = content
            for old_name, new_name in rename_map.items():
                updated = updated.replace(f"images/{old_name}", f"images/{new_name}")
            if updated != content:
                atomic_write_text(markdown_path, updated)
                result.markdown_updated += 1

    summarizer._persist_compression_manifest(images_dir, manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress cached summary images and rewrite markdown references.")
    parser.add_argument(
        "--roots",
        nargs="+",
        choices=["html_md", "mineru"],
        default=["html_md", "mineru"],
        help="Cache roots to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many cache directories (0 = no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) // 2 or 1)),
        help="Parallel worker count",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run offline compression even when automatic image compression is disabled.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    force = bool(getattr(args, "force", False))
    original_enabled = paper_summarizer_module._summary_image_compression_enabled
    if force:
        paper_summarizer_module._summary_image_compression_enabled = lambda: True

    try:
        summarizer = PaperSummarizer()
        data_dir = Path(settings.data_dir)

        task_groups = []
        if "html_md" in args.roots:
            task_groups.extend(_iter_html_tasks(data_dir / "html_md"))
        if "mineru" in args.roots:
            task_groups.extend(_iter_mineru_tasks(data_dir / "mineru"))

        if args.limit and args.limit > 0:
            task_groups = task_groups[: args.limit]

        total_dirs = len(task_groups)
        if total_dirs == 0:
            print("No cache directories with markdown+images found.", flush=True)
            return 0

        print(
            f"Compressing cached summary images: roots={','.join(args.roots)} "
            f"dirs={total_dirs} workers={args.workers} backend={summarizer.image_compression_backend_name()}"
            + (" force=true" if force else ""),
            flush=True,
        )
        start = time.time()
        summary = CompressionResult()

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            with tqdm(
                total=total_dirs,
                desc="Compressing",
                unit="dir",
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stdout,
            ) as progress:
                for idx, result in enumerate(
                    executor.map(
                        lambda task: _compress_task(summarizer, task[0], task[1]),
                        task_groups,
                    ),
                    start=1,
                ):
                    summary.merge(result)
                    if idx % 25 == 0 or idx == total_dirs:
                        saved = summary.bytes_before - summary.bytes_after
                        progress.set_postfix(
                            images=summary.scanned,
                            changed=summary.changed,
                            markdown=summary.markdown_updated,
                            saved=_human_bytes(saved),
                        )
                    progress.update(1)

        elapsed = time.time() - start
        saved = summary.bytes_before - summary.bytes_after
        print("\nCompression complete", flush=True)
        print(f"  scanned images:    {summary.scanned}", flush=True)
        print(f"  changed images:    {summary.changed}", flush=True)
        print(f"  markdown updated:  {summary.markdown_updated}", flush=True)
        print(f"  before:            {_human_bytes(summary.bytes_before)}", flush=True)
        print(f"  after:             {_human_bytes(summary.bytes_after)}", flush=True)
        print(f"  saved:             {_human_bytes(saved)}", flush=True)
        print(f"  elapsed:           {elapsed:.1f}s", flush=True)
        return 0
    finally:
        paper_summarizer_module._summary_image_compression_enabled = original_enabled


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level.upper())
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
