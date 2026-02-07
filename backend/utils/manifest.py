"""Static asset manifest utilities for cache-busting with hashed filenames."""

from __future__ import annotations

import glob
import json
import os

from loguru import logger

_MANIFEST_PATH: str | None = None
_MANIFEST_MTIME: float = 0.0
_MANIFEST_CACHE: dict[str, str] = {}
_DIST_DIR_MTIME: float = 0.0
_FALLBACK_CACHE: dict[str, str] = {}


def _get_manifest_path() -> str:
    """Get the path to the manifest.json file."""
    global _MANIFEST_PATH
    if _MANIFEST_PATH is None:
        # Go up two levels from backend/utils/ to project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _MANIFEST_PATH = os.path.join(root_dir, "static", "dist", "manifest.json")
    return _MANIFEST_PATH


def _load_manifest() -> dict[str, str]:
    """Load the manifest.json file, with caching based on mtime."""
    global _MANIFEST_MTIME, _MANIFEST_CACHE

    manifest_path = _get_manifest_path()

    if not os.path.isfile(manifest_path):
        # Fallback: no manifest, return empty dict (use original filenames)
        # Clear cached manifest so a newly-created file will be reloaded even if mtime resolution is coarse.
        _MANIFEST_MTIME = 0.0
        _MANIFEST_CACHE = {}
        logger.debug(f"Manifest file not found at {manifest_path}, using fallback resolution")
        return {}

    try:
        current_mtime = os.path.getmtime(manifest_path)
        if current_mtime != _MANIFEST_MTIME:
            with open(manifest_path, encoding="utf-8") as f:
                _MANIFEST_CACHE = json.load(f)
            _MANIFEST_MTIME = current_mtime
            logger.debug(f"Loaded manifest.json with {len(_MANIFEST_CACHE)} entries")
    except Exception as exc:
        logger.warning(f"Failed to load manifest.json: {exc}")
        return {}

    return _MANIFEST_CACHE


def _get_dist_dir() -> str:
    manifest_path = _get_manifest_path()
    return os.path.dirname(manifest_path)


def _refresh_dist_fallback_cache() -> None:
    """Refresh fallback cache for dist/ based on dist dir mtime."""
    global _DIST_DIR_MTIME, _FALLBACK_CACHE
    dist_dir = _get_dist_dir()
    try:
        current_mtime = os.path.getmtime(dist_dir)
    except Exception as exc:
        logger.debug(f"Failed to get mtime for dist directory {dist_dir}: {exc}")
        return
    if current_mtime != _DIST_DIR_MTIME:
        _DIST_DIR_MTIME = current_mtime
        _FALLBACK_CACHE = {}
        logger.debug(f"Cleared fallback cache due to dist directory mtime change")


def _fallback_hashed_filename(original_name: str) -> str | None:
    """Best-effort hashed filename lookup when manifest is missing/outdated.

    Looks for dist/<stem>-<hash>.<ext>. Returns the most recently modified match.
    """
    if not original_name:
        return None

    _refresh_dist_fallback_cache()
    cached = _FALLBACK_CACHE.get(original_name)
    if cached:
        return cached

    dist_dir = _get_dist_dir()
    try:
        direct_path = os.path.join(dist_dir, original_name)
        if os.path.isfile(direct_path):
            _FALLBACK_CACHE[original_name] = original_name
            return original_name
    except Exception as exc:
        logger.debug(f"Failed to check direct path for {original_name}: {exc}")

    stem, ext = os.path.splitext(original_name)
    if not stem or not ext:
        return None

    pattern = os.path.join(dist_dir, f"{stem}-*{ext}")
    try:
        matches = glob.glob(pattern)
    except Exception:
        matches = []
    if not matches:
        return None

    best = None
    best_mtime = -1.0
    for path in matches:
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = path
        elif mtime == best_mtime and best is not None and path > best:
            best = path

    if not best:
        return None

    name = os.path.basename(best)
    _FALLBACK_CACHE[original_name] = name
    return name


def get_hashed_filename(original_name: str) -> str:
    """
    Get the hashed filename for a static asset.

    Args:
        original_name: Original filename (e.g., 'paper_list.js')

    Returns:
        Hashed filename (e.g., 'paper_list-ABC123.js') or original if not found
    """
    manifest = _load_manifest()
    resolved = manifest.get(original_name)
    if resolved:
        # Guard against stale manifest entries (e.g., partial deploy/rollback).
        # If manifest points to a non-existent file, fall back to best-effort lookup.
        try:
            dist_dir = _get_dist_dir()
            if os.path.isfile(os.path.join(dist_dir, resolved)):
                return resolved
            logger.warning(f"Manifest entry missing on disk: {original_name} -> {resolved}. Falling back.")
        except Exception:
            # If anything goes wrong, fall back to best-effort lookup.
            pass

    fallback = _fallback_hashed_filename(original_name)
    return fallback or original_name


def static_url(filename: str) -> str:
    """
    Generate a URL for a static file with cache-busting hash.

    For files in dist/, looks up the hashed filename from manifest.json.
    For other files, returns the original path.

    Args:
        filename: Path relative to static folder (e.g., 'dist/paper_list.js')

    Returns:
        Path with hashed filename if available
    """
    if filename.startswith("dist/"):
        # Extract the original filename
        original_name = filename[5:]  # Remove 'dist/' prefix
        hashed_name = get_hashed_filename(original_name)
        return f"dist/{hashed_name}"

    return filename


def clear_manifest_cache() -> None:
    """Clear the manifest cache (useful for testing or hot-reload)."""
    global _MANIFEST_MTIME, _MANIFEST_CACHE, _DIST_DIR_MTIME, _FALLBACK_CACHE
    _MANIFEST_MTIME = 0.0
    _MANIFEST_CACHE = {}
    _DIST_DIR_MTIME = 0.0
    _FALLBACK_CACHE = {}
