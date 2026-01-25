"""Static asset manifest utilities for cache-busting with hashed filenames."""

from __future__ import annotations

import json
import os

from loguru import logger

_MANIFEST_PATH: str | None = None
_MANIFEST_MTIME: float = 0.0
_MANIFEST_CACHE: dict[str, str] = {}


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


def get_hashed_filename(original_name: str) -> str:
    """
    Get the hashed filename for a static asset.

    Args:
        original_name: Original filename (e.g., 'paper_list.js')

    Returns:
        Hashed filename (e.g., 'paper_list-ABC123.js') or original if not found
    """
    manifest = _load_manifest()
    return manifest.get(original_name, original_name)


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
    global _MANIFEST_MTIME, _MANIFEST_CACHE
    _MANIFEST_MTIME = 0.0
    _MANIFEST_CACHE = {}
