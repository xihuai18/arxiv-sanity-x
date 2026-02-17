"""Utilities for working with LiteLLM model ordering.

This project treats `config/llm.yml` (LiteLLM config) as the source of truth for
model display order and (optionally) fallback order.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:  # PyYAML is already present in most environments (LiteLLM depends on it).
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def default_llm_yml_path() -> Path:
    return Path(__file__).resolve().parent / "llm.yml"


def read_llm_yml_model_order(path: Path | None = None) -> list[str]:
    """Return unique `model_name` sequence as declared in `config/llm.yml`.

    Note: The result is cached and automatically invalidated when the file mtime changes.
    Missing/unreadable files are NOT cached (so they can become available later).
    """
    p = path or default_llm_yml_path()
    try:
        st = p.stat()
    except Exception:
        return []
    return _read_llm_yml_model_order_cached(str(p), int(getattr(st, "st_mtime_ns", 0)), int(st.st_size))


@lru_cache(maxsize=32)
def _read_llm_yml_model_order_cached(path_str: str, mtime_ns: int, size: int) -> list[str]:
    _ = (mtime_ns, size)  # part of cache key
    p = Path(path_str)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    order: list[str] = []
    seen: set[str] = set()

    # Prefer parsing as YAML for correctness (anchors, multi-line strings, etc.).
    if yaml is not None:
        try:
            cfg = yaml.safe_load(text)
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            model_list = cfg.get("model_list") or []
            if isinstance(model_list, list):
                for it in model_list:
                    if not isinstance(it, dict):
                        continue
                    name = str(it.get("model_name") or "").strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    order.append(name)
                return order

    # Fallback: best-effort line parsing (kept for minimal environments).
    for line in text.splitlines():
        if "model_name" not in line:
            continue
        # Keep this intentionally simple; YAML parsing above is the primary path.
        parts = line.split("model_name", 1)
        if len(parts) < 2:
            continue
        rhs = parts[1]
        if ":" not in rhs:
            continue
        name = rhs.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"")
        if not name or name in seen:
            continue
        seen.add(name)
        order.append(name)
    return order


def clear_llm_yml_model_order_cache() -> None:
    _read_llm_yml_model_order_cached.cache_clear()


# Compatibility: allow callers/tests to call `read_llm_yml_model_order.cache_clear()`.
read_llm_yml_model_order.cache_clear = clear_llm_yml_model_order_cache  # type: ignore[attr-defined]


def sort_models_by_preferred_order(models: list[dict], preferred_order: list[str]) -> list[dict]:
    """Stable-sort model dicts by preferred id order.

    - Models whose `id` appears in `preferred_order` come first, in that order.
    - All other models keep their relative order and are appended after.
    """
    if not models or not preferred_order:
        return models

    rank = {mid: i for i, mid in enumerate(preferred_order)}
    indexed = list(enumerate(models))

    def _key(item):
        idx, obj = item
        mid = ""
        try:
            mid = str((obj or {}).get("id") or "")
        except Exception:
            mid = ""
        r = rank.get(mid)
        if r is None:
            return (1, idx)
        return (0, r)

    indexed.sort(key=_key)
    return [obj for _idx, obj in indexed]


def compute_auto_fallback_models(
    *,
    yml_order: list[str],
    anchor: str,
    default_anchor: str | None = None,
) -> list[str]:
    """Compute auto fallback models based on `config/llm.yml` model order.

    Semantics:
    - The project treats `config/llm.yml` order as "progressively stronger".
    - When a model fails, fallback should try earlier (weaker) models first.

    Behavior:
    - If `anchor` exists in `yml_order`: return models *before* it, in reverse order.
    - Else if `default_anchor` exists in `yml_order`: fallback relative to `default_anchor`.
    - Else: return `yml_order` as-is (best-effort).
    """

    order = list(yml_order or [])
    a = (anchor or "").strip()
    d = (default_anchor or "").strip() if default_anchor is not None else ""

    def _before_in_reverse(model_id: str) -> list[str] | None:
        if not model_id:
            return None
        try:
            idx = order.index(model_id)
        except ValueError:
            return None
        return list(reversed(order[:idx]))

    out = _before_in_reverse(a)
    if out is not None:
        return out
    out = _before_in_reverse(d)
    if out is not None:
        return out
    return order
