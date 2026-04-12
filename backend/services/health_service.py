"""Health-check helpers for the OpenCode text-model service."""

from __future__ import annotations

from typing import Any

from .opencode_service import healthcheck as opencode_healthcheck


def check_text_model_service(
    default_model: str,
    *,
    timeout_s: float = 5.0,
    probe: bool = False,
) -> dict[str, Any]:
    """Probe OpenCode availability plus required model presence."""

    return opencode_healthcheck(
        default_model=default_model,
        timeout=timeout_s,
        probe=probe,
    )
