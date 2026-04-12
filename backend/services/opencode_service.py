"""OpenCode HTTP adapter for text-model operations."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from config import settings
from config.model_aliases import (
    build_model_candidate_chain,
    collapse_catalog_models,
    display_model_id,
    expand_model_alias_members,
    is_model_alias_id,
    model_alias_ids,
)

try:
    import requests
    from requests import Response, Session
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test envs
    requests = None
    Response = Any  # type: ignore[assignment]
    Session = Any  # type: ignore[assignment]


_MODEL_ID_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


class OpenCodeServiceError(RuntimeError):
    """Raised when the OpenCode adapter cannot fulfill a request."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.kind = str(kind or "unknown_error")
        self.status_code = status_code
        self.details = dict(details or {})
        super().__init__(str(message or self.kind))


def _opencode_timeout() -> float:
    return float(getattr(settings.opencode, "timeout", 600) or 600)


def _opencode_base_url() -> str:
    return str(getattr(settings.opencode, "resolved_base_url", "") or "").rstrip("/")


def _opencode_auth() -> tuple[str, str] | None:
    username = str(getattr(settings.opencode, "username", "") or "").strip()
    password = str(getattr(settings.opencode, "password", "") or "").strip()
    if not username and not password:
        return None
    return username, password


def normalize_model_id(model_id: str | None) -> str:
    """Validate and normalize an OpenCode canonical model id."""

    normalized = str(model_id or "").strip()
    if not normalized:
        raise OpenCodeServiceError("configuration_error", "OpenCode model id is empty")
    if not _MODEL_ID_RE.fullmatch(normalized):
        raise OpenCodeServiceError(
            "configuration_error",
            f"Invalid OpenCode model id: {normalized}",
            details={"model": normalized},
        )
    return normalized


def _split_model_id(model_id: str) -> tuple[str, str]:
    normalized = normalize_model_id(model_id)
    provider_id, raw_model_id = normalized.split("/", 1)
    return provider_id, raw_model_id


def _build_session() -> Any:
    if requests is None:
        raise OpenCodeServiceError("configuration_error", "requests is not installed")
    session = requests.Session()
    session.trust_env = False
    auth = _opencode_auth()
    if auth is not None:
        session.auth = auth
    return session


def _request_timeout(timeout: float | None = None) -> tuple[float, float]:
    total = float(timeout if timeout is not None else _opencode_timeout())
    total = max(1.0, total)
    return min(5.0, total), total


def _parse_response_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception as exc:
        raise OpenCodeServiceError(
            "invalid_response",
            f"Invalid JSON response from OpenCode: {exc}",
            status_code=getattr(response, "status_code", None),
        ) from exc


def _classify_http_error(
    response: Any, payload: Any | None = None
) -> OpenCodeServiceError:
    status_code = getattr(response, "status_code", None)
    message = "OpenCode request failed"
    if isinstance(payload, dict):
        error_value = payload.get("error")
        if isinstance(error_value, str) and error_value.strip():
            message = error_value.strip()
        elif isinstance(error_value, list) and error_value:
            message = "; ".join(
                str(item.get("message") or item)
                if isinstance(item, dict)
                else str(item)
                for item in error_value
            )
    elif response.text:
        message = response.text.strip() or message

    kind = "http_error"
    if status_code in {401, 403}:
        kind = "auth_failed"
    elif status_code == 404:
        kind = "not_found"

    return OpenCodeServiceError(
        kind,
        message,
        status_code=status_code,
        details={"payload": payload} if payload is not None else None,
    )


def _request_json(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    if requests is None:
        raise OpenCodeServiceError("configuration_error", "requests is not installed")

    base_url = _opencode_base_url()
    if not base_url:
        raise OpenCodeServiceError(
            "configuration_error", "OpenCode base URL is not configured"
        )

    url = f"{base_url}{path}"
    session = _build_session()
    try:
        response = session.request(
            method.upper(),
            url,
            json=json_body,
            timeout=_request_timeout(timeout),
        )
    except requests.exceptions.Timeout as exc:
        raise OpenCodeServiceError(
            "request_timeout",
            f"OpenCode request timed out: {method.upper()} {path}",
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise OpenCodeServiceError(
            "service_unreachable",
            f"OpenCode server is unreachable: {base_url}",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OpenCodeServiceError(
            "http_error",
            f"OpenCode request failed: {exc}",
        ) from exc
    finally:
        session.close()

    payload = _parse_response_json(response)
    if response.status_code >= 400:
        raise _classify_http_error(response, payload)
    return payload


def _create_session(*, title: str) -> str:
    payload = _request_json(
        "POST", "/session", json_body={"title": title}, timeout=15.0
    )
    if not isinstance(payload, dict):
        raise OpenCodeServiceError(
            "invalid_response", "OpenCode session response is not an object"
        )
    session_id = str(payload.get("id") or "").strip()
    if not session_id:
        raise OpenCodeServiceError(
            "invalid_response", "OpenCode session response missing id"
        )
    return session_id


def _delete_session(session_id: str) -> None:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return
    try:
        _request_json("DELETE", f"/session/{normalized_session_id}", timeout=15.0)
    except OpenCodeServiceError as exc:
        logger.debug(
            f"Failed to delete OpenCode session {normalized_session_id}: {exc}"
        )


def _normalize_usage(info: dict[str, Any]) -> dict[str, Any]:
    tokens = info.get("tokens")
    if isinstance(tokens, dict):
        return dict(tokens)
    return {}


def _extract_text_from_parts(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    return "".join(chunks).strip()


def _extract_retry_errors(parts: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "retry":
            continue
        error_info = part.get("error")
        if isinstance(error_info, dict):
            message = str(
                error_info.get("message") or error_info.get("name") or ""
            ).strip()
            if message:
                errors.append(message)
        elif error_info:
            errors.append(str(error_info))
    return errors


def _extract_json_from_text(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise OpenCodeServiceError(
            "invalid_response", "OpenCode returned empty JSON text"
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Any]] = []
    for index, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(raw[index:])
            candidates.append((index, index + end, value))
        except json.JSONDecodeError:
            continue

    top_level_candidates: list[tuple[int, int, Any]] = []
    for start, end, value in candidates:
        is_nested = any(
            outer_start <= start
            and end <= outer_end
            and (outer_start, outer_end) != (start, end)
            for outer_start, outer_end, _outer_value in candidates
        )
        if not is_nested:
            top_level_candidates.append((start, end, value))

    if top_level_candidates:
        return top_level_candidates[-1][2]

    raise OpenCodeServiceError(
        "invalid_response", "OpenCode did not return valid JSON text"
    )


def _normalize_model_response(
    *,
    requested_model: str,
    info: dict[str, Any],
) -> tuple[str, str]:
    provider_id = str(info.get("providerID") or "").strip()
    model_id = str(info.get("modelID") or "").strip()
    if provider_id and model_id:
        return normalize_model_id(f"{provider_id}/{model_id}"), provider_id
    requested_provider_id, _requested_model_id = _split_model_id(requested_model)
    return normalize_model_id(requested_model), requested_provider_id


def _send_message(
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    timeout: float | None = None,
    format_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_model = normalize_model_id(model)
    provider_id, model_id = _split_model_id(normalized_model)
    session_id = _create_session(title="arxiv-sanity llm")
    try:
        body: dict[str, Any] = {
            "model": {"providerID": provider_id, "modelID": model_id},
            "parts": [{"type": "text", "text": str(prompt or "")}],
        }
        if system:
            body["system"] = str(system)
        if format_payload is not None:
            body["format"] = dict(format_payload)

        payload = _request_json(
            "POST",
            f"/session/{session_id}/message",
            json_body=body,
            timeout=timeout,
        )
        if not isinstance(payload, dict):
            raise OpenCodeServiceError(
                "invalid_response", "OpenCode message response is not an object"
            )
        info = payload.get("info")
        parts = payload.get("parts")
        if not isinstance(info, dict) or not isinstance(parts, list):
            raise OpenCodeServiceError(
                "invalid_response", "OpenCode message response is missing info/parts"
            )
        normalized_parts = [part for part in parts if isinstance(part, dict)]
        return info, normalized_parts
    finally:
        _delete_session(session_id)


def list_models(*, timeout: float | None = None) -> list[dict[str, Any]]:
    """Return OpenCode models in provider/model canonical form."""

    payload = _request_json("GET", "/config/providers", timeout=timeout or 15.0)
    if not isinstance(payload, dict):
        raise OpenCodeServiceError(
            "invalid_response", "OpenCode providers response is not an object"
        )

    providers = payload.get("providers")
    default_map_raw = payload.get("default")
    default_map: dict[str, Any] = (
        default_map_raw if isinstance(default_map_raw, dict) else {}
    )
    if not isinstance(providers, list):
        raise OpenCodeServiceError(
            "invalid_response", "OpenCode providers response missing providers list"
        )

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_id = str(provider.get("id") or "").strip()
        provider_name = str(provider.get("name") or provider_id or "").strip()
        provider_models = provider.get("models")
        if not provider_id or not isinstance(provider_models, dict):
            continue
        default_model_id = str(default_map.get(provider_id) or "").strip()
        for model_key, model_info in provider_models.items():
            if not isinstance(model_info, dict):
                continue
            model_id = str(model_info.get("id") or model_key or "").strip()
            if not model_id:
                continue
            try:
                canonical_id = normalize_model_id(f"{provider_id}/{model_id}")
            except OpenCodeServiceError:
                # Some providers expose multi-segment ids that this repo does not
                # treat as selectable summary models. Skip them instead of failing
                # the whole model list or readiness probe.
                continue
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            models.append(
                {
                    "id": canonical_id,
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "model_id": model_id,
                    "name": str(model_info.get("name") or model_id),
                    "family": model_info.get("family"),
                    "status": model_info.get("status"),
                    "default": model_id == default_model_id,
                    "raw": {
                        "provider": {
                            "id": provider_id,
                            "name": provider_name,
                            "source": provider.get("source"),
                        },
                        "model": model_info,
                    },
                }
            )
    return models


def build_llm_models_api_payload(
    models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the legacy `/api/llm_models` compatible payload."""

    resolved_models = collapse_catalog_models(
        list(models) if models is not None else list_models()
    )
    available_aliases = {
        str(item.get("id") or "").strip()
        for item in resolved_models
        if is_model_alias_id(item.get("id"))
    }
    ordered_aliases = [
        alias_id for alias_id in model_alias_ids() if alias_id in available_aliases
    ]
    default_model = display_model_id(settings.llm.name)
    return {
        "models": [{"id": alias_id} for alias_id in ordered_aliases],
        "default": default_model if is_model_alias_id(default_model) else "",
    }


def generate_text(
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Generate text through OpenCode using a canonical provider/model id."""

    normalized_model = normalize_model_id(model)
    info, parts = _send_message(
        model=normalized_model,
        prompt=prompt,
        system=system,
        timeout=timeout,
        format_payload={"type": "text"},
    )

    text = _extract_text_from_parts(parts)
    retry_errors = _extract_retry_errors(parts)
    if not text:
        detail = (
            "; ".join(retry_errors)
            if retry_errors
            else "OpenCode returned no text content"
        )
        lowered = detail.lower()
        kind = (
            "auth_failed"
            if "401" in lowered or "unauthorized" in lowered or "auth" in lowered
            else "invalid_response"
        )
        raise OpenCodeServiceError(kind, detail, details={"info": info, "parts": parts})

    resolved_model, provider_id = _normalize_model_response(
        requested_model=normalized_model, info=info
    )
    return {
        "text": text,
        "resolved_model": resolved_model,
        "provider": provider_id,
        "usage": _normalize_usage(info),
        "raw": {"info": info, "parts": parts},
    }


def generate_structured_json(
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    system: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Generate structured JSON through OpenCode with strict fallback parsing."""

    normalized_model = normalize_model_id(model)
    info, parts = _send_message(
        model=normalized_model,
        prompt=prompt,
        system=system,
        timeout=timeout,
        format_payload={"type": "json_schema", "schema": schema, "retryCount": 0},
    )

    structured = info.get("structured")
    raw_text = _extract_text_from_parts(parts)
    retry_errors = _extract_retry_errors(parts)

    if structured is None:
        if not raw_text:
            detail = (
                "; ".join(retry_errors)
                if retry_errors
                else "OpenCode returned no structured JSON"
            )
            lowered = detail.lower()
            kind = (
                "auth_failed"
                if "401" in lowered or "unauthorized" in lowered or "auth" in lowered
                else "invalid_response"
            )
            raise OpenCodeServiceError(
                kind, detail, details={"info": info, "parts": parts}
            )
        structured = _extract_json_from_text(raw_text)

    resolved_model, provider_id = _normalize_model_response(
        requested_model=normalized_model, info=info
    )
    return {
        "json": structured,
        "text": raw_text,
        "resolved_model": resolved_model,
        "provider": provider_id,
        "usage": _normalize_usage(info),
        "raw": {"info": info, "parts": parts},
    }


def healthcheck(
    *,
    default_model: str | None = None,
    timeout: float | None = None,
    probe: bool = False,
) -> dict[str, Any]:
    """Validate OpenCode availability plus required model presence."""

    result: dict[str, Any] = {
        "ok": False,
        "base_url": _opencode_base_url(),
        "service": {},
        "models": {"available_count": 0},
        "required": {"required": [], "missing": []},
    }
    try:
        service_info = _request_json("GET", "/global/health", timeout=timeout or 5.0)
        if not isinstance(service_info, dict):
            raise OpenCodeServiceError(
                "invalid_response", "OpenCode health response is not an object"
            )
        result["service"] = {
            "reachable": True,
            "healthy": bool(service_info.get("healthy")),
            "version": service_info.get("version"),
        }

        models = list_models(timeout=timeout or 10.0)
        available_ids = {item["id"] for item in models if item.get("id")}
        runtime_candidates = build_model_candidate_chain(default_model, None)
        required_labels: list[str] = []
        missing: list[str] = []
        grouped_candidates: dict[str, list[str]] = {}
        for candidate in runtime_candidates:
            label = display_model_id(candidate)
            if not label:
                continue
            grouped_candidates.setdefault(label, []).append(
                normalize_model_id(candidate)
            )
        for label, members in grouped_candidates.items():
            required_labels.append(label)
            if not any(member in available_ids for member in members):
                missing.append(label)
        result["models"] = {"available_count": len(models)}
        result["required"] = {
            "required": required_labels,
            "missing": missing,
        }

        probe_result: dict[str, Any] | None = None
        if probe and default_model and not missing:
            probe_members = [
                normalize_model_id(item)
                for item in expand_model_alias_members(default_model)
            ]
            probe_model = next(
                (item for item in probe_members if item in available_ids), None
            )
            if probe_model is None:
                probe_model = normalize_model_id(default_model)
            probe_output = generate_text(
                model=probe_model,
                system="Reply with the single word OK.",
                prompt="Reply with the single word OK.",
                timeout=min(float(timeout or _opencode_timeout()), 30.0),
            )
            probe_result = {
                "ok": str(probe_output.get("text") or "").strip().upper() == "OK",
                "resolved_model": probe_output.get("resolved_model"),
            }
            result["probe"] = probe_result
            if not probe_result["ok"]:
                raise OpenCodeServiceError(
                    "invalid_response", "OpenCode probe returned unexpected content"
                )

        result["ok"] = bool(result["service"].get("healthy")) and not missing
        return result
    except OpenCodeServiceError as exc:
        result["error_kind"] = exc.kind
        result["error"] = str(exc)
        if exc.status_code is not None:
            result["status_code"] = exc.status_code
        if exc.details:
            result["details"] = exc.details
        return result
