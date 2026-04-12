from __future__ import annotations

from typing import Any

MODEL_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "gpt-5.4": ("openai/gpt-5.4", "rightcode-openai/gpt-5.4"),
    "gpt-5.4-mini": ("openai/gpt-5.4-mini", "rightcode-openai/gpt-5.4-mini"),
    "kimi-k2.5": ("kimi-for-coding/k2p5", "nvidia/moonshotai/kimi-k2.5"),
    "glm-5.1": ("zhipuai-coding-plan/glm-5.1",),
}


def model_alias_ids() -> list[str]:
    return list(MODEL_ALIAS_GROUPS.keys())


def _split_model_id(model: str | None) -> tuple[str, str]:
    text = str(model or "").strip()
    if not text or "/" not in text:
        return "", text
    provider_id, model_id = text.split("/", 1)
    return provider_id.strip(), model_id.strip()


def validate_model_aliases() -> list[str]:
    errors: list[str] = []
    seen_members: dict[str, str] = {}

    for alias_id, members in MODEL_ALIAS_GROUPS.items():
        alias_text = str(alias_id or "").strip()
        if not alias_text:
            errors.append("Model alias id must not be empty")
            continue
        if "/" in alias_text:
            errors.append(f"Model alias must not contain '/': {alias_text}")
        if not isinstance(members, tuple) or not members:
            errors.append(f"Model alias must declare at least one member: {alias_text}")
            continue

        local_seen: set[str] = set()
        for member in members:
            member_text = str(member or "").strip()
            if not member_text:
                errors.append(f"Alias member must not be empty: {alias_text}")
                continue
            provider_id, model_id = _split_model_id(member_text)
            if not provider_id or not model_id:
                errors.append(f"Alias member must use provider/model format: {alias_text} -> {member_text}")
                continue
            if member_text in local_seen:
                errors.append(f"Alias members must be unique within group: {alias_text} -> {member_text}")
                continue
            local_seen.add(member_text)
            owner = seen_members.get(member_text)
            if owner and owner != alias_text:
                errors.append(f"Alias member cannot belong to multiple aliases: {member_text} ({owner}, {alias_text})")
                continue
            seen_members[member_text] = alias_text

    return errors


def is_model_alias_id(model: str | None) -> bool:
    text = str(model or "").strip()
    return bool(text and text in MODEL_ALIAS_GROUPS)


def display_model_id(model: str | None) -> str:
    text = str(model or "").strip()
    if not text:
        return ""
    if text in MODEL_ALIAS_GROUPS:
        return text
    for alias_id, members in MODEL_ALIAS_GROUPS.items():
        if text in members:
            return alias_id
    return text


def expand_model_alias_members(model: str | None) -> list[str]:
    text = str(model or "").strip()
    if not text:
        return []
    if text in MODEL_ALIAS_GROUPS:
        return list(MODEL_ALIAS_GROUPS[text])
    return [text]


def model_lookup_ids(model: str | None) -> list[str]:
    text = str(model or "").strip()
    if not text:
        return []

    alias_id = display_model_id(text)
    members = expand_model_alias_members(text if text in MODEL_ALIAS_GROUPS else alias_id)

    ordered: list[str] = []
    for candidate in [text, alias_id, *members]:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def build_model_candidate_chain(
    requested: str | None,
    default_model: str | None,
) -> list[str]:
    requested_text = str(requested or "").strip()
    default_text = str(default_model or "").strip()
    primary = requested_text or default_text
    if primary:
        primary_members = expand_model_alias_members(primary)
        if is_model_alias_id(primary) or len(primary_members) > 1:
            return list(primary_members)

    candidates: list[str] = []
    for model_name in [requested_text, default_text]:
        text = str(model_name or "").strip()
        if not text:
            continue
        expanded = expand_model_alias_members(text)
        for candidate in expanded:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def collapse_catalog_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in models or []:
        model_id = display_model_id((item or {}).get("id"))
        if not model_id or model_id in seen:
            continue
        row = dict(item or {})
        row["id"] = model_id
        collapsed.append(row)
        seen.add(model_id)
    return collapsed


def merge_model_count_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows or []:
        model_id = display_model_id((row or {}).get("model"))
        if not model_id:
            continue
        counts[model_id] = counts.get(model_id, 0) + int((row or {}).get("count") or 0)
    return [
        {"model": model_id, "count": count}
        for model_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 0
    ]


def is_valid_model_selector(model: str | None) -> bool:
    text = str(model or "").strip()
    if not text:
        return False
    if is_model_alias_id(text):
        return True
    provider_id, model_id = _split_model_id(text)
    return bool(provider_id and model_id)
