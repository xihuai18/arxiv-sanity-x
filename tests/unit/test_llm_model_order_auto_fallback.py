"""Tests for LLM model auto-fallback ordering utilities."""

from __future__ import annotations


def test_compute_auto_fallback_models_anchor_found():
    """Auto fallback goes to earlier models in reverse order."""
    from config.llm_model_order import compute_auto_fallback_models

    order = ["qwen3.5-plus", "glm-5", "kimi-k2.5", "gpt-5.2", "claude-opus-4-6"]
    assert compute_auto_fallback_models(yml_order=order, anchor="gpt-5.2") == [
        "kimi-k2.5",
        "glm-5",
        "qwen3.5-plus",
    ]


def test_compute_auto_fallback_models_opus_to_gpt_then_weaker():
    """If anchor is the strongest model, fallback includes the next-strongest first."""
    from config.llm_model_order import compute_auto_fallback_models

    order = ["qwen3.5-plus", "glm-5", "kimi-k2.5", "gpt-5.2", "claude-opus-4-6"]
    assert compute_auto_fallback_models(yml_order=order, anchor="claude-opus-4-6") == [
        "gpt-5.2",
        "kimi-k2.5",
        "glm-5",
        "qwen3.5-plus",
    ]


def test_compute_auto_fallback_models_anchor_missing_uses_default_anchor():
    """When anchor is unknown, fallback is computed relative to the default model."""
    from config.llm_model_order import compute_auto_fallback_models

    order = ["qwen3.5-plus", "glm-5", "kimi-k2.5", "gpt-5.2", "claude-opus-4-6"]
    assert compute_auto_fallback_models(yml_order=order, anchor="unknown", default_anchor="gpt-5.2") == [
        "kimi-k2.5",
        "glm-5",
        "qwen3.5-plus",
    ]


def test_compute_auto_fallback_models_anchor_and_default_missing_returns_empty():
    """When both anchors are unknown, auto fallback should be conservative."""
    from config.llm_model_order import compute_auto_fallback_models

    order = ["qwen3.5-plus", "glm-5", "kimi-k2.5", "gpt-5.2", "claude-opus-4-6"]
    assert compute_auto_fallback_models(yml_order=order, anchor="unknown", default_anchor="also-unknown") == []


def test_llm_settings_auto_fallback_defaults_when_no_llm_yml(monkeypatch):
    """When llm.yml is missing/unreadable, settings uses a deterministic default fallback."""
    import config.llm_model_order as llm_model_order
    from config.settings import LLMSettings

    monkeypatch.setattr(llm_model_order, "read_llm_yml_model_order", lambda *args, **kwargs: [])

    llm = LLMSettings(fallback_models="auto", name="gpt-5.2")

    assert llm.fallback_model_list == ["glm-4.7"]
