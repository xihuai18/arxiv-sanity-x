from __future__ import annotations


def test_validate_model_aliases_passes_for_current_config():
    from config.model_aliases import validate_model_aliases

    assert validate_model_aliases() == []


def test_kimi_alias_expands_to_current_provider_model():
    from config.model_aliases import expand_model_alias_members

    assert expand_model_alias_members("kimi-k2.5") == ["kimi-for-coding/kimi-k2-thinking"]


def test_glm_alias_expands_to_current_provider_model():
    from config.model_aliases import expand_model_alias_members

    assert expand_model_alias_members("glm-5.1") == ["zhipuai-coding-plan/glm-5.1"]


def test_display_model_id_collapses_kimi_and_glm_aliases():
    from config.model_aliases import display_model_id

    assert display_model_id("kimi-for-coding/kimi-k2-thinking") == "kimi-k2.5"
    assert display_model_id("zhipuai-coding-plan/glm-5.1") == "glm-5.1"
