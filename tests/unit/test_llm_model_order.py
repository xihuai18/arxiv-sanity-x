from __future__ import annotations


def test_read_llm_yml_model_order_unique_and_ordered(tmp_path):
    from config.llm_model_order import read_llm_yml_model_order

    read_llm_yml_model_order.cache_clear()
    p = tmp_path / "llm.yml"
    p.write_text(
        """
model_list:
  - model_name: deepseek-v3.2  # comment
    litellm_params: { model: openai/deepseek-v3.2 }
  - model_name: "glm-5"
    litellm_params: { model: openai/z-ai/glm5 }
  - model_name: deepseek-v3.2
    litellm_params: { model: openai/deepseek-v3.2-alt }
  - model_name: 'claude-opus-4-6'
    litellm_params: { model: anthropic/claude-opus-4-6-20260205 }
""".strip(),
        encoding="utf-8",
    )

    order = read_llm_yml_model_order(p)
    assert order == ["deepseek-v3.2", "glm-5", "claude-opus-4-6"]


def test_sort_models_by_preferred_order_stable():
    from config.llm_model_order import sort_models_by_preferred_order

    models = [{"id": "b"}, {"id": "x"}, {"id": "a"}, {"id": "x2"}]
    preferred = ["a", "b"]
    sorted_models = sort_models_by_preferred_order(models, preferred)
    assert [m["id"] for m in sorted_models] == ["a", "b", "x", "x2"]
