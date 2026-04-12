"""Tests for stats page summary model filtering."""

from __future__ import annotations


def test_stats_filters_unsupported_summary_models(app, monkeypatch):
    from backend import legacy

    monkeypatch.setattr(legacy, "default_context", lambda: {})
    monkeypatch.setattr(legacy, "get_metas", lambda: {})
    monkeypatch.setattr(legacy, "_llm_name", lambda: "gpt-5.4")
    monkeypatch.setattr(legacy, "_supported_summary_model_ids", lambda: ["gpt-5.4", "gpt-4.1-mini"])
    monkeypatch.setattr(legacy, "_count_summary_cache_papers_for_models", lambda _models: 2)
    monkeypatch.setattr(
        legacy,
        "_get_summary_cache_stats",
        lambda: {
            "data": {
                "summary_cache_total": 5,
                "summary_cache_paper_count": 4,
                "summary_cache_model_counts": [
                    {"model": "legacy-model", "count": 2},
                    {"model": "gpt-5.4", "count": 2},
                    {"model": "gpt-4.1-mini", "count": 1},
                ],
            },
            "updated_time": 0.0,
            "in_progress": False,
            "duration": 0.0,
            "ttl": 300,
        },
    )
    monkeypatch.setattr(
        legacy.SummaryStatusRepository,
        "get_all_items",
        lambda: [
            ("paper-a::gpt-5.4", {"status": "ok"}),
            ("paper-b::legacy-model", {"status": "ok"}),
            ("paper-c::gpt-4.1-mini", {"status": "ok"}),
        ],
    )
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_items_with_prefix", lambda _prefix: [])

    captured = {}

    def _fake_render_template(_template, **context):
        captured["context"] = context
        return context

    monkeypatch.setattr(legacy, "render_template", _fake_render_template)

    with app.test_request_context("/stats"):
        legacy.stats()

    context = captured.get("context") or {}
    assert context["summary_cache_model_counts"] == [
        {"model": "gpt-5.4", "count": 2},
        {"model": "gpt-4.1-mini", "count": 1},
    ]
    assert context["summary_model_counts"] == [
        {"model": "gpt-5.4", "count": 1},
        {"model": "gpt-4.1-mini", "count": 1},
    ]
    assert context["summary_total_ok"] == 2
    assert context["summary_paper_count"] == 2
    assert context["summary_cache_total"] == 3
    assert context["summary_cache_paper_count"] == 2


def test_stats_collapses_provider_models_into_alias(app, monkeypatch):
    from backend import legacy

    monkeypatch.setattr(legacy, "default_context", lambda: {})
    monkeypatch.setattr(legacy, "get_metas", lambda: {})
    monkeypatch.setattr(legacy, "_supported_summary_model_ids", lambda: ["gpt-5.4"])
    monkeypatch.setattr(legacy, "_count_summary_cache_papers_for_models", lambda _models: 2)
    monkeypatch.setattr(
        legacy,
        "_get_summary_cache_stats",
        lambda: {
            "data": {
                "summary_cache_total": 2,
                "summary_cache_paper_count": 2,
                "summary_cache_model_counts": [
                    {"model": "openai/gpt-5.4", "count": 1},
                    {"model": "rightcode-openai/gpt-5.4", "count": 1},
                ],
            },
            "updated_time": 0.0,
            "in_progress": False,
            "duration": 0.0,
            "ttl": 300,
        },
    )
    monkeypatch.setattr(
        legacy.SummaryStatusRepository,
        "get_all_items",
        lambda: [
            ("paper-a::openai/gpt-5.4", {"status": "ok"}),
            ("paper-b::rightcode-openai/gpt-5.4", {"status": "ok"}),
        ],
    )
    monkeypatch.setattr(legacy.SummaryStatusRepository, "get_items_with_prefix", lambda _prefix: [])

    captured = {}

    def _fake_render_template(_template, **context):
        captured["context"] = context
        return context

    monkeypatch.setattr(legacy, "render_template", _fake_render_template)

    with app.test_request_context("/stats"):
        legacy.stats()

    context = captured.get("context") or {}
    assert context["summary_cache_model_counts"] == [{"model": "gpt-5.4", "count": 2}]
    assert context["summary_model_counts"] == [{"model": "gpt-5.4", "count": 2}]
