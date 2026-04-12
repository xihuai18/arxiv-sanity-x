from __future__ import annotations

from types import SimpleNamespace


def _make_summary_settings(*, default_model="openai/gpt-5.4", fallbacks=None):
    del fallbacks
    return SimpleNamespace(
        data_dir="/tmp/arxiv-sanity-test",
        llm=SimpleNamespace(
            name=default_model,
            summary_lang="en",
            timeout=123,
        ),
        summary=SimpleNamespace(
            markdown_source="html",
            html_sources="ar5iv,arxiv",
            min_chinese_ratio=0.25,
            image_compression_enabled=True,
            image_max_long_edge=0,
            image_webp_quality=86,
            image_min_savings_bytes=8192,
            image_skip_below_bytes=32768,
        ),
        mineru=SimpleNamespace(
            enabled=False,
            port=52000,
            backend="api",
            device="cpu",
            max_workers=1,
            max_vram=1,
            api_key="",
            api_poll_interval=5,
            api_timeout=60,
        ),
        main_content_min_ratio=0.1,
    )


def test_summarize_with_llm_uses_opencode_and_records_meta(monkeypatch):
    import tools.paper_summarizer as ps

    monkeypatch.setattr(ps, "settings", _make_summary_settings())
    summarizer = ps.PaperSummarizer()
    monkeypatch.setattr(summarizer, "_parse_summary_sections", lambda text: text)
    monkeypatch.setattr(summarizer, "_looks_like_valid_blog_summary", lambda text: True)
    monkeypatch.setattr(
        ps,
        "generate_opencode_text",
        lambda **kwargs: {
            "text": "# Title\n\n## TL;DR\n\nShort summary.",
            "resolved_model": "openai/gpt-5.4",
            "provider": "openai",
            "usage": {
                "input": 11,
                "output": 22,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
            "raw": {"info": {"finish": "stop", "id": "msg_test"}, "parts": []},
        },
    )

    result = summarizer.summarize_with_llm("paper markdown")

    assert result["content"].startswith("# Title")
    assert result["meta"]["llm_model"] == "gpt-5.4"
    assert result["meta"]["resolved_model"] == "openai/gpt-5.4"
    assert result["meta"]["llm"]["api"] == "opencode"
    assert result["meta"]["llm"]["provider"] == "openai"
    assert result["meta"]["llm"]["usage"]["input"] == 11


def test_summarize_with_llm_falls_back_between_models(monkeypatch):
    import tools.paper_summarizer as ps

    monkeypatch.setattr(
        ps,
        "settings",
        _make_summary_settings(
            default_model="gpt-5.4",
            fallbacks=["anthropic/claude-sonnet-4-6"],
        ),
    )
    summarizer = ps.PaperSummarizer()
    monkeypatch.setattr(summarizer, "_parse_summary_sections", lambda text: text)
    monkeypatch.setattr(summarizer, "_looks_like_valid_blog_summary", lambda text: True)

    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai/gpt-5.4":
            raise RuntimeError("provider internal error")
        return {
            "text": "# Title\n\n## TL;DR\n\nFallback summary.",
            "resolved_model": "rightcode-openai/gpt-5.4",
            "provider": "rightcode-openai",
            "usage": {
                "input": 1,
                "output": 2,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
            "raw": {"info": {"finish": "stop"}, "parts": []},
        }

    monkeypatch.setattr(ps, "generate_opencode_text", fake_generate)

    result = summarizer.summarize_with_llm("paper markdown")

    assert calls == ["openai/gpt-5.4", "rightcode-openai/gpt-5.4"]
    assert result["meta"]["llm_model"] == "gpt-5.4"
    assert result["meta"]["resolved_model"] == "rightcode-openai/gpt-5.4"
    assert result["meta"]["llm_fallback_attempts"][0]["model"] == "openai/gpt-5.4"
    assert result["meta"]["llm_fallback_attempts"][1]["success"] is True


def test_summarize_with_llm_retries_when_first_content_is_incomplete(monkeypatch):
    import tools.paper_summarizer as ps

    monkeypatch.setattr(ps, "settings", _make_summary_settings())
    summarizer = ps.PaperSummarizer()
    monkeypatch.setattr(summarizer, "_parse_summary_sections", lambda text: text)
    monkeypatch.setattr(
        summarizer,
        "_looks_like_valid_blog_summary",
        lambda text: "## TL;DR" in text,
    )

    responses = iter(
        [
            {
                "text": "# Title\n\nIncomplete",
                "resolved_model": "openai/gpt-5.4",
                "provider": "openai",
                "usage": {},
                "raw": {"info": {"finish": "stop"}, "parts": []},
            },
            {
                "text": "# Title\n\n## TL;DR\n\nComplete summary.",
                "resolved_model": "openai/gpt-5.4",
                "provider": "openai",
                "usage": {},
                "raw": {"info": {"finish": "stop"}, "parts": []},
            },
        ]
    )
    monkeypatch.setattr(ps, "generate_opencode_text", lambda **kwargs: next(responses))

    result = summarizer.summarize_with_llm("paper markdown")

    assert "## TL;DR" in result["content"]
