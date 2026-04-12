from tools.batch_paper_summarizer import BatchProcessor
from tools.paper_summarizer import is_error_summary_content


def test_compact_failure_message_strips_error_heading_and_truncates():
    processor = BatchProcessor(max_workers=1, model="gpt-5.4")

    message = processor._compact_failure_message("# Error\n\nSummary generation failed: " + "x" * 400)

    assert not message.startswith("# Error")
    assert message.startswith("Summary generation failed:")
    assert message.endswith("...")
    assert len(message) <= 240


def test_extract_summary_failure_detail_includes_last_attempt_model():
    processor = BatchProcessor(max_workers=1, model="gpt-5.4")

    detail = processor._extract_summary_failure_detail(
        "# Error\n\nSummary generation failed: provider timeout",
        {
            "llm_fallback_attempts": [
                {
                    "model": "gpt-5.4",
                    "error": "provider timeout (new.xychatai.com | 524: A timeout occurred)",
                }
            ]
        },
    )

    assert "provider timeout" in detail
    assert "last model=gpt-5.4" in detail


def test_progress_postfix_includes_generated_count_for_cached_result():
    processor = BatchProcessor(max_workers=1, model="gpt-5.4")
    processor.stats["success"] = 3

    text = processor._progress_postfix(
        round_success=5,
        round_failed=1,
        pid="2601.12345",
        message="Cached",
        success=True,
    )

    assert text == "✓5 new3 ✗1 | 2601.12345 (Cached)"


def test_is_error_summary_content_does_not_misclassify_errorllm_title():
    assert is_error_summary_content("# ErrorLLM: Title\n\n## TL;DR\n\nBody") is False
    assert is_error_summary_content("# Error\n\nSomething failed") is True


def test_classify_summary_failure_reason_detects_download_error():
    processor = BatchProcessor(max_workers=1, model="gpt-5.4")

    reason, detail = processor._classify_summary_failure_reason(
        "# Error\n\nUnable to download paper PDF: HTTP 404 from arXiv PDF endpoint",
        {},
    )

    assert reason == "download_failed"
    assert "HTTP 404" in detail


def test_classify_summary_failure_reason_detects_parse_error():
    processor = BatchProcessor(max_workers=1, model="gpt-5.4")

    reason, detail = processor._classify_summary_failure_reason(
        "# MinerU API Error\n\nFailed to parse PDF using MinerU API. Please try again later.",
        {},
    )

    assert reason == "parse_failed"
    assert "Failed to parse PDF" in detail


def test_classify_summary_failure_reason_detects_empty_content():
    processor = BatchProcessor(max_workers=1, model="gpt-5.4")

    reason, detail = processor._classify_summary_failure_reason(
        "# Error\n\nParsed Markdown content is empty",
        {},
    )

    assert reason == "empty_content"
    assert "Parsed Markdown content is empty" in detail
