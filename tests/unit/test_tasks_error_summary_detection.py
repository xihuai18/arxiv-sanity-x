from tasks import _is_error_summary


def test_is_error_summary_allows_normal_titles_starting_with_error_word():
    text = "# ErrorLLM: 用错误建模重塑 Text-to-SQL Refine\n\n## TL;DR\n\nBody" + "x" * 900

    assert _is_error_summary(text) is False
    assert _is_error_summary("# Error\n\nSummary generation failed") is True
