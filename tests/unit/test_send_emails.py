"""Unit tests for send_emails utility functions.

Tests helper functions in send_emails.py without actually sending emails.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class TestResolveNumThreads:
    """Tests for _resolve_num_threads function."""

    def test_resolve_num_threads_auto(self):
        """Test auto thread resolution."""
        from tools.send_emails import MAX_NUM_THREADS, _resolve_num_threads

        result = _resolve_num_threads(0)
        assert result > 0
        assert result <= MAX_NUM_THREADS

    def test_resolve_num_threads_explicit(self):
        """Test explicit thread count."""
        from tools.send_emails import MAX_NUM_THREADS, _resolve_num_threads

        # Use a value less than MAX_NUM_THREADS
        requested = min(4, MAX_NUM_THREADS)
        result = _resolve_num_threads(requested)
        assert result == requested

    def test_resolve_num_threads_capped(self):
        """Test that thread count is capped."""
        from tools.send_emails import MAX_NUM_THREADS, _resolve_num_threads

        result = _resolve_num_threads(MAX_NUM_THREADS + 100)
        assert result <= MAX_NUM_THREADS


class TestHtmlEscape:
    """Tests for _h function (HTML escape)."""

    def test_h_basic(self):
        """Test basic HTML escaping."""
        from tools.send_emails import _h

        assert _h("hello") == "hello"
        assert _h("<script>") == "&lt;script&gt;"
        assert _h('"quoted"') == "&quot;quoted&quot;"
        assert _h("a & b") == "a &amp; b"

    def test_h_non_string(self):
        """Test HTML escaping with non-string input."""
        from tools.send_emails import _h

        assert _h(123) == "123"
        assert _h(None) == "None"


class TestCropSummary:
    """Tests for _crop_summary function."""

    def test_crop_summary_short(self):
        """Test cropping short text."""
        from tools.send_emails import _crop_summary

        text = "Short text"
        result = _crop_summary(text, limit=500)
        assert result == text

    def test_crop_summary_long(self):
        """Test cropping long text."""
        from tools.send_emails import _crop_summary

        text = "A" * 600
        result = _crop_summary(text, limit=500)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_crop_summary_empty(self):
        """Test cropping empty text."""
        from tools.send_emails import _crop_summary

        assert _crop_summary("") == ""
        assert _crop_summary(None) == ""


class TestAbbrAuthorMiddle:
    """Tests for _abbr_author_middle function."""

    def test_abbr_author_middle_two_names(self):
        """Test abbreviation with two names."""
        from tools.send_emails import _abbr_author_middle

        assert _abbr_author_middle("John Doe") == "John Doe"
        assert _abbr_author_middle("Huilin Deng") == "Huilin Deng"

    def test_abbr_author_middle_three_names(self):
        """Test abbreviation with three names."""
        from tools.send_emails import _abbr_author_middle

        result = _abbr_author_middle("John Ronald Tolkien")
        assert result == "John R. Tolkien"

    def test_abbr_author_middle_four_names(self):
        """Test abbreviation with four names."""
        from tools.send_emails import _abbr_author_middle

        result = _abbr_author_middle("John Ronald Reuel Tolkien")
        assert result == "John R. R. Tolkien"

    def test_abbr_author_middle_single_name(self):
        """Test abbreviation with single name."""
        from tools.send_emails import _abbr_author_middle

        assert _abbr_author_middle("Madonna") == "Madonna"

    def test_abbr_author_middle_empty(self):
        """Test abbreviation with empty input."""
        from tools.send_emails import _abbr_author_middle

        assert _abbr_author_middle("") == ""
        assert _abbr_author_middle(None) == ""


class TestApiWorkerCount:
    """Tests for _api_worker_count function."""

    def test_api_worker_count_small(self):
        """Test worker count for small task count."""
        from tools.send_emails import _api_worker_count

        result = _api_worker_count(5)
        assert result >= 1
        assert result <= 5

    def test_api_worker_count_large(self):
        """Test worker count for large task count."""
        from tools.send_emails import _api_worker_count

        result = _api_worker_count(100)
        assert result >= 1


class TestBuildParser:
    """Tests for build_parser function."""

    def test_build_parser_returns_parser(self):
        """Test that build_parser returns an ArgumentParser."""
        from tools.send_emails import build_parser

        parser = build_parser()
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_build_parser_has_required_args(self):
        """Test that parser has required arguments."""
        from tools.send_emails import build_parser

        parser = build_parser()

        # Parse with defaults
        args = parser.parse_args([])

        # Check some expected attributes exist
        assert hasattr(args, "time_delta") or hasattr(args, "t")
        assert hasattr(args, "dry_run")


class TestRecoHyperparams:
    """Tests for RecoHyperparams dataclass."""

    def test_reco_hyperparams_creation(self):
        """Test RecoHyperparams creation."""
        from tools.send_emails import RecoHyperparams

        params = RecoHyperparams(api_limit=100, model_C=0.1)
        assert params.api_limit == 100
        assert params.model_C == 0.1

    def test_reco_hyperparams_frozen(self):
        """Test that RecoHyperparams is frozen."""
        from tools.send_emails import RecoHyperparams

        params = RecoHyperparams(api_limit=100, model_C=0.1)

        with pytest.raises(Exception):  # FrozenInstanceError
            params.api_limit = 200


class TestConfigureThreadEnvVars:
    """Tests for _configure_thread_env_vars function."""

    def test_configure_thread_env_vars(self):
        """Test that thread env vars are set."""
        import os

        from tools.send_emails import _configure_thread_env_vars

        _configure_thread_env_vars(4)

        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "4"
        assert os.environ.get("NUMEXPR_NUM_THREADS") == "4"


def test_main_supports_keyword_only_user(monkeypatch):
    import tools.send_emails as se

    monkeypatch.setattr(se.TagRepository, "get_all_tags", lambda: {})
    monkeypatch.setattr(se.CombinedTagRepository, "get_all_combined_tags", lambda: {})
    monkeypatch.setattr(se.KeywordRepository, "get_all_keywords", lambda: {"alice": {"llm": ["p1"]}})
    monkeypatch.setattr(
        se.UserRepository,
        "get_all_email_lists",
        lambda: {"alice": ["alice@example.com"]},
    )
    monkeypatch.setattr(se, "calculate_recommendation", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(se, "calculate_ctag_recommendation", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(
        se,
        "search_keywords_recommendations",
        lambda *_a, **_k: ({"llm": ["2401.00001"]}, {"llm": [1.0]}),
    )
    monkeypatch.setattr(se, "render_recommendations", lambda *_a, **_k: "<p>ok</p>")

    sent = []
    monkeypatch.setattr(se, "send_email", lambda to_email, _html: sent.append(to_email))

    rc = se.main(["--user", "alice"])

    assert rc == 0
    assert sent == ["alice@example.com"]


def test_main_returns_nonzero_when_email_send_fails(monkeypatch):
    import tools.send_emails as se

    monkeypatch.setattr(se.TagRepository, "get_all_tags", lambda: {"alice": {"tag": {"p1"}}})
    monkeypatch.setattr(se.CombinedTagRepository, "get_all_combined_tags", lambda: {})
    monkeypatch.setattr(se.KeywordRepository, "get_all_keywords", lambda: {})
    monkeypatch.setattr(
        se.UserRepository,
        "get_all_email_lists",
        lambda: {"alice": ["alice@example.com"]},
    )
    monkeypatch.setattr(
        se,
        "calculate_recommendation",
        lambda *_a, **_k: ({"tag": ["2401.00001"]}, {"tag": [1.0]}),
    )
    monkeypatch.setattr(se, "calculate_ctag_recommendation", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(se, "search_keywords_recommendations", lambda *_a, **_k: ({}, {}))
    monkeypatch.setattr(se, "render_recommendations", lambda *_a, **_k: "<p>ok</p>")

    def _boom(*_a, **_k):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(se, "send_email", _boom)

    rc = se.main(["--user", "alice"])

    assert rc == 1


def test_send_email_uses_starttls_for_non_ssl_ports(monkeypatch):
    import tools.send_emails as se

    calls = []

    class _FakeSMTP:
        def __init__(self, host, port):
            calls.append(("smtp", host, port))

        def ehlo(self):
            calls.append(("ehlo",))

        def has_extn(self, name):
            return name == "starttls"

        def starttls(self):
            calls.append(("starttls",))

        def login(self, username, password):
            calls.append(("login", username, password))

        def sendmail(self, from_email, recipients, _msg):
            calls.append(("sendmail", from_email, tuple(recipients)))

        def quit(self):
            calls.append(("quit",))

    monkeypatch.setattr(se.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(
        se.smtplib,
        "SMTP_SSL",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not use SMTP_SSL")),
    )
    monkeypatch.setattr(se, "args", SimpleNamespace(dry_run=False))
    monkeypatch.setattr(se, "tnow_str", "Mar 14")
    monkeypatch.setattr(se.settings.email, "smtp_server", "smtp.example.com", raising=False)
    monkeypatch.setattr(se.settings.email, "smtp_port", 587, raising=False)
    monkeypatch.setattr(se.settings.email, "username", "alice", raising=False)
    monkeypatch.setattr(se.settings.email, "password", "secret", raising=False)
    monkeypatch.setattr(se.settings.email, "from_email", "from@example.com", raising=False)

    se.send_email("to@example.com", "<p>ok</p>")

    assert ("smtp", "smtp.example.com", 587) in calls
    assert ("starttls",) in calls
    assert ("login", "alice", "secret") in calls
    assert ("sendmail", "from@example.com", ("to@example.com",)) in calls
