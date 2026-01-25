"""Unit tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestSummarySchemas:
    """Tests for summary-related schemas."""

    def test_summary_get_request_valid(self):
        """Test SummaryGetRequest with valid data."""
        from backend.schemas import SummaryGetRequest

        req = SummaryGetRequest(pid="2301.00001")
        assert req.pid == "2301.00001"
        assert req.model is None

    def test_summary_get_request_with_model(self):
        """Test SummaryGetRequest with model specified."""
        from backend.schemas import SummaryGetRequest

        req = SummaryGetRequest(pid="2301.00001", model="gpt-4")
        assert req.pid == "2301.00001"
        assert req.model == "gpt-4"

    def test_summary_get_request_missing_pid(self):
        """Test SummaryGetRequest rejects missing pid."""
        from backend.schemas import SummaryGetRequest

        with pytest.raises(ValidationError):
            SummaryGetRequest()

    def test_summary_trigger_request_valid(self):
        """Test SummaryTriggerRequest with valid data."""
        from backend.schemas import SummaryTriggerRequest

        req = SummaryTriggerRequest(pid="2301.00001")
        assert req.pid == "2301.00001"

    def test_summary_status_request_with_pids(self):
        """Test SummaryStatusRequest with pids list."""
        from backend.schemas import SummaryStatusRequest

        req = SummaryStatusRequest(pids=["2301.00001", "2301.00002"])
        assert req.pids == ["2301.00001", "2301.00002"]

    def test_summary_status_request_with_single_pid(self):
        """Test SummaryStatusRequest with single pid."""
        from backend.schemas import SummaryStatusRequest

        req = SummaryStatusRequest(pid="2301.00001")
        assert req.pid == "2301.00001"

    def test_summary_clear_model_request_valid(self):
        """Test SummaryClearModelRequest with valid data."""
        from backend.schemas import SummaryClearModelRequest

        req = SummaryClearModelRequest(pid="2301.00001", model="gpt-4")
        assert req.pid == "2301.00001"
        assert req.model == "gpt-4"

    def test_summary_clear_model_request_missing_model(self):
        """Test SummaryClearModelRequest rejects missing model."""
        from backend.schemas import SummaryClearModelRequest

        with pytest.raises(ValidationError):
            SummaryClearModelRequest(pid="test")


class TestReadingListSchemas:
    """Tests for reading list schemas."""

    def test_reading_list_pid_request_valid(self):
        """Test ReadingListPidRequest with valid data."""
        from backend.schemas import ReadingListPidRequest

        req = ReadingListPidRequest(pid="2301.00001")
        assert req.pid == "2301.00001"

    def test_reading_list_pid_request_missing_pid(self):
        """Test ReadingListPidRequest rejects missing pid."""
        from backend.schemas import ReadingListPidRequest

        with pytest.raises(ValidationError):
            ReadingListPidRequest()


class TestTagSchemas:
    """Tests for tag-related schemas."""

    def test_tag_feedback_request_valid(self):
        """Test TagFeedbackRequest with valid data."""
        from backend.schemas import TagFeedbackRequest

        req = TagFeedbackRequest(pid="2301.00001", tag="ml", label=1)
        assert req.pid == "2301.00001"
        assert req.tag == "ml"
        assert req.label == 1

    def test_tag_feedback_request_missing_tag(self):
        """Test TagFeedbackRequest rejects missing tag."""
        from backend.schemas import TagFeedbackRequest

        with pytest.raises(ValidationError):
            TagFeedbackRequest(pid="test", label=1)

    def test_tag_feedback_request_invalid_label(self):
        """Test TagFeedbackRequest rejects invalid label."""
        from backend.schemas import TagFeedbackRequest

        with pytest.raises(ValidationError):
            TagFeedbackRequest(pid="test", tag="ml", label=999)


class TestPaperSchemas:
    """Tests for paper-related schemas."""

    def test_paper_titles_request_valid(self):
        """Test PaperTitlesRequest with valid data."""
        from backend.schemas import PaperTitlesRequest

        req = PaperTitlesRequest(pids=["2301.00001", "2301.00002"])
        assert req.pids == ["2301.00001", "2301.00002"]


class TestSchemaExtraFields:
    """Tests for schema extra field handling."""

    def test_schema_ignores_extra_fields(self):
        """Test that schemas ignore extra fields like csrf_token."""
        from backend.schemas import SummaryGetRequest

        req = SummaryGetRequest(pid="test", csrf_token="should_be_ignored", extra_field="also_ignored")
        assert req.pid == "test"
        # Extra fields should not be accessible
        assert not hasattr(req, "csrf_token") or getattr(req, "csrf_token", None) is None
