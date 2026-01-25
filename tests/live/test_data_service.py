"""Live tests for data availability.

These tests run when data files are available.
Tests are automatically skipped if data files are not present.
"""

from __future__ import annotations

from tests.service_detection import is_data_available, requires_data


class TestDataAvailability:
    """Tests for data availability detection."""

    def test_can_detect_data_availability(self):
        """Test that we can detect data availability."""
        result = is_data_available()
        assert isinstance(result, bool)
        if result:
            print("✓ Data files are available")
        else:
            print("✗ Data files are not available")


@requires_data
class TestDataServiceLive:
    """Live tests for data service (requires data files)."""

    def test_get_pids_returns_data(self):
        """Test that get_pids returns actual data."""
        from backend.services.data_service import get_pids

        pids = get_pids()
        assert isinstance(pids, list)
        assert len(pids) > 0
        print(f"  Found {len(pids)} papers")

    def test_get_metas_returns_data(self):
        """Test that get_metas returns actual data."""
        from backend.services.data_service import get_metas

        metas = get_metas()
        assert isinstance(metas, dict)
        assert len(metas) > 0
        print(f"  Found {len(metas)} paper metadata entries")

    def test_paper_exists_for_real_paper(self):
        """Test paper_exists for a real paper."""
        from backend.services.data_service import get_pids, paper_exists

        pids = get_pids()
        if pids:
            # Test with first available paper
            assert paper_exists(pids[0]) is True

    def test_get_paper_returns_data(self):
        """Test that get_paper returns actual paper data."""
        from backend.services.data_service import get_paper, get_pids

        pids = get_pids()
        if pids:
            paper = get_paper(pids[0])
            assert paper is not None
            assert isinstance(paper, dict)

    def test_get_papers_bulk_returns_data(self):
        """Test that get_papers_bulk returns actual data."""
        from backend.services.data_service import get_papers_bulk, get_pids

        pids = get_pids()
        if len(pids) >= 3:
            papers = get_papers_bulk(pids[:3])
            assert isinstance(papers, dict)
            assert len(papers) > 0


@requires_data
class TestRenderServiceLive:
    """Live tests for render service (requires data files)."""

    def test_render_pid_returns_data(self):
        """Test that render_pid returns actual rendered data."""
        from backend.services.data_service import get_pids
        from backend.services.render_service import render_pid

        pids = get_pids()
        if pids:
            rendered = render_pid(pids[0])
            assert isinstance(rendered, dict)
            # Should have basic fields
            assert "pid" in rendered or "id" in rendered

    def test_build_paper_text_fields_with_real_data(self):
        """Test build_paper_text_fields with real paper data."""
        from backend.services.data_service import get_paper, get_pids
        from backend.services.render_service import build_paper_text_fields

        pids = get_pids()
        if pids:
            paper = get_paper(pids[0])
            if paper:
                fields = build_paper_text_fields(paper)
                assert isinstance(fields, dict)


@requires_data
class TestSearchServiceLive:
    """Live tests for search service (requires data files)."""

    def test_keyword_search_returns_results(self):
        """Test that keyword search returns results."""
        # This would need the full search infrastructure
        # For now, just test the parsing functions
        from backend.services.search_service import normalize_text, parse_search_query

        query = "machine learning transformer"
        parsed = parse_search_query(query)
        assert "terms" in parsed
        assert "norm" in parsed

        normalized = normalize_text(query)
        assert isinstance(normalized, str)
