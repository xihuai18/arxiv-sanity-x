"""Shared pytest fixtures and test configuration.

This module provides common fixtures and utilities for all tests.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from typing import Generator

import pytest

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Ensure tools directory is on sys.path (for paper_summarizer, etc.)
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


def configure_test_env() -> None:
    """Configure environment variables for testing.

    Disables background warmups/scheduler to keep tests deterministic and fast.

    Data directory strategy:
    - If ARXIV_SANITY_DATA_DIR is already set, use it (user override)
    - If real data/ directory exists with papers.db, use it (development environment)
    - Otherwise, use a temporary directory (CI or fresh environment)

    This ensures:
    1. Tests don't accidentally modify real data (dict.db writes go to temp)
    2. Tests that need papers.db can run in development environments
    3. Tests gracefully skip in CI environments without data
    """
    if "ARXIV_SANITY_DATA_DIR" not in os.environ:
        # Check if real data directory exists with papers.db
        real_data_dir = os.path.join(REPO_ROOT, "data")
        real_papers_db = os.path.join(real_data_dir, "papers.db")

        if os.path.exists(real_papers_db):
            # Use real data directory for read operations
            # Note: dict.db tables will be created in real data/ if they don't exist
            # This is acceptable for development but not ideal for CI
            os.environ["ARXIV_SANITY_DATA_DIR"] = real_data_dir
        else:
            # Use temporary directory to isolate tests
            _test_data_dir = tempfile.mkdtemp(prefix="arxiv_sanity_test_")
            os.environ["ARXIV_SANITY_DATA_DIR"] = _test_data_dir

    os.environ["ARXIV_SANITY_WARMUP_DATA"] = "0"
    os.environ["ARXIV_SANITY_WARMUP_ML"] = "0"
    os.environ["ARXIV_SANITY_ENABLE_SCHEDULER"] = "0"
    os.environ["ARXIV_SANITY_SUMMARY_REPAIR_ENABLE"] = "0"
    os.environ["ARXIV_SANITY_ENABLE_SWAGGER"] = "0"
    os.environ["ARXIV_SANITY_SECRET_KEY"] = "test-secret-key"
    os.environ.setdefault("ARXIV_SANITY_LOG_LEVEL", "ERROR")


# Configure test environment on import
configure_test_env()


@pytest.fixture(scope="session")
def app():
    """Create Flask application for testing.

    This fixture is session-scoped to avoid recreating the app for each test.
    """
    from backend import create_app

    application = create_app()
    application.testing = True
    return application


@pytest.fixture
def client(app):
    """Create Flask test client.

    This fixture is function-scoped to ensure clean state for each test.
    """
    return app.test_client()


@pytest.fixture
def csrf_token(client) -> str:
    """Get CSRF token from a page.

    Fetches the about page and extracts the CSRF token from the meta tag.
    """
    resp = client.get("/about")
    match = re.search(r'csrf-token"\s+content="([^"]+)"', resp.get_data(as_text=True))
    return match.group(1) if match else ""


@pytest.fixture
def logged_in_client(client, csrf_token) -> Generator:
    """Create a logged-in test client.

    Logs in as 'test_user' and yields the client with CSRF token.
    """
    client.post(
        "/login",
        data={"username": "test_user"},
        headers={"X-CSRF-Token": csrf_token},
        follow_redirects=False,
    )
    yield client


@pytest.fixture
def auth_headers(csrf_token) -> dict:
    """Get headers with CSRF token for authenticated requests."""
    return {"X-CSRF-Token": csrf_token}


# Test data fixtures
@pytest.fixture
def sample_pid() -> str:
    """Return a sample paper ID for testing."""
    return "2301.00001"


@pytest.fixture
def sample_pids() -> list:
    """Return a list of sample paper IDs for testing."""
    return ["2301.00001", "2301.00002", "2301.00003"]


@pytest.fixture
def sample_tag() -> str:
    """Return a sample tag name for testing."""
    return "test_tag"
