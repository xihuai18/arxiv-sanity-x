"""Test package for arxiv-sanity.

This package contains tests organized into four categories:

- **unit/**: Unit tests for individual functions and classes
  - test_cache.py: Cache utility tests
  - test_data_service.py: Data service tests
  - test_manifest.py: Static asset manifest tests
  - test_render_service.py: Render service tests
  - test_schemas.py: Pydantic schema tests
  - test_search_service.py: Search service function tests
  - test_sse.py: SSE event utility tests
  - test_summary_service.py: Summary service function tests
  - test_user_service.py: User service tests
  - test_validation.py: Validation function tests

- **integration/**: Integration tests using Flask test client
  - test_app.py: Application setup, blueprints, routes
  - test_api_auth.py: Authentication and CSRF protection
  - test_api_login.py: Login/logout API tests
  - test_api_papers.py: Paper-related APIs
  - test_api_readinglist.py: Reading list APIs
  - test_api_search.py: Search APIs (keyword/tag search)
  - test_api_summary.py: Summary APIs (status/clear/check)
  - test_api_tags.py: Tag APIs
  - test_api_tag_management.py: Tag/keyword management APIs
  - test_api_user.py: User APIs

- **live/**: Live service tests (auto-skip if service unavailable)
  - test_llm_service.py: LLM/LiteLLM service tests
  - test_web_server.py: Web server tests
  - test_data_service.py: Data service tests (requires data files)

- **e2e/**: End-to-end tests against a live server
  - test_api_e2e.py: Full API test suite using requests

Running tests:
    # Run all tests (unit + integration + live)
    pytest tests/

    # Run only unit tests
    pytest tests/unit/

    # Run only integration tests
    pytest tests/integration/

    # Run live service tests (auto-skips unavailable services)
    pytest tests/live/

    # Run e2e tests (requires running server)
    python tests/e2e/test_api_e2e.py --host localhost --port 55555

These tests are intentionally lightweight and avoid heavy ML/data warmups.
"""
