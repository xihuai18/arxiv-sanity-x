# Test Directory Structure

This directory contains all tests for the arxiv-sanity project, organized by type into three subdirectories.

## Directory Structure

```
tests/
├── conftest.py              # Shared pytest fixtures and configuration
├── service_detection.py     # Service detection utilities
├── __init__.py              # Package documentation
├── README.md                # This file
│
├── unit/                    # Unit tests
│   ├── test_cache.py        # Cache utility tests
│   ├── test_data_service.py # Data service tests
│   ├── test_manifest.py     # Static resource manifest tests
│   ├── test_render_service.py   # Render service tests
│   ├── test_schemas.py      # Pydantic schema tests
│   ├── test_search_service.py   # Search service function tests
│   ├── test_sse.py          # SSE event utility tests
│   ├── test_summary_service.py  # Summary service function tests
│   ├── test_user_service.py # User service tests
│   └── test_validation.py   # Validation function tests
│
├── integration/             # Integration tests (using Flask test_client)
│   ├── test_app.py          # App creation, blueprints, route tests
│   ├── test_api_auth.py     # Authentication and CSRF protection tests
│   ├── test_api_login.py    # Login/logout API tests
│   ├── test_api_papers.py   # Paper-related API tests
│   ├── test_api_readinglist.py  # Reading list API tests
│   ├── test_api_search.py   # Search API tests (keyword/tag search)
│   ├── test_api_summary.py  # Summary API tests (status/clear/check)
│   ├── test_api_tags.py     # Tag API tests
│   ├── test_api_tag_management.py  # Tag/keyword management API tests
│   └── test_api_user.py     # User API tests
│
├── live/                    # Live service tests (requires running services)
│   ├── test_llm_service.py  # LLM/LiteLLM service tests
│   ├── test_web_server.py   # Web server tests
│   └── test_data_service.py # Data service tests (requires data files)
│
└── e2e/                     # End-to-end tests (requires running server)
    └── test_api_e2e.py      # Complete API test suite
```

## Running Tests

### Data Directory Strategy

The test framework automatically selects the data directory:

1. **User specified**: If `ARXIV_SANITY_DATA_DIR` environment variable is set, use that directory
2. **Development environment**: If `data/papers.db` exists, use the real `data/` directory
3. **CI/New environment**: Otherwise use a temporary directory (`/tmp/arxiv_sanity_test_*`)

This ensures:
- Tests in development environment can use real data
- Tests in CI environment won't crash due to missing data (tests requiring data will be skipped)
- Real data directory won't be accidentally polluted

To force using a temporary directory for isolated testing:

```bash
ARXIV_SANITY_DATA_DIR=$(mktemp -d) pytest tests/unit/ tests/integration/ -v
```

### Run All Unit and Integration Tests

```bash
cd /path/to/arxiv-sanity
conda activate sanity
pytest tests/unit/ tests/integration/ -v
```

### Run Only Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Only Integration Tests

```bash
pytest tests/integration/ -v
```

### Run Specific Test File

```bash
pytest tests/unit/test_cache.py -v
pytest tests/integration/test_api_auth.py -v
```

### Run End-to-End Tests

End-to-end tests require a running server:

```bash
# Start the server first
./up.sh

# In another terminal, run the tests
python tests/e2e/test_api_e2e.py --host localhost --port 55555

# Skip summary tests (faster)
python tests/e2e/test_api_e2e.py --no-summary
```

## Test Type Descriptions

### Unit Tests (unit/)

- Test individual functions or classes
- Don't require Flask application context
- Fast execution
- High isolation

### Integration Tests (integration/)

- Test APIs using Flask test_client
- Test collaboration of multiple components
- Verify HTTP status codes, response formats
- Test authentication and CSRF protection

### Live Service Tests (live/)

- Test features that require running services
- Automatically detect if services are available
- Automatically skip tests when services are unavailable
- Includes Embedding, LLM, Web server tests

Running live service tests:

```bash
# Run all live service tests (automatically skips unavailable services)
pytest tests/live/ -v

# See which tests are skipped
pytest tests/live/ -v --tb=no
```

### End-to-End Tests (e2e/)

- Use requests library to send requests to real server
- Test complete user flows
- Includes login, tag operations, reading list, etc.
- Requires a running server

## Writing New Tests

### Using Shared Fixtures

`conftest.py` provides commonly used fixtures:

```python
def test_example(app, client, csrf_token, logged_in_client):
    # app: Flask application instance
    # client: Flask test_client
    # csrf_token: CSRF token string
    # logged_in_client: Logged-in test_client
    pass
```

### Test Naming Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<behavior>`

### Example

```python
class TestMyFeature:
    """Tests for my feature."""

    def test_feature_works(self, client):
        """Test that feature works correctly."""
        resp = client.get("/api/my_endpoint")
        assert resp.status_code == 200
```
