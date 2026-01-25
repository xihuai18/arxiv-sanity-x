"""Service availability detection utilities for conditional tests.

This module provides utilities to detect whether various services are available,
allowing tests to be skipped when required services are not running.
"""

from __future__ import annotations

import socket
import sys
from functools import lru_cache
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).parent.parent.absolute()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_http_service(url: str, timeout: float = 2.0) -> bool:
    """Check if an HTTP service is responding."""
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_vars_config() -> dict:
    """Load configuration from config.settings."""
    try:
        from config import settings

        return {
            "DATA_DIR": str(settings.data_dir),
            "EMBED_PORT": settings.embedding.port,
            "LITELLM_PORT": settings.litellm_port,
            "MINERU_PORT": settings.mineru.port,
            "SERVE_PORT": settings.serve_port,
            "LLM_BASE_URL": settings.llm.base_url,
        }
    except ImportError:
        return {}


def is_embedding_service_available() -> bool:
    """Check if the embedding service is available."""
    config = get_vars_config()
    port = config.get("EMBED_PORT", 54000)
    return check_port_open("localhost", port)


def is_litellm_service_available() -> bool:
    """Check if the LiteLLM service is available."""
    config = get_vars_config()
    port = config.get("LITELLM_PORT", 53000)
    if not check_port_open("localhost", port):
        return False
    # Also check the health endpoint
    return check_http_service(f"http://localhost:{port}/v1/models")


def is_mineru_service_available() -> bool:
    """Check if the MinerU service is available."""
    config = get_vars_config()
    port = config.get("MINERU_PORT", 52000)
    return check_port_open("localhost", port)


def is_web_server_available() -> bool:
    """Check if the main web server is available."""
    config = get_vars_config()
    port = config.get("SERVE_PORT", 55555)
    if not check_port_open("localhost", port):
        return False
    return check_http_service(f"http://localhost:{port}/about")


def is_data_available() -> bool:
    """Check if data files are available."""
    config = get_vars_config()
    data_dir = config.get("DATA_DIR", "")
    if not data_dir:
        return False

    data_path = Path(data_dir)
    # Check for essential data files
    papers_db = data_path / "papers.db"
    features_file = data_path / "features.p"

    return papers_db.exists() or features_file.exists()


def get_available_llm_models() -> list:
    """Get list of available LLM models from LiteLLM."""
    if not is_litellm_service_available():
        return []

    try:
        import requests

        config = get_vars_config()
        port = config.get("LITELLM_PORT", 53000)
        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("id") for m in data.get("data", [])]
    except Exception:
        pass
    return []


# Pytest skip markers
requires_embedding_service = pytest.mark.skipif(
    not is_embedding_service_available(), reason="Embedding service not available"
)

requires_litellm_service = pytest.mark.skipif(
    not is_litellm_service_available(), reason="LiteLLM service not available"
)

requires_mineru_service = pytest.mark.skipif(not is_mineru_service_available(), reason="MinerU service not available")

requires_web_server = pytest.mark.skipif(not is_web_server_available(), reason="Web server not available")

requires_data = pytest.mark.skipif(not is_data_available(), reason="Data files not available")
