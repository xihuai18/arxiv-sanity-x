"""Service availability detection utilities for conditional tests."""

from __future__ import annotations

import socket
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).parent.parent.absolute()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _opencode_auth() -> tuple[str, str] | None:
    config = get_vars_config()
    username = str(config.get("OPENCODE_USERNAME", "") or "")
    password = str(config.get("OPENCODE_PASSWORD", "") or "")
    if not username and not password:
        return None
    return username, password


def check_http_service(url: str, timeout: float = 2.0, auth: tuple[str, str] | None = None) -> bool:
    try:
        import requests

        resp = requests.get(url, timeout=timeout, auth=auth)
        return resp.status_code < 500
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_vars_config() -> dict:
    try:
        from config import settings

        return {
            "DATA_DIR": str(settings.data_dir),
            "EMBED_PORT": settings.embedding.port,
            "OPENCODE_PORT": settings.opencode.port,
            "OPENCODE_BASE_URL": settings.opencode.resolved_base_url,
            "OPENCODE_USERNAME": settings.opencode.username,
            "OPENCODE_PASSWORD": settings.opencode.password,
            "MINERU_PORT": settings.mineru.port,
            "SERVE_PORT": settings.serve_port,
        }
    except ImportError:
        return {}


def is_embedding_service_available() -> bool:
    config = get_vars_config()
    port = config.get("EMBED_PORT", 54000)
    return check_port_open("localhost", port)


def is_opencode_service_available() -> bool:
    config = get_vars_config()
    base_url = str(config.get("OPENCODE_BASE_URL", "http://127.0.0.1:53000") or "").rstrip("/")
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    port = parsed.port or int(config.get("OPENCODE_PORT", 53000))
    if hostname in {"localhost", "127.0.0.1"} and port and not check_port_open(hostname, port):
        return False
    return check_http_service(f"{base_url}/global/health", auth=_opencode_auth())


def is_mineru_service_available() -> bool:
    config = get_vars_config()
    port = config.get("MINERU_PORT", 52000)
    return check_port_open("localhost", port)


def is_web_server_available() -> bool:
    config = get_vars_config()
    port = config.get("SERVE_PORT", 55555)
    if not check_port_open("localhost", port):
        return False
    return check_http_service(f"http://localhost:{port}/about")


def is_data_available() -> bool:
    config = get_vars_config()
    data_dir = config.get("DATA_DIR", "")
    if not data_dir:
        return False

    data_path = Path(data_dir)
    papers_db = data_path / "papers.db"
    features_file = data_path / "features.p"

    return papers_db.exists() or features_file.exists()


def get_available_text_models() -> list[str]:
    if not is_opencode_service_available():
        return []

    try:
        import requests

        config = get_vars_config()
        base_url = str(config.get("OPENCODE_BASE_URL", "http://127.0.0.1:53000") or "").rstrip("/")
        resp = requests.get(f"{base_url}/config/providers", timeout=5, auth=_opencode_auth())
        if resp.status_code != 200:
            return []
        payload = resp.json()
        providers = payload.get("providers") if isinstance(payload, dict) else []
        models: list[str] = []
        if isinstance(providers, list):
            for provider in providers:
                if not isinstance(provider, dict):
                    continue
                provider_id = str(provider.get("id") or "").strip()
                provider_models = provider.get("models")
                if not provider_id or not isinstance(provider_models, dict):
                    continue
                for model_key, model_info in provider_models.items():
                    model_id = str((model_info or {}).get("id") or model_key or "").strip()
                    if model_id:
                        models.append(f"{provider_id}/{model_id}")
        return list(dict.fromkeys(models))
    except Exception:
        return []


requires_embedding_service = pytest.mark.skipif(
    not is_embedding_service_available(), reason="Embedding service not available"
)

requires_opencode_service = pytest.mark.skipif(
    not is_opencode_service_available(), reason="OpenCode service not available"
)

requires_mineru_service = pytest.mark.skipif(not is_mineru_service_available(), reason="MinerU service not available")

requires_web_server = pytest.mark.skipif(not is_web_server_available(), reason="Web server not available")

requires_data = pytest.mark.skipif(not is_data_available(), reason="Data files not available")
