from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_run_services_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "bin" / "run_services.py"
    spec = importlib.util.spec_from_file_location("test_run_services_module", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_web_readiness_env_flags(monkeypatch):
    module = _load_run_services_module()
    monkeypatch.delenv("ARXIV_SANITY_READY_REQUIRE_EMBEDDING", raising=False)
    monkeypatch.delenv("ARXIV_SANITY_READY_REQUIRE_MINERU", raising=False)

    module._configure_web_readiness_env(no_embed=True, no_mineru=False)
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_EMBEDDING") == "0"
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_MINERU") == "1"

    # setdefault semantics: existing operator overrides are preserved.
    module._configure_web_readiness_env(no_embed=False, no_mineru=True)
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_EMBEDDING") == "0"
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_MINERU") == "1"
