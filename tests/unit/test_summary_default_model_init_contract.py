"""Contract checks for summary page default-model initialization.

These checks guard against async state timing regressions where
`selectInitialModel` reads stale `defaultModel` / `models` values.
"""

from __future__ import annotations

from pathlib import Path


def _read_summary_js() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    js_path = repo_root / "static" / "paper_summary.js"
    return js_path.read_text(encoding="utf-8", errors="ignore")


def test_load_models_updates_model_state_synchronously():
    text = _read_summary_js()

    start = text.find("summaryApp.loadModels = async function")
    assert start != -1
    end = text.find("summaryApp.selectInitialModel", start)
    assert end != -1
    block = text[start:end]

    assert "this.setStateSync({ models, modelsError: null })" in block


def test_init_summary_app_sets_default_model_synchronously():
    text = _read_summary_js()

    start = text.find("async function initSummaryApp()")
    assert start != -1
    end = text.find("await summaryApp.loadModels()", start)
    assert end != -1
    block = text[start:end]

    assert "summaryApp.setStateSync({" in block
    assert "selectedModel: initialDefaultModel" in block
    assert "defaultModel: initialDefaultModel" in block
