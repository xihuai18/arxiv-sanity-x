"""Live browser checks for summary page behavior.

These tests exercise real page scripts against a running localhost instance.
They are intentionally separate from the lightweight source-contract tests.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid

import pytest

from tests.service_detection import get_vars_config, requires_web_server


def _has_playwright_cli() -> bool:
    return shutil.which("playwright-cli") is not None


requires_playwright_cli = pytest.mark.skipif(not _has_playwright_cli(), reason="playwright-cli not available")


def _run_playwright(session: str, *args: str, timeout: int = 120) -> str:
    result = subprocess.run(
        ["playwright-cli", f"-s={session}", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(
            "playwright-cli command failed:\n"
            f"cmd={['playwright-cli', f'-s={session}', *args]}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def _extract_result_payload(stdout: str) -> dict:
    marker = "### Result"
    idx = stdout.find(marker)
    if idx < 0:
        pytest.fail(f"No result payload found in playwright output:\n{stdout}")

    lines = stdout[idx + len(marker) :].splitlines()
    payload_lines: list[str] = []
    started = False
    for line in lines:
        if line.startswith("### "):
            break
        if not started and not line.strip():
            continue
        started = True
        payload_lines.append(line)

    payload_text = "\n".join(payload_lines).strip()
    if not payload_text:
        pytest.fail(f"Empty result payload in playwright output:\n{stdout}")
    return json.loads(payload_text)


@requires_web_server
@requires_playwright_cli
class TestSummaryFrontendBehaviorLive:
    @pytest.fixture
    def base_url(self) -> str:
        config = get_vars_config()
        port = config.get("SERVE_PORT", 55555)
        return f"http://localhost:{port}"

    def test_resolved_model_sse_refreshes_current_summary_view(self, base_url: str):
        session = f"summary-fallback-{uuid.uuid4().hex[:8]}"
        requested_model = "requested-model"
        resolved_model = "fallback-model"

        try:
            _run_playwright(session, "open", f"{base_url}/")
            pid_stdout = _run_playwright(
                session,
                "run-code",
                "async page => { await page.waitForLoadState('networkidle'); return await page.evaluate(() => { const el = document.querySelector('a[href*=\"/summary?pid=\"]'); return { href: el ? el.getAttribute('href') || '' : '' }; }); }",
            )
            pid_payload = _extract_result_payload(pid_stdout)
            match = re.search(r"/summary\?pid=([^\"'&<>\s]+)", str(pid_payload.get("href") or ""))
            if not match:
                pytest.skip("Homepage does not expose a public summary pid in this environment")
            pid = match.group(1)

            script = f"""async page => {{
                const baseUrl = {json.dumps(base_url)};
                const pid = {json.dumps(pid)};
                const requestedModel = {json.dumps(requested_model)};
                const resolvedModel = {json.dumps(resolved_model)};

                await page.waitForLoadState('networkidle');
                const loginStatus = await page.evaluate(async () => {{
                    const token = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
                    const body = new URLSearchParams({{ username: 'live_test_user' }});
                    const resp = await fetch('/login', {{
                        method: 'POST',
                        headers: {{
                            'X-CSRF-Token': token,
                            'Content-Type': 'application/x-www-form-urlencoded',
                        }},
                        body: body.toString(),
                        credentials: 'same-origin',
                    }});
                    return resp.status;
                }});

                await page.goto(baseUrl + '/summary?pid=' + encodeURIComponent(pid));
                await page.waitForFunction(() => Boolean(window.summaryApp && window.summaryApp.pid));

                const result = await page.evaluate(async (args) => {{
                    const app = window.summaryApp;
                    const common = window.ArxivSanityCommon;
                    const calls = [];
                    app.selectedModel = args.resolvedModel;
                    app.getCurrentModel = () => args.resolvedModel;
                    app.loadSummary = function(loadPid, options) {{
                        calls.push({{
                            pid: String(loadPid || ''),
                            model: String((options && options.model) || ''),
                            cache_only: Boolean(options && options.cache_only),
                        }});
                    }};

                    common.dispatchUserEvent({{
                        type: 'summary_status',
                        pid: app.pid,
                        model: args.requestedModel,
                        status: 'ok',
                        resolved_model: args.resolvedModel,
                    }});

                    for (let i = 0; i < 20; i += 1) {{
                        if (calls.length > 0) break;
                        await new Promise(resolve => setTimeout(resolve, 50));
                    }}

                    return {{
                        calls,
                        loginStatus: args.loginStatus,
                        cachedRequested: app.summaryStatusCacheByModel[args.requestedModel] || null,
                        cachedResolved: app.summaryStatusCacheByModel[args.resolvedModel] || null,
                    }};
                }}, {{ requestedModel, resolvedModel, loginStatus }});

                return result;
            }}"""

            _run_playwright(session, "goto", f"{base_url}/profile")
            stdout = _run_playwright(session, "run-code", script, timeout=180)
            payload = _extract_result_payload(stdout)

            assert payload["loginStatus"] in {200, 302, 303}
            assert payload["calls"] == [
                {
                    "pid": pid,
                    "model": resolved_model,
                    "cache_only": True,
                }
            ]
            assert payload["cachedRequested"]["resolved_model"] == resolved_model
            assert payload["cachedResolved"]["status"] == "ok"
        finally:
            subprocess.run(
                ["playwright-cli", f"-s={session}", "close"],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
