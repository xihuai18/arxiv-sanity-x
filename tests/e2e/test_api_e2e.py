#!/usr/bin/env python3
"""End-to-end API tests.

These tests run against a live server using the requests library.
They test the full stack including network communication.

Usage:
    cd /path/to/arxiv-sanity
    conda activate sanity
    python tests/e2e/test_api_e2e.py [--host HOST] [--port PORT]
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import requests


class APITester:
    """API tester class for end-to-end testing."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.csrf_token: str | None = None
        self.results: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}

    def _get_csrf_token(self) -> str:
        """Fetch CSRF token from homepage."""
        r = self.session.get(f"{self.base_url}/")
        match = re.search(r'csrf-token" content="([^"]+)"', r.text)
        if match:
            self.csrf_token = match.group(1)
            return self.csrf_token
        raise Exception("Failed to get CSRF token")

    def _log(self, test_name: str, status: str, message: str = "") -> None:
        """Log test result."""
        icons = {"pass": "✅", "fail": "❌", "skip": "⏭️", "info": "ℹ️"}
        icon = icons.get(status, "")
        if message:
            print(f"{icon} {test_name}: {message}")
        else:
            print(f"{icon} {test_name}")

        if status == "pass":
            self.results["passed"] += 1
        elif status == "fail":
            self.results["failed"] += 1
        elif status == "skip":
            self.results["skipped"] += 1

    def _post(self, endpoint: str, json_data: dict, timeout: int = 30) -> dict:
        """Send POST request."""
        headers = {"X-CSRF-Token": self.csrf_token} if self.csrf_token else {}
        r = self.session.post(
            f"{self.base_url}{endpoint}",
            json=json_data,
            headers=headers,
            timeout=timeout,
        )
        return r.json()

    def _get(self, endpoint: str, params: dict | None = None, timeout: int = 10) -> dict:
        """Send GET request."""
        r = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=timeout)
        return r.json()

    def _get_page(self, endpoint: str) -> int:
        """Get page status code."""
        r = self.session.get(f"{self.base_url}{endpoint}")
        return r.status_code

    def login(self, username: str = "testuser") -> None:
        """Login as user."""
        self._get_csrf_token()
        self.session.post(
            f"{self.base_url}/login",
            data={"username": username},
            headers={"X-CSRF-Token": self.csrf_token},
        )

    # ==================== Page Tests ====================

    def test_pages(self) -> None:
        """Test page access."""
        print("\n" + "=" * 50)
        print("📄 Page Access Tests")
        print("=" * 50)

        pages = [
            ("/", "Home"),
            ("/about", "About"),
            ("/stats", "Stats"),
            ("/profile", "Profile"),
            ("/readinglist", "Reading List"),
            ("/inspect?pid=2507.18405", "Paper Detail"),
            ("/summary?pid=2507.18405", "Paper Summary"),
        ]

        for path, name in pages:
            try:
                status = self._get_page(path)
                if status == 200:
                    self._log(f"{name} ({path})", "pass", f"HTTP {status}")
                else:
                    self._log(f"{name} ({path})", "fail", f"HTTP {status}")
            except Exception as e:
                self._log(f"{name} ({path})", "fail", str(e))

    # ==================== Public API Tests ====================

    def test_public_apis(self) -> None:
        """Test public APIs (no login required)."""
        print("\n" + "=" * 50)
        print("🌐 Public API Tests (No Login Required)")
        print("=" * 50)

        # Keyword search
        try:
            result = self._post(
                "/api/keyword_search",
                {"keyword": "transformer", "time_delta": 365, "limit": 3},
            )
            if result.get("success"):
                pids = result.get("pids", [])
                self._log("Keyword Search API", "pass", f"Found {len(pids)} papers")
            else:
                self._log("Keyword Search API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Keyword Search API", "fail", str(e))

        # LLM models list
        try:
            result = self._get("/api/llm_models")
            if result.get("success"):
                models = result.get("models", [])
                model_names = [m.get("id") for m in models]
                self._log("LLM Models API", "pass", f"Available models: {', '.join(model_names)}")
            else:
                self._log("LLM Models API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("LLM Models API", "fail", str(e))

        # Queue stats
        try:
            result = self._get("/api/queue_stats")
            if result.get("success"):
                queued = result.get("queued", 0)
                running = result.get("running", 0)
                self._log("Queue Stats API", "pass", f"Queued: {queued}, Running: {running}")
            else:
                self._log("Queue Stats API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Queue Stats API", "fail", str(e))

        # Check paper summaries
        try:
            result = self._get("/api/check_paper_summaries", {"pid": "2507.18405"})
            if result.get("success"):
                models = result.get("available_models", [])
                self._log(
                    "Check Paper Summaries API", "pass", f"Available summary models: {models if models else 'None'}"
                )
            else:
                self._log("Check Paper Summaries API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Check Paper Summaries API", "fail", str(e))

    # ==================== Authenticated API Tests ====================

    def test_authenticated_apis(self) -> None:
        """Test APIs that require login."""
        print("\n" + "=" * 50)
        print("🔐 Authenticated API Tests")
        print("=" * 50)

        # User state
        try:
            result = self._get("/api/user_state")
            if result.get("success"):
                tags = result.get("tags", [])
                keys = result.get("keys", [])
                self._log("User State API", "pass", f"Tags: {len(tags)}, Keywords: {len(keys)}")
            else:
                self._log("User State API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("User State API", "fail", str(e))

        # Paper titles
        try:
            result = self._post("/api/paper_titles", {"pids": ["2507.18405", "2512.22212"]})
            if result.get("success"):
                titles = result.get("titles", {})
                self._log("Paper Titles API", "pass", f"Got {len(titles)} titles")
            else:
                self._log("Paper Titles API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Paper Titles API", "fail", str(e))

        # Tag members
        try:
            result = self._get("/api/tag_members", {"tag": "test_tag"})
            if result.get("success"):
                members = result.get("members", [])
                self._log("Tag Members API", "pass", f"Members: {len(members)}")
            else:
                self._log("Tag Members API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Tag Members API", "fail", str(e))

        # Reading list
        try:
            result = self._get("/api/readinglist/list")
            if result.get("success"):
                items = result.get("items", [])
                self._log("Reading List API", "pass", f"Total {len(items)} items")
            else:
                self._log("Reading List API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Reading List API", "fail", str(e))

        # Summary status
        try:
            result = self._post("/api/summary_status", {"pids": ["2507.18405"]})
            if result.get("success"):
                self._log("Summary Status API", "pass")
            else:
                self._log("Summary Status API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Summary Status API", "fail", str(e))

    # ==================== Tag Operation Tests ====================

    def test_tag_operations(self) -> None:
        """Test tag operations."""
        print("\n" + "=" * 50)
        print("🏷️ Tag Operation Tests")
        print("=" * 50)

        test_tag = "api_test_tag"
        test_pid = "2507.18405"

        # Add tag
        try:
            result = self._post("/api/tag_feedback", {"pid": test_pid, "tag": test_tag, "label": 1})
            if result.get("success"):
                self._log(f"Add tag '{test_tag}'", "pass")
            else:
                self._log(f"Add tag '{test_tag}'", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log(f"Add tag '{test_tag}'", "fail", str(e))

        # Tag search
        try:
            result = self._post(
                "/api/tag_search",
                {"tag_name": test_tag, "user": "testuser", "time_delta": 365, "limit": 10},
            )
            if result.get("success"):
                pids = result.get("pids", [])
                self._log("Tag Search API", "pass", f"Found {len(pids)} related papers")
            else:
                self._log("Tag Search API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Tag Search API", "fail", str(e))

        # Multi-tag search
        try:
            result = self._post(
                "/api/tags_search",
                {"tags": [test_tag], "user": "testuser", "time_delta": 365, "limit": 10},
            )
            if result.get("success"):
                pids = result.get("pids", [])
                self._log("Multi-tag Search API", "pass", f"Found {len(pids)} related papers")
            else:
                self._log("Multi-tag Search API", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Multi-tag Search API", "fail", str(e))

        # Remove tag
        try:
            result = self._post("/api/tag_feedback", {"pid": test_pid, "tag": test_tag, "label": 0})
            if result.get("success"):
                self._log(f"Remove tag '{test_tag}'", "pass")
            else:
                self._log(f"Remove tag '{test_tag}'", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log(f"Remove tag '{test_tag}'", "fail", str(e))

    # ==================== Reading List Operation Tests ====================

    def test_readinglist_operations(self) -> None:
        """Test reading list operations."""
        print("\n" + "=" * 50)
        print("📚 Reading List Operation Tests")
        print("=" * 50)

        test_pid = "2507.18405"

        # Add to reading list
        try:
            result = self._post("/api/readinglist/add", {"pid": test_pid})
            if result.get("success"):
                self._log("Add to Reading List", "pass")
            else:
                self._log("Add to Reading List", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Add to Reading List", "fail", str(e))

        # Verify reading list
        try:
            result = self._get("/api/readinglist/list")
            if result.get("success"):
                items = result.get("items", [])
                has_pid = any(item.get("pid") == test_pid for item in items)
                if has_pid:
                    self._log("Verify Reading List", "pass", f"Paper {test_pid} is in list")
                else:
                    self._log("Verify Reading List", "fail", f"Paper {test_pid} not in list")
            else:
                self._log("Verify Reading List", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Verify Reading List", "fail", str(e))

        # Remove from reading list
        try:
            result = self._post("/api/readinglist/remove", {"pid": test_pid})
            if result.get("success"):
                self._log("Remove from Reading List", "pass")
            else:
                self._log("Remove from Reading List", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Remove from Reading List", "fail", str(e))

    # ==================== Summary Function Tests ====================

    def test_summary_apis(self) -> None:
        """Test summary functions."""
        print("\n" + "=" * 50)
        print("📝 Summary Function Tests")
        print("=" * 50)

        test_pid = "2507.18405"

        # Get available models
        try:
            result = self._get("/api/llm_models")
            if not result.get("success") or not result.get("models"):
                self._log("Summary Test", "skip", "No available LLM models")
                return
            model = result["models"][0]["id"]
            self._log("Select Model", "info", model)
        except Exception as e:
            self._log("Get Model List", "fail", str(e))
            return

        # Trigger summary generation
        try:
            result = self._post(
                "/api/trigger_paper_summary",
                {"pid": test_pid, "model": model},
                timeout=10,
            )
            if result.get("success"):
                task_id = result.get("task_id")
                if task_id:
                    self._log("Trigger Summary Generation", "pass", f"Task ID: {task_id}")
                else:
                    self._log("Trigger Summary Generation", "pass", "Cached or processing")
            else:
                self._log("Trigger Summary Generation", "fail", result.get("error", "Unknown error"))
        except Exception as e:
            self._log("Trigger Summary Generation", "fail", str(e))

        # Get paper summary (may need to wait)
        print("    Waiting for summary generation...")
        max_wait = 120  # Max wait 120 seconds
        start_time = time.time()
        summary_ready = False

        while time.time() - start_time < max_wait:
            try:
                result = self._post(
                    "/api/get_paper_summary",
                    {"pid": test_pid, "model": model},
                    timeout=10,
                )
                if result.get("success"):
                    summary = result.get("summary") or result.get("summary_content", "")
                    if summary:
                        summary_preview = summary[:100] + "..." if len(summary) > 100 else summary
                        self._log("Get Paper Summary", "pass", f"Summary length: {len(summary)} chars")
                        print(f"    Summary preview: {summary_preview}")
                        summary_ready = True
                        break
                    else:
                        time.sleep(5)
                else:
                    error = result.get("error", "")
                    if "generating" in error.lower() or "processing" in error.lower():
                        time.sleep(5)
                    else:
                        self._log("Get Paper Summary", "fail", error)
                        break
            except requests.exceptions.Timeout:
                time.sleep(5)
            except Exception as e:
                self._log("Get Paper Summary", "fail", str(e))
                break

        if not summary_ready and time.time() - start_time >= max_wait:
            self._log("Get Paper Summary", "skip", f"Timeout ({max_wait}s)")

    # ==================== Run All Tests ====================

    def run_all_tests(self, include_summary: bool = True) -> bool:
        """Run all tests."""
        print("\n" + "=" * 60)
        print("🧪 arxiv-sanity API Tests")
        print(f"   Server: {self.base_url}")
        print("=" * 60)

        # Page tests
        self.test_pages()

        # Public API tests
        self.test_public_apis()

        # Login
        print("\n" + "-" * 50)
        print("🔑 Logging in test user...")
        self.login("testuser")
        print("   Logged in as: testuser")

        # Authenticated API tests
        self.test_authenticated_apis()

        # Tag operation tests
        self.test_tag_operations()

        # Reading list operation tests
        self.test_readinglist_operations()

        # Summary function tests
        if include_summary:
            self.test_summary_apis()
        else:
            print("\n" + "=" * 50)
            print("📝 Summary Function Tests")
            print("=" * 50)
            self._log("Summary Test", "skip", "Skipped (--no-summary flag)")

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        print(f"   ✅ Passed: {self.results['passed']}")
        print(f"   ❌ Failed: {self.results['failed']}")
        print(f"   ⏭️ Skipped: {self.results['skipped']}")
        print("=" * 60)

        return self.results["failed"] == 0


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="arxiv-sanity API Test Script")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=55555, help="Server port (default: 55555)")
    parser.add_argument("--no-summary", action="store_true", help="Skip summary tests")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    tester = APITester(base_url)
    success = tester.run_all_tests(include_summary=not args.no_summary)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
