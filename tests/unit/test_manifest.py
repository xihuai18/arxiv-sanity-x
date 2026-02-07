"""Unit tests for static asset manifest utilities."""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock


class TestManifestUtils:
    """Tests for manifest utility functions."""

    def test_static_url_fallback_when_manifest_missing(self):
        """Test that static_url returns original path when manifest is missing."""
        from backend.utils import manifest

        manifest.clear_manifest_cache()
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "no_such_manifest.json")
            with mock.patch.object(manifest, "_get_manifest_path", return_value=missing):
                assert manifest.static_url("dist/main.css") == "dist/main.css"
                assert manifest.static_url("dist/common_utils.js") == "dist/common_utils.js"
                assert manifest.static_url("style.css") == "style.css"

    def test_static_url_uses_manifest_mapping(self):
        """Test that static_url uses manifest mapping when available."""
        from backend.utils import manifest

        manifest.clear_manifest_cache()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"main.css": "main-ABCDEF1234.css", "common_utils.js": "common_utils-XYZ98765.js"}, f)

            # The resolver validates that mapped files exist on disk (guards against stale manifests).
            # Create placeholder files to simulate a real dist/ directory.
            open(os.path.join(td, "main-ABCDEF1234.css"), "a", encoding="utf-8").close()
            open(os.path.join(td, "common_utils-XYZ98765.js"), "a", encoding="utf-8").close()

            with mock.patch.object(manifest, "_get_manifest_path", return_value=path):
                assert manifest.static_url("dist/main.css") == "dist/main-ABCDEF1234.css"
                assert manifest.static_url("dist/common_utils.js") == "dist/common_utils-XYZ98765.js"

    def test_static_url_preserves_unmapped_files(self):
        """Test that unmapped files are returned as-is."""
        from backend.utils import manifest

        manifest.clear_manifest_cache()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"main.css": "main-ABCDEF1234.css"}, f)

            with mock.patch.object(manifest, "_get_manifest_path", return_value=path):
                # Unmapped file should be returned as-is
                assert manifest.static_url("dist/other.js") == "dist/other.js"

    def test_static_url_falls_back_when_manifest_points_to_missing_file(self):
        """Test that we fall back when manifest entry points to a non-existent file on disk."""
        from backend.utils import manifest

        manifest.clear_manifest_cache()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"main.css": "main-STALE123.css"}, f)

            # Create an alternative hashed file that matches the fallback glob.
            open(os.path.join(td, "main-REAL999.css"), "a", encoding="utf-8").close()

            with mock.patch.object(manifest, "_get_manifest_path", return_value=path):
                assert manifest.static_url("dist/main.css") == "dist/main-REAL999.css"
                assert manifest.static_url("dist/main.css") != "dist/main-STALE123.css"

    def test_clear_manifest_cache(self):
        """Test that clear_manifest_cache clears the cache."""
        from backend.utils import manifest

        manifest.clear_manifest_cache()
        # Should not raise any errors
        manifest.clear_manifest_cache()
