"""Unit tests for upload utility functions.

Tests upload utility functions including PID validation, generation,
file hashing, and path utilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


class TestIsUploadPid:
    """Tests for is_upload_pid function."""

    def test_valid_upload_pid(self):
        """Test that valid upload PIDs are recognized."""
        from backend.utils.upload_utils import is_upload_pid

        assert is_upload_pid("up_V1StGXR8_Z5j") is True
        assert is_upload_pid("up_abcdefghijkl") is True
        assert is_upload_pid("up_123456789012") is True

    def test_invalid_upload_pid(self):
        """Test that invalid PIDs are rejected."""
        from backend.utils.upload_utils import is_upload_pid

        assert is_upload_pid("") is False
        assert is_upload_pid(None) is False
        assert is_upload_pid("2301.00001") is False
        assert is_upload_pid("arxiv_123") is False
        assert is_upload_pid("upload_123") is False

    def test_prefix_only(self):
        """Test that prefix alone is recognized as upload PID."""
        from backend.utils.upload_utils import is_upload_pid

        # Prefix only should still return True (starts with up_)
        assert is_upload_pid("up_") is True
        assert is_upload_pid("up_short") is True


class TestValidateUploadPid:
    """Tests for validate_upload_pid function."""

    def test_valid_format(self):
        """Test that valid format PIDs pass validation."""
        from backend.utils.upload_utils import validate_upload_pid

        assert validate_upload_pid("up_V1StGXR8_Z5j") is True
        assert validate_upload_pid("up_abcdefghijkl") is True
        assert validate_upload_pid("up_ABCDEFGHIJKL") is True
        assert validate_upload_pid("up_123456789012") is True
        assert validate_upload_pid("up_abc-def_12AB") is True

    def test_invalid_format(self):
        """Test that invalid format PIDs fail validation."""
        from backend.utils.upload_utils import validate_upload_pid

        assert validate_upload_pid("") is False
        assert validate_upload_pid(None) is False
        assert validate_upload_pid("up_") is False
        assert validate_upload_pid("up_short") is False
        assert validate_upload_pid("up_toolongstring123") is False
        assert validate_upload_pid("2301.00001") is False
        assert validate_upload_pid("up_invalid!char") is False


class TestGenerateUploadPid:
    """Tests for generate_upload_pid function."""

    def test_generates_valid_pid(self):
        """Test that generated PIDs are valid."""
        from backend.utils.upload_utils import generate_upload_pid, validate_upload_pid

        pid = generate_upload_pid()
        assert pid.startswith("up_")
        assert len(pid) == 15
        assert validate_upload_pid(pid) is True

    def test_generates_unique_pids(self):
        """Test that generated PIDs are unique."""
        from backend.utils.upload_utils import generate_upload_pid

        pids = [generate_upload_pid() for _ in range(100)]
        assert len(set(pids)) == 100

    def test_fallback_without_nanoid(self):
        """Test fallback when nanoid is not available."""
        import sys

        # Temporarily remove nanoid from modules to test fallback
        nanoid_module = sys.modules.get("nanoid")
        sys.modules["nanoid"] = None

        try:
            # Force reimport to trigger fallback
            import importlib

            from backend.utils import upload_utils

            importlib.reload(upload_utils)

            pid = upload_utils.generate_upload_pid()
            assert pid.startswith("up_")
            assert len(pid) == 15
            assert upload_utils.validate_upload_pid(pid) is True
        finally:
            # Restore nanoid module
            if nanoid_module is not None:
                sys.modules["nanoid"] = nanoid_module
            elif "nanoid" in sys.modules:
                del sys.modules["nanoid"]


class TestComputeFileSha256:
    """Tests for compute_file_sha256 function."""

    def test_computes_hash(self):
        """Test that file hash is computed correctly."""
        from backend.utils.upload_utils import compute_file_sha256

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            path = Path(f.name)

        try:
            hash_value = compute_file_sha256(path)
            assert len(hash_value) == 64
            assert hash_value.isalnum()
        finally:
            path.unlink()

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        from backend.utils.upload_utils import compute_file_sha256

        content = b"identical content"
        paths = []

        for _ in range(2):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(content)
                f.flush()
                paths.append(Path(f.name))

        try:
            hashes = [compute_file_sha256(p) for p in paths]
            assert hashes[0] == hashes[1]
        finally:
            for p in paths:
                p.unlink()

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        from backend.utils.upload_utils import compute_file_sha256

        paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(f"content {i}".encode())
                f.flush()
                paths.append(Path(f.name))

        try:
            hashes = [compute_file_sha256(p) for p in paths]
            assert hashes[0] != hashes[1]
        finally:
            for p in paths:
                p.unlink()


class TestComputeBytesSha256:
    """Tests for compute_bytes_sha256 function."""

    def test_computes_hash(self):
        """Test that bytes hash is computed correctly."""
        from backend.utils.upload_utils import compute_bytes_sha256

        hash_value = compute_bytes_sha256(b"test content")
        assert len(hash_value) == 64
        assert hash_value.isalnum()

    def test_same_bytes_same_hash(self):
        """Test that same bytes produce same hash."""
        from backend.utils.upload_utils import compute_bytes_sha256

        data = b"identical content"
        assert compute_bytes_sha256(data) == compute_bytes_sha256(data)

    def test_different_bytes_different_hash(self):
        """Test that different bytes produce different hash."""
        from backend.utils.upload_utils import compute_bytes_sha256

        assert compute_bytes_sha256(b"content 1") != compute_bytes_sha256(b"content 2")

    def test_empty_bytes(self):
        """Test hash of empty bytes."""
        from backend.utils.upload_utils import compute_bytes_sha256

        hash_value = compute_bytes_sha256(b"")
        assert len(hash_value) == 64


class TestGetUploadPdfPath:
    """Tests for get_upload_pdf_path function."""

    def test_returns_correct_path(self):
        """Test that correct path is returned."""
        from backend.utils.upload_utils import get_upload_pdf_path

        path = get_upload_pdf_path("up_V1StGXR8_Z5j", "/data")
        assert path == Path("/data/uploads/up_V1StGXR8_Z5j/original.pdf")

    def test_handles_trailing_slash(self):
        """Test that trailing slash in data_dir is handled."""
        from backend.utils.upload_utils import get_upload_pdf_path

        path = get_upload_pdf_path("up_V1StGXR8_Z5j", "/data/")
        assert "uploads" in str(path)
        assert "original.pdf" in str(path)


class TestGetUploadDir:
    """Tests for get_upload_dir function."""

    def test_returns_correct_path(self):
        """Test that correct directory path is returned."""
        from backend.utils.upload_utils import get_upload_dir

        path = get_upload_dir("up_V1StGXR8_Z5j", "/data")
        assert path == Path("/data/uploads/up_V1StGXR8_Z5j")


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_normal_filename(self):
        """Test that normal filenames are preserved."""
        from backend.utils.upload_utils import sanitize_filename

        assert sanitize_filename("paper.pdf") == "paper.pdf"
        assert sanitize_filename("my_paper_2024.pdf") == "my_paper_2024.pdf"

    def test_removes_path_separators(self):
        """Test that path separators are replaced."""
        from backend.utils.upload_utils import sanitize_filename

        assert "/" not in sanitize_filename("path/to/file.pdf")
        assert "\\" not in sanitize_filename("path\\to\\file.pdf")

    def test_removes_parent_directory_references(self):
        """Test that .. is replaced."""
        from backend.utils.upload_utils import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result

    def test_removes_null_bytes(self):
        """Test that null bytes are removed."""
        from backend.utils.upload_utils import sanitize_filename

        result = sanitize_filename("file\x00name.pdf")
        assert "\x00" not in result

    def test_strips_whitespace_and_dots(self):
        """Test that leading/trailing whitespace and dots are stripped."""
        from backend.utils.upload_utils import sanitize_filename

        assert sanitize_filename("  file.pdf  ") == "file.pdf"
        # Note: .. is replaced with _ first, then dots are stripped
        result = sanitize_filename("...file.pdf...")
        assert "file.pdf" in result

    def test_truncates_long_filename(self):
        """Test that long filenames are truncated."""
        from backend.utils.upload_utils import sanitize_filename

        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_empty_filename(self):
        """Test that empty filename returns 'unnamed'."""
        from backend.utils.upload_utils import sanitize_filename

        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"
        # Note: ... becomes _ after .. replacement, not empty
        result = sanitize_filename("...")
        assert result  # Should not be empty

    def test_preserves_extension_on_truncation(self):
        """Test that extension is preserved when truncating."""
        from backend.utils.upload_utils import sanitize_filename

        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert result.endswith(".pdf")
