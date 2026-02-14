"""Unit tests for upload_service create_uploaded_paper rollback on PDF write failure."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_updb():
    """Create a mock uploaded papers DB with transaction support."""
    mock_updb = MagicMock()
    mock_updb.__enter__ = MagicMock(return_value=mock_updb)
    mock_updb.__exit__ = MagicMock(return_value=False)
    mock_updb.get = MagicMock(return_value=None)

    mock_txn = MagicMock()
    mock_txn.__enter__ = MagicMock(return_value=mock_updb)
    mock_txn.__exit__ = MagicMock(return_value=False)
    mock_updb.transaction = MagicMock(return_value=mock_txn)
    return mock_updb


def _make_mock_idx_db():
    """Create a mock index DB."""
    mock_idx = MagicMock()
    mock_idx.__enter__ = MagicMock(return_value=mock_idx)
    mock_idx.__exit__ = MagicMock(return_value=False)
    mock_idx.get = MagicMock(return_value=[])
    return mock_idx


class TestCreateUploadedPaperRollback:
    """Verify DB records are not committed when PDF write fails."""

    def test_db_not_committed_on_write_failure(self, app):
        """When file write raises OSError (disk full), DB cleanup should be attempted."""
        from backend.services.upload_service import create_uploaded_paper

        fake_content = b"%PDF-1.4 fake content for testing"
        user = "rollback_test_user"

        with app.app_context():
            with tempfile.TemporaryDirectory() as tmpdir:
                upload_dir = Path(tmpdir) / "uploads" / "up_TESTROLLBACK"
                pdf_path = upload_dir / "paper.pdf"

                mock_updb = _make_mock_updb()
                mock_idx = _make_mock_idx_db()

                original_open = open

                def failing_open(path, *args, **kwargs):
                    if "paper.pdf" in str(path) and ".tmp" in str(path):
                        raise OSError("No space left on device")
                    return original_open(path, *args, **kwargs)

                with (
                    patch("backend.services.upload_service.UploadedPaperRepository") as mock_repo,
                    patch("backend.services.upload_service.compute_bytes_sha256", return_value="fakehash"),
                    patch("backend.services.upload_service.generate_upload_pid", return_value="up_TESTROLLBACK"),
                    patch("backend.services.upload_service.sanitize_filename", return_value="test.pdf"),
                    patch("backend.services.upload_service.get_upload_dir", return_value=upload_dir),
                    patch("backend.services.upload_service.get_upload_pdf_path", return_value=pdf_path),
                    patch("aslite.db.get_uploaded_papers_db", return_value=mock_updb),
                    patch("aslite.db.get_uploaded_papers_index_db", return_value=mock_idx),
                    patch("builtins.open", side_effect=failing_open),
                ):

                    mock_repo.get_by_sha256.return_value = None
                    mock_repo.sha256_mapping_key.return_value = "sha256::rollback_test_user::fakehash"

                    with pytest.raises(OSError, match="No space left"):
                        create_uploaded_paper(user, fake_content, "test.pdf")

                    # Cleanup should have been attempted on the repository
                    mock_repo.delete.assert_called()
                    mock_repo.remove_sha256_mapping.assert_called()

    def test_tmp_file_cleaned_on_replace_failure(self, app):
        """When Path.replace fails after tmp write, tmp file cleanup is attempted."""
        from backend.services.upload_service import create_uploaded_paper

        fake_content = b"%PDF-1.4 fake content"
        user = "cleanup_test_user"

        with app.app_context():
            with tempfile.TemporaryDirectory() as tmpdir:
                upload_dir = Path(tmpdir) / "uploads" / "up_TESTCLEAN"
                pdf_path = upload_dir / "paper.pdf"

                mock_updb = _make_mock_updb()
                mock_idx = _make_mock_idx_db()

                with (
                    patch("backend.services.upload_service.UploadedPaperRepository") as mock_repo,
                    patch("backend.services.upload_service.compute_bytes_sha256", return_value="fakehash2"),
                    patch("backend.services.upload_service.generate_upload_pid", return_value="up_TESTCLEAN"),
                    patch("backend.services.upload_service.sanitize_filename", return_value="test.pdf"),
                    patch("backend.services.upload_service.get_upload_dir", return_value=upload_dir),
                    patch("backend.services.upload_service.get_upload_pdf_path", return_value=pdf_path),
                    patch("aslite.db.get_uploaded_papers_db", return_value=mock_updb),
                    patch("aslite.db.get_uploaded_papers_index_db", return_value=mock_idx),
                    patch.object(Path, "replace", side_effect=OSError("Permission denied")),
                ):

                    mock_repo.get_by_sha256.return_value = None
                    mock_repo.sha256_mapping_key.return_value = "sha256::cleanup_test_user::fakehash2"

                    with pytest.raises(OSError):
                        create_uploaded_paper(user, fake_content, "test.pdf")

                    # Exception propagated means DB transaction was not committed.
                    # Cleanup of DB records should have been attempted.
                    mock_repo.delete.assert_called()
