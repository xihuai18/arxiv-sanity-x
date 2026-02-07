"""Integration tests for uploaded papers API endpoints.

Tests the /api/upload_pdf and /api/uploaded_papers/* endpoints.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch


class TestUploadPdfApi:
    """Tests for POST /api/upload_pdf endpoint."""

    def test_upload_pdf_without_login_returns_401(self, client, csrf_token):
        """Test that upload without login returns 401."""
        data = {"file": (io.BytesIO(b"%PDF-1.4 test content"), "test.pdf")}
        resp = client.post(
            "/api/upload_pdf",
            data=data,
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    def test_upload_pdf_without_csrf_returns_403(self, logged_in_client):
        """Test that upload without CSRF returns 403."""
        data = {"file": (io.BytesIO(b"%PDF-1.4 test content"), "test.pdf")}
        resp = logged_in_client.post(
            "/api/upload_pdf",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 403

    def test_upload_pdf_without_file_returns_400(self, logged_in_client, csrf_token):
        """Test that upload without file returns 400."""
        resp = logged_in_client.post(
            "/api/upload_pdf",
            data={},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is False

    def test_upload_pdf_non_pdf_returns_400(self, logged_in_client, csrf_token):
        """Test that non-PDF file returns 400."""
        data = {"file": (io.BytesIO(b"not a pdf"), "test.txt")}
        resp = logged_in_client.post(
            "/api/upload_pdf",
            data=data,
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        json_data = resp.get_json(silent=True) or {}
        assert json_data.get("success") is False
        assert "PDF" in json_data.get("error", "")

    def test_upload_pdf_invalid_magic_bytes_returns_400(self, logged_in_client, csrf_token):
        """Test that file without PDF magic bytes returns 400."""
        data = {"file": (io.BytesIO(b"not a real pdf content here"), "fake.pdf")}
        resp = logged_in_client.post(
            "/api/upload_pdf",
            data=data,
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400
        json_data = resp.get_json(silent=True) or {}
        assert json_data.get("success") is False

    def test_upload_pdf_too_small_returns_400(self, logged_in_client, csrf_token):
        """Test that file too small returns 400."""
        data = {"file": (io.BytesIO(b"%PDF"), "tiny.pdf")}
        resp = logged_in_client.post(
            "/api/upload_pdf",
            data=data,
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_upload_pdf_too_large_returns_413(self, logged_in_client, csrf_token, monkeypatch):
        """Test that oversized file returns 413."""
        monkeypatch.setattr("backend.blueprints.api_uploads.MAX_UPLOAD_SIZE", 8)
        data = {"file": (io.BytesIO(b"%PDF-1.4 too large content"), "large.pdf")}
        resp = logged_in_client.post(
            "/api/upload_pdf",
            data=data,
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 413

    def test_reupload_failed_duplicate_sets_queued_once(self, logged_in_client, csrf_token):
        """Re-uploading a failed duplicate should atomically set queued and avoid duplicate enqueues."""
        from aslite.repositories import UploadedPaperRepository

        user = "test_user"
        pid = "up_reupload12345"

        # Seed an existing failed record in db so the endpoint can CAS-update it.
        try:
            UploadedPaperRepository.save(
                pid,
                {
                    "pid": pid,
                    "owner": user,
                    "created_time": 1.0,
                    "updated_time": 1.0,
                    "original_filename": "test.pdf",
                    "size_bytes": 123,
                    "sha256": "dummy",
                    "parse_status": "failed",
                    "parse_error": "boom",
                    "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
                    "meta_extracted_ok": False,
                    "meta_override": {},
                    "summary_task_id": None,
                },
            )

            # Mock create_uploaded_paper to always return the same stale failed status.
            with patch("backend.services.upload_service.create_uploaded_paper") as create_mock:
                create_mock.return_value = (
                    pid,
                    {
                        "pid": pid,
                        "owner": user,
                        "original_filename": "test.pdf",
                        "parse_status": "failed",
                    },
                    False,
                )

                with patch("tasks.huey") as huey_mock, patch("tasks.process_uploaded_pdf_task") as task_mock:
                    huey_mock.enqueue = MagicMock()
                    task_mock.s = MagicMock(return_value="task")

                    pdf_bytes = b"%PDF-1.4 " + (b"x" * 256)
                    data = {"file": (io.BytesIO(pdf_bytes), "test.pdf")}
                    resp1 = logged_in_client.post(
                        "/api/upload_pdf",
                        data=data,
                        content_type="multipart/form-data",
                        headers={"X-CSRF-Token": csrf_token},
                    )
                    assert resp1.status_code == 200
                    j1 = resp1.get_json(silent=True) or {}
                    assert j1.get("success") is True
                    assert j1.get("parse_status") == "queued"
                    assert huey_mock.enqueue.call_count == 1

                    # Second immediate re-upload should not enqueue again.
                    data2 = {"file": (io.BytesIO(pdf_bytes), "test.pdf")}
                    resp2 = logged_in_client.post(
                        "/api/upload_pdf",
                        data=data2,
                        content_type="multipart/form-data",
                        headers={"X-CSRF-Token": csrf_token},
                    )
                    assert resp2.status_code == 200
                    j2 = resp2.get_json(silent=True) or {}
                    assert j2.get("success") is True
                    assert j2.get("parse_status") == "queued"
                    assert huey_mock.enqueue.call_count == 1
        finally:
            UploadedPaperRepository.delete(pid)

    def test_upload_pdf_success_returns_task_id(self, logged_in_client, csrf_token):
        """Successful upload should include task_id when Huey provides one."""

        class DummyTask:
            id = "task_upload_001"

        pid = "up_taskupload01"
        with patch("backend.services.upload_service.create_uploaded_paper") as create_mock:
            create_mock.return_value = (
                pid,
                {
                    "pid": pid,
                    "owner": "test_user",
                    "original_filename": "test.pdf",
                    "parse_status": "queued",
                },
                True,
            )

            with patch("tasks.huey") as huey_mock, patch("tasks.process_uploaded_pdf_task") as task_mock:
                task_mock.s = MagicMock(return_value=DummyTask())
                huey_mock.enqueue = MagicMock(return_value=DummyTask())

                pdf_bytes = b"%PDF-1.4 " + (b"x" * 256)
                data = {"file": (io.BytesIO(pdf_bytes), "test.pdf")}
                resp = logged_in_client.post(
                    "/api/upload_pdf",
                    data=data,
                    content_type="multipart/form-data",
                    headers={"X-CSRF-Token": csrf_token},
                )

                assert resp.status_code == 200
                payload = resp.get_json(silent=True) or {}
                assert payload.get("success") is True
                assert payload.get("task_id") == "task_upload_001"


class TestUploadedPapersParseApi:
    """Tests for POST /api/uploaded_papers/parse endpoint."""

    def test_parse_success_returns_task_id(self, logged_in_client, csrf_token):
        """Parse endpoint should return task_id on successful enqueue."""
        with patch("backend.services.upload_service.trigger_parse_only") as parse_mock:
            parse_mock.return_value = (True, "queued", "task_parse_001")

            resp = logged_in_client.post(
                "/api/uploaded_papers/parse",
                json={"pid": "up_testpaper123"},
                headers={"X-CSRF-Token": csrf_token},
            )

        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("parse_status") == "queued"
        assert payload.get("task_id") == "task_parse_001"


class TestUploadedPapersListApi:
    """Tests for GET /api/uploaded_papers/list endpoint."""

    def test_list_without_login_returns_401(self, client):
        """Test that list without login returns 401."""
        resp = client.get("/api/uploaded_papers/list")
        assert resp.status_code == 401

    def test_list_returns_success(self, logged_in_client):
        """Test that list returns success structure."""
        resp = logged_in_client.get("/api/uploaded_papers/list")
        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        assert "papers" in data
        assert isinstance(data["papers"], list)

    def test_list_includes_upload_task_ids(self, logged_in_client):
        """List endpoint should expose parse/extract task ids for observability."""
        mocked_item = {
            "id": "up_testpaper_list1",
            "kind": "upload",
            "title": "Test",
            "authors": "Author",
            "time": "",
            "summary": "",
            "utags": [],
            "ntags": [],
            "parse_status": "queued",
            "summary_status": "",
            "created_time": 1.0,
            "original_filename": "test.pdf",
            "parse_task_id": "task_parse_list_1",
            "extract_task_id": "task_extract_list_1",
            "summary_task_id": "task_summary_list_1",
        }

        with patch("backend.services.upload_service.get_uploaded_papers_list", return_value=[mocked_item]):
            resp = logged_in_client.get("/api/uploaded_papers/list")

        assert resp.status_code == 200
        data = resp.get_json(silent=True) or {}
        assert data.get("success") is True
        papers = data.get("papers") or []
        assert len(papers) == 1
        assert papers[0].get("parse_task_id") == "task_parse_list_1"
        assert papers[0].get("extract_task_id") == "task_extract_list_1"
        assert papers[0].get("summary_task_id") == "task_summary_list_1"


class TestUploadedPapersUpdateMetaApi:
    """Tests for POST /api/uploaded_papers/update_meta endpoint."""

    def test_update_meta_without_login_returns_401(self, client, csrf_token):
        """Test that update_meta without login returns 401."""
        resp = client.post(
            "/api/uploaded_papers/update_meta",
            json={"pid": "up_V1StGXR8_Z5j", "title": "New Title"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    def test_update_meta_without_csrf_returns_403(self, logged_in_client):
        """Test that update_meta without CSRF returns 403."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/update_meta",
            json={"pid": "up_V1StGXR8_Z5j", "title": "New Title"},
        )
        assert resp.status_code == 403

    def test_update_meta_invalid_pid_returns_400(self, logged_in_client, csrf_token):
        """Test that invalid PID returns 400."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/update_meta",
            json={"pid": "invalid_pid", "title": "New Title"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_update_meta_missing_pid_returns_400(self, logged_in_client, csrf_token):
        """Test that missing PID returns 400."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/update_meta",
            json={"title": "New Title"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_update_meta_nonexistent_paper_returns_404(self, logged_in_client, csrf_token):
        """Test that nonexistent paper returns 404."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/update_meta",
            json={"pid": "up_abcdefghijkl", "title": "New Title"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404


class TestUploadedPapersExtractInfoApi:
    """Tests for POST /api/uploaded_papers/extract_info endpoint."""

    def test_extract_info_not_owner_does_not_leak(self, logged_in_client, csrf_token):
        """Non-owners should get 404 to avoid PID enumeration."""
        from aslite.repositories import UploadedPaperRepository

        pid = "up_aaaaaaaaaaaa"
        try:
            UploadedPaperRepository.save(
                pid,
                {
                    "pid": pid,
                    "owner": "other_user",
                    "created_time": 1.0,
                    "updated_time": 1.0,
                    "original_filename": "x.pdf",
                    "size_bytes": 123,
                    "sha256": "dummy",
                    "parse_status": "ok",
                    "parse_error": None,
                    "meta_extracted": {"title": "", "authors": [], "year": None, "abstract": None},
                    "meta_extracted_ok": False,
                    "meta_override": {},
                    "summary_task_id": None,
                },
            )

            resp = logged_in_client.post(
                "/api/uploaded_papers/extract_info",
                json={"pid": pid},
                headers={"X-CSRF-Token": csrf_token},
            )
            assert resp.status_code == 404
            data = resp.get_json(silent=True) or {}
            assert data.get("success") is False
        finally:
            UploadedPaperRepository.delete(pid)

    def test_extract_info_success_returns_task_id(self, logged_in_client, csrf_token, monkeypatch):
        """Extract-info endpoint should return task_id on successful enqueue."""
        from aslite import repositories

        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda _pid: {
                "owner": "test_user",
                "parse_status": "ok",
                "meta_extracted_ok": False,
            },
        )

        with patch("backend.services.upload_service.trigger_extract_info") as extract_mock:
            extract_mock.return_value = (True, "task_extract_001")

            resp = logged_in_client.post(
                "/api/uploaded_papers/extract_info",
                json={"pid": "up_testpaper123"},
                headers={"X-CSRF-Token": csrf_token},
            )

        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("task_id") == "task_extract_001"


class TestUploadedPapersDeleteApi:
    """Tests for POST /api/uploaded_papers/delete endpoint."""

    def test_delete_without_login_returns_401(self, client, csrf_token):
        """Test that delete without login returns 401."""
        resp = client.post(
            "/api/uploaded_papers/delete",
            json={"pid": "up_V1StGXR8_Z5j"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    def test_delete_without_csrf_returns_403(self, logged_in_client):
        """Test that delete without CSRF returns 403."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/delete",
            json={"pid": "up_V1StGXR8_Z5j"},
        )
        assert resp.status_code == 403

    def test_delete_invalid_pid_returns_400(self, logged_in_client, csrf_token):
        """Test that invalid PID returns 400."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/delete",
            json={"pid": "invalid_pid"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_delete_nonexistent_paper_returns_404(self, logged_in_client, csrf_token):
        """Test that nonexistent paper returns 404."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/delete",
            json={"pid": "up_abcdefghijkl"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404


class TestUploadedPapersRetryParseApi:
    """Tests for POST /api/uploaded_papers/retry_parse endpoint."""

    def test_retry_parse_without_login_returns_401(self, client, csrf_token):
        """Test that retry_parse without login returns 401."""
        resp = client.post(
            "/api/uploaded_papers/retry_parse",
            json={"pid": "up_V1StGXR8_Z5j"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 401

    def test_retry_parse_without_csrf_returns_403(self, logged_in_client):
        """Test that retry_parse without CSRF returns 403."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/retry_parse",
            json={"pid": "up_V1StGXR8_Z5j"},
        )
        assert resp.status_code == 403

    def test_retry_parse_invalid_pid_returns_400(self, logged_in_client, csrf_token):
        """Test that invalid PID returns 400."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/retry_parse",
            json={"pid": "invalid_pid"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 400

    def test_retry_parse_nonexistent_paper_returns_404(self, logged_in_client, csrf_token):
        """Test that nonexistent paper returns 404."""
        resp = logged_in_client.post(
            "/api/uploaded_papers/retry_parse",
            json={"pid": "up_abcdefghijkl"},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404

    def test_retry_parse_success_returns_task_id(self, logged_in_client, csrf_token):
        """Retry endpoint should return task_id on successful enqueue."""
        with patch("backend.services.upload_service.retry_parse_uploaded_paper") as retry_mock:
            retry_mock.return_value = (True, "queued", "task_retry_001")

            resp = logged_in_client.post(
                "/api/uploaded_papers/retry_parse",
                json={"pid": "up_testpaper123"},
                headers={"X-CSRF-Token": csrf_token},
            )

        assert resp.status_code == 200
        payload = resp.get_json(silent=True) or {}
        assert payload.get("success") is True
        assert payload.get("parse_status") == "queued"
        assert payload.get("task_id") == "task_retry_001"


class TestUploadTaskStatusApi:
    """Tests for upload-related task status query."""

    def test_task_status_can_query_upload_task(self, logged_in_client):
        """Upload task stored in task::* should be queryable via /api/task_status."""
        from aslite.db import get_summary_status_db
        from aslite.repositories import SummaryStatusRepository

        task_id = "task_upload_status_001"
        SummaryStatusRepository.set_task_status(
            task_id,
            "queued",
            None,
            pid="up_testpaper123",
            model="upload_parse",
            user="test_user",
        )

        try:
            resp = logged_in_client.get(f"/api/task_status/{task_id}")
            assert resp.status_code == 200

            payload = resp.get_json(silent=True) or {}
            assert payload.get("success") is True
            assert payload.get("task_id") == task_id
            assert payload.get("status") == "queued"
            assert payload.get("pid") == "up_testpaper123"
            assert payload.get("model") == "upload_parse"
        finally:
            with get_summary_status_db(flag="c") as sdb:
                key = f"task::{task_id}"
                if key in sdb:
                    del sdb[key]


class TestUploadedPapersPdfApi:
    """Tests for GET /api/uploaded_papers/pdf/<pid> endpoint."""

    def test_pdf_without_login_returns_401(self, client):
        """Test that PDF download without login returns 401."""
        resp = client.get("/api/uploaded_papers/pdf/up_V1StGXR8_Z5j")
        assert resp.status_code == 401

    def test_pdf_invalid_pid_returns_404(self, logged_in_client):
        """Test that invalid PID returns 404."""
        resp = logged_in_client.get("/api/uploaded_papers/pdf/invalid_pid")
        assert resp.status_code == 404

    def test_pdf_nonexistent_paper_returns_404(self, logged_in_client):
        """Test that nonexistent paper returns 404."""
        resp = logged_in_client.get("/api/uploaded_papers/pdf/up_abcdefghijkl")
        assert resp.status_code == 404


class TestUploadedPapersSimilarApi:
    """Tests for GET /api/uploaded_papers/similar/<pid> endpoint."""

    def test_similar_without_login_returns_401(self, client):
        """Test that similar without login returns 401."""
        resp = client.get("/api/uploaded_papers/similar/up_V1StGXR8_Z5j")
        assert resp.status_code == 401

    def test_similar_invalid_pid_returns_400(self, logged_in_client):
        """Test that invalid PID returns 400."""
        resp = logged_in_client.get("/api/uploaded_papers/similar/invalid_pid")
        assert resp.status_code == 400

    def test_similar_nonexistent_paper_returns_404(self, logged_in_client):
        """Test that nonexistent paper returns 404."""
        resp = logged_in_client.get("/api/uploaded_papers/similar/up_abcdefghijkl")
        assert resp.status_code == 404

    def test_similar_with_limit_param(self, logged_in_client, monkeypatch):
        """Test that limit parameter is respected."""
        from aslite import repositories
        from backend.services import upload_similarity_service

        # Mock repository to return a paper owned by test user (test_user from conftest)
        mock_record = {
            "owner": "test_user",
            "parse_status": "ok",
        }
        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda pid: mock_record,
        )

        # Mock find_similar_papers to capture the limit
        captured_limit = [None]

        def mock_find_similar(pid, limit=20):
            captured_limit[0] = limit
            return []

        monkeypatch.setattr(
            upload_similarity_service,
            "find_similar_papers",
            mock_find_similar,
        )

        # Use valid PID format: up_ + 12 chars = 15 total
        resp = logged_in_client.get("/api/uploaded_papers/similar/up_testpaper123?limit=50")
        assert resp.status_code == 200
        assert captured_limit[0] == 50

    def test_similar_limit_capped_at_100(self, logged_in_client, monkeypatch):
        """Test that limit is capped at 100."""
        from aslite import repositories
        from backend.services import upload_similarity_service

        mock_record = {
            "owner": "test_user",
            "parse_status": "ok",
        }
        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda pid: mock_record,
        )

        captured_limit = [None]

        def mock_find_similar(pid, limit=20):
            captured_limit[0] = limit
            return []

        monkeypatch.setattr(
            upload_similarity_service,
            "find_similar_papers",
            mock_find_similar,
        )

        # Use valid PID format: up_ + 12 chars = 15 total
        resp = logged_in_client.get("/api/uploaded_papers/similar/up_testpaper123?limit=500")
        assert resp.status_code == 200
        assert captured_limit[0] == 100  # Should be capped

    def test_similar_paper_not_parsed_returns_400(self, logged_in_client, monkeypatch):
        """Test that unparsed paper returns 400."""
        from aslite import repositories

        mock_record = {
            "owner": "test_user",
            "parse_status": "pending",  # Not parsed yet
        }
        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda pid: mock_record,
        )

        # Use valid PID format: up_ + 12 chars = 15 total
        resp = logged_in_client.get("/api/uploaded_papers/similar/up_testpaper123")
        assert resp.status_code == 400
        data = resp.get_json(silent=True) or {}
        assert "parsed" in data.get("error", "").lower()

    def test_similar_other_user_paper_returns_404(self, logged_in_client, monkeypatch):
        """Test that accessing another user's paper returns 404."""
        from aslite import repositories

        mock_record = {
            "owner": "other_user",  # Different user
            "parse_status": "ok",
        }
        monkeypatch.setattr(
            repositories.UploadedPaperRepository,
            "get",
            lambda pid: mock_record,
        )

        # Use valid PID format: up_ + 12 chars = 15 total
        resp = logged_in_client.get("/api/uploaded_papers/similar/up_testpaper123")
        assert resp.status_code == 404
