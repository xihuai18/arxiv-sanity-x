"""Unit tests for image serving path handling."""

from __future__ import annotations

from pathlib import Path

from flask import Flask

from backend.services.render_service import serve_paper_image


def test_serve_paper_image_relative_base_dir_works(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    app_root = tmp_path / "app_root"
    cwd.mkdir(parents=True, exist_ok=True)
    app_root.mkdir(parents=True, exist_ok=True)

    images_dir = cwd / "data" / "html_md" / "1234.5678" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    # Put the file under CWD but not under Flask root_path to ensure
    # serve_paper_image uses an absolute path when calling send_file.
    monkeypatch.chdir(cwd)
    app = Flask("test_app", root_path=str(app_root))

    with app.test_request_context("/"):
        resp = serve_paper_image("1234.5678", "test.png", Path("data") / "html_md")
        assert resp.status_code == 200
        assert resp.mimetype == "image/png"
