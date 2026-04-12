from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tools.compress_summary_images import _compress_task, run
from tools.paper_summarizer import PaperSummarizer


def _make_summarizer(tmp_path, monkeypatch, *, enabled=True) -> PaperSummarizer:
    import tools.paper_summarizer as ps

    monkeypatch.setattr(ps, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ps, "_summary_image_compression_enabled", lambda: enabled)
    monkeypatch.setattr(ps, "_summary_image_max_long_edge", lambda: 0)
    monkeypatch.setattr(ps, "_summary_image_min_savings_bytes", lambda: 0)
    monkeypatch.setattr(ps, "_summary_image_webp_quality", lambda: 86)
    monkeypatch.setattr(ps, "_summary_image_skip_below_bytes", lambda: 0)
    summarizer = PaperSummarizer()
    monkeypatch.setattr(summarizer, "_pyvips_available", lambda: False)
    return summarizer


def _write_noise_png(path: Path, size: tuple[int, int] = (1200, 900)) -> None:
    image = Image.effect_noise(size, 96).convert("RGB")
    image.save(path, format="PNG")


def test_compress_task_rewrites_markdown_references(tmp_path, monkeypatch):
    summarizer = _make_summarizer(tmp_path, monkeypatch)
    images_dir = tmp_path / "html_md" / "2501.00001" / "images"
    images_dir.mkdir(parents=True)
    image_path = images_dir / "figure.png"
    _write_noise_png(image_path)

    markdown_path = images_dir.parent / "2501.00001.md"
    markdown_path.write_text("![A](images/figure.png)\n", encoding="utf-8")

    result = _compress_task(summarizer, images_dir, [markdown_path])

    assert result.scanned == 1
    assert result.changed == 1
    assert result.markdown_updated == 1
    assert markdown_path.read_text(encoding="utf-8") == "![A](images/figure.webp)\n"
    assert (images_dir / "figure.webp").exists()


def test_run_processes_html_and_mineru_roots(tmp_path, monkeypatch, capsys):
    summarizer = _make_summarizer(tmp_path, monkeypatch)

    html_images = tmp_path / "html_md" / "2501.00002" / "images"
    html_images.mkdir(parents=True)
    _write_noise_png(html_images / "html.png")
    (html_images.parent / "2501.00002.md").write_text("![H](images/html.png)\n", encoding="utf-8")

    mineru_images = tmp_path / "mineru" / "2501.00003" / "api" / "images"
    mineru_images.mkdir(parents=True)
    _write_noise_png(mineru_images / "mineru.png")
    (mineru_images.parent / "2501.00003.md").write_text("![M](images/mineru.png)\n", encoding="utf-8")

    import tools.compress_summary_images as csi

    monkeypatch.setattr(csi, "settings", SimpleNamespace(data_dir=tmp_path, log_level="WARNING"))
    monkeypatch.setattr(csi, "PaperSummarizer", lambda: summarizer)

    rc = run(SimpleNamespace(roots=["html_md", "mineru"], limit=0, workers=1))
    captured = capsys.readouterr().out

    assert rc == 0
    assert "Compression complete" in captured
    assert "backend=" in captured
    assert (html_images / "html.webp").exists()
    assert (mineru_images / "mineru.webp").exists()


def test_run_force_overrides_disabled_auto_compression(tmp_path, monkeypatch, capsys):
    summarizer = _make_summarizer(tmp_path, monkeypatch, enabled=False)

    html_images = tmp_path / "html_md" / "2501.00004" / "images"
    html_images.mkdir(parents=True)
    _write_noise_png(html_images / "forced.png")
    (html_images.parent / "2501.00004.md").write_text("![F](images/forced.png)\n", encoding="utf-8")

    import tools.compress_summary_images as csi

    monkeypatch.setattr(csi, "settings", SimpleNamespace(data_dir=tmp_path, log_level="WARNING"))
    monkeypatch.setattr(csi, "PaperSummarizer", lambda: summarizer)

    rc = run(SimpleNamespace(roots=["html_md"], limit=0, workers=1, force=True))
    captured = capsys.readouterr().out

    assert rc == 0
    assert "force=true" in captured
    assert (html_images / "forced.webp").exists()
