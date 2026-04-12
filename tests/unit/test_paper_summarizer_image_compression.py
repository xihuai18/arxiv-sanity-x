from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.paper_summarizer import PaperSummarizer


def _make_summarizer(tmp_path, monkeypatch) -> PaperSummarizer:
    import tools.paper_summarizer as ps

    monkeypatch.setattr(ps, "_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ps, "_summary_image_compression_enabled", lambda: True)
    monkeypatch.setattr(ps, "_summary_image_max_long_edge", lambda: 1024)
    monkeypatch.setattr(ps, "_summary_image_min_savings_bytes", lambda: 0)
    monkeypatch.setattr(ps, "_summary_image_webp_quality", lambda: 86)
    monkeypatch.setattr(ps, "_summary_image_skip_below_bytes", lambda: 0)
    summarizer = PaperSummarizer()
    monkeypatch.setattr(summarizer, "_pyvips_available", lambda: False)
    return summarizer


def _write_noise_png(path: Path, size: tuple[int, int] = (2200, 1600)) -> int:
    image = Image.effect_noise(size, 96).convert("RGB")
    image.save(path, format="PNG")
    return path.stat().st_size


def test_compress_cached_image_converts_large_png_to_webp(tmp_path, monkeypatch):
    summarizer = _make_summarizer(tmp_path, monkeypatch)
    image_path = tmp_path / "figure.png"
    original_size = _write_noise_png(image_path)

    optimized_path = summarizer._compress_cached_image(image_path)

    assert optimized_path.suffix == ".webp"
    assert optimized_path.exists()
    assert not image_path.exists()
    assert optimized_path.stat().st_size < original_size

    with Image.open(optimized_path) as optimized_image:
        assert max(optimized_image.size) <= 1024


def test_normalize_mineru_images_updates_markdown_after_compression(tmp_path, monkeypatch):
    summarizer = _make_summarizer(tmp_path, monkeypatch)
    output_path = tmp_path / "2501.00001" / "api"
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True)

    original_image = images_dir / "hash-figure.png"
    _write_noise_png(original_image)

    markdown_path = output_path / "2501.00001.md"
    markdown_path.write_text("![Figure](images/hash-figure.png)\n", encoding="utf-8")

    summarizer._normalize_mineru_images(output_path, markdown_path)

    updated_markdown = markdown_path.read_text(encoding="utf-8")
    assert "images/image-1.webp" in updated_markdown
    assert "hash-figure.png" not in updated_markdown
    assert (images_dir / "image-1.webp").exists()
    assert sorted(p.name for p in images_dir.iterdir()) == [
        ".image_compression_manifest.json",
        "image-1.webp",
    ]


def test_compress_cached_image_skips_webp_and_small_files(tmp_path, monkeypatch):
    import tools.paper_summarizer as ps

    summarizer = _make_summarizer(tmp_path, monkeypatch)
    monkeypatch.setattr(ps, "_summary_image_skip_below_bytes", lambda: 1024 * 1024)

    webp_path = tmp_path / "figure.webp"
    Image.new("RGB", (32, 32), "white").save(webp_path, format="WEBP")
    assert summarizer._compress_cached_image(webp_path) == webp_path

    small_png = tmp_path / "small.png"
    Image.new("RGB", (16, 16), "black").save(small_png, format="PNG")
    assert summarizer._compress_cached_image(small_png) == small_png


def test_compress_cached_image_skips_processed_entry(tmp_path, monkeypatch):
    summarizer = _make_summarizer(tmp_path, monkeypatch)
    image_path = tmp_path / "figure.png"
    _write_noise_png(image_path, size=(512, 512))

    optimized_path = summarizer._compress_cached_image(image_path)
    first_mtime = optimized_path.stat().st_mtime_ns
    manifest_path = optimized_path.parent / ".image_compression_manifest.json"
    assert manifest_path.exists()

    skipped_path = summarizer._compress_cached_image(optimized_path)

    assert skipped_path == optimized_path
    assert optimized_path.stat().st_mtime_ns == first_mtime


def test_build_image_candidates_prefers_pyvips_when_available(tmp_path, monkeypatch):
    summarizer = _make_summarizer(tmp_path, monkeypatch)
    image_path = tmp_path / "figure.png"
    _write_noise_png(image_path, size=(512, 512))

    calls = []

    monkeypatch.setattr(summarizer, "_pyvips_available", lambda: True)
    monkeypatch.setattr(
        summarizer,
        "_build_image_candidates_vips",
        lambda path: calls.append(("vips", path.name)) or ({".webp": b"abc"}, False),
    )
    monkeypatch.setattr(
        summarizer,
        "_build_image_candidates_pillow",
        lambda path: calls.append(("pillow", path.name)) or ({".png": b"abcd"}, False),
    )

    candidates, resized, backend = summarizer._build_image_candidates(image_path)

    assert backend == "pyvips"
    assert candidates == {".webp": b"abc"}
    assert resized is False
    assert calls == [("vips", "figure.png")]
