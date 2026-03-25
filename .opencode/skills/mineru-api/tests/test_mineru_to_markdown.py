import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mineru_to_markdown.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mineru_to_markdown", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_module()


class MineruMarkdownTests(unittest.TestCase):
    def test_mapping_uses_reading_order_and_kind_suffixes(self):
        refs = [
            "images/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg",
            "images/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png",
            "images/cccccccccccccccccccccccccccccccc.jpg",
        ]
        kinds = {
            refs[0]: "image",
            refs[1]: "table",
            refs[2]: "equation",
        }

        result = mod.mapping(refs, kinds)

        self.assertEqual(result[refs[0]], "001-image.jpg")
        self.assertEqual(result[refs[1]], "002-table.png")
        self.assertEqual(result[refs[2]], "003-equation.jpg")

    def test_process_raw_rewrites_markdown_and_copies_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            raw = temp_path / "raw"
            images = raw / "images"
            out = temp_path / "out"
            images.mkdir(parents=True)

            img1 = "images/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"
            img2 = "images/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png"

            (raw / "demo.md").write_text(
                "# Demo\n\n" f"![fig]({img1})\n\n" f"Table preview: ![tbl]({img2})\n\n" f"Repeat: ![fig2]({img1})\n",
                encoding="utf-8",
            )
            (raw / "demo_content_list.json").write_text(
                json.dumps(
                    [
                        {"type": "image", "img_path": img1},
                        {"type": "table", "img_path": img2},
                    ]
                ),
                encoding="utf-8",
            )
            (images / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg").write_bytes(b"jpg-data")
            (images / "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png").write_bytes(b"png-data")

            md, assets = mod.process_raw(raw, out, asset_dir="assets", markdown_name="clean.md", keep_raw=False)

            text = md.read_text(encoding="utf-8")
            self.assertIn("assets/001-image.jpg", text)
            self.assertIn("assets/002-table.png", text)
            self.assertEqual(text.count("assets/001-image.jpg"), 2)
            self.assertTrue((assets / "001-image.jpg").is_file())
            self.assertTrue((assets / "002-table.png").is_file())

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["assets"][0]["target"], "assets/001-image.jpg")
            self.assertEqual(manifest["assets"][1]["target"], "assets/002-table.png")

    def test_resolve_token_reads_skill_local_env_file(self):
        env_path = ROOT / ".env.mineru.local"
        original = env_path.read_text(encoding="utf-8") if env_path.exists() else None
        old_token = os.environ.pop("MINERU_API_TOKEN", None)
        old_token_file = os.environ.pop("MINERU_API_TOKEN_FILE", None)

        try:
            env_path.write_text("MINERU_API_TOKEN=test-from-skill-local\n", encoding="utf-8")
            args = type("A", (), {"token_file": None})()
            self.assertEqual(mod.resolve_token(args), "test-from-skill-local")
        finally:
            if original is None:
                env_path.unlink(missing_ok=True)
            else:
                env_path.write_text(original, encoding="utf-8")
            if old_token is not None:
                os.environ["MINERU_API_TOKEN"] = old_token
            if old_token_file is not None:
                os.environ["MINERU_API_TOKEN_FILE"] = old_token_file


if __name__ == "__main__":
    unittest.main()
