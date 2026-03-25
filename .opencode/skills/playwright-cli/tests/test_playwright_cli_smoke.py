import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PlaywrightCliSmokeTests(unittest.TestCase):
    SESSION = "skill-smoke"

    def setUp(self):
        self.cli = shutil.which("playwright-cli.cmd") or shutil.which("playwright-cli") or shutil.which("npx.cmd")
        if self.cli is None:
            self.skipTest("playwright-cli is not installed")
        assert self.cli is not None
        self.tempdir = tempfile.TemporaryDirectory(prefix="pwcli-skill-test-")
        self.workdir = Path(self.tempdir.name)

    def tearDown(self):
        if hasattr(self, "tempdir"):
            self.run_cli("close", check=False)
            self.tempdir.cleanup()

    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cli = self.cli or ""
        if Path(cli).name.lower().startswith("npx"):
            cmd = [cli, "playwright-cli", f"-s={self.SESSION}", *args]
        else:
            cmd = [cli, f"-s={self.SESSION}", *args]
        return subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True, check=check)

    def test_help_is_available(self):
        cli = self.cli or ""
        if Path(cli).name.lower().startswith("npx"):
            cmd = [cli, "playwright-cli", "--help"]
        else:
            cmd = [cli, "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertIn("playwright-cli", result.stdout)
        self.assertIn("snapshot", result.stdout)

    def test_basic_browser_workflow(self):
        self.run_cli("open", "https://example.com")
        self.run_cli("snapshot", "--filename=example.yaml")
        title = self.run_cli("eval", "document.title")
        self.run_cli("close")

        self.assertIn("Example Domain", title.stdout)
        self.assertTrue((self.workdir / "example.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
