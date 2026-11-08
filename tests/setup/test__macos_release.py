"""Check native macOS artifact gates with no Apple credentials.

The real macOS artifact validator runs on an Apple Silicon runner. These
portable tests protect the command's missing-artifact failure and ensure
pull-request automation does not depend on signing credentials.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import yaml


_ROOT = Path(__file__).resolve().parents[2]


class TestMacOSRelease(unittest.TestCase):
    """Check release safety without a native application or credentials.

    """

    def test_native_checker_requires_existing_artifacts(self) -> None:
        """Fail before invoking macOS tools when no .app/.dmg exists.

        """
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checker = root / "tests/setup/check_macos_app.sh"
            checker.parent.mkdir(parents=True)
            checker.write_bytes(
                (_ROOT / "tests/setup/check_macos_app.sh").read_bytes()
            )
            result = subprocess.run(
                ["bash", str(checker)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing macOS distribution", result.stderr)

    def test_native_checker_rejects_unknown_options(self) -> None:
        """Reject unsupported public-release validation modes.

        """
        result = subprocess.run(
            ["bash", "tests/setup/check_macos_app.sh", "--unknown"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_native_ci_is_unprivileged_and_no_secrets(self) -> None:
        """Verify PR builds check a real app without Apple credentials.

        """
        workflow = _ROOT / ".github/workflows/macos_app.yml"
        text = workflow.read_text(encoding="utf-8")
        parsed = yaml.safe_load(text)
        self.assertEqual(parsed["permissions"], {"contents": "read"})
        job = parsed["jobs"]["app"]
        self.assertEqual(job["runs-on"], "macos-15")
        commands = "\n".join(
            step.get("run", "") for step in job["steps"]
        )
        self.assertIn("build_app.sh --wheel", commands)
        self.assertIn("check_macos_app.sh", commands)
        self.assertNotIn("--notarize", commands)
        self.assertNotIn("notarytool", text)
        self.assertNotIn("KILIN_NOTARY_PROFILE", text)
        self.assertNotIn("secrets.", text)
        self.assertEqual(job["env"]["KILIN_SKIP_FINDER_LAYOUT"], "1")


if __name__ == "__main__":
    unittest.main()
