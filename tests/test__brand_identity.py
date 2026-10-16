"""Verify the public KíLín identity throughout the source distribution.

A complete prerelease rename must not retain an import alias, launcher,
configuration filename, or textual reference to the former project name.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest

import kilin


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    ROOT / "kilin",
    ROOT / "tests",
    ROOT / "setup",
    ROOT / "docs",
    ROOT / "tutorials",
    ROOT / "assets",
)
SKIPPED_PARTS = frozenset({"__pycache__", ".pytest_cache"})
FORMER_NAME = re.compile(b"qa" + b"[_ -]?chat", re.IGNORECASE)


class TestBrandIdentity(unittest.TestCase):
    """Require consistent public and packaged names without legacy aliases."""

    def test_distribution_import_and_launcher(self) -> None:
        """Keep the PyPI distribution, import package, and CLI aligned."""

        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["name"], "kilin")
        self.assertEqual(
            metadata["project"]["scripts"],
            {"kilin": "kilin.cli:main"},
        )
        self.assertEqual(Path(kilin.__file__).parent.name, "kilin")
        self.assertTrue((ROOT / "kilin/kilin_defaults.json").is_file())

    def test_gui_branding_assets(self) -> None:
        """Keep application labels and bundled icons under the new name."""

        html = (ROOT / "kilin/gui/web/index.html").read_text()
        self.assertIn("<title>KíLín</title>", html)
        self.assertIn("<h1>KíLín</h1>", html)
        icons = ROOT / "kilin/gui/icons"
        self.assertTrue((icons / "kilin-light.png").is_file())
        self.assertTrue((icons / "kilin-dark.png").is_file())

    def test_no_former_name_in_sources_or_paths(self) -> None:
        """Reject obsolete identifiers in source, tests, and resources."""

        files = [
            path
            for root in SOURCE_ROOTS
            for path in root.rglob("*")
            if path.is_file()
            and not any(part in SKIPPED_PARTS for part in path.parts)
        ]
        files.extend((ROOT / "README.md", ROOT / "pyproject.toml",
                      ROOT / "uv.lock", ROOT / ".gitignore"))
        violations = [
            str(path.relative_to(ROOT))
            for path in files
            if FORMER_NAME.search(path.relative_to(ROOT).as_posix().encode())
            or FORMER_NAME.search(path.read_bytes())
        ]
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
