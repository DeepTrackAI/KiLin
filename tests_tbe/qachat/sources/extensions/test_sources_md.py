"""Tests for Markdown natural-unit extraction."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.sources.sources import (
    extract_source_units,
    read_repository_files,
)


class TestSourcesMd(unittest.TestCase):

    def test_markdown_file_becomes_heading_sections(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text(
                "Introduction.\n\n# Setup\nRun setup.\n",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".md"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(
            [unit.kind for unit in units],
            ["markdown_preamble", "markdown_section"],
        )
        self.assertEqual(units[0].text, "Introduction.\n\n")
        self.assertEqual(units[1].section, "Setup")
        self.assertEqual(units[1].start_line, 3)
        self.assertEqual(units[1].text, "# Setup\nRun setup.\n")


if __name__ == "__main__":
    unittest.main()
