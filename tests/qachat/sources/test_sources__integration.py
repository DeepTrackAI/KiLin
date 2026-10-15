"""Integration tests for repository scanning and source extraction.

A small synthetic repository is built from committed format fixtures. The
complete scan-read-extract workflow must not execute any source code or
require network access or external example repositories.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from qachat.repository import scan_repository
from qachat.sources.sources import (
    SourceFile,
    extract_source_units,
    read_repository_files,
)
from tests.qachat.sources._support import fixture_path


class TestSourceIntegration(unittest.TestCase):
    """Verify source provenance through the complete local workflow."""

    def setUp(self) -> None:
        """Create a small repository of independent synthetic fixtures."""

        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "repository"
        self.root.mkdir()
        fixtures = {
            "src/sample.py": "sample.py",
            "matlab/sample.m": "sample.m",
            "docs/sample.md": "sample.md",
            "notes/sample.txt": "sample.txt",
            "notebooks/sample.ipynb": "sample.ipynb",
        }
        for relative, name in fixtures.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture_path(name), target)
        (self.root / "docs" / "guide.rst").write_text(
            "Whole-file fallback.\n", encoding="utf-8"
        )

    def test_scan_read_extract_preserves_all_format_provenance(self) -> None:
        """Read every selected format and retain relative paths and units."""

        report = scan_repository(self.root)
        read_progress = Mock()
        extract_progress = Mock()
        files = read_repository_files(
            report, progress_callback=read_progress
        )
        units = extract_source_units(
            files, progress_callback=extract_progress
        )

        self.assertTrue(report["complete"])
        self.assertEqual(
            [file.path.as_posix() for file in files],
            sorted(report["selected_files"]),
        )
        self.assertEqual(read_progress.call_count, len(files))
        self.assertEqual(extract_progress.call_count, len(files))
        read_progress.assert_any_call(len(files), len(files))
        extract_progress.assert_any_call(len(files), len(files))
        self.assertEqual(
            {unit.path.as_posix() for unit in units},
            set(report["selected_files"]),
        )
        self.assertTrue(all(unit.start_line >= 1 for unit in units))
        self.assertTrue(
            all(unit.end_line >= unit.start_line for unit in units)
        )
        self.assertTrue(all(unit.text.strip() for unit in units))
        self.assertIn(
            ("python_function", "calculate"),
            {(unit.kind, unit.symbol) for unit in units},
        )
        self.assertIn(
            ("matlab_class", "Sample"),
            {(unit.kind, unit.symbol) for unit in units},
        )
        self.assertIn(
            "Installation",
            {unit.section for unit in units},
        )
        self.assertIn(
            ("notebook_code", 3),
            {(unit.kind, unit.cell) for unit in units},
        )
        self.assertEqual(
            [unit.kind for unit in units if unit.path.suffix == ".rst"],
            ["file"],
        )
        self.assertNotIn(
            "DO NOT INDEX OUTPUTS",
            "".join(unit.text for unit in units),
        )

    def test_selected_subset_ignores_unselected_sources(self) -> None:
        """Honor scan selection without rediscovering other files."""

        report = scan_repository(self.root, include_paths=["docs"])
        files = read_repository_files(report)
        units = extract_source_units(files)
        self.assertEqual(
            report["selected_files"],
            ["docs/guide.rst", "docs/sample.md"],
        )
        self.assertEqual(
            {unit.path.as_posix() for unit in units},
            {"docs/guide.rst", "docs/sample.md"},
        )

    def test_selected_file_disappearing_after_scan_raises(self) -> None:
        """Do not silently omit files deleted after repository scanning."""

        report = scan_repository(
            self.root, include_paths=["docs/guide.rst"]
        )
        (self.root / "docs" / "guide.rst").unlink()
        with self.assertRaises(FileNotFoundError):
            read_repository_files(report)

    def test_invalid_utf8_after_scan_is_rejected(self) -> None:
        """Reject selected files that cannot be decoded as UTF-8."""

        (self.root / "invalid.rst").write_bytes(b"\xff\xfe")
        report = scan_repository(
            self.root, include_paths=["invalid.rst"]
        )
        with self.assertRaisesRegex(ValueError, "not UTF-8"):
            read_repository_files(report)

    def test_incomplete_report_is_currently_accepted_by_reader(self) -> None:
        """Record the current reader contract pending policy review."""

        report = scan_repository(
            self.root, include_paths=["docs/guide.rst"]
        )
        report["complete"] = False
        self.assertEqual(
            read_repository_files(report),
            [SourceFile(Path("docs/guide.rst"), "Whole-file fallback.\n")],
        )


if __name__ == "__main__":
    unittest.main()
