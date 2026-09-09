"""Tests for safe source reading and extension coordination."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.repository import scan_repository
from qachat.sources.sources import (
    SourceFile,
    extract_source_units,
    read_repository_files,
)


class TestSources(unittest.TestCase):

    def test_read_repository_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text(
                "value = 1\n",
                encoding="utf-8",
            )
            (root / "binary.py").write_bytes(b"\xff\xfe")

            report = scan_repository(root)
            with self.assertRaisesRegex(ValueError, "not UTF-8"):
                read_repository_files(report)

            report = scan_repository(
                root,
                include_paths=["module.py"],
            )
            files = read_repository_files(report)

        self.assertEqual(
            files,
            [SourceFile(Path("module.py"), "value = 1\n")],
        )

    def test_paths_must_stay_inside_repository(self) -> None:
        with TemporaryDirectory() as directory:
            report = {
                "repository": directory,
                "selected_files": ["../outside.py"],
            }
            with self.assertRaisesRegex(ValueError, "inside"):
                read_repository_files(report)

    def test_unknown_text_type_stays_whole(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text(
                "Keep this whole.\n",
                encoding="utf-8",
            )
            report = scan_repository(
                root,
                extensions=[".txt"],
            )
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "file")
        self.assertEqual(units[0].path, Path("notes.txt"))
        self.assertEqual(units[0].text, "Keep this whole.\n")


if __name__ == "__main__":
    unittest.main()
