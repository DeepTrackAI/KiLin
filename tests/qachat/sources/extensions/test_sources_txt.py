"""Tests for plain-text natural-unit extraction."""

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


class TestSourcesTxt(unittest.TestCase):

    def test_text_file_becomes_paragraph_units(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text(
                "First paragraph.\nStill first.\n\nSecond paragraph.\n",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".txt"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(
            [unit.kind for unit in units],
            ["text_paragraph", "text_paragraph"],
        )
        self.assertEqual(units[0].text, "First paragraph.\nStill first.\n")
        self.assertEqual(units[0].start_line, 1)
        self.assertEqual(units[0].end_line, 2)
        self.assertEqual(units[1].text, "Second paragraph.\n")
        self.assertEqual(units[1].start_line, 4)
        self.assertEqual(units[1].end_line, 4)

    def test_text_file_without_blank_lines_is_one_unit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "short.txt").write_text(
                "One line.\nAnother line, no blank separator.\n",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".txt"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual([unit.kind for unit in units], ["text_paragraph"])
        self.assertEqual(
            units[0].text,
            "One line.\nAnother line, no blank separator.\n",
        )

    def test_multiple_blank_lines_between_paragraphs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "spaced.txt").write_text(
                "Para one.\n\n\n\nPara two.\n",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".txt"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(
            [unit.kind for unit in units],
            ["text_paragraph", "text_paragraph"],
        )
        self.assertEqual(units[0].text, "Para one.\n")
        self.assertEqual(units[1].text, "Para two.\n")

    def test_leading_and_trailing_blank_lines_are_dropped(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "padded.txt").write_text(
                "\n\nActual content.\n\n\n",
                encoding="utf-8",
            )
            report = scan_repository(root, extensions=[".txt"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual([unit.kind for unit in units], ["text_paragraph"])
        self.assertEqual(units[0].text, "Actual content.\n")

    def test_blank_file_produces_no_units(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.txt").write_text("\n\n\n", encoding="utf-8")
            report = scan_repository(root, extensions=[".txt"])
            files = read_repository_files(report)
            units = extract_source_units(files)

        self.assertEqual(units, [])


if __name__ == "__main__":
    unittest.main()
