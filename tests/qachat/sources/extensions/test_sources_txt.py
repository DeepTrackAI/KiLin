"""Unit tests for blank-line-delimited plain-text extraction.

Tests cover paragraph boundaries, whitespace, Unicode, line endings, and
exact source coordinates without filesystem or external dependencies.

"""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.sources.extensions import sources_txt
from qachat.sources.sources import SourceFile
from tests.qachat.sources._support import fixture_text


class TestTextExtraction(unittest.TestCase):
    """Verify plain-text paragraph extraction and provenance."""

    def test_fixture_paragraphs_and_unicode(self) -> None:
        """Split on blank lines and preserve non-ASCII source text."""

        text = fixture_text("sample.txt")
        units = sources_txt.extract_source_units(
            SourceFile(Path("sample.txt"), text)
        )
        self.assertEqual(len(units), 3)
        self.assertEqual(
            [(unit.start_line, unit.end_line) for unit in units],
            [(1, 2), (4, 4), (7, 7)],
        )
        self.assertIn("café", units[1].text)
        self.assertTrue(all(unit.kind == "text_paragraph" for unit in units))

    def test_without_blank_lines_is_one_paragraph(self) -> None:
        """Retain contiguous nonblank lines as one unit."""

        text = "One\nTwo\n"
        units = sources_txt.extract_source_units(
            SourceFile(Path("short.txt"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text, text)

    def test_multiple_blank_lines_are_not_included(self) -> None:
        """Drop delimiters rather than merging them into paragraphs."""

        text = "First\n\n\n\nSecond\n"
        units = sources_txt.extract_source_units(
            SourceFile(Path("notes.txt"), text)
        )
        self.assertEqual([unit.text for unit in units],
                         ["First\n", "Second\n"])
        self.assertEqual([unit.start_line for unit in units], [1, 5])

    def test_leading_and_trailing_blank_lines_are_omitted(self) -> None:
        """Preserve coordinates after discarding surrounding whitespace."""

        units = sources_txt.extract_source_units(SourceFile(
            Path("notes.txt"), "\n\nContent\n\n\n"
        ))
        self.assertEqual(len(units), 1)
        self.assertEqual((units[0].start_line, units[0].end_line), (3, 3))
        self.assertEqual(units[0].text, "Content\n")

    def test_crlf_boundaries_are_preserved(self) -> None:
        """Keep original CRLF line endings in source units."""

        text = "First\r\n\r\nSecond\r\n"
        units = sources_txt.extract_source_units(
            SourceFile(Path("notes.txt"), text)
        )
        self.assertEqual([unit.text for unit in units],
                         ["First\r\n", "Second\r\n"])
        self.assertEqual([unit.start_line for unit in units], [1, 3])

    def test_tabs_and_spaces_separate_paragraphs(self) -> None:
        """Treat lines containing only whitespace as separators."""

        units = sources_txt.extract_source_units(SourceFile(
            Path("notes.txt"), "First\n\t \nSecond\n"
        ))
        self.assertEqual([unit.text for unit in units],
                         ["First\n", "Second\n"])

    def test_single_line_without_newline(self) -> None:
        """Accept a final paragraph without a trailing newline."""

        units = sources_txt.extract_source_units(
            SourceFile(Path("notes.txt"), "One line")
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text, "One line")
        self.assertEqual((units[0].start_line, units[0].end_line), (1, 1))

    def test_empty_and_blank_text_produces_no_units(self) -> None:
        """Avoid empty paragraph units."""

        for text in ("", "\n\n", "  \n\t\n"):
            with self.subTest(text=text):
                self.assertEqual(
                    sources_txt.extract_source_units(
                        SourceFile(Path("blank.txt"), text)
                    ),
                    [],
                )

    def test_unit_text_matches_original_line_ranges(self) -> None:
        """Verify every paragraph's inclusive one-based provenance."""

        text = fixture_text("sample.txt")
        lines = text.splitlines(keepends=True)
        units = sources_txt.extract_source_units(
            SourceFile(Path("sample.txt"), text)
        )
        for unit in units:
            self.assertEqual(
                unit.text,
                "".join(lines[unit.start_line - 1:unit.end_line]),
            )


if __name__ == "__main__":
    unittest.main()
