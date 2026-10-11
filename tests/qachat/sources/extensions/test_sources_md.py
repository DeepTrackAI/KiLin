"""Unit tests for lightweight Markdown heading extraction.

The extractor recognizes ATX headings outside fenced code without parsing
nested document structure. Tests preserve this lightweight contract.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.sources.extensions import sources_md
from qachat.sources.sources import SourceFile
from tests.qachat.sources._support import fixture_text


class TestMarkdownExtraction(unittest.TestCase):
    """Verify heading boundaries, text preservation, and provenance."""

    def test_fixture_sections_and_preamble(self) -> None:
        """Partition preamble and successive heading sections."""

        text = fixture_text("sample.md")
        units = sources_md.extract_source_units(
            SourceFile(Path("sample.md"), text)
        )
        self.assertEqual(
            [unit.kind for unit in units],
            ["markdown_preamble"] + ["markdown_section"] * 3,
        )
        self.assertEqual(
            [unit.section for unit in units],
            [None, "Installation", "Usage", "Details"],
        )
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_single_heading_without_preamble(self) -> None:
        """Avoid an empty preamble when the first line is a heading."""

        units = sources_md.extract_source_units(SourceFile(
            Path("readme.md"), "# Title\nBody\n"
        ))
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].section, "Title")
        self.assertEqual((units[0].start_line, units[0].end_line), (1, 2))

    def test_no_heading_is_one_preamble(self) -> None:
        """Retain heading-free Markdown as a preamble unit."""

        text = "An introduction.\nAnother line.\n"
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "markdown_preamble")
        self.assertEqual(units[0].text, text)

    def test_hashes_without_whitespace_are_not_headings(self) -> None:
        """Require whitespace after one to six leading hash marks."""

        text = "#No space\n####### Too many hashes\n"
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "markdown_preamble")
        self.assertEqual(units[0].text, text)

    def test_heading_levels_do_not_create_nested_units(self) -> None:
        """Keep the documented flat section model for nested headings."""

        text = "# Main\nOne\n## Subsection\nTwo\n### Detail\nThree\n"
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual(
            [unit.section for unit in units],
            ["Main", "Subsection", "Detail"],
        )
        self.assertTrue(all(not unit.children for unit in units))

    def test_heading_title_strips_closing_hashes(self) -> None:
        """Normalize heading labels without rewriting source text."""

        text = "## Heading ###\nContent\n"
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual(units[0].section, "Heading")
        self.assertEqual(units[0].text, text)

    def test_headings_inside_backtick_fences_are_not_sections(self) -> None:
        """Keep fenced Python comments inside the surrounding section."""

        text = (
            "Preamble\n# Intro\n```python\n# Comment\n"
            "```\n## Usage\nExample\n"
        )
        units = sources_md.extract_source_units(
            SourceFile(Path("readme.md"), text)
        )
        self.assertEqual([unit.section for unit in units],
                         [None, "Intro", "Usage"])
        self.assertEqual([(u.start_line, u.end_line) for u in units],
                         [(1, 1), (2, 5), (6, 7)])
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_tilde_fence_closes_only_with_matching_marker(self) -> None:
        """Require a matching closer of at least the opener's length."""

        text = (
            "# Start\n~~~~ text\n# Fake one\n~~~\n"
            "# Fake two\n````\n# Fake three\n"
            "~~~~ extra\n# Fake four\n~~~~  \n## End\n"
        )
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual([unit.section for unit in units],
                         ["Start", "End"])
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_unclosed_fence_hides_all_later_headings(self) -> None:
        """Do not end an open backtick fence at a shorter delimiter."""

        text = "# Section\n````js\n# Fake\n```\n## Also fake\n"
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].section, "Section")
        self.assertEqual(units[0].text, text)

    def test_fence_in_preamble_does_not_create_a_section(self) -> None:
        """Keep heading-shaped lines in initial fenced text as preamble."""

        text = "   ```md\n# Fake\n   ```\n# Actual\nText\n"
        units = sources_md.extract_source_units(
            SourceFile(Path("notes.md"), text)
        )
        self.assertEqual([unit.section for unit in units],
                         [None, "Actual"])
        self.assertEqual("".join(unit.text for unit in units), text)

    def test_blank_markdown_produces_no_units(self) -> None:
        """Avoid creating units from whitespace-only documents."""

        for text in ("", " \n\n"):
            with self.subTest(text=text):
                self.assertEqual(
                    sources_md.extract_source_units(
                        SourceFile(Path("blank.md"), text)
                    ),
                    [],
                )

    def test_section_coordinates_match_original_lines(self) -> None:
        """Preserve exact one-based source coordinates."""

        text = fixture_text("sample.md")
        lines = text.splitlines(keepends=True)
        units = sources_md.extract_source_units(
            SourceFile(Path("sample.md"), text)
        )
        for unit in units:
            with self.subTest(section=unit.section):
                self.assertEqual(
                    unit.text,
                    "".join(lines[unit.start_line - 1:unit.end_line]),
                )


if __name__ == "__main__":
    unittest.main()
