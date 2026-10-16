"""Unit tests for JSON notebook cell extraction and provenance.

Tests use synthetic notebook JSON and never execute code cells. Outputs,
metadata, and raw cells are intentionally excluded from source units.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import unittest

from kilin.sources.extensions import sources_ipynb
from kilin.sources.sources import SourceFile, SourceUnit
from tests.kilin.sources._support import fixture_text


class TestNotebookExtraction(unittest.TestCase):
    """Verify notebook validation, visible cells, and local line ranges."""

    @staticmethod
    def _extract(cells: list[object]) -> list[SourceUnit]:
        """Extract units from a minimal synthetic notebook.

        Parameters
        ----------
        cells: list[object]
            Notebook cell objects to serialize.

        Returns
        -------
        list[SourceUnit]
            Extracted source units.

        """

        notebook = {"cells": cells}
        return sources_ipynb.extract_source_units(SourceFile(
            Path("sample.ipynb"), json.dumps(notebook)
        ))

    def test_fixture_excludes_outputs_metadata_and_raw_cells(self) -> None:
        """Index only visible code and Markdown cell sources."""

        units = sources_ipynb.extract_source_units(SourceFile(
            Path("sample.ipynb"), fixture_text("sample.ipynb")
        ))
        self.assertEqual(
            [(unit.kind, unit.cell) for unit in units],
            [
                ("notebook_markdown", 1),
                ("notebook_code", 3),
                ("notebook_code", 5),
            ],
        )
        self.assertEqual(units[0].section, "Notebook overview")
        self.assertEqual((units[1].start_line, units[1].end_line), (1, 2))
        text = "".join(unit.text for unit in units)
        for marker in (
            "DO NOT INDEX OUTPUTS",
            "DO NOT INDEX METADATA",
            "DO NOT INDEX RAW CELLS",
        ):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)

    def test_string_and_list_sources_are_supported(self) -> None:
        """Join string-array sources and preserve plain string sources."""

        units = self._extract([
            {"cell_type": "code", "source": ["a = 1\n", "b = 2"]},
            {"cell_type": "markdown", "source": "# Heading\nBody"},
        ])
        self.assertEqual([unit.text for unit in units],
                         ["a = 1\nb = 2", "# Heading\nBody"])
        self.assertEqual([unit.end_line for unit in units], [2, 2])
        self.assertEqual(units[1].section, "Heading")

    def test_cell_number_counts_skipped_cells(self) -> None:
        """Keep original cell indices after raw and empty cells."""

        units = self._extract([
            {"cell_type": "raw", "source": "ignored"},
            {"cell_type": "code", "source": "  \n"},
            {"cell_type": "markdown", "source": "Visible"},
        ])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].cell, 3)

    def test_first_markdown_heading_sets_section(self) -> None:
        """Use the first visible ATX heading as a cell section label."""

        units = self._extract([
            {"cell_type": "markdown", "source":
             "Intro\n## First ##\n### Second\n"},
        ])
        self.assertEqual(units[0].section, "First")
        self.assertEqual(units[0].text,
                         "Intro\n## First ##\n### Second\n")

    def test_markdown_without_heading_has_no_section(self) -> None:
        """Leave section absent when no heading is present."""

        units = self._extract([
            {"cell_type": "markdown", "source": "Just prose."},
        ])
        self.assertIsNone(units[0].section)

    def test_empty_and_blank_cells_produce_no_units(self) -> None:
        """Skip cells with empty or whitespace-only source text."""

        units = self._extract([
            {"cell_type": "code", "source": ""},
            {"cell_type": "markdown", "source": ["  \n"]},
            {"cell_type": "code"},
        ])
        self.assertEqual(units, [])

    def test_invalid_json_and_missing_cells_are_rejected(self) -> None:
        """Translate malformed notebook roots into ValueError."""

        for text in ("{", "{}", "[]", "null", "42"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
                    sources_ipynb.extract_source_units(
                        SourceFile(Path("broken.ipynb"), text)
                    )

    def test_non_list_cells_are_rejected(self) -> None:
        """Require the notebook cells member to be a list."""

        for cells in (None, "cells", {}, 42):
            with self.subTest(cells=cells):
                text = json.dumps({"cells": cells})
                with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
                    sources_ipynb.extract_source_units(
                        SourceFile(Path("broken.ipynb"), text)
                    )

    def test_non_object_cells_are_rejected(self) -> None:
        """Reject invalid cell records rather than skipping them."""

        for cell in (None, "cell", [], 42):
            with self.subTest(cell=cell):
                with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
                    self._extract([cell])

    def test_invalid_source_types_are_rejected(self) -> None:
        """Require code and Markdown source to be strings or string lists."""

        for source in (None, 42, {}, ["valid", 2], [None]):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
                    self._extract([
                        {"cell_type": "code", "source": source},
                    ])

    def test_malformed_cell_types_are_rejected(self) -> None:
        """Reject non-string cell types with a consistent ValueError."""

        for cell_type in (None, 7, 1.5, False, [], {}, ["code"]):
            with self.subTest(cell_type=cell_type):
                with self.assertRaisesRegex(ValueError, "Invalid Jupyter"):
                    self._extract([
                        {"cell_type": cell_type, "source": "text"},
                    ])

    def test_markdown_cell_ignores_headings_inside_code_fences(self) -> None:
        """Find the first real heading, not a comment inside a fence."""

        units = self._extract([
            {"cell_type": "markdown", "source":
             "```python\n# Fake\n```\n## Actual\nBody\n"},
            {"cell_type": "markdown", "source":
             "~~~text\n# Still fake\n~~~\n"},
            {"cell_type": "markdown", "source":
             "````\n# Fake\n```\n# Still fake\n"},
        ])
        self.assertEqual([unit.section for unit in units],
                         ["Actual", None, None])
        self.assertEqual([unit.cell for unit in units], [1, 2, 3])

    def test_unknown_cell_type_is_ignored(self) -> None:
        """Ignore unrecognized but hashable cell types."""

        units = self._extract([
            {"cell_type": "raw", "source": "raw"},
            {"cell_type": "unknown", "source": "unknown"},
            {"cell_type": "code", "source": "answer = 42"},
        ])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].cell, 3)

    def test_empty_notebook_has_no_units(self) -> None:
        """Return no units for a valid notebook with no cells."""

        self.assertEqual(self._extract([]), [])


if __name__ == "__main__":
    unittest.main()
