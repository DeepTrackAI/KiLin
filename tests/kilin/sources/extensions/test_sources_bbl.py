"""Verify compiled bibliography entries have searchable source units.

Selected ``.bbl`` files preserve original LaTeX, printed item order and
file-relative line coordinates for retrieval and source previews.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.project_type import project_type
from kilin.repository import scan_repository
from kilin.sources.extensions import sources_bbl
from kilin.sources.sources import (
    SourceFile, extract_source_units, read_repository_files,
)


class TestBblSource(unittest.TestCase):
    """Verify entries and actual format dispatch."""

    def test_item_units_and_coordinates(self) -> None:
        data = (
            "\\begin{thebibliography}{9}\n"
            "\\bibitem{lee} A study.\n"
            "More detail.\n"
            "\\bibitem[Kim(2021)]{kim} B study.\n"
            "\\end{thebibliography}\n"
        )
        units = sources_bbl.extract_source_units(
            SourceFile(Path("paper.bbl"), data)
        )
        self.assertEqual(
            [unit.kind for unit in units],
            ["bbl_preamble", "bbl_entry", "bbl_entry"],
        )
        self.assertEqual([unit.symbol for unit in units],
                         [None, "lee", "kim"])
        self.assertEqual(
            [(unit.start_line, unit.end_line) for unit in units],
            [(1, 1), (2, 3), (4, 5)],
        )
        lines = data.splitlines(keepends=True)
        for unit in units:
            self.assertEqual(
                unit.text,
                "".join(lines[unit.start_line - 1:unit.end_line]),
            )

    def test_empty_and_unstructured_bbl(self) -> None:
        self.assertEqual(sources_bbl.extract_source_units(
            SourceFile(Path("a.bbl"), "")), [])
        units = sources_bbl.extract_source_units(
            SourceFile(Path("a.bbl"), "No bibitem\n")
        )
        self.assertEqual(units[0].kind, "bbl_text")

    def test_selection_and_real_dispatch(self) -> None:
        self.assertIn(".bbl", project_type("literature").extensions)
        with TemporaryDirectory() as folder:
            (Path(folder) / "paper.bbl").write_text(
                "\\bibitem{lee} A study.\n", encoding="utf-8"
            )
            report = scan_repository(folder, extensions=[".bbl"])
            self.assertEqual(report["selected_files"], ["paper.bbl"])
            units = extract_source_units(read_repository_files(report))
            self.assertEqual(units[0].kind, "bbl_entry")
            self.assertEqual(units[0].symbol, "lee")


if __name__ == "__main__":
    unittest.main()
