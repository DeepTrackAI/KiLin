"""Verify that selected BibTeX entries are independently indexable.

Test exact source text, citation keys, malformed content and selection
through the normal repository scan and extraction dispatcher.

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
from kilin.sources.extensions import sources_bib
from kilin.sources.sources import (
    SourceFile, extract_source_units, read_repository_files,
)


class TestBibSource(unittest.TestCase):
    """Verify entry boundaries and selected source dispatch."""

    def test_entry_units_preserve_exact_text_and_provenance(self) -> None:
        data = (
            "% a database\n"
            "@string{jn = {Nature}}\n"
            "@article{lee2020,\n"
            "  author={Lee, A.},\n"
            "  title={A study}\n"
            "}\n\n"
            "@book{kim2021, title={B book}}\n"
        )
        units = sources_bib.extract_source_units(
            SourceFile(Path("refs.bib"), data)
        )
        self.assertEqual(
            [unit.kind for unit in units],
            ["bib_preamble", "bib_definition", "bib_entry", "bib_entry"],
        )
        self.assertEqual(
            [unit.symbol for unit in units],
            [None, "jn", "lee2020", "kim2021"],
        )
        self.assertEqual(
            [(unit.start_line, unit.end_line) for unit in units],
            [(1, 1), (2, 2), (3, 7), (8, 8)],
        )
        lines = data.splitlines(keepends=True)
        for unit in units:
            self.assertEqual(
                unit.text,
                "".join(lines[unit.start_line - 1:unit.end_line]),
            )

    def test_empty_and_unstructured_bib(self) -> None:
        self.assertEqual(sources_bib.extract_source_units(
            SourceFile(Path("a.bib"), "")), [])
        units = sources_bib.extract_source_units(
            SourceFile(Path("a.bib"), "an incomplete database\n")
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "bib_text")

    def test_selection_and_real_dispatch(self) -> None:
        self.assertIn(".bib", project_type("literature").extensions)
        with TemporaryDirectory() as folder:
            (Path(folder) / "refs.bib").write_text(
                "@article{lee, title={A study}}\n", encoding="utf-8"
            )
            report = scan_repository(folder, extensions=[".bib"])
            self.assertEqual(report["selected_files"], ["refs.bib"])
            files = read_repository_files(report)
            units = extract_source_units(files)
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].kind, "bib_entry")
            self.assertEqual(units[0].symbol, "lee")


if __name__ == "__main__":
    unittest.main()
