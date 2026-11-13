"""Verify that selected bibliography files enter managed search indexes.

A literature import can contain BibTeX, compiled bibliography and LaTeX
files as ordinary selected sources. Unselected bibliographies must not be
copied solely because a selected LaTeX file mentions their name.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.chunking import chunk_source_units
from kilin.repository import scan_repository
from kilin.sources.sources import extract_source_units, read_repository_files
from kilin.workspace import prepare_sources_workspace


class TestBibliographyImport(unittest.TestCase):
    """Exercise the actual project import, extraction and chunking paths."""

    def test_selected_bibliographies_are_copied_and_chunked(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            originals = root / "originals"
            originals.mkdir()
            text = {
                "paper.tex": "\\begin{document} Text.\\end{document}\n",
                "refs.bib": "@article{lee, title={An important paper}}\n",
                "paper.bbl": "\\bibitem{lee} An important paper.\n",
                "unselected.bib": "@article{private, title={Private}}\n",
            }
            for name, contents in text.items():
                (originals / name).write_text(
                    contents, encoding="utf-8"
                )
            project = prepare_sources_workspace(
                [originals], "literature", root / "workspace",
                selected_files={str(originals): [
                    "paper.tex", "refs.bib", "paper.bbl",
                ]},
            )
            self.assertTrue((project.source / "originals/refs.bib").is_file())
            self.assertTrue((project.source / "originals/paper.bbl").is_file())
            self.assertFalse(
                (project.source / "originals/unselected.bib").exists()
            )
            report = scan_repository(
                project.source, extensions=project.profile.extensions,
                max_file_bytes=project.profile.max_file_bytes,
            )
            self.assertEqual(
                set(report["selected_files"]),
                {"originals/paper.tex", "originals/refs.bib",
                 "originals/paper.bbl"},
            )
            units = extract_source_units(read_repository_files(report))
            chunks = chunk_source_units(
                units, count_tokens=lambda data: len(data.split()),
                target_tokens=100, max_tokens=150, overlap_tokens=0,
            )
            kinds = {(c.path.name, c.kind) for c in chunks}
            self.assertIn(("refs.bib", "bib_entry"), kinds)
            self.assertIn(("paper.bbl", "bbl_entry"), kinds)
            self.assertTrue(any(
                "important paper" in c.text for c in chunks
                if c.path.suffix in {".bib", ".bbl"}
            ))


if __name__ == "__main__":
    unittest.main()
