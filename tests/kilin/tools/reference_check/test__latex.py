"""Unit tests for reading a LaTeX paper's citations and bibliography.

Check the BibTeX parser, references written from ``.bib`` fields, LaTeX
lines reduced to readable text with citations as numbers, the three
bibliography sources, and a full deterministic check of a LaTeX paper.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.chunking import Chunk
from kilin.tools.reference_check import tool
from kilin.tools.reference_check._latex import (
    _readable_line,
    bib_entry_text,
    latex_references,
    parse_bib,
)

BIB = r"""@string{plos = "PLoS One"}
@comment{Exported from a reference manager.}
@article{rubinov10,
  author = {Rubinov, Mikail and Sporns, Olaf},
  title = {Complex network measures of brain connectivity: {U}ses},
  journal = {NeuroImage}, volume = 52, pages = {1059--1069},
  year = 2010, doi = {10.1016/j.neuroimage.2009.10.003}
}
@article{mij17, author = "Mijalkov, Mite and Kakaei, Ehsan",
  title = "{BRAPH}: A graph theory software", journal = plos,
  year = {2017}}
@misc{unused, author = {Nobody, A.}, title = {Never cited}, year = 2000}
"""

PAPER = r"""\documentclass{article}
\usepackage{natbib}
\begin{document}
\section{Introduction}
Graph theory describes brain networks well \cite{rubinov10}.
BRAPH implements these measures \citep[see][p.~3]{mij17, rubinov10}.
A missing work is cited here too \cite{ghost}. % \cite{commented}

\section{Methods}
We used the toolbox for all analyses \citet{mij17}.\nocite{unused}
\bibliography{refs}
\end{document}
"""

BIBLIOGRAPHY = r"""\begin{thebibliography}{9}
\bibitem{mij17} M. Mijalkov, \emph{BRAPH: A graph theory software},
\newblock PLoS One (2017).
\bibitem[Rubinov(2010)]{rubinov10} M. Rubinov and O. Sporns, NeuroImage
(2010).
\bibitem{spare} An entry never cited (2001).
\end{thebibliography}
"""


def _chunks(text: str, name: str = "draft.tex") -> list[Chunk]:
    """Return the text after ``\\begin{document}`` as two passages."""
    lines = text.split("\n")
    first = next(
        index for index, line in enumerate(lines) if "\\begin{document}"
        in line
    ) + 1
    middle = (first + len(lines)) // 2
    return [
        Chunk("c2", Path(name), "\n".join(lines[middle:]), middle + 1,
              len(lines), "latex_section", None, "Methods"),
        Chunk("c1", Path(name), "\n".join(lines[first:middle]), first + 1,
              middle, "latex_section", None, "Introduction"),
    ]


class TestBib(unittest.TestCase):

    def test_parse_bib(self) -> None:
        entries = parse_bib(BIB)
        self.assertEqual(list(entries), ["rubinov10", "mij17", "unused"])
        self.assertEqual(entries["rubinov10"]["volume"], "52")
        self.assertEqual(entries["rubinov10"]["type"], "article")
        self.assertEqual(
            entries["rubinov10"]["title"],
            "Complex network measures of brain connectivity: {U}ses",
        )
        # Abbreviations are expanded, "#" joins parts.
        self.assertEqual(entries["mij17"]["journal"], "PLoS One")
        self.assertEqual(
            parse_bib('@string{n = "Nature"}@misc{a, x = n # " Methods"}')
            ["a"]["x"], "Nature Methods",
        )
        # An entry that does not close is skipped.
        self.assertEqual(list(parse_bib("@article{a, title = {Open")), [])

    def test_entry_text(self) -> None:
        entries = parse_bib(BIB)
        self.assertEqual(
            bib_entry_text(entries["rubinov10"]),
            "Rubinov, Mikail; Sporns, Olaf (2010). Complex network "
            "measures of brain connectivity: Uses. NeuroImage, 52, "
            "1059--1069. doi:10.1016/j.neuroimage.2009.10.003",
        )
        many = " and ".join(f"Author{i}, A." for i in range(8))
        text = bib_entry_text(
            {"author": many, "title": "T", "eprint": "2101.00001"}
        )
        self.assertIn("Author5, A.; et al.", text)
        self.assertNotIn("Author6", text)
        self.assertTrue(text.endswith("arXiv:2101.00001"))
        self.assertEqual(bib_entry_text({}), "")


class TestReadableLines(unittest.TestCase):

    def test_lines(self) -> None:
        numbers = {"a": 1, "b": 2}
        for line, readable in (
            ("Known \\cite{a,b}.", "Known [1, 2]."),
            ("Notes \\citep[see][p.~2]{b}.", "Notes [2]."),
            ("Unknown \\parencite{zzz}.", "Unknown [?]."),
            ("Listed\\nocite{a} only.", "Listed only."),
            ("Range $[0,1]$ here.", "Range (0,1) here."),
            ("Text 5\\% % \\cite{a} commented", "Text 5% "),
            ("Na\\\"{i}ve \\emph{bold} Fig.~\\ref{f}",
             "Naive bold Fig. (ref)"),
            ("\\section{The \\emph{X} tool}\\label{s}", "The X tool."),
            ("\\textbf{Bold} \\v{c}a \\bibitem", "Bold ca "),
        ):
            with self.subTest(line=line):
                self.assertEqual(_readable_line(line, numbers), readable)


class TestLatexReferences(unittest.TestCase):

    def test_bib_file_numbered_by_first_citation(self) -> None:
        with TemporaryDirectory() as folder:
            file = Path(folder) / "draft.tex"
            file.write_text(PAPER, encoding="utf-8")
            (Path(folder) / "refs.bib").write_text(BIB, encoding="utf-8")
            found = latex_references(_chunks(PAPER), file)
        self.assertEqual(found.keys, {1: "rubinov10", 2: "mij17"})
        self.assertTrue(found.entries[1].text.startswith("Mijalkov, Mite"))
        self.assertTrue(all(entry.printed for entry in found.entries))
        self.assertEqual(found.source, "refs.bib")
        self.assertEqual(found.undefined, ("ghost",))
        self.assertIn("entries never cited are not listed", found.note)
        self.assertIn("`ghost`", found.note)
        # Passages in line order, line numbers kept, citations as numbers.
        self.assertEqual([chunk.identifier for chunk in found.body],
                         ["c1", "c2"])
        text = found.body[0].text
        self.assertIn("brain networks well [1].", text)
        self.assertIn("these measures [2, 1].", text)
        self.assertIn("cited here too [?].", text)
        self.assertNotIn("commented", text)
        self.assertEqual(text.count("\n"),
                         found.body[0].end_line - found.body[0].start_line)

    def test_thebibliography_keeps_printed_order(self) -> None:
        paper = PAPER.replace("\\bibliography{refs}", BIBLIOGRAPHY)
        with TemporaryDirectory() as folder:
            file = Path(folder) / "draft.tex"
            file.write_text(paper, encoding="utf-8")
            found = latex_references(_chunks(paper), file)
        self.assertEqual(found.keys,
                         {1: "mij17", 2: "rubinov10", 3: "spare"})
        self.assertEqual(
            found.entries[0].text,
            "M. Mijalkov, BRAPH: A graph theory software, PLoS One (2017).",
        )
        self.assertEqual(found.source, "thebibliography")
        body = "".join(chunk.text for chunk in found.body)
        self.assertIn("these measures [1, 2].", body)
        self.assertNotIn("never cited", body)

    def test_bbl_file_beside_the_paper(self) -> None:
        with TemporaryDirectory() as folder:
            file = Path(folder) / "draft.tex"
            file.write_text(PAPER, encoding="utf-8")
            (Path(folder) / "draft.bbl").write_text(
                BIBLIOGRAPHY, encoding="utf-8"
            )
            found = latex_references(_chunks(PAPER), file)
        self.assertEqual(found.source, "draft.bbl")
        self.assertEqual(found.keys[2], "rubinov10")

    def test_no_bibliography(self) -> None:
        found = latex_references(_chunks(PAPER), None)
        self.assertEqual(found.entries, ())
        self.assertEqual(found.source, "")
        self.assertEqual(found.undefined, ("rubinov10", "mij17", "ghost"))
        self.assertIn("No bibliography found", found.note)


class TestLatexCheck(unittest.TestCase):

    def test_check_references_of_a_latex_paper(self) -> None:
        paper = PAPER.replace("\\bibliography{refs}", BIBLIOGRAPHY)
        with TemporaryDirectory() as folder:
            file = Path(folder) / "draft.tex"
            file.write_text(paper, encoding="utf-8")
            report = tool.check_references(
                Path("draft.tex"), _chunks(paper), file, []
            )
        self.assertEqual(len(report.entries), 3)
        self.assertEqual(report.scan.uncited, (3,))
        cited = {
            number: [sentence.text for sentence in sentences]
            for number, sentences in report.cited_by.items()
        }
        self.assertEqual(
            cited[1],
            ["BRAPH implements these measures [1, 2].",
             "We used the toolbox for all analyses [1]."],
        )
        # The section title is a sentence of its own.
        self.assertEqual(
            cited[2][0],
            "Graph theory describes brain networks well [2].",
        )
        self.assertIn("`ghost`", report.source_note)
        answer, _ = tool.chat_answer(report, 10, None)
        self.assertIn("References from thebibliography.", answer)

    def test_latex_paper_without_bibliography(self) -> None:
        report = tool.check_references(
            Path("draft.tex"), _chunks(PAPER), None, []
        )
        answer, quoted = tool.chat_answer(report, 10, None)
        self.assertTrue(answer.startswith(
            "I found no references in draft.tex. No bibliography found"
        ))
        self.assertEqual(quoted, ())


if __name__ == "__main__":
    unittest.main()
