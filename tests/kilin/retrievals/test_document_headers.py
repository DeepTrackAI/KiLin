"""Verify document titles, section headings, and paper search document_search.

Titles come from real PDF metadata in the test fixtures, Markdown headings,
and file names; headings come from synthetic passages.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilin import rag
from kilin.chunking import Chunk
from kilin.retrievals import document_headers as document_search
from kilin.retrievals.document_headers import DocumentInfo
from tests.kilin import _rag_support as fixtures

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pdfs"


def _page(identifier: str, text: str, page: int, start: int = 1,
          path: str = "paper.pdf") -> Chunk:
    """Return one PDF passage on a page."""
    return Chunk(identifier, Path(path), text, start, start + 5,
                 "pdf_page", None, f"Page {page}")


class TestDocumentInfo(unittest.TestCase):

    def test_pdf_metadata_gives_title_and_first_author(self):
        info = document_search.document_info(
            FIXTURES, Path("mijalkov17braph.pdf"), ()
        )
        self.assertEqual(
            info.title,
            "BRAPH: A graph theory software for the analysis of brain "
            "connectivity",
        )
        self.assertEqual(info.first_author, "Mijalkov")

    def test_semicolon_separated_authors(self):
        info = document_search.document_info(
            FIXTURES, Path("patil25deep.pdf"), ()
        )
        self.assertEqual(info.first_author, "Patil")

    def test_unreadable_pdf_falls_back_to_the_file_name(self):
        info = document_search.document_info(
            FIXTURES, Path("missing-paper.pdf"), ()
        )
        self.assertEqual(info, DocumentInfo("missing-paper", None))

    def test_placeholder_metadata_titles_are_ignored(self):
        for title in ("Untitled", "draft.docx", "Microsoft Word - x", "ab"):
            with self.subTest(title=title):
                self.assertIsNone(document_search._clean_title(title))
        self.assertEqual(document_search._clean_title(" A  long\ntitle "),
                         "A long title")

    def test_word_metadata_and_heading_fallback(self):
        """Prefer real Word properties, then a textual heading."""

        fixture = (
            Path(__file__).resolve().parents[1]
            / "sources/extensions/fixtures/sample.docx"
        )
        info = document_search.document_info(
            fixture.parent, Path(fixture.name), ()
        )
        self.assertEqual(
            info, DocumentInfo("A Sample Study of Word Documents", "Lovelace")
        )

        title = Chunk(
            "word", Path("missing.docx"), "# Study of Sleep\nResults", 1,
            2, "markdown_section", None, "Study of Sleep",
        )
        self.assertEqual(
            document_search.document_info(
                fixture.parent, title.path, [title]
            ),
            DocumentInfo("Study of Sleep", None),
        )

    def test_markdown_title_and_plain_fallback(self):
        note = Chunk("m", Path("notes/review.md"), "Intro\n# Sleep and age\n",
                     1, 3, "markdown_preamble", None, None)
        self.assertEqual(
            document_search.document_info(Path("."), note.path, [note]),
            DocumentInfo("Sleep and age", None),
        )
        self.assertEqual(
            document_search.document_info(Path("."), Path("a/b.txt"), []),
            DocumentInfo("b", None),
        )

    def test_latex_title_from_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "draft.tex").write_text(
                "\\documentclass{article}\n"
                "\\title{Brain networks with \\emph{BRAPH}}\n"
                "\\begin{document}\nText.\n\\end{document}\n",
                encoding="utf-8",
            )
            (source / "chapter.tex").write_text(
                "\\section{Results}\nText.\n", encoding="utf-8"
            )
            self.assertEqual(
                document_search.document_info(
                    source, Path("draft.tex"), []
                ),
                DocumentInfo("Brain networks with BRAPH", None),
            )
            self.assertEqual(
                document_search.document_info(
                    source, Path("chapter.tex"), []
                ),
                DocumentInfo("chapter", None),
            )
            # A missing file falls back to the file name.
            self.assertEqual(
                document_search.document_info(
                    source, Path("gone.tex"), []
                ),
                DocumentInfo("gone", None),
            )

class TestSectionHeadings(unittest.TestCase):

    def test_headings_follow_passages_in_reading_order(self):
        chunks = [
            # Deliberately out of order: reading order is page, then line.
            _page("c3", "More results here.", 2, 1),
            _page("c1", "Title\nAbstract\nWe study sleep.", 1, 1),
            _page("c2", "Text before.\nmore\nmore\n2. Methods\nWe used",
                  1, 20),
        ]
        headings = document_search.section_headings(chunks)
        self.assertEqual(headings["c1"], "Abstract")
        # A heading late in a passage starts the next passages' section.
        self.assertEqual(headings["c2"], "Abstract")
        self.assertEqual(headings["c3"], "2. Methods")

    def test_numbers_and_sentences_are_not_headings(self):
        for line in ("0.91. Furthermore, the model", "07.003 PMID: 239",
                     "75 ROIs in the next run",
                     "3.1. This sentence ends with a period."):
            with self.subTest(line=line):
                self.assertFalse(document_search._is_heading(line))
        for line in ("3.1. Software architecture", "Data availability",
                     "Materials and methods", "4 Discussion"):
            with self.subTest(line=line):
                self.assertTrue(document_search._is_heading(line))

    def test_documents_are_followed_separately(self):
        chunks = [
            _page("a1", "Results\nA.", 1, path="a.pdf"),
            _page("b1", "No heading yet.", 1, path="b.pdf"),
        ]
        headings = document_search.section_headings(chunks)
        self.assertEqual(headings, {"a1": "Results"})

    def test_markdown_passages_keep_their_section(self):
        chunk = Chunk("m", Path("n.md"), "Text", 4, 6,
                      "markdown_section", None, "Findings")
        self.assertEqual(
            document_search.section_headings([chunk]), {"m": "Findings"}
        )

    def test_latex_passages_keep_their_section(self):
        chunks = [
            Chunk("f", Path("d.tex"), "\\maketitle", 7, 9,
                  "latex_front", None, None),
            Chunk("s", Path("d.tex"), "\\section{Methods}\nText", 10, 12,
                  "latex_section", None, "Methods"),
        ]
        self.assertEqual(
            document_search.section_headings(chunks), {"s": "Methods"}
        )


class TestDocumentHeaders(unittest.TestCase):

    def test_header_lines_and_scope(self):
        chunks = [
            _page("p", "Results\nWe found.", 3),
            Chunk("t", Path("notes.txt"), "Plain", 1, 1, "text_paragraph",
                  None, None),
            Chunk("code", Path("a.py"), "x = 1", 1, 1, "python_file",
                  None, None),
        ]
        headers = document_search.document_headers(chunks, {
            Path("paper.pdf"): DocumentInfo("A study of sleep", "Smith"),
            Path("notes.txt"): DocumentInfo("notes", None),
            Path("a.py"): DocumentInfo("a", None),
        })
        self.assertEqual(
            headers["p"],
            "A study of sleep\nSmith et al. · paper.pdf\nResults · page 3",
        )
        self.assertEqual(headers["t"], "notes\nnotes.txt")
        self.assertNotIn("code", headers)

    def test_documents_without_info_get_no_header(self):
        self.assertEqual(
            document_search.document_headers([_page("p", "Text", 1)], {}), {}
        )


class TestPaperRetrievalText(unittest.TestCase):

    def test_paper_passages_are_searched_with_their_header(self):
        passage = _page("p", "We found.", 2)
        self.assertEqual(
            rag._retrieval_text(passage, {}, {"p": "Title\npaper.pdf"}),
            "Title\npaper.pdf\n\nWe found.",
        )
        self.assertEqual(rag._retrieval_text(passage, {}), "We found.")

    def test_headers_and_metadata_only_for_paper_projects(self):
        session = fixtures._session()
        passage = _page("p", "Results\nWe found.", 1)
        for type_id, indexed in (("literature", True), ("code", False)):
            with self.subTest(type_id=type_id):
                session.project.profile = SimpleNamespace(
                    **{**vars(fixtures._profile()), "type_id": type_id}
                )
                with patch.object(
                    rag, "document_info",
                    return_value=DocumentInfo("Title", None),
                ):
                    headers = session._document_headers([passage])
                self.assertEqual(bool(headers), indexed)
                self.assertEqual(
                    "document_headers" in session._index_metadata(), indexed
                )


if __name__ == "__main__":
    unittest.main()
