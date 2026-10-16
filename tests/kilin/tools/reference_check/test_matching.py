"""Unit tests for matching reference entries to project files.

Check the three matching routes (DOI, trusted title, first page), the
first-author check that separates other versions of a work, and the
statuses reported for missing and unreadable entries, independently of
the tool's configuration.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pypdf import PdfWriter

from kilin.tools.reference_check._matching import (
    MATCH_FOUND,
    MATCH_MISSING,
    MATCH_PROBABLE,
    MATCH_UNREADABLE,
    MatchRules,
    PaperIdentity,
    first_author,
    match_reference,
    match_references,
    paper_identity,
)
from tests.kilin.tools._shared._support import chunk

RULES = MatchRules(
    first_characters=1500,
    title_words_min=3,
    title_on_page=0.6,
    title_found=0.8,
    title_probable=0.5,
    page_found=0.85,
    page_probable=0.65,
    entry_words_min=4,
    common_words=frozenset({"journal", "study", "analysis"}),
)

BRAPH = PaperIdentity(
    path=Path("mijalkov17braph.pdf"),
    title="BRAPH: A graph theory software for the analysis of brain "
    "connectivity",
    authors="Mite Mijalkov, Ehsan Kakaei, Joana B. Pereira",
    doi="10.1371/journal.pone.0178798",
    first_page="RESEARCH ARTICLE BRAPH: A graph theory software for the "
    "analysis of brain connectivity Mite Mijalkov1, Ehsan Kakaei1",
)
BRAPH2 = PaperIdentity(
    path=Path("tiunn26braph2.pdf"),
    title="BRAPH 2: A flexible, open-source, reproducible, "
    "community-oriented, easy-to-use framework for network analyses in "
    "neurosciences",
    authors="Iok-ui Tiunn",
    doi="",
    first_page="BRAPH 2: A flexible, open-source, reproducible, "
    "community-oriented, easy-to-use framework for network analyses in "
    "neurosciences Iok-ui Tiunn, Blanca Zufiria-Gerboles",
)
UNTITLED = PaperIdentity(
    path=Path("storm.pdf"),
    title="",
    authors="",
    doi="",
    first_page="Computational memory capacity predicts aging and "
    "cognitive decline M i t e M i j a l k o v, L u d v i g S t o r m "
    "Nature Communications",
)
PAPERS = (BRAPH, BRAPH2, UNTITLED)


class TestFirstAuthor(unittest.TestCase):
    """Read the first author's surname in common citation styles."""

    def test_styles(self) -> None:
        for entry, surname in (
            ("Mijalkov, M. et al. BRAPH: A graph theory ...", "Mijalkov"),
            ("Mijalkov M, Kakaei E, Pereira JB. BRAPH ...", "Mijalkov"),
            ("van den Heuvel MP, Hulshoff Pol HE. Exploring ...",
             "Heuvel"),
            ("Chang, Y.-W. et al. BRAPH 2 ...", "Chang"),
            ("Lee, K. & Kim, H. Another study ...", "Lee"),
            ("Smith (2020). A title ...", "Smith"),
            ("Barabási AL. Network science. Cambridge ...", "Barabási"),
        ):
            with self.subTest(entry=entry):
                self.assertEqual(first_author(entry), surname)

    def test_no_person(self) -> None:
        self.assertIsNone(
            first_author("order based on functional brain networks with "
                         "deep learning, J. Comput Biol.")
        )


class TestMatchReference(unittest.TestCase):
    """Match one entry by DOI, title, or first page."""

    def test_title_and_first_author(self) -> None:
        match = match_reference(
            "Mijalkov, M. et al. BRAPH: A graph theory software for the "
            "analysis of brain connectivity. PLoS ONE 12, e0178798 (2017).",
            7, PAPERS, RULES,
        )
        self.assertEqual(match.status, MATCH_FOUND)
        self.assertEqual(match.path, BRAPH.path)
        self.assertEqual(match.position, 7)
        self.assertEqual(match.reason, "title and first author")

    def test_doi_settles_the_match(self) -> None:
        match = match_reference(
            "Unrelated words here. https://doi. org/10.1371/journal.pone."
            "0178798",
            1, PAPERS, RULES,
        )
        self.assertEqual((match.status, match.reason), (MATCH_FOUND, "DOI"))
        self.assertEqual(match.score, 1.0)

    def test_other_first_author_is_another_version(self) -> None:
        match = match_reference(
            "Chang, Y.-W. et al. BRAPH 2: a ﬂexible, open-source, "
            "reproducible, community-oriented, easy-to-use framework for "
            "network analyses in neurosciences. bioRxiv (2025).",
            1, PAPERS, RULES,
        )
        self.assertEqual(match.status, MATCH_PROBABLE)
        self.assertEqual(match.path, BRAPH2.path)
        self.assertIn("another version", match.reason)

    def test_part_of_the_title_is_probable(self) -> None:
        match = match_reference(
            "Mijalkov, M. et al. BRAPH: a graph theory software for the "
            "analysis of",
            1, PAPERS, RULES,
        )
        self.assertEqual(match.status, MATCH_PROBABLE)
        self.assertIn("part of the title", match.reason)

    def test_first_page_with_letter_spaced_names(self) -> None:
        match = match_reference(
            "Storm, L. et al. Computational memory capacity predicts aging "
            "and cognitive decline. Nature Communications 16 (2025).",
            1, PAPERS, RULES,
        )
        self.assertEqual(match.status, MATCH_FOUND)
        self.assertEqual(match.path, UNTITLED.path)
        self.assertEqual(match.reason, "first page and first author")

    def test_missing_and_unreadable(self) -> None:
        missing = match_reference(
            "LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, "
            "436–444 (2015).",
            1, PAPERS, RULES,
        )
        self.assertEqual(missing.status, MATCH_MISSING)
        self.assertIsNone(missing.path)
        short = match_reference("Nat. Commun. 8 (2017).", 2, PAPERS, RULES)
        self.assertEqual(short.status, MATCH_UNREADABLE)

    def test_duplicates_are_alternatives(self) -> None:
        copy = replace(BRAPH, path=Path("copy/braph.pdf"))
        match = match_reference(
            "Mijalkov, M. et al. BRAPH: A graph theory software for the "
            "analysis of brain connectivity. PLoS ONE (2017).",
            1, (*PAPERS, copy), RULES,
        )
        self.assertEqual(match.status, MATCH_FOUND)
        self.assertEqual(len(match.alternatives), 1)

    def test_entries_keep_their_positions(self) -> None:
        matches = match_references(
            ["Nat. Commun. 8 (2017).",
             "Mijalkov M, Kakaei E. BRAPH: a graph theory software for "
             "the analysis of brain connectivity. PLoS One 2017."],
            PAPERS, RULES,
        )
        self.assertEqual([m.position for m in matches], [1, 2])
        self.assertEqual(matches[1].status, MATCH_FOUND)


class TestPaperIdentity(unittest.TestCase):
    """Trust only metadata titles printed on the first page."""

    def _pdf(self, folder: str, title: str) -> Path:
        path = Path(folder) / "paper.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_metadata({"/Title": title, "/Author": "Jane Smith"})
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    def test_title_doi_and_authors(self) -> None:
        passages = (chunk(
            "a", "Article https://doi.org/10.1038/s41467-025-57995-0\n"
            "Counting cells in growing tissue\nJane Smith\n",
            path="paper.pdf",
        ),)
        with TemporaryDirectory() as folder:
            file = self._pdf(folder, "Counting cells in growing tissue")
            identity = paper_identity(
                Path("paper.pdf"), passages, file, RULES
            )
        self.assertEqual(identity.title, "Counting cells in growing tissue")
        self.assertEqual(identity.authors, "Jane Smith")
        self.assertEqual(identity.doi, "10.1038/s41467-025-57995-0")

    def test_junk_title_is_not_trusted(self) -> None:
        passages = (chunk("a", "Counting cells in growing tissue\n"),)
        with TemporaryDirectory() as folder:
            file = self._pdf(folder, "Microsoft Word - final draft v3.docx")
            identity = paper_identity(
                Path("paper.pdf"), passages, file, RULES
            )
        self.assertEqual(identity.title, "")

    def test_no_file_uses_text_only(self) -> None:
        passages = (chunk("a", "Some first page text\n"),)
        identity = paper_identity(Path("notes.md"), passages, None, RULES)
        self.assertEqual((identity.title, identity.authors), ("", ""))
        self.assertEqual(identity.first_page, "Some first page text\n")


if __name__ == "__main__":
    unittest.main()
