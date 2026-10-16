"""Unit tests for finding numbered in-text citations.

Check bracketed and superscript markers, the font confirmation of
superscripts, the rejection of author, affiliation and exponent markers,
and the diagnostics for uncited and out-of-list numbers, independently of
the tool's configuration.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools.reference_check._citations import (
    STYLE_BRACKETS,
    STYLE_NONE,
    STYLE_SUPERSCRIPT,
    CitationRules,
    cited_numbers,
    find_citations,
    superscript_runs,
)
from tests.kilin.tools._shared._support import chunk

RULES = CitationRules(
    bracket_markers_min=2,
    superscript_ratio=0.85,
    signature_characters=8,
    max_range=50,
    unit_words=frozenset({"cm", "log"}),
    byline_share=0.5,
)
PAPERS = Path(__file__).resolve().parents[4] / "example_repos" / (
    "literature_repo/papers"
)


class TestCitedNumbers(unittest.TestCase):
    """Expand one marker into reference numbers."""

    def test_lists_and_ranges(self) -> None:
        self.assertEqual(cited_numbers("3", 50), (3,))
        self.assertEqual(cited_numbers("3,4", 50), (3, 4))
        self.assertEqual(cited_numbers("2, 5–7", 50), (2, 5, 6, 7))
        self.assertEqual(cited_numbers("1-3", 50), (1, 2, 3))

    def test_invalid_markers(self) -> None:
        for marker in ("0", "7–3", "1–90", "1–2–3"):
            with self.subTest(marker=marker):
                self.assertIsNone(cited_numbers(marker, 50))


class TestBrackets(unittest.TestCase):
    """Bracketed markers need no confirmation."""

    def test_bracket_citations_and_diagnostics(self) -> None:
        passages = (chunk(
            "a",
            "Brains are networks [1]. Graphs measure them [2, 4–5].\n"
            "A later study disagrees [7].\n",
        ),)
        scan = find_citations(passages, 6, None, RULES)
        self.assertEqual(scan.style, STYLE_BRACKETS)
        self.assertEqual(
            [c.numbers for c in scan.citations], [(1,), (2, 4, 5), (7,)]
        )
        self.assertEqual(
            scan.citations[1].sentence.text, "Graphs measure them [2, 4–5]."
        )
        self.assertEqual(scan.uncited, (3, 6))
        self.assertEqual(scan.beyond_list, (7,))

    def test_few_brackets_without_superscripts(self) -> None:
        # Fewer than bracket_markers_min, but nothing else to look for.
        rules = CitationRules(
            bracket_markers_min=5, superscript_ratio=0.85,
            signature_characters=8, max_range=50,
            unit_words=frozenset({"cm"}), byline_share=0.5,
        )
        passages = (chunk("a", "Only one citation here [2].\n"),)
        scan = find_citations(passages, 3, None, rules)
        self.assertEqual(scan.style, STYLE_BRACKETS)
        self.assertEqual([c.numbers for c in scan.citations], [(2,)])
        # With superscripts in the PDF, one bracket is not enough.
        scan = find_citations(passages, 3, {"7": ("other",)}, rules)
        self.assertEqual(scan.style, STYLE_SUPERSCRIPT)

    def test_no_style_without_brackets_or_pdf(self) -> None:
        passages = (chunk("a", "Cells grow in 3 groups of 12 samples.\n"),)
        scan = find_citations(passages, 3, None, RULES)
        self.assertEqual(scan.style, STYLE_NONE)
        self.assertEqual(scan.citations, ())
        self.assertEqual(scan.uncited, (1, 2, 3))


class TestSuperscripts(unittest.TestCase):
    """Superscripts count only when the PDF prints them small."""

    TEXT = (
        "Schizophrenia is notably polygenic15 with many variants12,14,16. "
        "Raman systems are portable 17,18, and spectra peak near 1150 "
        "cm 1 in leaves. Samples were split in 3 groups of animals. "
        "The GWAS Catalog72 lists loci.\n"
    )

    def test_font_confirmed_markers(self) -> None:
        superscripts = {
            "15": ("polygenic",),
            "121416": ("variants",),
            "1718": ("portable",),
            "1": ("cm",),
            "72": ("scatalog",),
        }
        passages = (chunk("a", self.TEXT),)
        scan = find_citations(passages, 80, superscripts, RULES)
        self.assertEqual(scan.style, STYLE_SUPERSCRIPT)
        # "cm 1" is an exponent; "3 groups" is not printed small.
        self.assertEqual(
            [c.marker for c in scan.citations],
            ["15", "12,14,16", "17,18", "72"],
        )

    def test_version_numbers_but_not_powers(self) -> None:
        passages = (chunk(
            "a",
            "Networks were built with BRAPH 2.087 for all subjects. "
            "Values of 6.87 and counts of 105 cells per well.\n",
        ),)
        # The PDF prints 87 small after "BRAPH 2.0", and 5 after "10" (a
        # power); "6.87" and "105" are not printed as citations.
        superscripts = {"87": ("braph20",), "5": ("countsof10",)}
        scan = find_citations(passages, 90, superscripts, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(87,)])

    def test_shorter_pdf_context_at_line_start(self) -> None:
        # The PDF line started at "ples", the text has "samples".
        passages = (chunk("a", "Measured in many samples 19,24,25.\n"),)
        scan = find_citations(passages, 30, {"192425": ("ples",)}, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(19, 24, 25)])

    def test_bylines_and_exponents_are_not_citations(self) -> None:
        passages = (chunk(
            "a",
            "Mite Mijalkov1, Ludvig Storm2, Anna Canal-Garcia1, Hang Zhao2. "
            "The hash has spaces256 of keys.\n",
        ),)
        superscripts = {"1": ("mijalkov", "garcia"), "2": ("storm", "zhao"),
                        "256": ("spaces",)}
        # Ten references: 256 is far beyond the list, so an exponent.
        scan = find_citations(passages, 10, superscripts, RULES)
        self.assertEqual(scan.citations, ())
        # With brackets, numbers beyond the list are kept and reported.
        brackets = (chunk("a", "One [1]. Two [2]. Twelve [12].\n"),)
        self.assertEqual(
            find_citations(brackets, 10, None, RULES).beyond_list, (12,)
        )

    @unittest.skipUnless(
        (PAPERS / "patil25deep.pdf").is_file(), "example paper missing"
    )
    def test_real_pdf_superscripts(self) -> None:
        runs = superscript_runs(PAPERS / "patil25deep.pdf", RULES)
        # "...real-time monitoring of plants in modern agricultural
        # practices 1,8–14." prints the marker in superscript.
        self.assertIn("1814", runs)
        self.assertTrue(any(c.endswith("ractices") for c in runs["1814"]))

    def test_unreadable_pdf_has_no_superscripts(self) -> None:
        self.assertEqual(superscript_runs(Path("missing.pdf"), RULES), {})


if __name__ == "__main__":
    unittest.main()
