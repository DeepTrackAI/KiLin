"""Unit tests for finding numbered in-text citations.

Check bracketed, written-out and superscript markers, font confirmation
of superscripts, rejection of author, affiliation and exponent markers,
and diagnostics for uncited and out-of-list numbers, independently of
the tool's configuration.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools.reference_check._citations import (
    STYLE_AUTHOR_YEAR,
    STYLE_BRACKETS,
    STYLE_NONE,
    STYLE_PARENTHESES,
    STYLE_SUPERSCRIPT,
    STYLE_WRITTEN,
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
    unit_words=frozenset({"cm", "log", "nm"}),
    byline_share=0.5,
)
PAPER_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "pdfs"


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
        passages = (
            chunk(
                "a",
                "Brains are networks [1]. Graphs measure them [2, 4–5].\n"
                "A later study disagrees [7].\n",
            ),
        )
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
            bracket_markers_min=5,
            superscript_ratio=0.85,
            signature_characters=8,
            max_range=50,
            unit_words=frozenset({"cm"}),
            byline_share=0.5,
        )
        passages = (chunk("a", "Only one citation here [2].\n"),)
        scan = find_citations(passages, 3, None, rules)
        self.assertEqual(scan.style, STYLE_BRACKETS)
        self.assertEqual([c.numbers for c in scan.citations], [(2,)])
        # With superscripts in the PDF, one bracket is not enough.
        scan = find_citations(passages, 3, {"7": ("other",)}, rules)
        self.assertEqual(scan.style, STYLE_SUPERSCRIPT)

    def test_parenthesized_citations(self) -> None:
        passages = (
            chunk(
                "a",
                "Single molecules move in cells (1). Motion is anomalous "
                "(6,7), as shown (15,30–32). The MSD is 4Dt (2). See "
                "Equation (3) and Eqs. (4) and (5). The steps are: (1) "
                "detection, then linking. Earlier work (8), and later.\n",
            ),
        )
        scan = find_citations(passages, 40, None, RULES)
        self.assertEqual(scan.style, STYLE_PARENTHESES)
        self.assertEqual(
            [c.numbers for c in scan.citations],
            [(1,), (6, 7), (15, 30, 31, 32), (2,), (8,)],
        )

    def test_parentheses_lose_to_superscripts(self) -> None:
        # A superscript paper numbers its equations in parentheses.
        passages = (
            chunk(
                "a",
                "Diffusion is anomalous in cells12,13,14. It is described by "
                "Equation (1), by the model (2), and in the methods (3).\n",
            ),
        )
        scan = find_citations(passages, 40, {"121314": ("cells",)}, RULES)
        self.assertEqual(scan.style, STYLE_SUPERSCRIPT)
        self.assertEqual([c.numbers for c in scan.citations], [(12, 13, 14)])

    def test_written_references_without_other_citation_styles(self) -> None:
        passages = (
            chunk(
                "a",
                "Earlier work in Ref. 2 established this. Another method "
                "follows refs. 3,4 and refs 6–7. The grant was Ref. "
                "001-P-00038, not a citation.\n",
            ),
        )
        scan = find_citations(passages, 6, None, RULES)
        self.assertEqual(scan.style, STYLE_WRITTEN)
        self.assertEqual(
            [citation.numbers for citation in scan.citations],
            [(2,), (3, 4), (6, 7)],
        )
        self.assertEqual(scan.uncited, (1, 5))
        self.assertEqual(scan.beyond_list, (7,))
        self.assertEqual(
            [
                citation.sentence.text[citation.start : citation.end]
                for citation in scan.citations
            ],
            ["Ref. 2", "refs. 3,4", "refs 6–7"],
        )

    def test_written_references_with_unrelated_pdf_superscripts(self) -> None:
        passages = (
            chunk("a", "These results follow Ref. 2 and refs. 3,4.\n"),
        )
        scan = find_citations(passages, 5, {"99": ("unrelated",)}, RULES)
        self.assertEqual(scan.style, STYLE_WRITTEN)
        self.assertEqual(
            [citation.numbers for citation in scan.citations],
            [(2,), (3, 4)],
        )

    def test_no_style_without_brackets_or_pdf(self) -> None:
        passages = (chunk("a", "Cells grow in 3 groups of 12 samples.\n"),)
        scan = find_citations(passages, 3, None, RULES)
        self.assertEqual(scan.style, STYLE_NONE)
        self.assertEqual(scan.citations, ())
        self.assertEqual(scan.uncited, (1, 2, 3))


class TestAuthorYear(unittest.TestCase):
    """Author-year citations of an unnumbered reference list."""

    ENTRIES = (
        (1, "Brown, A. (2018). Networks. J 1, 2."),
        (2, "Lee, B. (2019). One. J 2, 3."),
        (3, "Lee, B. (2019). Two. J 3, 4."),
        (4, "Smith, J., Wu, C., & Ode, D. (2020). Graphs. J 4, 5."),
        (5, "Wu, C. (2020). Unused. J 5, 6."),
        (6, "Ode, D. (2021). One. J 6, 7."),
        (7, "Ode, D. (2021). Two. J 7, 8."),
    )

    def test_resolved_ambiguous_missing_and_uncited(self) -> None:
        passages = (chunk(
            "a",
            "Brains are networks (Brown 2018; Lee 2019b).\n"
            "Smith et al. (2020) measured them, as did others (Ode 2021).\n"
            "Graphs help (e.g., Smith et al., 2020, p. 4; Gray et al. "
            "2017).\nPhases differ (Phase 2020).\n",
        ),)
        scan = find_citations(passages, 7, None, RULES, self.ENTRIES)
        self.assertEqual(scan.style, STYLE_AUTHOR_YEAR)
        self.assertEqual(
            [(c.marker, c.numbers) for c in scan.citations],
            [("(Brown 2018; Lee 2019b)", (1, 3)),
             ("Smith et al. (2020)", (4,)),
             ("(e.g., Smith et al., 2020, p. 4; Gray et al. 2017)", (4,))],
        )
        citation = scan.citations[1]
        self.assertTrue(citation.author_year)
        self.assertEqual(
            citation.sentence.text[citation.start:citation.end],
            "Smith et al. (2020)",
        )
        self.assertEqual(scan.ambiguous, (("Ode, 2021", (6, 7)),))
        self.assertEqual(scan.unresolved, ("Gray et al., 2017",))
        # Entry 2 is never cited; 6 and 7 may be, by the ambiguous one.
        self.assertEqual(scan.uncited, (2, 5))

    def test_numbered_citations_win(self) -> None:
        passages = (chunk(
            "a", "Brains are networks [1]. Graphs [2]. Again [4].\n"
            "Brown (2018) agrees.\n",
        ),)
        scan = find_citations(passages, 7, None, RULES, self.ENTRIES)
        self.assertEqual(scan.style, STYLE_BRACKETS)
        self.assertFalse(scan.citations[0].author_year)
        # Without entries, author-year citations are not looked for.
        passages = (chunk("a", "Brown (2018) agrees.\n"),)
        scan = find_citations(passages, 7, None, RULES)
        self.assertEqual(scan.style, STYLE_NONE)

    def test_incompatible_author_lists_are_not_cited(self) -> None:
        entries = (
            (1, "Brown, A. (2018). A network study. J 1, 2."),
            (2, "Smith, J., & Wu, C. (2020). A different study. J 2, 3."),
        )
        passages = (chunk(
            "a",
            "Brown (2018) identified networks.\n"
            "Smith and Lee (2020) investigated them too.\n",
        ),)
        scan = find_citations(passages, 2, None, RULES, entries)
        self.assertEqual(scan.style, STYLE_AUTHOR_YEAR)
        self.assertEqual([citation.numbers for citation in scan.citations],
                         [(1,)])
        self.assertEqual(scan.unresolved, ("Smith and Lee, 2020",))
        self.assertEqual(scan.uncited, (2,))


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
        passages = (
            chunk(
                "a",
                "Networks were built with BRAPH 2.087 for all subjects. "
                "Values of 6.87 and counts of 105 cells per well.\n",
            ),
        )
        # The PDF prints 87 small after "BRAPH 2.0", and 5 after "10" (a
        # power); "6.87" and "105" are not printed as citations.
        superscripts = {"87": ("braph2.0",), "5": ("countsof10",)}
        scan = find_citations(passages, 90, superscripts, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(87,)])

    def test_greek_letters_symbols_and_written_references(self) -> None:
        passages = (
            chunk(
                "a",
                "Values depend on the anomalous exponent α64. The exponent "
                "relates as H = α/212,67. Together with Ref. 50, "
                "these agree. "
                "A grant, Ref. 001-P-00038, paid for it.\n",
            ),
        )
        superscripts = {"64": ("xponentα",), "1267": ("h=α/2",)}
        scan = find_citations(passages, 70, superscripts, RULES)
        self.assertEqual(
            [c.numbers for c in scan.citations], [(64,), (12, 67), (50,)]
        )
        written = scan.citations[2]
        self.assertEqual(
            written.sentence.text[written.start : written.end], "Ref. 50"
        )

    def test_superscript_after_a_symbol(self) -> None:
        passages = (
            chunk(
                "a", "One method based on UNet3+86,87 outperformed others.\n"
            ),
        )
        scan = find_citations(passages, 90, {"8687": ("unet3+",)}, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(86, 87)])

    def test_exponents_and_subscripts_in_formulas(self) -> None:
        passages = (
            chunk(
                "a",
                "The latent vector follows N (µi, σ2 i) in the encoder. "
                "The variance is log(σ2) for every pixel of the image. "
                "ðsÞ = 1 2\n",
            ),
        )
        superscripts = {"2": ("(µi,σ", "log(σ", "=1"), "1": ("ðsÞ=",)}
        scan = find_citations(passages, 70, superscripts, RULES)
        self.assertEqual(scan.citations, ())

    def test_shorter_pdf_context_at_line_start(self) -> None:
        # The PDF line started at "ples", the text has "samples".
        passages = (chunk("a", "Measured in many samples 19,24,25.\n"),)
        scan = find_citations(passages, 30, {"192425": ("ples",)}, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(19, 24, 25)])

    def test_bylines_and_exponents_are_not_citations(self) -> None:
        passages = (
            chunk(
                "a",
                "Mite Mijalkov1, Ludvig Storm2, Anna Canal-Garcia1, "
                "Hang Zhao2. The hash has spaces256 of keys.\n",
            ),
        )
        superscripts = {
            "1": ("mijalkov", "garcia"),
            "2": ("storm", "zhao"),
            "256": ("spaces",),
        }
        # Ten references: 256 is far beyond the list, so an exponent.
        scan = find_citations(passages, 10, superscripts, RULES)
        self.assertEqual(scan.citations, ())
        # With brackets, numbers beyond the list are kept and reported.
        brackets = (chunk("a", "One [1]. Two [2]. Twelve [12].\n"),)
        self.assertEqual(
            find_citations(brackets, 10, None, RULES).beyond_list, (12,)
        )

    def test_nature_style_references_units_and_dashes(self) -> None:
        # From three Nature Communications papers: "refs" without a full
        # stop, a citation after a unit, and an em dash after a range.
        passages = (
            chunk(
                "a",
                "The layers number at most three (refs 8,19,20). It turns "
                "for angles above 13 degrees (refs 19,21). Films transmit 97% "
                "at a wavelength above 500 nm23. Here we use reservoir "
                "computing10—a framework that reproduces biological "
                "networks11–14—to study aging. "
                "Areas of 4 cm2 were used.\n",
            ),
        )
        superscripts = {
            "23": ("500nm",),
            "10": ("omputing",),
            "1114": ("networks",),
            "2": ("4cm",),
        }
        scan = find_citations(passages, 40, superscripts, RULES)
        self.assertEqual(
            [c.numbers for c in scan.citations],
            [(8, 19, 20), (19, 21), (23,), (10,), (11, 12, 13, 14)],
        )

    def test_count_in_parentheses_before_a_superscript(self) -> None:
        # "high (3)⁴⁵,⁴⁶": the parenthesis follows no function name, so
        # the digits are a citation; "f(0)²" stays an exponent.
        passages = (
            chunk(
                "a",
                "Expression was high (3)45,46 in the leaves. The value "
                "of f(0)2 was small in every sample.\n",
            ),
        )
        superscripts = {"4546": ("shigh(3)",), "2": ("ofuef(0)",)}
        scan = find_citations(passages, 60, superscripts, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(45, 46)])

    def test_marker_broken_over_two_lines(self) -> None:
        # "…accumulation⁵⁹⁻" ends a line and "⁶¹" starts the next.
        passages = (
            chunk(
                "a",
                "Stress led to protein accumulation59–\n61 in plants and "
                "more.\n",
            ),
        )
        superscripts = {"59": ("mulation",), "61": ("^inplants",)}
        scan = find_citations(passages, 70, superscripts, RULES)
        self.assertEqual(
            [c.numbers for c in scan.citations], [(59, 60, 61)]
        )
        self.assertEqual(scan.citations[0].marker, "59– 61")
        # The rest not printed small at a line start: only the first part.
        scan = find_citations(passages, 70, {"59": ("mulation",)}, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(59,)])

    def test_sentence_broken_by_a_page_number(self) -> None:
        # The page number "13" sits between the halves of the sentence,
        # and the marker is printed first on the new page.
        passages = (
            chunk("a", "Cells respond to stress through protein\n"),
            chunk("b", "13\n45,46 in leaves. A new sentence starts.\n",
                  line=2),
        )
        superscripts = {"4546": ("^inleaves",)}
        scan = find_citations(passages, 60, superscripts, RULES)
        self.assertEqual([c.numbers for c in scan.citations], [(45, 46)])
        self.assertEqual(
            scan.citations[0].sentence.text,
            "Cells respond to stress through protein 45,46 in leaves.",
        )
        # Inside one passage, with blank lines around the number.
        passages = (
            chunk(
                "a",
                "Cells respond to stress through protein\n\n13\n\n"
                "accumulation45,46 in leaves.\n",
            ),
        )
        scan = find_citations(passages, 60, {"4546": ("mulation",)}, RULES)
        self.assertEqual(
            scan.citations[0].sentence.text,
            "Cells respond to stress through protein accumulation45,46 "
            "in leaves.",
        )

    def test_line_start_digits_need_the_same_following_text(self) -> None:
        # Small digits first on a PDF line ("^…") count only when the text
        # after them agrees: "day 6" elsewhere is not a citation.
        passages = (
            chunk("a", "Samples were taken on day 6 after the stress.\n"),
        )
        superscripts = {"6": ("^departme",)}
        scan = find_citations(passages, 60, superscripts, RULES)
        self.assertEqual(scan.citations, ())

    def test_real_pdf_superscripts(self) -> None:
        runs = superscript_runs(PAPER_FIXTURES / "patil25deep.pdf", RULES)
        # "...real-time monitoring of plants in modern agricultural
        # practices 1,8–14." prints the marker in superscript.
        self.assertIn("1814", runs)
        self.assertTrue(any(c.endswith("ractices") for c in runs["1814"]))

    def test_unreadable_pdf_has_no_superscripts(self) -> None:
        self.assertEqual(superscript_runs(Path("missing.pdf"), RULES), {})


if __name__ == "__main__":
    unittest.main()
