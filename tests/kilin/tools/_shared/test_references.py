"""Unit tests for the shared reference-list extractor.

Check that a paper's reference list is found by its heading and split into
the entries it prints, for numbered lists and for two-column layouts that
print the numbers apart from the entries, independently of any registered
tool's configuration.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest

from kilin.tools._shared.references import (
    ReferenceList,
    ReferenceLayout,
    joined_text,
    numbered_reference_entries,
    pdf_text,
    reference_body,
    reference_entries,
    reference_list,
    reference_lists,
)
from tests.kilin.tools._shared._support import chunk

LAYOUT = ReferenceLayout(
    max_characters=40000,
    numbered_entries_min=5,
    running_line_repeats=3,
    running_line_words=12,
    entry_min_words=5,
)

PAPER_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "pdfs"

NUMBERED = """Introduction
Some text cites earlier work [1].

References
1. Smith, J. A study of cells in culture. Nature 1, 2-3 (2019).
2. Lee, K. & Kim, H. Another study of tissue. Science 4, 5-6 (2020).
3. Wu, X. A third study about imaging methods. Cell 7, 8-9 (2021).
[4] Rossi, M. A fourth study about microscopy. PNAS 10, 11-12 (2022).
5) Garcia, L. A fifth study about dynamics. eLife 13, 14 (2023).
Acknowledgements We thank the facility.
6. Not a reference, this follows the list.
"""

TWO_COLUMN = """References
1.
2.
Smith, J. A study of cells in
culture. Nature 1, 2-3 (2019).
Lee, K. & Kim, H. Another study of tis-
sue samples. Science 4, 5-6 (2020).
https://doi.org/10.1000/xyz123
Wu, X. A third study about imaging. Cell 7, 8-9 (2021).
"""


class TestReferenceList(unittest.TestCase):
    """Find the reference list by its heading."""

    def test_no_heading_gives_empty_list(self) -> None:
        passages = (chunk("a", "Introduction\nNo list here.\n"),)
        self.assertEqual(reference_list(passages, LAYOUT), "")
        self.assertEqual(reference_entries(passages, LAYOUT), [])

    def test_last_heading_wins_across_passages(self) -> None:
        passages = (
            chunk("a", "References\nCited in the intro.\n"),
            chunk("b", "## Bibliography\nThe real list.\n", line=3),
        )
        self.assertEqual(reference_list(passages, LAYOUT), "The real list.")

    def test_heading_variants(self) -> None:
        for heading in (
            "References",
            "REFERENCES:",
            "# Literature cited",
            "Works Cited",
            "Bibliography",
        ):
            with self.subTest(heading=heading):
                passages = (chunk("a", f"{heading}\nEntry text.\n"),)
                self.assertEqual(
                    reference_list(passages, LAYOUT), "Entry text."
                )

    def test_max_characters_cuts_the_list(self) -> None:
        passages = (chunk("a", "References\n" + "x" * 100),)
        layout = replace(LAYOUT, max_characters=10)
        # The limit counts from the end of the heading, its newline included.
        cut = reference_list(passages, layout)
        self.assertEqual(cut, "x" * 9)


class TestReferenceBody(unittest.TestCase):
    """Return the text before the reference list."""

    def test_cut_before_the_last_heading(self) -> None:
        passages = (
            chunk("a", "Introduction\nCells grow [1].\n"),
            chunk("b", "Results hold [2].\nReferences\n1. Smith\n", line=3),
            chunk("c", "2. Lee\n", line=6),
        )
        body = reference_body(passages)
        self.assertEqual(len(body), 2)
        self.assertEqual(body[1].text, "Results hold [2].\n")
        self.assertEqual((body[1].start_line, body[1].end_line), (3, 4))

    def test_heading_at_passage_start_drops_it(self) -> None:
        passages = (
            chunk("a", "Text [1].\n"),
            chunk("b", "References\n1. Smith\n", line=2),
        )
        self.assertEqual(reference_body(passages), passages[:1])

    def test_no_heading_keeps_everything(self) -> None:
        passages = (chunk("a", "Text only.\n"),)
        self.assertEqual(reference_body(passages), passages)


class TestReferenceEntries(unittest.TestCase):
    """Split the reference list into the entries it prints."""

    def test_numbered_list_splits_at_numbers_and_stops(self) -> None:
        entries = reference_entries((chunk("a", NUMBERED),), LAYOUT)
        self.assertEqual(len(entries), 5)
        self.assertTrue(entries[0].startswith("Smith, J. A study"))
        self.assertTrue(entries[3].startswith("Rossi, M."))
        self.assertTrue(entries[4].startswith("Garcia, L."))
        self.assertFalse(any("Not a reference" in e for e in entries))

    def test_two_column_list_splits_at_entry_ends(self) -> None:
        entries = reference_entries((chunk("a", TWO_COLUMN),), LAYOUT)
        self.assertEqual(
            entries,
            [
                "Smith, J. A study of cells in culture. Nature 1, 2-3 "
                "(2019).",
                "Lee, K. & Kim, H. Another study of tissue samples. "
                "Science 4, 5-6 (2020). https://doi.org/10.1000/xyz123",
                "Wu, X. A third study about imaging. Cell 7, 8-9 (2021).",
            ],
        )

    def test_page_furniture_is_dropped(self) -> None:
        running = "J. Smith et al. Journal 35 (2026)"
        text = (
            f"{running}\nIntro.\n{running}\nReferences\n"
            "Smith, J. A study of cells. Nature 1, 2-3 (2019).\n"
            "12\n"
            f"{running}\n"
            "PLOS ONE | https://doi.org/10.1371/x\n"
            "Lee, K. Another study of tissue. Science 4, 5-6 (2020).\n"
            "3 / 25\n"
            "Too short (2021).\n"
        )
        self.assertEqual(
            reference_entries((chunk("a", text),), LAYOUT),
            [
                "Smith, J. A study of cells. Nature 1, 2-3 (2019).",
                "Lee, K. Another study of tissue. Science 4, 5-6 (2020).",
            ],
        )

    def test_layout_thresholds_are_the_callers(self) -> None:
        # With fewer numbered lines than required, entries split at their
        # ends instead; a high word minimum drops every entry.
        passages = (chunk("a", NUMBERED),)
        strict = replace(LAYOUT, numbered_entries_min=50)
        self.assertEqual(len(reference_entries(passages, strict)), 5)
        long_only = replace(LAYOUT, entry_min_words=50)
        self.assertEqual(reference_entries(passages, long_only), [])

    def test_year_and_pages_end_an_entry(self) -> None:
        # Two-column layout: the lone numbers force splitting at entry
        # ends, and "2016, 160422." or "1995; 289–300." end an entry.
        text = (
            "References\n1.\n2.\n3.\n"
            "Ji GJ, Yu Y. Decreased network efficiency in epilepsy.\n"
            "Radiology 2016, 160422.\n"
            "Liu Y, Liang M. Disrupted small-world networks in\n"
            "schizophrenia. Brain 2008, 131, 945–961. PMID: 18299296\n"
            "Benjamini Y, Hochberg Y. Controlling the false discovery\n"
            "rate. J Royal Stat Soc 1995; 289–300.\n"
        )
        entries = reference_entries((chunk("a", text),), LAYOUT)
        self.assertEqual(len(entries), 3)
        self.assertTrue(entries[1].startswith("Liu Y"))

    def test_explicit_short_references_use_opt_in_filter(self) -> None:
        text = (
            "References\n"
            "1. Nature 2020;12:22.\n"
            "2. Garcia L. A full reference about tissue dynamics. "
            "Science 2021;13:33.\n"
        )
        passages = (chunk("a", text),)
        standard = numbered_reference_entries(passages, LAYOUT)
        self.assertEqual([entry.number for entry in standard], [2])
        seen = []

        def accept_short(entry: str) -> bool:
            seen.append(entry)
            return entry.startswith("Nature 2020;")

        extended = numbered_reference_entries(
            passages, LAYOUT, accept_short=accept_short
        )
        self.assertEqual(
            [entry.number for entry in extended], [1, 2]
        )
        self.assertTrue(extended[0].printed)
        self.assertEqual(seen, ["Nature 2020;12:22."])

    def test_lone_numbers_go_to_unnumbered_entries(self) -> None:
        # PLOS prints some numbers apart from their entries ("6.", "7."),
        # others inline ("8. Baggio"); a DOI broken after "2013." must not
        # start an entry that takes a number.
        text = (
            "References\n"
            "4. Meunier D, Tyler LK. Age-related reorganization. Neurobiol\n"
            "Aging 2014; 35: 42–54. https://doi.org/10.1016/j.neuro.2013.\n"
            "07.003 PMID: 23942392\n"
            "6.\n7.\n"
            "van den Heuvel MP, Pol HE. Exploring the brain network.\n"
            "Eur Neuropsychopharmacol 2010; 20: 519–534. PMID: 20471808\n"
            "Tijms BM, Wink AM. Connecting findings from graph studies.\n"
            "Neurobiol Aging 2013; 34: 2023–2036. PMID: 23541878\n"
            "8. Baggio HC, Junque C. Resting-state networks in Parkinson.\n"
            "CNS Neurosci Ther 2015; 21: 793–801. PMID: 26224057\n"
        )
        entries = numbered_reference_entries((chunk("a", text),), LAYOUT)
        self.assertEqual(
            [(e.number, e.text.split()[0], e.printed) for e in entries],
            [
                (4, "Meunier", True),
                (6, "van", True),
                (7, "Tijms", True),
                (8, "Baggio", True),
            ],
        )
        self.assertIn("2013. 07.003 PMID", entries[0].text)

    def test_numbers_inside_an_entry_start_no_entry(self) -> None:
        # A page range wrapped after its dash ("919–\n927.") looks like an
        # entry number but does not follow the previous one.
        text = (
            "References\n"
            "46. Smith J, Lee K. A study of brain networks. Brain 2011; 95:\n"
            "629–635. PMID: 21930184\n"
            "47. Weintraub D, Simuni T. Cognitive performance in early\n"
            "Parkinson disease. Mov Disord 2015; 30: 919–\n"
            "927. https://doi.org/10.1002/mds.26170 PMID: 25737166\n"
            "48. Winblad B, Amouyel P. Defeating Alzheimer disease in\n"
            "Europe. Lancet Neurol 2016; 15: 455–532.\n"
        )
        entries = numbered_reference_entries((chunk("a", text),), LAYOUT)
        self.assertEqual([e.number for e in entries], [46, 47, 48])
        self.assertIn("919– 927. https://doi.org", entries[1].text)

    def test_captions_and_restarted_numbers_end_the_list(self) -> None:
        entries = "".join(
            f"{n}. Author{n} A. A study number {n} of cells. "
            f"Nature {n}, 2-3 (2019).\n"
            for n in range(1, 7)
        )
        cases = (
            "Table 1. Characteristics of the 40 participants.\n"
            "1. Age, years 65 ± 3 and more values per group\n",
            "Fig. 2 | The network of brain regions over the cortex.\n",
            "Supplementary Figure S3: Additional results for all.\n",
            "1. Patients were recruited at three clinical sites, all.\n",
        )
        for tail in cases:
            with self.subTest(tail=tail.split()[0]):
                found = numbered_reference_entries(
                    (chunk("a", f"References\n{entries}{tail}"),), LAYOUT
                )
                self.assertEqual(
                    [e.number for e in found], list(range(1, 7))
                )
                self.assertTrue(found[-1].text.endswith("(2019)."))

    def test_several_lists_are_found_and_chosen(self) -> None:
        main = "".join(
            f"{n}. Main{n} A. A main study {n} of cells. Nature {n}, 2 "
            f"(2019).\n" for n in range(1, 7)
        )
        extra = "".join(
            f"{n}. Extra{n} B. A supplementary study {n}. Cell {n}, 3 "
            f"(2020).\n" for n in range(1, 6)
        )
        text = (
            "Intro cites [1] and [6].\nReferences\n" + main
            + "Supplementary Information\nThe supplement cites [5].\n"
            "Supplementary References\n" + extra
        )
        passages = (chunk("a", text),)
        lists = reference_lists(passages, LAYOUT)
        self.assertEqual(
            [(found.index, found.heading, len(found.entries))
             for found in lists],
            [(0, "References", 6), (1, "Supplementary References", 5)],
        )
        self.assertIsInstance(lists[0], ReferenceList)
        self.assertEqual(
            lists[1].label, "Supplementary References (5 entries, 1–5)"
        )
        first = numbered_reference_entries(passages, LAYOUT, which=0)
        self.assertTrue(first[0].text.startswith("Main1"))
        self.assertTrue(
            numbered_reference_entries(passages, LAYOUT)[0]
            .text.startswith("Extra1")
        )
        # Each list's citing text: the main text, or the supplement.
        self.assertEqual(
            reference_body(passages, 0)[0].text.strip(),
            "Intro cites [1] and [6].",
        )
        supplement = " ".join(
            passage.text for passage in reference_body(passages, 1)
        )
        self.assertIn("The supplement cites [5].", supplement)
        self.assertNotIn("Main1", supplement)
        self.assertNotIn("Intro", supplement)

    def test_supplementary_methods_ends_main_reference_list(self) -> None:
        """A supplementary section is neither an entry nor lost body."""

        main = "".join(
            f"{n}. Main{n} A. A study of cells {n}. Nature {n}, 3 "
            f"(2019).\n" for n in range(1, 7)
        )
        text = (
            "Main work cites [1].\nReferences\n" + main
            + "Supplementary Methods\nAdditional experiments cite [1].\n"
            "Supplementary References\n"
            "1. Extra B. A supplementary study of cells. Cell 1, 3 "
            "(2020).\n"
        )
        passages = (chunk("a", text),)
        self.assertEqual(
            [len(ref.entries) for ref in reference_lists(passages, LAYOUT)],
            [6, 1],
        )
        self.assertTrue(all(
            "Supplementary" not in entry.text
            for entry in numbered_reference_entries(
                passages, LAYOUT, which=0
            )
        ))
        supplement = " ".join(
            part.text for part in reference_body(passages, 1)
        )
        self.assertIn("Additional experiments cite [1].", supplement)
        self.assertNotIn("Main1", supplement)

    def test_unprinted_numbers_follow_the_previous(self) -> None:
        entries = numbered_reference_entries(
            (chunk("a", TWO_COLUMN.replace("1.\n2.\n", "")),), LAYOUT
        )
        self.assertEqual(
            [(e.number, e.printed) for e in entries],
            [(1, False), (2, False), (3, False)],
        )

    def test_overlapping_passages_are_read_once(self) -> None:
        # Lines 5-6 ("1. Smith", "2. Lee") are in both passages.
        lines = NUMBERED.splitlines()
        passages = (
            chunk("a", "\n".join(lines[:6])),
            chunk("b", "\n".join(lines[4:]), line=5),
        )
        self.assertEqual(passages[0].end_line, 6)
        self.assertEqual(joined_text(passages), "\n".join(lines))
        self.assertEqual(len(reference_entries(passages, LAYOUT)), 5)

    def test_differing_overlap_is_kept(self) -> None:
        passages = (
            chunk("a", "one\ntwo"),
            chunk("b", "TWO\nthree", line=2),
        )
        self.assertEqual(joined_text(passages), "one\ntwo\nTWO\nthree")

    def test_layout_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            LAYOUT.entry_min_words = 1  # type: ignore[misc]


class TestPdfText(unittest.TestCase):
    """Read PDF pages in column order."""

    def test_two_column_reference_list_in_order(self) -> None:
        file = PAPER_FIXTURES / "mijalkov25computational.pdf"
        text = pdf_text(file)
        self.assertIn("\f", text)
        # Extracted text interleaves references 1-4 and 23-28; read in
        # column order, the 96 printed numbers come out 1, 2, ..., 96.
        entries = numbered_reference_entries((), LAYOUT, file)
        self.assertEqual([e.number for e in entries], list(range(1, 97)))
        self.assertTrue(all(e.printed for e in entries))
        self.assertTrue(entries[26].text.startswith("LeCun, Y."))

    def test_other_committed_pdf_reference_lists(self) -> None:
        papers = (
            ("mijalkov17braph.pdf", 56, "Sporns O."),
            ("tiunn26braph2.pdf", 25, "Seguin C"),
        )
        for name, count, first_author in papers:
            with self.subTest(name=name):
                entries = numbered_reference_entries(
                    (), LAYOUT, PAPER_FIXTURES / name
                )
                self.assertEqual(
                    [entry.number for entry in entries],
                    list(range(1, count + 1)),
                )
                self.assertTrue(all(entry.printed for entry in entries))
                self.assertTrue(entries[0].text.startswith(first_author))

    def test_other_files_use_the_passages(self) -> None:
        self.assertEqual(pdf_text(Path("notes.md")), "")
        self.assertEqual(pdf_text(Path("missing.pdf")), "")
        passages = (chunk("a", NUMBERED),)
        self.assertEqual(
            reference_list(passages, LAYOUT, Path("missing.pdf")),
            reference_list(passages, LAYOUT),
        )


if __name__ == "__main__":
    unittest.main()
