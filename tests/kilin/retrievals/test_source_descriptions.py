"""Unit tests for resolving papers a question describes.

Check year and title-word matching against source-card metadata, earlier
and later papers, file-name years, and that ambiguous or generic
descriptions resolve to nothing.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.retrievals.source_descriptions import described_sources

CARDS = {
    Path("mijalkov17braph.pdf"): {
        "title": "BRAPH: A graph theory software for the analysis of brain "
                 "connectivity",
        "year": "2017",
    },
    Path("tiunn26braph2.pdf"): {
        "title": "BRAPH 2: A flexible, open-source, reproducible framework",
        "year": "2026",
    },
    Path("patil25deep.pdf"): {
        "title": "Deep-Learning Investigation of Vibrational Raman Spectra",
        "year": "",
    },
    Path("yao25connecting.pdf"): {
        "title": "Connecting genomic results to human brain cell types",
        "year": "2024",
    },
}


def _names(question: str, cards=CARDS) -> list[str]:
    return [path.name for path in described_sources(question, cards)]


class TestDescribedSources(unittest.TestCase):

    def test_year_and_title_words(self):
        cases = {
            "How does it compare with the 2017 paper?":
                ["mijalkov17braph.pdf"],
            "Give me an overview of the BRAPH 2 paper.":
                ["tiunn26braph2.pdf"],
            "Now tell me about the deep learning article.":
                ["patil25deep.pdf"],
            "Compare the 2017 paper with the deep learning article.":
                ["mijalkov17braph.pdf", "patil25deep.pdf"],
            "What does the study published in 2024 find?":
                ["yao25connecting.pdf"],
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(_names(question), expected)

    def test_earlier_and_later_papers(self):
        self.assertEqual(
            _names("How does BRAPH 2 differ from the earlier BRAPH "
                   "framework?"),
            ["mijalkov17braph.pdf"],
        )
        self.assertEqual(
            _names("How is it different from the original BRAPH software?"),
            ["mijalkov17braph.pdf"],
        )
        self.assertEqual(
            _names("Does the newer BRAPH paper support Python?"),
            ["tiunn26braph2.pdf"],
        )

    def test_year_from_the_file_name_and_unknown_years(self):
        # patil25deep has no card year: its file name says 2025.
        self.assertEqual(_names("Summarize the 2025 paper."),
                         ["patil25deep.pdf"])
        # With no year anywhere, it may be any year: nothing resolves.
        cards = {**CARDS, Path("notes.pdf"): {"title": "Lab notes"}}
        self.assertEqual(_names("Summarize the 2025 paper.", cards), [])

    def test_ambiguous_or_generic_descriptions_resolve_to_nothing(self):
        for question in (
            "Summarize the brain paper.",  # two titles say brain
            "What are the limitations of this paper?",
            "Compare it with the previous article.",
            "What did the DIVA paper use as input?",  # not in any title
            "Which paper from 2024 or 2025 studies aging?",
        ):
            with self.subTest(question=question):
                self.assertEqual(_names(question), [])


if __name__ == "__main__":
    unittest.main()
