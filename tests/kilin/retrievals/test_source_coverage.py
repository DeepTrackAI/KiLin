"""Exercise explicit source-coverage wording without paper retrieval.

The same heuristic is used by source-first answering intents, regardless
of whether sources have scientific-paper cards or document headers.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.retrievals.source_coverage import (
    asks_for_all_sources,
    requested_source_count,
)


class TestSourceCoverage(unittest.TestCase):

    def test_explicit_all_source_requests(self):
        for question in (
            "Compare the datasets used in all five project papers.",
            "What does each study conclude?",
            "Review every document in this folder.",
            "What is common across the articles?",
            "Summarize all of them.",
            "Compare all of these.",
            "Compare all indexed sources.",
            "Evaluate all PDFs.",
            "ACROSS THESE WORKS, what changes?",
        ):
            with self.subTest(question=question):
                self.assertTrue(asks_for_all_sources(question))

    def test_subset_and_unrelated_questions_are_not_exhaustive(self):
        for question in (
            "Compare two papers.",
            "Review the five articles.",
            "How does this compare with the 2017 paper?",
            "Compare a.pdf and b.pdf.",
            "What is the evidence for memory consolidation?",
            "Review all the evidence in one specific paper.",
            "What is the most recent study?",
        ):
            with self.subTest(question=question):
                self.assertFalse(asks_for_all_sources(question))


class TestRequestedSourceCount(unittest.TestCase):

    def test_counts_of_sources(self):
        cases = {
            "Review the five articles for their relevance.": 5,
            "Review the zero papers.": 0,
            "Review the one paper.": 0,
            "Summarize twenty articles.": 20,
            "Compare 12 articles.": 12,
            "Compare two papers on memory.": 2,
            "Do all 5 project papers use the same data?": 5,
            "How does it compare with the 2017 paper?": 0,
            "Which of the 3 groups grew fastest?": 0,
            "Summarize the paper.": 0,
        }
        for question, count in cases.items():
            with self.subTest(question=question):
                self.assertEqual(requested_source_count(question), count)


if __name__ == "__main__":
    unittest.main()
