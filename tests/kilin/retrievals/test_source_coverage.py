"""Exercise explicit source-coverage wording without paper retrieval.

The same heuristic is used by source-first answering intents, regardless
of whether sources have scientific-paper cards or document headers.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.retrievals.source_coverage import asks_for_all_sources


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


if __name__ == "__main__":
    unittest.main()
