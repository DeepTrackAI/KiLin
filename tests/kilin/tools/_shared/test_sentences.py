"""Unit tests for splitting indexed text into sentences.

Check sentence boundaries, abbreviations that do not end a sentence, and
repeated sentences in overlapping passages.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools._shared.sentences import split_sentences
from tests.kilin.tools._shared._support import chunk


class TestSplitSentences(unittest.TestCase):
    """Split passages at sentence ends only."""

    def test_sentence_ends(self) -> None:
        sentences = split_sentences(
            [chunk("a", "Cells grow fast. They divide often! Do they stop?")],
            min_words=1,
        )
        self.assertEqual(
            [s.text for s in sentences],
            ["Cells grow fast.", "They divide often!", "Do they stop?"],
        )

    def test_abbreviations_do_not_end_sentences(self) -> None:
        text = (
            "Together with Ref. 50, these data agree (Fig. 5A). As in "
            "Eq. 3 and refs. 12,13, the fit holds. A new sentence starts."
        )
        sentences = split_sentences([chunk("a", text)], min_words=1)
        self.assertEqual(
            [s.text for s in sentences],
            [
                "Together with Ref. 50, these data agree (Fig. 5A).",
                "As in Eq. 3 and refs. 12,13, the fit holds.",
                "A new sentence starts.",
            ],
        )

    def test_overlapping_passages_repeat_nothing(self) -> None:
        passages = [
            chunk("a", "First sentence here now. Second sentence here now."),
            chunk("b", "Second sentence here now. Third sentence here now.",
                  line=2),
        ]
        self.assertEqual(len(split_sentences(passages)), 3)


    def test_sentence_cut_by_a_passage_boundary_is_joined(self) -> None:
        # The first passage stops inside a sentence; the second starts
        # inside it, repeating a few words (the overlap).
        passages = [
            chunk("a", "Cells grow fast. The genu and splenium link "
                       "frontal regions and occipital-"),
            chunk("b", "frontal regions and occipital- parietal cortex "
                       "across brain regions60. Next one starts here.",
                  line=2),
        ]
        self.assertEqual(
            [s.text for s in split_sentences(passages, min_words=1)],
            [
                "Cells grow fast.",
                "The genu and splenium link frontal regions and occipital- "
                "parietal cortex across brain regions60.",
                "Next one starts here.",
            ],
        )

    def test_whole_sentence_repeated_replaces_its_cut_start(self) -> None:
        passages = [
            chunk("a", "Cells grow fast. The genu and splenium link"),
            chunk("b", "The genu and splenium link frontal regions. "
                       "Next one starts here.", line=2),
        ]
        self.assertEqual(
            [s.text for s in split_sentences(passages, min_words=1)],
            [
                "Cells grow fast.",
                "The genu and splenium link frontal regions.",
                "Next one starts here.",
            ],
        )

    def test_sentence_over_three_passages(self) -> None:
        passages = [
            chunk("a", "One sentence ends. A very long sentence starts"),
            chunk("b", "sentence starts and goes on without"),
            chunk("c", "on without an end until here. Done now, all.",
                  line=3),
        ]
        sentences = split_sentences(passages, min_words=1)
        self.assertEqual(
            sentences[1].text,
            "A very long sentence starts and goes on without an end until "
            "here.",
        )
        self.assertIs(sentences[1].chunk, passages[0])

if __name__ == "__main__":
    unittest.main()
