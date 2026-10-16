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


if __name__ == "__main__":
    unittest.main()
