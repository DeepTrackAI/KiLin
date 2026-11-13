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

    def test_page_number_between_the_halves_of_a_sentence(self) -> None:
        within = [
            chunk("a", "Cells respond to stress through the\n\n13\n\n"
                       "accumulation of proteins. Next one starts.")
        ]
        across = [
            chunk("a", "Cells respond to stress through the"),
            chunk("b", "13\naccumulation of proteins. Next one starts.",
                  line=2),
        ]
        for passages in (within, across):
            with self.subTest(passages=len(passages)):
                self.assertEqual(
                    [s.text for s in split_sentences(passages, min_words=1)],
                    [
                        "Cells respond to stress through the accumulation "
                        "of proteins.",
                        "Next one starts.",
                    ],
                )

    def test_paragraphs_and_numbers_stay_apart(self) -> None:
        # A finished sentence, a heading or a capitalized start is not
        # joined across a blank line, and a lone number stays when it is
        # not between the halves of a sentence.
        passages = [
            chunk("a", "Cells grow fast.\n\n13\n\nresults follow here. "
                       "Methods\n\nWe measured cells.")
        ]
        self.assertEqual(
            [s.text for s in split_sentences(passages, min_words=1)],
            ["Cells grow fast.", "13", "results follow here.",
             "Methods", "We measured cells."],
        )

    def test_overlapping_passage_starting_earlier_is_not_joined(self) -> None:
        # The next passage starts in the overlap, before the cut sentence:
        # its lowercase start belongs to an earlier sentence.
        passages = [
            chunk("a", "Methods include (in alpha-\nbetical order): A, B and "
                       "C19–26. These methods evaluate the association of\n\n",
                  line=1),
            chunk("b", "betical order): A, B and C19–26. These methods "
                       "evaluate the association of\n\ncells with traits. "
                       "Done here.", line=2),
        ]
        texts = [s.text for s in split_sentences(passages, min_words=1)]
        self.assertEqual(
            texts,
            [
                "Methods include (in alpha- betical order): A, B and "
                "C19–26.",
                "These methods evaluate the association of cells with "
                "traits.",
                "Done here.",
            ],
        )

    def test_links_and_table_text_survive_page_reconstruction(self) -> None:
        """Ordinary document content is not classified as page furniture."""

        for meaningful in (
            "https://github.com/example/repository",
            "https://doi.org/10.1000/example",
            "Signal | Noise",
        ):
            with self.subTest(content=meaningful):
                passage = chunk(
                    "p", "We release the source at\n\n" + meaningful
                    + "\n\nand evaluated it across datasets."
                )
                sentences = split_sentences((passage,), min_words=1)
                self.assertTrue(any(
                    meaningful in sentence.text for sentence in sentences
                ))

    def test_standalone_link_between_passages_is_preserved(self) -> None:
        """A lone URL without a repeated page header remains searchable."""

        passages = (
            chunk("p1", "The paper provides data at"),
            chunk(
                "p2", "https://github.com/example/source\nand shares "
                "the results.", line=2,
            ),
        )
        texts = [
            sentence.text for sentence in split_sentences(
                passages, min_words=1
            )
        ]
        self.assertIn("https://github.com/example/source", " ".join(texts))

    def test_footers_between_the_halves_are_left_out(self) -> None:
        footer = "Nature Communications |\n\n (2025) 16:2748\n\n"
        pages = [
            chunk("a", f"Intro text here. {footer}11\n\n"),
            chunk("b", f"Article\n\nMore text here. {footer}12\n\n"),
            chunk("c", "Article\n\nSoftware ran on Numpy90 for data\n\n"
                       f"{footer}13\n\n"),
            chunk("d", "Article\n\nhttps://doi.org/10.1038/x\n\n"
                       "handling and analysis91. Next one."),
        ]
        texts = [s.text for s in split_sentences(pages, min_words=1)]
        self.assertIn(
            "Software ran on Numpy90 for data handling and analysis91.",
            texts,
        )

if __name__ == "__main__":
    unittest.main()
