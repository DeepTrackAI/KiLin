"""Test the check that the passages found are enough to answer.

The router model is replaced by plain functions, so no model backend or
source file is needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import Mock

from kilin.tools._shared.enough_evidence import (
    passages_answer,
    rank_until_answered,
)


class TestPassagesAnswer(unittest.TestCase):
    """Read the router's yes or no, and nothing else."""

    def test_yes_and_no_replies(self) -> None:
        """Read the first word, ignoring case and punctuation."""

        for reply, expected in (
            ("Yes.", True),
            (" no, the methods are missing", False),
            ("Maybe", None),
            ("", None),
        ):
            with self.subTest(reply=reply):
                ask = Mock(return_value=reply)
                self.assertIs(
                    passages_answer("Which software?", ["MATLAB"], ask),
                    expected,
                )

    def test_prompt_numbers_the_passages(self) -> None:
        """Show every passage once, numbered, with the question."""

        ask = Mock(return_value="yes")
        passages_answer("Which software?", [" Intro ", "MATLAB"], ask)
        _, prompt, tokens = ask.call_args.args
        self.assertIn("[1] Intro\n\n[2] MATLAB", prompt)
        self.assertIn("Question: Which software?", prompt)
        self.assertLess(tokens, 16)

    def test_failed_call_is_unclear(self) -> None:
        """A router error never stops the answer."""

        ask = Mock(side_effect=RuntimeError("out of memory"))
        self.assertIsNone(passages_answer("Q", ["text"], ask))


class TestRankUntilAnswered(unittest.TestCase):
    """Widen the ranked passages one page at a time."""

    def setUp(self) -> None:
        """Create five passages ranked in reverse order."""

        self.texts = ["a", "b", "c", "d", "e"]
        self.ranked = [4, 3, 2, 1, 0]

    def _rank(self, replies: list[str], rounds: int = 2):
        """Run the search with fixed router replies.

        Parameters
        ----------
        replies: list[str]
            Router replies, one per check.
        rounds: int, optional
            Most searches again. Defaults to ``2``.

        Returns
        -------
        tuple[list[int], int, Mock]
            Picked indices, searches made, and the router mock.

        """

        ask = Mock(side_effect=replies)
        picked, searches = rank_until_answered(
            "Q", self.ranked, self.texts, page=2, rounds=rounds, ask=ask
        )
        return picked, searches, ask

    def test_enough_passages_are_kept(self) -> None:
        """Keep the first page when the router says yes."""

        picked, searches, ask = self._rank(["yes"])
        self.assertEqual((picked, searches), ([4, 3], 0))
        ask.assert_called_once()

    def test_not_enough_reads_further_down(self) -> None:
        """Add one page per "no", up to the rounds."""

        picked, searches, ask = self._rank(["no", "no"])
        self.assertEqual((picked, searches), ([4, 3, 2, 1, 0], 2))
        self.assertEqual(ask.call_count, 2)

    def test_unclear_reply_stops_searching(self) -> None:
        """An unclear check never costs more reading."""

        picked, searches, _ = self._rank(["perhaps"])
        self.assertEqual((picked, searches), ([4, 3], 0))

    def test_zero_rounds_and_short_rankings_skip_the_check(self) -> None:
        """Ask nothing when no search again is possible."""

        picked, searches, ask = self._rank([], rounds=0)
        self.assertEqual((picked, searches), ([4, 3], 0))
        ask.assert_not_called()

        ask = Mock()
        picked, searches = rank_until_answered(
            "Q", [1, 0], self.texts, page=2, rounds=2, ask=ask
        )
        self.assertEqual((picked, searches), ([1, 0], 0))
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
