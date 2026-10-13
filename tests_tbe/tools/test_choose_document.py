"""Tests for how tools settle which documents a request is about."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.tools import EarlierTurn, ToolRequest
from qachat.tools.choose_document import document_scope
from tests.qachat.tools.helpers import chunk, context

_PAPER = Path("repo/chen25.pdf")
_OTHER = Path("repo/park26.pdf")


class TestDocumentScope(unittest.TestCase):
    """Named files first; otherwise the model judges the earlier paper."""

    def scope(self, question, reply, earlier=True):
        """Return the scope for a question and the prompts the router saw."""

        asked = []

        def router(system, prompt, tokens):
            asked.append(prompt)
            return reply

        built, _ = context(
            [chunk(str(_PAPER), ["A."]), chunk(str(_OTHER), ["B."])],
            router=router,
        )
        turns = (EarlierTurn(3, "summary of that paper", (_PAPER,) * 5),)
        request = ToolRequest(
            question, question, earlier=turns if earlier else ()
        )
        sources = [_PAPER, _OTHER]
        return document_scope(request, built.ask_router, sources), asked

    def test_a_named_file_needs_no_model(self) -> None:
        (scope, why), asked = self.scope("does park26 say MRI?", "ALL")
        self.assertEqual(scope, [_OTHER])
        self.assertEqual(why, "named in the question")
        self.assertEqual(asked, [])

    def test_the_model_decides_about_the_earlier_paper(self) -> None:
        (scope, why), asked = self.scope("go through that paper", "SAME")
        self.assertEqual(scope, [_PAPER])
        self.assertEqual(why, "the file discussed in turn 3")
        self.assertIn("chen25.pdf", asked[0])
        (scope, _), _ = self.scope("across all papers", "ALL")
        self.assertEqual(scope, [])
        (scope, _), _ = self.scope("in another paper", "DIFFERENT")
        self.assertEqual(scope, [])

    def test_a_turn_with_few_passages_from_one_file_counts(self) -> None:
        built, _ = context(
            [chunk(str(_PAPER), ["A."]), chunk(str(_OTHER), ["B."])],
            router=lambda system, prompt, tokens: "SAME",
        )
        turns = (EarlierTurn(2, "does this paper use it?", (_PAPER,) * 2),)
        request = ToolRequest("in that paper", "in that paper", earlier=turns)
        scope, why = document_scope(
            request, built.ask_router, [_PAPER, _OTHER]
        )
        self.assertEqual(scope, [_PAPER])
        self.assertEqual(why, "the file discussed in turn 2")

    def test_the_newest_paper_wins_over_an_older_related_one(self) -> None:
        # The router related an older turn on the same topic, but the last
        # turn just named another paper: "that paper" means the newest.
        turns = (
            EarlierTurn(
                5, "which paper is about X?", (_OTHER,) * 4, linked=False
            ),
            EarlierTurn(3, "summary of that paper", (_PAPER,) * 5),
        )
        request = ToolRequest("in that paper", "in that paper", earlier=turns)
        scope, why = document_scope(
            request, lambda *_: "SAME", [_PAPER, _OTHER]
        )
        self.assertEqual(scope, [_OTHER])
        self.assertIn("turn 5", why)

    def test_without_earlier_turns_the_whole_project(self) -> None:
        (scope, _), asked = self.scope("list MRI", "SAME", earlier=False)
        self.assertEqual(scope, [])
        self.assertEqual(asked, [])


if __name__ == "__main__":
    unittest.main()
