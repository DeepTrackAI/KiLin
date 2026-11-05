"""Verify which documents a request is limited to after earlier turns.

Clear wording decides without the router model; the router is a mock that
records whether it was asked.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock

from kilin.tools._contract import EarlierTurn, ToolRequest
from kilin.tools._shared.choose_document import document_scope

SOURCES = [Path("tiunn26braph2.pdf"), Path("patil25deep.pdf")]
EARLIER = (EarlierTurn(1, "Summarize the BRAPH 2 article.",
                       paths=(Path("tiunn26braph2.pdf"),) * 4),)


class TestDocumentScopeWording(unittest.TestCase):

    def _scope(self, question, reply="ALL"):
        router = Mock(return_value=reply)
        scope, _ = document_scope(
            ToolRequest(question, question, earlier=EARLIER), router,
            SOURCES,
        )
        return scope, router

    def test_words_for_one_document_mean_the_earlier_file(self):
        for question in ("What are its main limitations?",
                         "What kind of measurements does that study "
                         "analyze?", "How does it compare with the 2017 "
                         "paper?"):
            with self.subTest(question=question):
                scope, router = self._scope(question)
                self.assertEqual(scope, [Path("tiunn26braph2.pdf")])
                router.assert_not_called()

    def test_words_for_several_documents_mean_the_project(self):
        for question in ("Which of these articles mention BRAPH?",
                         "Compare its methods with the other papers.",
                         "Summarize all of them."):
            with self.subTest(question=question):
                scope, router = self._scope(question, reply="SAME")
                self.assertEqual(scope, [])
                router.assert_not_called()

    def test_other_requests_ask_the_router(self):
        scope, router = self._scope("Summarize the main findings.",
                                    reply="SAME")
        self.assertEqual(scope, [Path("tiunn26braph2.pdf")])
        router.assert_called_once()

    def test_named_documents_and_no_earlier_file(self):
        router = Mock()
        scope, why = document_scope(
            ToolRequest("Its methods in patil25deep.pdf?", "x"), router,
            SOURCES,
        )
        self.assertEqual((scope, why),
                         ([Path("patil25deep.pdf")], "named in the question"))
        scope, _ = document_scope(ToolRequest("Its methods?", "x"), router,
                                  SOURCES)
        self.assertEqual(scope, [])
        router.assert_not_called()


if __name__ == "__main__":
    unittest.main()
