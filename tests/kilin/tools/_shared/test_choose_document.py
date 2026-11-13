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

    def test_described_documents_resolve_through_cards(self):
        sources = [*SOURCES, Path("mijalkov17braph.pdf")]
        cards = {
            Path("mijalkov17braph.pdf"): {"title": "BRAPH: graph software",
                                          "year": "2017"},
            Path("tiunn26braph2.pdf"): {"title": "BRAPH 2: a framework",
                                        "year": "2026"},
            Path("patil25deep.pdf"): {"title": "Deep learning of spectra",
                                      "year": "2025"},
        }
        router = Mock(return_value="ALL")
        cases = (
            # "it" keeps the earlier file next to the described one.
            ("How does it compare with the 2017 paper?",
             [Path("tiunn26braph2.pdf"), Path("mijalkov17braph.pdf")]),
            ("How does BRAPH 2 differ from the earlier BRAPH framework?",
             [Path("mijalkov17braph.pdf")]),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                scope, why = document_scope(
                    ToolRequest(question, question, earlier=EARLIER),
                    router, sources, cards,
                )
                self.assertEqual(scope, expected)
                self.assertEqual(why, "described in the question")
        router.assert_not_called()
        # Without cards, descriptions are not resolved.
        scope, why = document_scope(
            ToolRequest("Summarize the 2017 paper.",
                        "Summarize the 2017 paper.", earlier=EARLIER),
            router, sources,
        )
        self.assertNotEqual(why, "described in the question")

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


    def test_generic_file_words_do_not_limit_the_search(self):
        # Issue 156: "configuration file" is not docs/configuration.md.
        sources = [Path("docs/architecture.md"),
                   Path("docs/configuration.md"),
                   Path("docs/module_map.md")]
        router = Mock(return_value="ALL")
        question = ("Revise this section about the configuration file "
                    "(\\texttt{profile.ini}) and the architecture.")
        scope, why = document_scope(ToolRequest(question, question),
                                    router, sources)
        self.assertEqual((scope, why), ([], None))
        for question, expected in (
            ("Summarize docs/configuration.md.", [sources[1]]),
            ("Explain configuration.md.", [sources[1]]),
            ("Compare architecture.md and module_map.md.",
             [sources[0], sources[2]]),
        ):
            with self.subTest(question=question):
                scope, why = document_scope(
                    ToolRequest(question, question), router, sources,
                )
                self.assertEqual((scope, why),
                                 (expected, "named in the question"))
        router.assert_not_called()

    def test_code_words_do_not_limit_the_search(self):
        sources = [Path("kilin/chat.py"), Path("kilin/config.py")]
        router = Mock(return_value="ALL")
        for question, expected in (
            ("Which config file does the app read?", []),
            ("How does the chat window scroll?", []),
            ("What does the chat module do?", [sources[0]]),
        ):
            with self.subTest(question=question):
                scope, _ = document_scope(
                    ToolRequest(question, question), router, sources,
                )
                self.assertEqual(scope, expected)
        router.assert_not_called()

if __name__ == "__main__":
    unittest.main()
