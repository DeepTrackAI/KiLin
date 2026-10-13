"""Check show_source: the cited sentence in context, explained."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.tools import ToolRequest
from qachat.tools.contract import EarlierTurn
from qachat.tools.show_source import SourceSettings, run
from tests.qachat.tools.helpers import chunk, context

_PAPER = "repo/kim25.pdf"


def _chunks():
    return [
        chunk(_PAPER, ["The data come from a public cohort of adults.",
                       "Scans were preprocessed with a standard pipeline."]),
        chunk(_PAPER, ["The network has two hidden layers of units.",
                       "The model was trained for 100 epochs in total.",
                       "Early stopping was not used in the experiments."],
              start=3),
        chunk(_PAPER, ["Results are reported on a held-out test set.",
                       "The code is available from the authors online."],
              start=6),
        chunk("repo/other.pdf", ["Another paper trained for 5 epochs."]),
    ]


class TestShowSource(unittest.TestCase):
    """The cited text is copied with its neighbours; nothing is searched."""

    def ask(self, question, earlier, settings=SourceSettings()):
        prompts = []

        def model(system, prompt, tokens):
            prompts.append(prompt)
            return "It describes how long the model was trained [1]."

        ranked = []
        built, _ = context(
            _chunks(), model=model,
            rank=lambda text: ranked.append(text) or [],
        )
        request = ToolRequest(question, question, earlier=earlier)
        return run(settings, request, built), prompts, ranked

    def turn(self, **fields):
        cited = _chunks()[1]
        return EarlierTurn(
            7, "what epoch number does that paper use?",
            (cited.path,), (cited.text,), (cited.start_line,),
            answer="The model was trained for 100 epochs [54].", **fields,
        )

    def test_the_cited_sentence_is_shown_with_its_neighbours(self) -> None:
        result, prompts, ranked = self.ask(
            "show me that sentence about epoch", (self.turn(),)
        )
        self.assertEqual(ranked, [])  # No new search.
        self.assertIn("**kim25.pdf, line 2** [1]", result.answer)
        self.assertIn(
            "**The model was trained for 100 epochs in total.**",
            result.answer,
        )
        # Two sentences before (one from the passage before) and after.
        self.assertIn("standard pipeline.", result.answer)
        self.assertIn("held-out test set.", result.answer)
        self.assertNotIn("public cohort", result.answer)
        self.assertNotIn("5 epochs", result.answer)
        self.assertIn(">>The model was trained for 100 epochs", prompts[0])
        self.assertTrue(prompts[0].endswith(
            "Question: show me that sentence about epoch"
        ))
        self.assertIn("It describes how long", result.answer)
        self.assertEqual(result.evidence[0]["citation"], 1)
        self.assertEqual(result.evidence[0]["path"], Path(_PAPER))

    def test_no_context_sentences_shows_only_the_sentence(self) -> None:
        settings = SourceSettings(context_sentences=0)
        result, _, _ = self.ask(
            "show me that sentence", (self.turn(),), settings
        )
        self.assertIn("> **The model was trained", result.answer)
        self.assertNotIn("hidden layers", result.answer)

    def test_the_passage_the_question_means_is_picked(self) -> None:
        files = _chunks()
        texts = tuple(c.text for c in files)
        turn = EarlierTurn(
            2, "list the sentences about training",
            tuple(c.path for c in files), texts,
            tuple(c.start_line for c in files),
        )
        settings = SourceSettings(max_passages=1)
        result, _, _ = self.ask(
            "show me the one about the held-out test set", (turn,), settings
        )
        self.assertIn("**Results are reported on a held-out", result.answer)
        self.assertEqual(len(result.evidence), 1)

    def test_without_earlier_sources_it_explains(self) -> None:
        result, prompts, _ = self.ask(
            "show me that sentence", (EarlierTurn(1, "hello"),)
        )
        self.assertEqual(prompts, [])
        self.assertIn("couldn't find an earlier answer", result.answer)


if __name__ == "__main__":
    unittest.main()
