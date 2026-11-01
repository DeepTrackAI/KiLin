"""Keep incomplete model calls from being swallowed by batch processors.

Exercise real tool helper paths with deterministic model failures and
explicit negative replies. No additional retries are permitted here.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin._model_completion import IncompleteModelResponse
from kilin.tools import EarlierTurn, ToolRequest
from kilin.tools.filter_results import FilterSettings, run as filter_items
from kilin.tools._shared.document_notes import take_notes
from kilin.tools._shared.claims import _judge as judge_claim
from kilin.tools.batch_classify.tool import _label_batch
from kilin.tools.find_passages.tool import _judge, split_sentences
from tests.kilin.tools._shared._support import chunk


class TestToolCompletion(unittest.TestCase):
    """Propagate completion failures to the coordinator without retrying."""

    def setUp(self) -> None:
        """Create one failing model service and source passage."""

        self.failure = IncompleteModelResponse(
            "Token limit", reason="token_limit", finish_reason="length"
        )
        self.context = SimpleNamespace(
            ask_model=Mock(side_effect=self.failure), progress=Mock(),
            trace=SimpleNamespace(emit=Mock(), include_content=True),
            trace_turn_id="t",
        )
        self.chunks = [
            chunk("1", "Deep learning is useful for classification.")
        ]

    def test_sentence_judgement_propagates_incompletion(self) -> None:
        """Never turn a failed batch into an empty sentence selection."""

        batch = split_sentences(self.chunks)
        with self.assertRaises(IncompleteModelResponse):
            _judge("learning", [batch, batch], "Find", self.context)
        self.context.ask_model.assert_called_once()

    def test_sentence_judgement_rejects_failed_backend_and_bad_text(self):
        """Neither a backend error nor arbitrary prose proves no matches."""

        batch = split_sentences(self.chunks)
        for error, reply in ((RuntimeError("offline"), None), (None, "maybe")):
            with self.subTest(reply=reply):
                self.context.ask_model = Mock(side_effect=error,
                                              return_value=reply)
                with self.assertRaises(IncompleteModelResponse):
                    _judge("learning", [batch], "Find", self.context)
                self.context.ask_model.assert_called_once()

    def test_notes_propagate_incompletion(self) -> None:
        """Do not continue reading or write a report after failed notes."""

        with self.assertRaises(IncompleteModelResponse):
            take_notes("Task", "paper", ["text"], [[0], [0]], 200,
                       self.context)
        self.context.ask_model.assert_called_once()

    def test_notes_require_citations_or_explicit_nothing(self) -> None:
        """An unparseable reply is distinct from a relevant-evidence miss."""

        self.context.ask_model = Mock(return_value="An uncited note")
        with self.assertRaises(IncompleteModelResponse):
            take_notes("Task", "paper", ["text"], [[0]], 200, self.context)
        self.context.ask_model.return_value = "NOTHING"
        self.assertEqual(
            take_notes("Task", "paper", ["text"], [[0]], 200, self.context),
            [(1, [])],
        )

    def test_classification_does_not_retry_incomplete_batch(self) -> None:
        """Never relabel truncated output as an unclassified negative."""

        with self.assertRaises(IncompleteModelResponse):
            _label_batch("Q", ["term"], ["yes"], ["text"], [0],
                         self.context)
        self.context.ask_model.assert_called_once()

    def test_claim_checking_propagates_incompletion(self) -> None:
        """Do not convert a completion failure into a scientific verdict."""

        with self.assertRaises(IncompleteModelResponse):
            judge_claim(("s", "p"), self.chunks, self.context.ask_model, None)
        self.context.ask_model.assert_called_once()


    def test_filter_success_and_explicit_empty_results(self) -> None:
        """Exercise real parsing and evidence construction, not just errors."""

        request = ToolRequest("Keep relevant", "Keep relevant", earlier=(
            EarlierTurn(1, "List", (Path("paper.md"),), ("Some evidence",)),
        ))
        for reply, matched in (("1", 1), ("NONE", 0)):
            with self.subTest(reply=reply):
                context = SimpleNamespace(
                    ask_model=Mock(return_value=reply), progress=Mock(),
                    done=Mock(),
                )
                result = filter_items(FilterSettings(), request, context)
                self.assertEqual(result.summary["matched"], matched)
                self.assertEqual(len(result.evidence), matched)
                context.ask_model.assert_called_once()
                context.done.assert_called_once()
                if matched:
                    self.assertEqual(
                        result.evidence[0]["text"], "Some evidence"
                    )
                else:
                    self.assertIn("None of the items matches", result.answer)

    def test_filter_invalid_output_is_not_an_empty_selection(self) -> None:
        """Reject empty and malformed replies at the real tool boundary."""

        request = ToolRequest("Keep relevant", "Keep relevant", earlier=(
            EarlierTurn(1, "List", (Path("paper.md"),), ("Some evidence",)),
        ))
        for reply in ("", "maybe", "999", "1,"):
            with self.subTest(reply=reply):
                context = SimpleNamespace(
                    ask_model=Mock(return_value=reply), progress=Mock(),
                    done=Mock(),
                )
                with self.assertRaises(IncompleteModelResponse):
                    filter_items(FilterSettings(), request, context)
                context.ask_model.assert_called_once()
                context.done.assert_not_called()


if __name__ == "__main__":
    unittest.main()
