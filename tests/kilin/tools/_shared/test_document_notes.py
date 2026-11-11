"""Test shared document-reading helpers independently of executable tools.

The shared notes service groups passages, validates citations and
calls a fake model without needing a running backend.

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
from kilin.chunking import Chunk
from kilin.tools._shared.document_notes import (
    NOTHING,
    answer_citations,
    clean_notes,
    notes_prompt,
    plan_parts,
    quote_citations,
    take_notes,
)


class TestDocumentNotes(unittest.TestCase):
    """Exercise shared note-taking with deterministic synthetic data."""

    @staticmethod
    def _chunk(number: int, text: str) -> Chunk:
        """Construct one source passage.

        Parameters
        ----------
        number: int
            Stable passage identifier.
        text: str
            Passage text.

        Returns
        -------
        Chunk
            Source fragment with deterministic metadata.

        """

        return Chunk(
            identifier=str(number), path=Path("paper.md"), text=text,
            start_line=number, end_line=number, kind="markdown",
            symbol=None, section=None,
        )

    def test_plan_parts_preserves_all_passages(self) -> None:
        """Keep consecutive source order and assign no passage twice.

        """

        chunks = [self._chunk(i, "many text words " * 5)
                  for i in range(4)]
        parts = plan_parts(chunks, 8)
        self.assertEqual([i for part in parts for i in part], list(range(4)))
        self.assertEqual(plan_parts([], 100), [])

    def test_note_prompts_keep_original_passage_numbers(self) -> None:
        """Prompts keep the citation contract across tools.

        """

        system, prompt = notes_prompt(
            "Summarize", "paper.md", 2, 3, [(5, "Physical result.")]
        )
        self.assertIn("short bullet notes", system)
        self.assertIn("part 2 of 3", prompt)
        self.assertIn("[5] Physical result.", prompt)
        self.assertIn(NOTHING, prompt)

    def test_clean_notes_checks_scope_and_sentinel(self) -> None:
        """Discard uncited notes and notes citing the wrong part.

        """

        self.assertEqual(clean_notes(NOTHING, [1, 2]), [])
        self.assertEqual(
            clean_notes(
                "- Fact [2]\n- Fake [3]\n- No citation\n- Both [1, 2]",
                [1, 2],
            ),
            ["- Fact [2]", "- Both [1, 2]"],
        )

    def test_ranges_in_notes_are_checked_against_part(self) -> None:
        """Accept supported ranges and reject partly unsupported ones.

        """

        reply = "- Supported [1-2]\n- Wrong [1, 3-9]"
        self.assertEqual(clean_notes(reply, [1, 2]), ["- Supported [1-2]"])

    def test_invalid_answer_citations(self) -> None:
        """Report only citation numbers outside the available range.

        """

        self.assertEqual(answer_citations("Found [2], [3, 9]", 3), [9])
        self.assertEqual(answer_citations("No citations", 2), [])
        self.assertEqual(answer_citations("Found [3-9]", 3), [])

    def test_quote_citations_move_to_the_passage_with_the_quote(self) -> None:
        """Cite the one given passage that contains a quoted phrase.

        """

        passages = {
            1: "BRAPH is a graph theory software.",
            10: "It provides a clear modular\nstruc- ture for new objects.",
            12: "Results in patients with MCI.",
        }
        answer, counts = quote_citations(
            'It has a "clear modular structure" [1]. It is software [1].',
            passages,
        )

        self.assertEqual(
            answer,
            'It has a "clear modular structure" [10]. It is software [1].',
        )
        self.assertEqual(counts, {"moved": 1, "unplaced": 0})

    def test_quote_citations_leave_supported_or_unclear_quotes(self) -> None:
        """Change nothing when the cited passage, or no single one, fits.

        """

        passages = {1: "one two three four", 2: "one two three five"}
        cases = (
            ('Says “two three four” [1].', {"moved": 0, "unplaced": 0}),
            ('Says "one two three" [9].', {"moved": 0, "unplaced": 0}),
            ('Says "six seven eight" [1].', {"moved": 0, "unplaced": 1}),
            ('Says "BRAPH" [2].', {"moved": 0, "unplaced": 0}),
            ('Says "two three four" [2] and [1].',
             {"moved": 0, "unplaced": 0}),
        )
        for text, counts in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    quote_citations(text, passages), (text, counts)
                )

    def test_quote_citations_do_not_rewrite_invalid_ranges(self) -> None:
        """Leave incorrect spans for explicit validation reporting.

        """

        passages = {1: "a clear modular structure"}
        answer = 'It has a "clear modular structure" [1-999999999].'
        self.assertEqual(quote_citations(answer, passages)[0], answer)

    def test_take_notes_uses_fake_model_and_traces_results(self) -> None:
        """Generate cited notes with callbacks and no real model.

        """

        progress, emit = Mock(), Mock()
        context = SimpleNamespace(
            ask_model=Mock(return_value="- Observation [1]"),
            progress=progress,
            trace=SimpleNamespace(include_content=True, emit=emit),
            trace_turn_id="turn-1",
        )
        notes = take_notes(
            "Read a paper", "paper.md", ["Observation"],
            [[0]], context, what="paper",
        )
        self.assertEqual(notes, [(1, ["- Observation [1]"])])
        self.assertEqual(context.ask_model.call_count, 1)
        self.assertIsNone(context.ask_model.call_args.args[-1])
        progress.assert_called_once()
        emit.assert_called_once()
        self.assertEqual(emit.call_args.kwargs["data"]["notes"], 1)
        self.assertIn("raw", emit.call_args.kwargs["data"])

    def test_failed_model_call_is_recorded_without_notes(self) -> None:
        """Record a failed reading part without treating it as empty notes.

        """

        context = SimpleNamespace(
            ask_model=Mock(side_effect=RuntimeError("offline")),
            progress=Mock(),
            trace=SimpleNamespace(include_content=False, emit=Mock()),
            trace_turn_id=None,
        )
        with self.assertRaises(IncompleteModelResponse) as failure:
            take_notes(
                "Summarize", "paper.md", ["Example"], [[0]], context
            )
        self.assertEqual(failure.exception.reason, "model_error")
        context.ask_model.assert_called_once()
        event = context.trace.emit.call_args.kwargs["data"]
        self.assertEqual(event["notes"], 0)
        self.assertNotIn("raw", event)


if __name__ == "__main__":
    unittest.main()
