"""Preserve independent tool work and distinguish failures from negatives.

Use real note, selector, source-display and reference-processing paths with
simulated model responses. Never load a model or use the network.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin._model_completion import IncompleteModelResponse
from kilin.tools import EarlierTurn, ToolRequest
from kilin.tools._shared.document_notes import take_notes
from kilin.tools._shared.mentions import MentionCollection, scan_mentions
from kilin.tools._shared.outputs import save_output_file
from kilin.tools.batch_review import tool as review_tool
from kilin.tools.find_fact import tool as fact_tool
from kilin.tools.batch_classify import tool as classify_tool
from kilin.tools._shared.claims import Claim, VERDICT_FAILED, VERDICT_SUPPORTED
from kilin.tools._shared._rating import citation_suitability
from kilin.tools.filter_results import tool as filtering
from kilin.tools.find_passages import tool as passages_tool
from kilin.tools.show_source import tool as source_tool
from kilin.tools.reference_check import tool as reference_tool
from kilin.tools.suggest_reference import tool as suggest_tool
from tests.kilin.tools._shared._support import chunk
from tests.kilin.tools.reference_check import test_tool as reference_fixture


def _failure():
    """Return a token failure without exposing its unusable raw text."""

    return IncompleteModelResponse(
        "Output-token limit exhausted.", reason="token_limit",
        finish_reason="length", raw_text="UNSAFE TRUNCATED DATA",
    )


def _context(replies):
    """Construct the common plain-callable services for tool tests."""

    return SimpleNamespace(
        ask_model=Mock(side_effect=replies), progress=Mock(), done=Mock(),
        trace=SimpleNamespace(emit=Mock(), include_content=True),
        trace_turn_id="test", model_id="fake", confirm=None,
    )


class TestIndependentRecovery(unittest.TestCase):
    """Check partial state, continuation and strict absence semantics."""

    def test_notes_keep_successes_without_inventing_empty_parts(self):
        context = _context(["Observation [1]", _failure(), "NOTHING"])
        failures = []
        notes = take_notes("Task", "paper", ["one", "two", "three"],
                           [[0], [1], [2]], context, failures=failures)
        self.assertEqual(notes, [(1, ["- Observation [1]"]), (3, [])])
        self.assertEqual(context.ask_model.call_count, 3)
        self.assertEqual(failures[0]["part"], 2)
        self.assertEqual(failures[0]["reason"], "token_limit")
        self.assertNotIn("UNSAFE", repr(notes) + repr(failures))

    def test_sentence_batches_continue_after_failure(self):
        pieces = [chunk(str(i), f"Sentence {i} contains useful results.")
                  for i in range(1, 4)]
        batches = [[s] for s in passages_tool.split_sentences(pieces)]
        context = _context(["1", _failure(), "NONE"])
        failures = []
        selected = passages_tool._judge("results", batches, "Find", context,
                                        failures)
        self.assertEqual(selected, batches[0])
        self.assertEqual(context.ask_model.call_count, 3)
        self.assertEqual(failures[0]["batch"], 2)
        self.assertEqual(failures[0]["sentences"], 1)

    def test_filter_retains_other_items_and_marks_unknown_batch(self):
        context = _context(["1", _failure(), "3"])
        request = ToolRequest("Keep useful", "Keep useful", earlier=(
            EarlierTurn(1, "List", (Path("a"), Path("b"), Path("c")),
                        ("First evidence", "Second evidence",
                         "Third evidence")),
        ))
        with patch.object(filtering, "_batches", return_value=[
            [(1, "First evidence")], [(2, "Second evidence")],
            [(3, "Third evidence")],
        ]):
            result = filtering.run(
                filtering.FilterSettings(), request, context
            )
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(result.summary["matched"], 2)
        self.assertEqual(context.ask_model.call_count, 3)
        self.assertEqual([e["path"] for e in result.evidence],
                         [Path("a"), Path("c")])
        self.assertEqual(result.summary["incomplete_items"][0]["items"], [2])
        self.assertNotIn("UNSAFE", result.answer)

    def test_failed_source_explanation_keeps_verbatim_text(self):
        text = "The method produces independently measured observations."
        passage = chunk("1", text)
        context = _context([_failure()])
        context.chunks = [passage]
        request = ToolRequest("Show the source", "Show the source", earlier=(
            EarlierTurn(1, "Explain method", (passage.path,), (text,), (1,)),
        ))
        result = source_tool.run(
            source_tool.SourceSettings(), request, context
        )
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]["text"], text)
        self.assertIn("independently measured", result.answer)
        self.assertIn("Source explanation", result.answer)
        self.assertNotIn("UNSAFE", result.answer)
        context.ask_model.assert_called_once()

    def test_reference_claim_failure_does_not_stop_other_claims(self):
        passage = chunk("1", "The method produces measured observations.",
                        path="cited.md")
        sentence = passages_tool.split_sentences([passage])[0]
        claims = [Claim(i, sentence, passage.path) for i in range(1, 4)]
        context = _context([
            _failure(),
            "VERDICT: supported\nQUOTE: The method produces measured "
            "observations.\nCOMMENT: Direct evidence.",
            "VERDICT: not found\nQUOTE: NONE\nCOMMENT: Unrelated.",
        ])
        context.rank = lambda text: [SimpleNamespace(chunk=passage)]
        report = reference_tool.ReferenceReport(
            Path("main.md"), (), (), SimpleNamespace(),
        )
        with (
            patch.object(reference_tool, "report_claims", return_value=claims),
            patch.object(reference_tool, "confirm_plan",
                         return_value=("continue", False)),
        ):
            result = reference_tool._verify(
                report, {passage.path: [passage]},
                reference_tool.ReferenceCheckSettings(), context,
            )
        self.assertEqual(context.ask_model.call_count, 3)
        self.assertEqual(len(result.verdicts), 3)
        self.assertEqual(result.verdicts[0].verdict, VERDICT_FAILED)
        self.assertIn("token_limit", result.verdicts[0].comment)
        self.assertIsNone(citation_suitability(result.verdicts[0]))
        self.assertEqual(result.verdicts[1].verdict, VERDICT_SUPPORTED)
        self.assertEqual(result.verdicts[0].quote, "")

    def test_suggest_reference_keeps_success_after_failed_candidate(self):
        pieces = [chunk("1", "Cells exhibit diffusion in living tissue.",
                        path="first.md"),
                  chunk("2", "Cells exhibit diffusion in living tissue.",
                        path="second.md")]
        context = _context([
            _failure(), "VERDICT: supported\nQUOTE: Cells exhibit diffusion "
            "in living tissue.\nCOMMENT: Direct evidence.",
        ])
        context.chunks = pieces
        context.rank = lambda text: [SimpleNamespace(chunk=p) for p in pieces]
        context.ask_router = Mock(return_value="NONE")
        question = "Suggest a reference for: cells exhibit diffusion in tissue"
        result = suggest_tool.run(
            suggest_tool.SuggestReferenceSettings(),
            ToolRequest(question, question), context,
        )
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(result.summary["suggested"], 1)
        self.assertEqual(result.summary["verdicts"]["failed"], 1)
        self.assertEqual([e["path"] for e in result.evidence],
                         [Path("second.md")])
        self.assertNotIn("UNSAFE", result.answer)
        self.assertNotIn("first.md", result.summary["suitability"])

    def test_classification_skips_failed_batch_without_schema_retry(self):
        pieces = [chunk("1", "BRAPH provides a graphical interface."),
                  chunk("2", "BRAPH was described in earlier work.", line=10)]
        scan = scan_mentions(pieces, ("BRAPH",))
        found = MentionCollection(scan, None, (), {"name": "batch_classify"},
                                  "")
        context = _context([
            _failure(),
            '[{"id": 2, "category": "citation", "reason": "Earlier work"}]',
        ])
        with (
            patch.object(
                classify_tool, "collect_mentions", return_value=found
            ),
            patch.object(
                classify_tool, "plan_batches", return_value=[[0], [1]]
            ),
            patch.object(classify_tool, "confirm_plan",
                         return_value=("continue", False)),
        ):
            result = classify_tool.run(
                classify_tool.ClassifySettings(
                    categories=("method", "citation")
                ),
                ToolRequest("Classify BRAPH", "Classify BRAPH"), context,
            )
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertEqual(result.summary["model_calls"], 2)
        self.assertEqual(result.summary["labelled"], 1)
        self.assertEqual(result.summary["not_classified"], 1)
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(result.summary["incomplete_items"][0]["items"], [1])
        self.assertNotIn("UNSAFE", result.answer)

    def test_passage_run_exports_partial_coverage_and_keeps_successes(self):
        pieces = [chunk("1", "This sentence reports a useful result."),
                  chunk("2", "This sentence needs further assessment.",
                        line=10)]
        sentences = passages_tool.split_sentences(pieces)
        context = _context(["1", _failure()])
        context.chunks = pieces
        context.ask_router = Mock(return_value="NONE")
        context.rank = Mock(return_value=[])
        context.choose = None
        with TemporaryDirectory() as directory:
            context.project_root = Path(directory)

            def export(ctx, **kwargs):
                return save_output_file(ctx.project_root, "partial.md",
                                        kwargs["markdown"])

            with (
                patch.object(passages_tool, "_topic",
                             return_value=("results", None, False)),
                patch.object(passages_tool, "document_scope",
                             return_value=([], None)),
                patch.object(
                    passages_tool, "term_hits", return_value=([], [])
                ),
                patch.object(passages_tool, "_batches", side_effect=[
                    [[sentences[0]], [sentences[1]]], [],
                ]),
                patch.object(passages_tool, "confirm_plan",
                             return_value=("continue", False)),
                patch.object(
                    passages_tool, "offer_export", side_effect=export
                ),
            ):
                result = passages_tool.run(
                    passages_tool.PassageSettings(),
                    ToolRequest("Find results", "Find results"), context,
                )
            self.assertEqual(result.summary["status"], "partial")
            self.assertEqual(result.summary["read"], 1)
            self.assertEqual(result.summary["found"], 1)
            self.assertEqual(result.summary["batches_completed"], 1)
            self.assertEqual(context.ask_model.call_count, 2)
            text = (context.project_root / result.summary["saved"]).read_text(
                encoding="utf-8",
            )
            self.assertIn("useful result", text)
            self.assertIn("Incomplete items", text)
            self.assertIn("partial list", result.answer)
            self.assertNotIn("UNSAFE", text)

    def test_fact_reading_omits_failed_document_not_negative_notes(self):
        context = _context([_failure(), "Observation [2]", "NOTHING"])
        failures = []
        material = fact_tool._notes(
            "Task", [(Path("failed.md"), [[0]]), (Path("valid.md"), [[1]]),
                     (Path("empty.md"), [[2]])],
            ["one", "two", "three"], fact_tool.FactSettings(), context,
            whole=True, failures=failures,
        )
        self.assertEqual(material,
                         [("valid.md", ["- Observation [2]"]),
                          ("empty.md", [])])
        self.assertEqual(failures[0]["passage_numbers"], [1])
        prompt = fact_tool.fact_prompt("Task", material, None)[1]
        self.assertNotIn("failed.md", prompt)
        self.assertIn("empty.md:\nNOTHING FOUND", prompt)

    def test_batch_review_keeps_good_notes_and_marks_failed_part(self):
        pieces = [chunk("1", "First observation."),
                  chunk("2", "Second observation.", line=10)]
        context = _context([_failure(), "Second observation [2]"])
        context.chunks, context.graph = pieces, None
        context.retrieval_count = 5
        context.rank = Mock(return_value=[])
        context.write = Mock(return_value="Second observation [2].")
        selection = SimpleNamespace(chunks=pieces, path=Path("paper.md"),
                                    functions=(), reason="named",
                                    estimated_tokens=64)
        with (
            patch.object(
                review_tool, "read_document", return_value=selection
            ),
            patch.object(review_tool, "plan_parts", return_value=[[0], [1]]),
            patch.object(review_tool, "term_hits", return_value=([], [])),
            patch.object(review_tool, "confirm_plan",
                         return_value=("continue", False)),
        ):
            result = review_tool.run(review_tool.ReviewSettings(),
                                     ToolRequest("Read", "Read"), context)
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(result.summary["passages_read"], 1)
        self.assertEqual(result.summary["parts_without_notes"], [])
        self.assertEqual([e["citation"] for e in result.evidence], [2])
        self.assertEqual(context.ask_model.call_count, 2)
        prompt = context.write.call_args.args[1]
        self.assertIn("Some reading parts failed", prompt)
        self.assertNotIn("UNSAFE", prompt)
        self.assertNotIn("model-call limit was reached", result.answer)

    def test_reference_report_records_failed_claim_without_losing_others(self):
        pieces = [chunk("1", reference_fixture.PAPER, path="paper.md"),
                  chunk("2", reference_fixture.BRAPH, path="braph.md"),
                  chunk("3", reference_fixture.BRAPH2, path="braph2.md")]
        context = _context([
            _failure(),
            "VERDICT: not found\nQUOTE: NONE\nCOMMENT: Unrelated.",
            "VERDICT: not found\nQUOTE: NONE\nCOMMENT: Unrelated.",
        ])
        context.chunks, context.choose = pieces, None
        context.rank = lambda text: [SimpleNamespace(chunk=p) for p in pieces]
        with TemporaryDirectory() as directory:
            context.project_root = Path(directory)
            with patch.object(reference_tool, "choose_document",
                              return_value=(Path("paper.md"), "named", False)):
                # Without splitting sentences, so the replies map to the
                # three claims in order.
                result = reference_tool.run(
                    reference_tool.ReferenceCheckSettings(focus_claims=False),
                    ToolRequest("Check references", "Check references"),
                    context,
                )
            self.assertEqual(context.ask_model.call_count, 3)
            self.assertEqual(result.summary["status"], "partial")
            self.assertEqual(result.summary["verdicts"]["failed"], 1)
            report_path = context.project_root / result.summary["report"]
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("Incomplete items", text)
            self.assertIn("token_limit", text)
            self.assertNotIn("UNSAFE", text)

    def test_all_reference_candidates_failed_are_not_zero_support(self):
        passage = chunk("1", "Cells exhibit diffusion in living tissue.")
        context = _context([_failure()])
        context.chunks = [passage]
        context.rank = lambda text: [SimpleNamespace(chunk=passage)]
        context.ask_router = Mock(return_value="NONE")
        question = "Suggest a reference for: cells exhibit diffusion in tissue"
        result = suggest_tool.run(
            suggest_tool.SuggestReferenceSettings(),
            ToolRequest(question, question), context,
        )
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertEqual(result.summary["verdicts"], {"failed": 1})
        self.assertIn("inconclusive", result.answer)
        self.assertEqual(result.evidence, ())
        self.assertIn("Output-token limit", result.answer)


if __name__ == "__main__":
    unittest.main()
