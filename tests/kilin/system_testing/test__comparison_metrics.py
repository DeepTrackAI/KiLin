"""Measure known numerical, ranking, citation, and trace boundaries."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.system_testing._comparison_metrics import (
    cosine, evidence_metrics, file_changes, inventory, model_calls,
    overlap, passage_identity, ranking_metrics, stage_metrics, text_metrics,
)
from kilin.system_testing.timeline import TimelineOperation
from kilin.system_testing.traces import TraceEvent


def _passage(path, text="text", *, cell=None, citation=1):
    return {"path": path, "text": text, "start_line": 1, "end_line": 2,
            "symbol": None, "cell": cell, "citation": citation}


def _event(sequence, kind, details, *, exact=None, duration=None):
    metadata = {"event_type": kind, "sequence": sequence, "trace_id": "run",
                "component": "test", "status": "ok"}
    if duration is not None:
        metadata["duration_ms"] = duration
    return TraceEvent("trace.json", str(sequence), metadata, details,
                      exact or {})


def _op(*events):
    return TimelineOperation("ask", 1, len(events), "completed", {}, events)


class TestComparisonMetrics(unittest.TestCase):
    def test_text_normalization_is_not_case_or_punctuation_folding(self):
        values = text_metrics("<think>hidden</think>cafe\u0301\n  value [1]",
                              "caf\u00e9 value [1]")
        self.assertTrue(values["normalized_equal"])
        self.assertFalse(values["exact_equal"])
        self.assertEqual(values["token_f1"], 1.)
        self.assertEqual(text_metrics("a a b", "a b b")["token_f1"], 2/3)
        self.assertEqual(text_metrics("A!", "a?")["token_f1"], 0.)
        self.assertIsNone(text_metrics("", "")["token_f1"])

    def test_exact_and_source_overlap_and_rank_displacement(self):
        a, b, c = [passage_identity(_passage(name))
                   for name in ("a", "b", "c")]
        result = overlap([a, b, c], [b, a])
        self.assertEqual(result["jaccard"], 2/3)
        self.assertEqual(result["prefix_overlap"], 1.)
        self.assertEqual(result["mean_absolute_rank_shift"], 1.)
        self.assertIsNone(overlap([], [])["jaccard"])
        self.assertEqual(overlap([a], [])["jaccard"], 0.)
        changed = passage_identity(_passage("a", "changed"))
        self.assertEqual(overlap([a], [changed])["source_jaccard"], 1.)
        self.assertEqual(overlap([a], [changed])["jaccard"], 0.)

    def test_citations_match_passage_not_number_or_notebook_cell(self):
        original = {"answer": "Claim [1, 8]", "evidence": [
            _passage("note.ipynb", cell=1),
            _passage("note.ipynb", cell=2, citation=2),
        ]}
        replayed = {"answer": "<think>[9]</think>Claim [2]", "evidence": [
            _passage("note.ipynb", cell=2),
            _passage("note.ipynb", cell=1, citation=2),
        ]}
        result = evidence_metrics(original, replayed)
        self.assertEqual(result["overlap"]["jaccard"], 1.)
        self.assertEqual(result["citations"]["overlap"]["jaccard"], 1.)
        self.assertEqual(result["citations"]["original_unresolved"], ["8"])
        self.assertEqual(result["citations"]["replayed_unresolved"], [])

    def test_missing_ranking_is_not_empty_and_source_ranking_supported(self):
        rows = [{"rank": 1, "chunk": passage_identity(_passage("a"))}]
        ranked = _event(1, "retrieval.semantic.ranked", {
            "count": 1, "results": rows,
        })
        source = _event(2, "retrieval.sources.ranked", {
            "sources": [{"path": "a"}, {"path": "b"}],
        })
        other = _event(1, "retrieval.sources.ranked", {
            "sources": [{"path": "b"}, {"path": "a"}],
        })
        first, second = ranking_metrics(_op(ranked, source), _op(other))
        self.assertIsNone(first["replayed"])
        self.assertIsNone(first["overlap"])
        self.assertEqual(second["identity"], "source_path")
        self.assertEqual(second["overlap"]["mean_absolute_rank_shift"], 1.)

    def test_model_calls_correlate_outputs_and_leave_unknown_counts(self):
        first = _event(1, "llm.call.input", {
            "call_id": "a", "input_token_count": 7,
        }, exact={"02_rendered_prompt": {"value": "prompt"}})
        second = _event(2, "llm.call.input", {
            "call_id": "b", "input_token_count": 4,
        })
        output = _event(3, "llm.call.output", {
            "call_id": "b", "output_token_count": 5,
        }, exact={"01_generated_answer": {"value": "answer"}})
        a, b = model_calls(_op(first, second, output))
        self.assertIsNone(a["output_tokens"])
        self.assertEqual(a["prompt"], "prompt")
        self.assertEqual(b["output_tokens"], 5)
        self.assertEqual(b["answer"], "answer")

    def test_stage_totals_require_every_duration(self):
        a = _op(_event(1, "stage", {}, duration=1.5),
                _event(2, "stage", {}))
        b = _op(_event(1, "stage", {}, duration=0))
        result = stage_metrics(a, b)[0]
        self.assertIsNone(result["original"]["total_duration_ms"])
        self.assertEqual(result["replayed"]["total_duration_ms"], 0)

    def test_cosine_validates_dimensions_and_norm(self):
        self.assertAlmostEqual(cosine([1., 0.], [1., 1.]), 2**-0.5)
        self.assertEqual(cosine([1., 0.], [-1., 0.]), -1.)
        for first, second in (([0.], [1.]), ([1.], [1., 2.]),
                              ([float("nan")], [1.])):
            with self.subTest(first=first), self.assertRaises(ValueError):
                cosine(first, second)

    def test_inventory_refuses_links_and_partial_equality(self):
        with TemporaryDirectory() as name:
            root = Path(name).resolve()
            folder = root / "outputs"
            folder.mkdir()
            (folder / "file").write_bytes(b"content")
            first = inventory(root, "outputs")
            self.assertEqual(first["files"]["outputs/file"]["sha256"],
                             sha256(b"content").hexdigest())
            try:
                (folder / "link").symlink_to(folder / "file")
            except OSError as error:
                self.skipTest(f"Symbolic links unavailable: {error}")
            second = inventory(root, "outputs")
            self.assertTrue(second["issues"])
            self.assertIsNone(file_changes(first, second)["changes"])


if __name__ == "__main__":
    unittest.main()
