"""Measure known numerical, ranking, citation, and trace boundaries."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.system_testing._comparison_metrics import (
    cosine, evidence_metrics, file_changes, inventory, model_calls,
    inference_comparison, inference_summary, overlap, passage_identity,
    performance_summary, ranking_metrics, stage_metrics, text_metrics,
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


class TestInferenceAndPerformance(unittest.TestCase):
    def setUp(self):
        self.input = {
            "model_id": "org/model", "backend": "mlx", "device": "metal",
            "model_source": "/cache/models--org--model/snapshots/" + "a" * 40,
            "thinking": False, "input_token_count": 20,
            "generation": {"temperature": 0., "sampling": False,
                           "max_new_tokens": 30,
                           "cjk_blocking_requested": False,
                           "cjk_tokens_blocked": False,
                           "cjk_suppressed_token_count": 0},
            "chat_template": {"add_generation_prompt": True,
                              "enable_thinking": False},
            "call_id": "call",
        }
        self.output = {"call_id": "call", "output_token_count": 10,
                       "output_token_count_source": "generated_tokens",
                       "reply_duration_ms": 600, "finish_reason": "stop"}
        self.text = {"prompt": {"exact_equal": True},
                     "answer": {"exact_equal": True},
                     "structured_normalized_equal": True}

    def _row(self, data=None, result=None):
        return model_calls(_op(
            _event(1, "llm.call.input", data or self.input,
                   exact={"02_rendered_prompt": {"value": "prompt"}}),
            _event(2, "llm.call.output", result or self.output,
                   duration=500,
                   exact={"01_generated_answer": {"value": "answer"}})
        ))[0]

    def test_equal_recorded_inputs_across_cache_roots(self):
        a = self._row()
        b = self._row({**self.input, "model_source":
                       "C:\\cache\\models--org--model\\snapshots\\"
                       + "a" * 40})
        self.assertEqual(a["resolved_revision"], "a" * 40)
        self.assertEqual(b["resolved_revision"], "a" * 40)
        self.assertEqual(a["generation_duration_ms"], 500)
        value = inference_comparison(a, b, self.text)
        self.assertTrue(value["model_equal"])
        self.assertTrue(value["settings_equal"])
        self.assertEqual(value["conditional_category"], "zero_temperature")
        self.assertTrue(value["finish_reason_equal"])
        b["finish_reason"] = "length"
        self.assertFalse(inference_comparison(a, b, self.text)[
            "finish_reason_equal"])

    def test_missing_revision_settings_and_prompt_never_qualify(self):
        a = self._row()
        for field in ("device", "resolved_revision", "thinking",
                      "generation", "chat_template", "input_tokens"):
            with self.subTest(field=field):
                b = deepcopy(a)
                b.pop(field)
                self.assertIsNone(inference_comparison(a, b, self.text)[
                    "conditional_category"])
        b = self._row({**self.input, "model_source": "/unversioned/local"})
        self.assertIsNone(b["resolved_revision"])
        self.assertIsNone(inference_comparison(a, b, self.text)["model_equal"])
        b = deepcopy(a)
        b["generation"]["cjk_suppressed_token_count"] = 1
        self.assertFalse(inference_comparison(a, b, self.text)[
            "settings_equal"])
        b["generation"]["temperature"] = float("nan")
        self.assertIsNone(inference_comparison(a, b, self.text)[
            "settings_equal"])
        text = {**self.text, "prompt": {"exact_equal": False}}
        self.assertIsNone(inference_comparison(a, a, text)[
            "conditional_category"])

    def test_output_denominators_and_nonzero_temperature(self):
        a = self._row()
        b = deepcopy(a)
        flags = inference_comparison(a, b, self.text)
        absent = inference_comparison(a, b, {**self.text, "answer": None})
        b["generation"].update(temperature=.3, sampling=True)
        stochastic = inference_comparison(b, b, {
            **self.text, "answer": {"exact_equal": False}})
        rows = [{"original": a, "replayed": a, "inference": flags},
                {"original": a, "replayed": a, "inference": absent},
                {"original": b, "replayed": b, "inference": stochastic},
                {"original": None, "replayed": b}]
        summary = inference_summary([{"model_calls": rows},
                                     {"model_calls": None}])
        self.assertEqual(summary["paired_calls"], 3)
        self.assertEqual(summary["unpaired_inputs"],
                         {"original": 0, "replayed": 1})
        self.assertEqual(summary["unavailable_operations"], 1)
        self.assertEqual(summary["conditional"]["zero_temperature"],
                         dict(identical_inputs=2, available_outputs=1,
                              equal_outputs=1))
        self.assertEqual(summary["conditional"]["nonzero_temperature"],
                         dict(identical_inputs=1, available_outputs=1,
                              equal_outputs=0))

    def _span(self, seq, kind, span, start, duration=None, trace="run"):
        event = _event(seq, kind + (".started" if duration is None else
                                    ".completed"), {}, duration=duration)
        event.metadata.update(trace_id=trace, span_id=span,
                              started_elapsed_ms=start)
        return event

    def test_timing_merges_nested_spans_and_reports_missing_samples(self):
        events = (
            self._span(1, "retrieval.semantic", "a", 100),
            self._span(2, "embedding.calculate", "b", 110),
            self._span(3, "embedding.calculate", "b", 110, 30),
            self._span(4, "retrieval.semantic", "a", 100, 100),
            self._span(5, "embedding.calculate", "c", 250),
            self._span(6, "embedding.calculate", "d", 10, trace="other"),
            self._span(7, "embedding.calculate", "d", 10, 15, "other"),
        )
        timing = performance_summary(events)["spans"]
        self.assertEqual(timing["retrieval_and_embeddings"],
                         dict(duration_ms=115., available=3, total=4))
        self.assertIsNone(timing["indexing"]["duration_ms"])

    def test_throughput_uses_measured_tokens_and_positive_durations(self):
        second = {**self.input, "call_id": "other"}
        events = (_event(1, "llm.call.input", self.input),
                  _event(2, "llm.call.output", self.output, duration=500),
                  _event(3, "llm.call.input", second),
                  _event(4, "llm.call.output", {**self.output,
                         "call_id": "other", "output_token_count_source":
                         "retokenized_decoded_text"}, duration=500))
        result = performance_summary(events)
        self.assertEqual(result["output_tokens"]["recorded_total"], 20)
        self.assertEqual(result["throughput"]["generation"],
                         dict(tokens=10, duration_ms=500.,
                              tokens_per_second=20., available=1, total=2))
        self.assertAlmostEqual(result["throughput"]["reply"][
            "tokens_per_second"], 1000 * 10 / 600)


if __name__ == "__main__":
    unittest.main()
