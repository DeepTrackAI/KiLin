"""Check identity mappings and narrow context-metadata normalization."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
import unittest

from kilin.system_testing._comparison_diagnostics import (
    context_comparison, structured_prompt, tool_comparison,
)
from kilin.system_testing.timeline import TimelineOperation
from tests.kilin.system_testing.test__comparison_metrics import _event


def _op(conversation, *events):
    """Construct a recorded command with explicit conversation identity."""
    return TimelineOperation("ask", 1, len(events), "completed",
                             {"conversation_id": conversation}, events)


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        self.maps = ({"old": "new", "branch": "fork"}, {"a": "b"})
        self.catalogs = ({"old": ["a"]}, {"new": ["b"]})
        self.old = {"turn_uuid": "a", "role": "user", "label": "M1",
                    "origins": [{"conversation_id": "old", "turn_number": 1},
                                {"conversation_id": "branch",
                                 "turn_number": 1}],
                    "fingerprint": "one", "priority": 2}
        self.new = deepcopy(self.old)
        self.new.update(turn_uuid="b", fingerprint="two")
        self.new["origins"] = [
            {"conversation_id": "fork", "turn_number": 1},
            {"conversation_id": "new", "turn_number": 1}]

    def test_inventory_maps_origins_without_erasing_content_changes(self):
        a = _op("old", _event(1, "memory.sync.result",
                              {"records": [self.old]}))
        b = _op("new", _event(1, "memory.sync.result",
                              {"records": [self.new]}))
        row = context_comparison(a, b, self.catalogs, self.maps)[0]
        self.assertTrue(row["identity_equal"])
        self.assertFalse(row["fingerprints_equal"])
        self.assertTrue(row["priorities_equal"])
        self.new["role"] = "assistant"
        self.assertFalse(context_comparison(
            a, b, self.catalogs, self.maps)[0]["identity_equal"])
        self.new["turn_uuid"] = "unknown"
        row = context_comparison(a, b, self.catalogs, self.maps)[0]
        self.assertIsNone(row["identity_equal"])
        self.assertTrue(row["issues"])
        row = context_comparison(a, _op("new"), self.catalogs, self.maps)[0]
        self.assertIsNone(row["replayed"])
        self.assertIsNone(row["identity_equal"])

    def test_history_final_selection_is_separate(self):
        history = dict(recent_turn_ids=[1], related_turn_ids=[],
                       final_turn_ids=[1])
        saved = _event(2, "gui.question.saved", {"turn_count": 2})
        a = _op("old", _event(1, "history.selection.result", history), saved)
        b = _op("new", _event(1, "history.selection.result", {
            **history, "related_turn_ids": [1]}), saved)
        row = context_comparison(a, b, self.catalogs, self.maps)[0]
        self.assertFalse(row["identity_equal"])
        self.assertTrue(row["selection_equal"]["final_turn_ids"])
        history["final_turn_ids"] = [2]
        row = context_comparison(a, b, self.catalogs, self.maps)[0]
        self.assertIsNone(row["original"])
        self.assertTrue(row["issues"])

    def _prompt(self, record, *, text="    Saved text", stamp=None):
        """Create the producer's memory header and selection record."""
        origins = ", ".join(f"{o['conversation_id']} turn {o['turn_number']}"
                            for o in record["origins"])
        timestamp = stamp or "2026-11-01T09:00:00+00:00"
        content = ("Question: What?\n\nCONVERSATION MEMORY (previous "
                   "discussion, NOT source evidence):\n\n"
                   f"Memory M1 | user | UUID {record['turn_uuid']} | "
                   f"{timestamp} | {origins}\n{text}")
        event = _event(2, "llm.call.input", {}, exact={
            "01_structured_messages": {"value": [
                {"role": "user", "content": content}]}})
        op = _op("old", _event(1, "conversation_memory.retrieved", {
            "selected_memories": [record]}), event)
        return event, op

    def test_prompt_only_normalizes_corroborated_metadata(self):
        a, before = self._prompt(self.old)
        b, after = self._prompt(self.new, stamp="2026-11-02T09:00:00+00:00")
        left = structured_prompt(a, before, {"a": "a"},
                                 {"old": "old", "branch": "branch"})
        right = structured_prompt(b, after, {"b": "a"},
                                  {"new": "old", "fork": "branch"})
        self.assertNotEqual(left["raw"], right["raw"])
        self.assertEqual(left["normalized"], right["normalized"])
        b, after = self._prompt(self.new, text="    Changed content")
        right = structured_prompt(b, after, {"b": "a"},
                                  {"new": "old", "fork": "branch"})
        self.assertNotEqual(left["normalized"], right["normalized"])
        b, after = self._prompt(self.new, stamp="invalid")
        right = structured_prompt(b, after, {"b": "a"},
                                  {"new": "old", "fork": "branch"})
        self.assertIsNone(right["normalized"])
        self.assertTrue(right["issue"])
        self.assertIsNone(structured_prompt(
            a, _op("old", a), {"a": "a"}, {"old": "old"})["normalized"])

    def test_timing_only_and_equal_failure(self):
        a = {"name": "summary", "status": "incomplete", "valid": False,
             "seconds": 20., "plan": {"estimated_seconds": 2., "items": 1}}
        b = deepcopy(a)
        b["seconds"] = 3.
        b["plan"]["estimated_seconds"] = 1.
        completion = dict(original=None, replayed=None)
        row = tool_comparison(dict(original=a, replayed=b), completion)
        self.assertTrue(row["functional_equal"])
        self.assertNotEqual(row["timing"]["original"],
                            row["timing"]["replayed"])
        self.assertEqual(row["outcomes"]["original"]["status"], "incomplete")
        self.assertIn("seconds", a)
        b["plan"]["items"] = 2
        self.assertFalse(tool_comparison(
            dict(original=a, replayed=b), completion)["functional_equal"])

    def test_tool_outcome_transitions_and_unknown_execution(self):
        completion = dict(original=None, replayed=None)
        cases = (
            (None, None, 0, 0, "not_applicable"),
            (None, None, 1, 1, "unknown"),
            ({"name": "summary", "saved": "outputs/file.md"},
             {"name": "summary"}, 1, 1, "unknown"),
            ({"valid": True}, {"valid": False}, 1, 1,
             "potential_regression"),
            ({"valid": False}, {"valid": True}, 1, 1,
             "possible_improvement"),
            ({"valid": True}, {"status": "completed", "valid": True}, 1, 1,
             "complete_reproduced"),
        )
        for a, b, first, second, expected in cases:
            with self.subTest(expected=expected):
                row = tool_comparison(
                    dict(original=a, replayed=b), completion,
                    executions=dict(original=first, replayed=second))
                self.assertEqual(row["transition"], expected)
        row = tool_comparison(dict(original=None, replayed=None),
                              dict(original={"status": "complete"},
                                   replayed={"status": "partial"}))
        self.assertEqual(row["transition"], "unknown")
        row = tool_comparison(dict(original={"valid": True},
                                   replayed={"valid": True}),
                              dict(original={"status": "partial"},
                                   replayed=None))
        self.assertEqual(row["transition"], "possible_improvement")
        row = tool_comparison(dict(original={"valid": True},
                                   replayed={"valid": True}), completion,
                              finish_reasons=dict(original="length",
                                                  replayed="stop"))
        self.assertEqual(row["transition"], "possible_improvement")


if __name__ == "__main__":
    unittest.main()
