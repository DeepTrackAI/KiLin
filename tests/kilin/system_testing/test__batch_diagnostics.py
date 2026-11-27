"""Check that the batch views preserve independent diagnostic dimensions."""

# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
import unittest

from kilin.system_testing.analysis import (
    batch_overview, event_inventory, question_explorer,
    _release_agreement_rows, _release_totals,
)
from kilin.system_testing.batch import (
    collect_issues, group_issues, _event_differences,
)


def report():
    """Minimal complete report with independent event-count coverage."""
    side = {"valid": True, "event_count": 2, "event_counts": {"ask": 2},
            "saved_questions": [], "validation_warnings": [],
            "validation_issues": []}
    return {"alignment": {"status": "complete"}, "operations": [],
            "original": deepcopy(side), "replayed": deepcopy(side),
            "report_issues": [], "final_state": {}}


class TestDiagnostics(unittest.TestCase):
    def test_incomplete_identical_tool_and_timing(self):
        data = report()
        data["operations"] = [{"original_sequence": 1, "question": {
            "tool": {"original": {"seconds": 3},
                     "replayed": {"seconds": 9}},
            "tool_comparison": {"functional_equal": True, "outcomes": {
                "original": {"status": "incomplete"},
                "replayed": {"status": "incomplete"}}}}}]
        issues = collect_issues(data, {})
        self.assertEqual([i["code"] for i in issues],
                         ["capability.incomplete"])
        self.assertEqual(issues[0]["severity"], "review")

    def test_missing_replay_preserves_independent_answers(self):
        data = report()
        data["alignment"].update(status="invalid_recording",
                                 reason="Both must validate.")
        data["original"].update(valid=False, saved_questions=[{
            "sequence": 2, "turn_uuid": "turn", "question": "My question",
            "answer": "<script>sample</script>", "validated": False,
            "tool_outcome": {"status": "incomplete"}}])
        data["replayed"].update(valid=False, event_count=0, event_counts={})
        self.assertTrue(event_inventory(data)["complete"])
        html = question_explorer(data)
        self.assertIn("My question", html)
        self.assertIn("unverified recording", html)
        self.assertNotIn("<script>sample</script>", html)
        issues = collect_issues(data, {"replay_error": "Invalid original"})
        groups = group_issues(issues, data)
        blocking = [g for g in groups if g["severity"] == "critical"]
        self.assertEqual(len(blocking), 1)
        self.assertEqual(len(blocking[0]["issues"]), 4)
        self.assertTrue(any(i["code"] == "capability.incomplete"
                            for i in issues))

    def test_expected_events_and_generated_names(self):
        data = report()
        data["original"].update(event_count=3, event_counts={
            "ask": 2, "gui.project.rename.completed": 1})
        data["expected_event_differences"] = ["gui.project.rename.completed"]
        data["final_state"]["conversations"] = [dict(name_equal=False,
            name_origin={"original": "generated", "replayed": "generated"})]
        issues = collect_issues(data, {})
        self.assertEqual(len(issues), 2)
        self.assertTrue(all(i["severity"] == "info" for i in issues))
        data["expected_event_differences"] = []
        data["final_state"]["conversations"][0]["name_origin"][
            "original"] = "explicit"
        issues = collect_issues(data, {})
        self.assertTrue(all(i["severity"] == "review" for i in issues))


    def test_outcome_transition_not_obscured_by_publication_hint(self):
        data = report()
        data["reference_publication"] = {"coverage": ["REVIEW: partial"]}
        data["operations"] = [{"original_sequence": 7, "question": {
            "tool_comparison": {"functional_equal": False, "outcomes": {
                "original": {"status": "incomplete"},
                "replayed": {"status": "complete"}},
                "transition": "possible_improvement"}}}]
        issues = collect_issues(data, {})
        by_code = {i["code"]: i for i in issues}
        self.assertEqual(by_code["generator.coverage"]["severity"], "info")
        self.assertIn("capability.changed", by_code)
        self.assertNotIn("capability.incomplete", by_code)
        groups = group_issues(issues, data)
        group = next(g for g in groups if g["code"] == "capability.changed")
        self.assertEqual(group["question_number"], 1)

    def test_event_attribution_with_cancelling_counts(self):
        data = report()
        data["operations"] = [{"original_sequence": seq, "stages": [{
            "event_type": "ask", "original": {"count": a},
            "replayed": {"count": b}}]} for seq, a, b in (
                (1, 0, 1), (2, 1, 0))]
        inventory = event_inventory(data)
        self.assertEqual(inventory["differences"], [])
        rows = _event_differences(data, inventory)
        self.assertEqual([(r["sequence"], r["original"], r["replayed"])
                          for r in rows], [(1, 0, 1), (2, 1, 0)])
        issues = collect_issues(data, {})
        self.assertEqual(len(group_issues(issues, data)), 2)
        data["replayed"].update(event_count=3, event_counts={"ask": 3})
        inventory = event_inventory(data)
        rows = _event_differences(data, inventory)
        self.assertEqual(rows[-1], dict(event_type="ask", original=1,
                                       replayed=2, sequence=None))
        data["operations"][0]["stages"][0]["original"]["count"] = 9
        rows = _event_differences(data, inventory)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attribution"], "unavailable")
        self.assertIn("attribution unavailable", collect_issues(data, {})[
            -1]["message"])

    def test_paired_citations_and_independent_sequence_identity(self):
        data = report()
        data["operations"] = [{"original_sequence": 7, "question": {
            "original_turn_uuid": "a", "replayed_turn_uuid": "b",
            "evidence": {"citations": {
                "original_unresolved": [1], "replayed_unresolved": [2]}}}}]
        data["replayed"]["saved_questions"] = [{
            "sequence": 7, "turn_uuid": "c", "completion": {
                "status": "partial"}, "tool_outcome": {"status": "unknown"}}]
        groups = group_issues(collect_issues(data, {}), data)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["code"], "question.citations")
        self.assertEqual(groups[0]["question_number"], 1)
        self.assertEqual(len(groups[0]["issues"]), 2)
        self.assertNotIn("question_number", groups[1])
        self.assertEqual(groups[1]["issues"][0]["side"], "replayed")

    def test_conditional_reproducibility_classification(self):
        data = report()
        call = {"original": {}, "replayed": {}, "stage": "answer",
                "inference": {"output_equal": False,
                              "conditional_category": "zero_temperature"}}
        data["operations"] = [{"original_sequence": 1,
                               "model_calls": [call]}]
        issues = collect_issues(data, {})
        self.assertEqual(issues[0]["severity"], "review")
        call["inference"]["conditional_category"] = "nonzero_temperature"
        issues = collect_issues(data, {})
        self.assertEqual(issues[0]["severity"], "info")
        call["inference"]["conditional_category"] = None
        self.assertEqual(collect_issues(data, {}), [])

    def test_current_aggregate_uses_counts_not_saved_snapshots(self):
        rows = [{"snapshot": status, "execution_status": execution,
                 "release_measurements": {"agreement": {
                     "metric": dict(equal=a, available=b, total=b)}}}
                for status, execution, a, b in (
                    ("current", "complete", 1, 2),
                    ("current", "complete", 8, 8),
                    ("saved", "saved snapshot", 40, 40))]
        rows.append({"snapshot": "current", "execution_status": "blocked",
                     "release_measurements": {}})
        self.assertEqual(_release_agreement_rows(rows), [
            ("metric", 9, 10, 10, "2/3")])
        totals = _release_totals(rows)
        self.assertEqual(totals["complete"], 2)
        self.assertEqual(totals["saved"], 1)
        self.assertEqual(totals["blocked"], 1)
        self.assertEqual(totals["fresh"], 0)
        html = batch_overview({}, runs=rows)
        self.assertIn("Current complete comparisons", html)
        self.assertIn("saved snapshots: 1", html)
        self.assertIn("2/3", html)

    def test_differences_include_incomplete_equal_answers(self):
        data = report()
        question = {"answer": {"exact_equal": True},
                    "original_turn_uuid": "a", "replayed_turn_uuid": "b"}
        op = {"original_sequence": 1, "question": question, "stages": [{
            "event_type": "tool", "original": {"count": 1},
            "replayed": {"count": 1}}]}
        data["operations"] = [op]
        self.assertIn("Showing 0 of 1", question_explorer(
            data, differences_only=True))
        question["tool_comparison"] = {"functional_equal": True,
                                      "outcomes": {"original": {
                                          "status": "incomplete"}}}
        self.assertIn("Showing 1 of 1", question_explorer(
            data, differences_only=True))
        question.pop("tool_comparison")
        question["evidence"] = {"citations": {"original_unresolved": [1]}}
        self.assertIn("Showing 1 of 1", question_explorer(
            data, differences_only=True))


if __name__ == "__main__":
    unittest.main()
