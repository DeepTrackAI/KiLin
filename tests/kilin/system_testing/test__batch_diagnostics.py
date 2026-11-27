"""Check that the batch views preserve independent diagnostic dimensions."""

# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
import unittest

from kilin.system_testing.analysis import event_inventory, question_explorer
from kilin.system_testing.batch import collect_issues, group_issues


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


if __name__ == "__main__":
    unittest.main()
