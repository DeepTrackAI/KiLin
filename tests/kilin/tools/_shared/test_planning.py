"""Verify the recording boundary for tool plans and interactive choices.

Use deterministic callbacks and a mock diagnostic sink to test the exact
proposal/outcome contract independently of the GUI and inference models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.tools._shared.planning import (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_NEVER,
    ToolPlan,
    ask_choice,
    confirm_plan,
)


class TestInteractionRecording(unittest.TestCase):

    def setUp(self):
        self.context = SimpleNamespace(
            trace=Mock(), trace_turn_id="turn_1", progress=Mock(),
            choose=Mock(), confirm=Mock(),
        )
        self.plan = ToolPlan("tool", "Work", "All", 2, 2, 2, 2.0, 1, 1.0)

    def _records(self, proposal, terminal):
        """Require a single correlated proposal and terminal record."""
        proposed, decided = self.context.trace.emit.call_args_list
        self.assertEqual(proposed.args, (proposal,))
        self.assertEqual(decided.args, (terminal,))
        for call in (proposed, decided):
            self.assertEqual(call.kwargs["turn_id"], "turn_1")
        before, after = (call.kwargs["data"] for call in (proposed, decided))
        self.assertEqual(before["interaction_id"], after["interaction_id"])
        return before, after

    def test_labels_survive_reordered_choices(self):
        """Record the actual label and index, once per interaction."""
        identities = []
        for options, index in ((('a', 'b'), 1), (('b', 'a'), 0)):
            self.context.trace.reset_mock()
            self.context.choose.reset_mock()
            self.context.choose.return_value = index
            self.assertEqual(ask_choice("Choose", options, self.context),
                             index)
            self.context.choose.assert_called_once_with("Choose", options)
            proposed, decided = self._records(
                "tool.choice.proposed", "tool.choice.decided"
            )
            self.assertTrue(proposed["asked"])
            self.assertTrue(decided["asked"])
            self.assertEqual(proposed["options"], list(options))
            self.assertEqual(decided["selected_option"], "b")
            self.assertEqual(decided["response"], index)
            self.assertEqual(decided["index"], index)
            identities.append(decided["interaction_id"])
        self.assertNotEqual(*identities)

    def test_automatic_choices_and_invalid_responses(self):
        """Distinguish an unasked prompt from a rejected callback value."""
        for options, callback, asked in (
            ((), Mock(), False), (("a",), None, False),
            (("a",), Mock(return_value=None), True),
            (("a",), Mock(return_value=2), True),
            (("a",), Mock(return_value=True), True),
        ):
            with self.subTest(options=options, asked=asked):
                self.context.trace.reset_mock()
                self.context.progress.reset_mock()
                self.context.choose = callback
                self.assertIsNone(ask_choice("Choose", options, self.context))
                proposed, decided = self._records(
                    "tool.choice.proposed", "tool.choice.decided"
                )
                self.assertIs(proposed["asked"], asked)
                self.assertIs(decided["asked"], asked)
                self.assertIsNone(decided["index"])
                self.assertIsNone(decided["selected_option"])
                if asked:
                    callback.assert_called_once_with("Choose", options)
                    self.assertEqual(
                        decided["response"], callback.return_value
                    )
                else:
                    self.context.progress.assert_not_called()
                    self.assertIsNone(decided["response"])
                    if callback is not None:
                        callback.assert_not_called()

    def test_plan_default_is_not_user_confirmation(self):
        """An automatic plan uses continue without invoking the callback."""
        self.assertEqual(confirm_plan(
            self.plan, CONFIRMATION_NEVER, 1, self.context
        ), ("continue", False))
        proposed, decided = self._records(
            "tool.plan.proposed", "tool.plan.decided"
        )
        self.assertFalse(proposed["asked"])
        self.assertFalse(decided["asked"])
        self.assertEqual(decided["choice"], "continue")
        self.assertIsNone(decided["response"])
        self.context.confirm.assert_not_called()

    def test_failed_callbacks_leave_correlated_failure_records(self):
        """Failures propagate without manufacturing successful decisions."""
        for callback, invoke, prefix in (
            (self.context.choose,
             lambda: ask_choice("Choose", ("a",), self.context),
             "tool.choice"),
            (self.context.confirm,
             lambda: confirm_plan(self.plan, CONFIRMATION_ALWAYS, 1,
                                  self.context), "tool.plan"),
        ):
            self.context.trace.reset_mock()
            callback.side_effect = RuntimeError("callback unavailable")
            with self.assertRaisesRegex(RuntimeError, "callback unavailable"):
                invoke()
            proposed, failed = self._records(
                prefix + ".proposed", prefix + ".failed"
            )
            self.assertTrue(proposed["asked"])
            self.assertEqual(failed["error_type"], "RuntimeError")
            self.assertEqual(failed["message"], "callback unavailable")
            self.assertEqual(self.context.trace.emit.call_args.kwargs[
                "status"
            ], "failed")


if __name__ == "__main__":
    unittest.main()
