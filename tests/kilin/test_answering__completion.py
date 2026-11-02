"""Check side-request completion and safe restoration of failed turns.

Use the same deterministic chat fixture as the answering tests. Preserve
router behavior while enforcing the stricter tool completion contract.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin import answering
from kilin._model_completion import IncompleteModelResponse
from tests.kilin.test_answering import _FakeChat, _turn


class TestSideReplyCompletion(unittest.TestCase):
    """Keep failed calls isolated from the ongoing conversation."""

    def test_failed_completion_restores_history_and_settings(self) -> None:
        """Capture metadata before restoring temporary session state."""

        for inherit in (False, True):
            with self.subTest(inherit=inherit):
                chat = _FakeChat(response="<think>unfinished")
                chat.last_finish_reason = "length"
                history = [dict(item) for item in chat.messages]
                streamed = []
                with self.assertRaises(IncompleteModelResponse) as failure:
                    answering.side_reply(
                        chat, "system", "prompt", max_new_tokens=120,
                        inherit_settings=inherit, preserve_thinking=True,
                        require_complete=True, token_callback=streamed.append,
                    )
                self.assertEqual(failure.exception.finish_reason, "length")
                self.assertEqual(chat.messages, history)
                self.assertTrue(chat.thinking)
                self.assertEqual(chat.temperature, 0.6)
                self.assertEqual(len(chat.observations), 1)
                self.assertEqual(streamed, ["<think>unfinished"])

    def test_complete_reply_keeps_thinking_preservation(self) -> None:
        """A successful user-facing tool reply remains unmodified."""

        chat = _FakeChat()
        chat.last_finish_reason = "stop"
        self.assertEqual(
            answering.side_reply(
                chat, "system", "prompt", max_new_tokens=500,
                inherit_settings=True, preserve_thinking=True,
                require_complete=True,
            ),
            chat.response,
        )

    def test_router_default_does_not_acquire_tool_failure_policy(self) -> None:
        """Retain existing router fallback behavior without stricter flags."""

        chat = _FakeChat(response="<think>unfinished")
        chat.last_finish_reason = "length"
        self.assertEqual(
            answering.side_reply(chat, "s", "q", max_new_tokens=64), ""
        )

    def test_history_omits_legacy_and_marked_incomplete_pairs(self) -> None:
        """Keep stored numbering but never restore an empty assistant."""

        turns = [
            _turn("Q1", "<think>unfinished"),
            _turn("Q2", "<think>complete</think>"),
            _turn("Q3", "Incomplete operation"),
            _turn("Q4", "Usable answer"),
        ]
        turns[2].parameters["completion"] = {"status": "incomplete"}
        history = answering.history_messages(turns, [1, 2, 3, 4])
        self.assertEqual(history, [
            {"role": "user", "content": "Q4"},
            {"role": "assistant", "content": "Usable answer"},
        ])
        self.assertEqual(len(turns), 4)
        self.assertEqual(turns[0].answer, "<think>unfinished")


if __name__ == "__main__":
    unittest.main()
