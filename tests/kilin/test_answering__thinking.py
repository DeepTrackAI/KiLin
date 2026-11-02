"""Verify request-local thinking overrides and restoration on every exit.

Tool requests retain the main model's sampling settings and token ceiling.
Router calls keep their separate nonthinking contract.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.answering import side_reply
from kilin._model_completion import IncompleteModelResponse
from tests.kilin.test_answering import _FakeChat


class TestSideReplyThinking(unittest.TestCase):
    """Apply explicit tool policies without leaking them into later calls."""

    def test_override_preserves_other_settings_and_history(self) -> None:
        for original in (True, False):
            for override in (None, True, False):
                with self.subTest(original=original, override=override):
                    chat = _FakeChat(response="Final text")
                    chat.thinking = original
                    chat.max_new_tokens = 100
                    history = [dict(item) for item in chat.messages]
                    reply = side_reply(
                        chat, "s", "p", max_new_tokens=120,
                        inherit_settings=True, thinking=override,
                        require_complete=True,
                    )
                    observed = chat.observations[0]
                    expected = original if override is None else override
                    self.assertIs(observed["thinking"], expected)
                    self.assertEqual(observed["temperature"], 0.6)
                    self.assertEqual(observed["max_new_tokens"], 100)
                    self.assertEqual(chat.max_new_tokens, 100)
                    self.assertEqual(chat.messages, history)
                    self.assertIs(chat.thinking, original)
                    self.assertEqual(chat.temperature, 0.6)
                    self.assertEqual(reply, "Final text")

    def test_invalid_override_is_rejected_before_mutation(self) -> None:
        for override, inherit in [("off", True), (0, True), (1, True),
                                  (True, False), (False, False)]:
            with self.subTest(override=override, inherit=inherit):
                chat = _FakeChat()
                history = list(chat.messages)
                with self.assertRaises(ValueError):
                    side_reply(
                        chat, "s", "p", max_new_tokens=120,
                        inherit_settings=inherit, thinking=override,
                    )
                self.assertEqual(chat.messages, history)
                self.assertEqual(chat.observations, [])

    def test_completion_failure_restores_override(self) -> None:
        chat = _FakeChat(response="<think>unfinished")
        chat.last_finish_reason = "length"
        history = list(chat.messages)
        with self.assertRaises(IncompleteModelResponse):
            side_reply(
                chat, "s", "p", max_new_tokens=120,
                inherit_settings=True, thinking=False, require_complete=True,
            )
        self.assertFalse(chat.observations[0]["thinking"])
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)
        self.assertEqual(chat.messages, history)
        self.assertEqual(len(chat.observations), 1)

    def test_backend_failure_also_restores_override(self) -> None:
        chat = _FakeChat(reply_error=RuntimeError("backend failed"))
        chat.thinking = False
        history = list(chat.messages)
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            side_reply(
                chat, "s", "p", max_new_tokens=120,
                inherit_settings=True, thinking=True,
            )
        self.assertTrue(chat.observations[0]["thinking"])
        self.assertFalse(chat.thinking)
        self.assertEqual(chat.messages, history)

    def test_router_and_unbounded_tool_contracts_are_unchanged(self) -> None:
        chat = _FakeChat(response="Result")
        side_reply(chat, "s", "route", max_new_tokens=64)
        self.assertFalse(chat.observations[0]["thinking"])
        self.assertEqual(chat.observations[0]["temperature"], 0.0)
        self.assertEqual(chat.observations[0]["max_new_tokens"], 64)
        side_reply(chat, "s", "judge", max_new_tokens=None,
                   inherit_settings=True)
        self.assertTrue(chat.observations[1]["thinking"])
        self.assertEqual(chat.observations[1]["temperature"], 0.6)
        self.assertIsNone(chat.observations[1]["max_new_tokens"])


if __name__ == "__main__":
    unittest.main()
