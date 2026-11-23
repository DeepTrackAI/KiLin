"""Verify one output-limit rule for isolated model calls.

Exercise user ceilings, nonthinking exceptions, explicit thinking overrides,
invalid inputs and restoration without loading a backend.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.answering import side_reply
from tests.kilin.test_answering import _FakeChat


class TestToolTokenBudget(unittest.TestCase):
    """Keep the selected maximum authoritative for each model call."""

    def test_resolution_matrix(self) -> None:
        """Apply caps only without thinking and never exceed the user limit.

        """

        for ceiling in (None, 1, 37, 128, 2048, 8192, 32768):
            for local in (None, 64, 128, 2048, 65536):
                for configured in (True, False):
                    for override in (None, True, False):
                        with self.subTest(
                            ceiling=ceiling, local=local,
                            configured=configured, override=override,
                        ):
                            chat = _FakeChat(response="Complete answer")
                            chat.max_new_tokens = ceiling
                            chat.thinking = configured
                            history = list(chat.messages)
                            active = (
                                configured if override is None else override
                            )
                            expected = ceiling
                            if not active and local is not None:
                                expected = (
                                    local if ceiling is None
                                    else min(local, ceiling)
                                )
                            side_reply(
                                chat, "System", "Question",
                                max_new_tokens=local, inherit_settings=True,
                                thinking=override, require_complete=True,
                            )
                            observed = chat.observations[0]
                            self.assertEqual(
                                observed["max_new_tokens"], expected
                            )
                            self.assertIs(observed["thinking"], active)
                            self.assertEqual(observed["temperature"], 0.6)
                            self.assertEqual(chat.max_new_tokens, ceiling)
                            self.assertIs(chat.thinking, configured)
                            self.assertEqual(chat.messages, history)
                            self.assertEqual(len(chat.observations), 1)

    def test_invalid_local_limits_fail_before_mutation(self) -> None:
        for limit in (True, False, 0, -1, 128.0, "128", []):
            for thinking in (True, False):
                with self.subTest(limit=limit, thinking=thinking):
                    chat = _FakeChat()
                    chat.thinking = thinking
                    history = list(chat.messages)
                    with self.assertRaisesRegex(ValueError, "positive"):
                        side_reply(
                            chat, "System", "Question",
                            max_new_tokens=limit, inherit_settings=True,
                        )
                    self.assertEqual(chat.observations, [])
                    self.assertEqual(chat.messages, history)

    def test_changes_take_effect_on_the_next_call(self) -> None:
        chat = _FakeChat(response="Complete answer")
        for ceiling in (2048, 16384, 256):
            chat.max_new_tokens = ceiling
            side_reply(chat, "System", "Question", max_new_tokens=128,
                       inherit_settings=True)
        self.assertEqual(
            [call["max_new_tokens"] for call in chat.observations],
            [2048, 16384, 256],
        )

    def test_router_keeps_its_separate_bounded_contract(self) -> None:
        chat = _FakeChat(response="Decision")
        chat.max_new_tokens = 16
        side_reply(chat, "Router", "Route", max_new_tokens=64)
        observed = chat.observations[0]
        self.assertEqual(observed["max_new_tokens"], 64)
        self.assertFalse(observed["thinking"])
        self.assertEqual(observed["temperature"], 0.0)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.max_new_tokens, 16)


if __name__ == "__main__":
    unittest.main()
