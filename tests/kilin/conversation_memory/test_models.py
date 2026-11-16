"""Verify stable turn UUIDs and strict per-message feedback values."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from uuid import UUID

from kilin.conversation_memory.models import (
    MessageFeedback,
    new_turn_uuid,
    require_feedback_role,
    require_saved_turn_ids,
    require_turn_uuid,
)


class TestConversationMemoryModels(unittest.TestCase):
    def test_new_turn_ids_are_distinct_canonical_uuid4(self):
        values = [new_turn_uuid() for _ in range(32)]
        self.assertEqual(len(values), len(set(values)))
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(len(value), 32)
                self.assertEqual(UUID(hex=value).version, 4)
                self.assertEqual(require_turn_uuid(value), value)

    def test_invalid_turn_ids_are_rejected_not_converted(self):
        for value in (None, 1, "", "some-uuid", "A" * 32,
                      "0" * 32, "00000000-0000-4000-8000-000000000000"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "UUID4"):
                    require_turn_uuid(value)

    def test_saved_turns_require_distinct_ids(self):
        first, second = new_turn_uuid(), new_turn_uuid()
        turns = [
            {"conversation_turn_uuid": first, "question": "Q"},
            {"conversation_turn_uuid": second, "question": "Q"},
        ]
        self.assertEqual(require_saved_turn_ids(turns), (first, second))
        for bad in ("not-a-list", [4], [{}], [turns[0], turns[0]]):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    require_saved_turn_ids(bad)

    def test_feedback_separates_save_like_and_dislike(self):
        self.assertFalse(MessageFeedback().selected)
        self.assertTrue(MessageFeedback(saved=True).selected)
        self.assertTrue(MessageFeedback(reaction="liked").selected)
        self.assertFalse(MessageFeedback(
            saved=True, reaction="disliked"
        ).selected)
        self.assertEqual(require_feedback_role("user"), "user")
        self.assertEqual(require_feedback_role("assistant"), "assistant")
        for invalid in ("system", "USER", None, 1):
            with self.subTest(role=invalid):
                with self.assertRaises(ValueError):
                    require_feedback_role(invalid)
        for values in ({"saved": 1}, {"saved": "false"},
                       {"reaction": "useful"}, {"reaction": None}):
            with self.subTest(value=values):
                with self.assertRaises(ValueError):
                    MessageFeedback(**values)


if __name__ == "__main__":
    unittest.main()
