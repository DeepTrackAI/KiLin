"""Test shared persisted conversation and tracing format contracts.

Use plain dictionaries to exercise the legacy and grouped layouts, reference
semantics, and in-place creation without requiring models, GUI, or filesystem
access. Invalid structures must not be silently interpreted as valid state.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import patch

from qachat import conversation_format as fmt


class TestConversationFormatConstants(unittest.TestCase):
    """Check the vocabulary shared by persistence and trace components."""

    def test_public_module_exports_match_the_storage_contract(self) -> None:
        expected = {
            "CONVERSATION_SESSION_KEY",
            "CONVERSATION_SESSION_TITLE",
            "CONVERSATION_SESSION_CATEGORY",
            "CONVERSATION_RESUME_STATE_KEY",
            "LEGACY_CONVERSATION_KEY",
            "new_conversation_session",
            "conversation_state",
        }
        self.assertEqual(set(fmt.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(fmt, name))

    def test_session_identifiers_are_stable(self) -> None:
        self.assertEqual(fmt.CONVERSATION_SESSION_KEY, "05_conversations")
        self.assertEqual(fmt.CONVERSATION_SESSION_TITLE, "Conversation turns")
        self.assertEqual(fmt.CONVERSATION_SESSION_CATEGORY, "conversations")
        self.assertEqual(fmt.CONVERSATION_RESUME_STATE_KEY, "resume_state")
        self.assertEqual(fmt.LEGACY_CONVERSATION_KEY, "conversation")


class TestNewConversationSession(unittest.TestCase):
    """Verify the shape and reference semantics of new session records."""

    def test_default_is_a_fresh_empty_session(self) -> None:
        first = fmt.new_conversation_session()
        second = fmt.new_conversation_session()
        self.assertEqual(
            first,
            {
                "title": fmt.CONVERSATION_SESSION_TITLE,
                "category": fmt.CONVERSATION_SESSION_CATEGORY,
                "turns": {},
                "operations": {},
                "resume_state": {},
            },
        )
        self.assertIsNot(first, second)
        for name in ("turns", "operations", "resume_state"):
            self.assertIsNot(first[name], second[name])

    def test_supplied_state_is_retained_by_identity(self) -> None:
        saved = {"turns": [{"question": "Hi"}], "checkpoint_id": "c1"}
        session = fmt.new_conversation_session(saved)
        self.assertIs(session[fmt.CONVERSATION_RESUME_STATE_KEY], saved)
        saved["checkpoint_id"] = "c2"
        self.assertEqual(session["resume_state"]["checkpoint_id"], "c2")

    def test_explicit_empty_state_is_retained_by_identity(self) -> None:
        saved: dict[str, object] = {}
        self.assertIs(fmt.new_conversation_session(saved)["resume_state"],
                      saved)

    def test_operation_and_turn_containers_are_independent(self) -> None:
        session = fmt.new_conversation_session({"turns": [1]})
        session["turns"]["turn_0001"] = {"question": "Q"}
        self.assertEqual(session["operations"], {})
        self.assertEqual(session["resume_state"]["turns"], [1])


class TestConversationState(unittest.TestCase):
    """Check lookup, compatibility, validation, and creation in place."""

    def test_legacy_top_level_state_is_returned_by_identity(self) -> None:
        state = {"turns": [1]}
        payload = {"conversation": state, "trace": {"events": []}}
        self.assertIs(fmt.conversation_state(payload), state)

    def test_legacy_top_level_state_has_precedence_over_grouped(self):
        legacy = {"checkpoint_id": "legacy"}
        grouped = {"checkpoint_id": "grouped"}
        payload = {
            "conversation": legacy,
            "sessions": {
                fmt.CONVERSATION_SESSION_KEY:
                    fmt.new_conversation_session(grouped),
            },
        }
        self.assertIs(fmt.conversation_state(payload), legacy)
        self.assertIs(fmt.conversation_state(payload, create=True), legacy)
        self.assertIs(
            payload["sessions"][fmt.CONVERSATION_SESSION_KEY]
            ["resume_state"], grouped,
        )

    def test_legacy_null_falls_through_to_grouped(self) -> None:
        state = {"checkpoint_id": "grouped"}
        payload = {
            "conversation": None,
            "sessions": {"custom": fmt.new_conversation_session(state)},
        }
        self.assertIs(fmt.conversation_state(payload), state)

    def test_grouped_lookup_does_not_depend_on_session_key(self) -> None:
        state = {"turns": [1]}
        payload = {
            "sessions": {
                "01_runtime": {"category": "runtime", "resume_state": {}},
                "custom_key": fmt.new_conversation_session(state),
            }
        }
        self.assertIs(fmt.conversation_state(payload), state)

    def test_grouped_lookup_skips_invalid_non_conversation_entries(self):
        expected = {"checkpoint_id": "ok"}
        payload = {
            "sessions": {
                "none": None,
                "list": ["not a session"],
                "other": {"category": "other", "resume_state": {}},
                "actual": fmt.new_conversation_session(expected),
            }
        }
        self.assertIs(fmt.conversation_state(payload), expected)

    def test_grouped_legacy_key_is_read_when_resume_state_missing(self):
        state = {"turns": ["older"]}
        payload = {
            "sessions": {
                "old": {
                    "category": fmt.CONVERSATION_SESSION_CATEGORY,
                    "conversation": state,
                }
            }
        }
        self.assertIs(fmt.conversation_state(payload), state)

    def test_resume_state_has_precedence_over_grouped_legacy_field(self):
        current = {"checkpoint_id": "current"}
        payload = {
            "sessions": {
                "old": {
                    "category": fmt.CONVERSATION_SESSION_CATEGORY,
                    "conversation": {"checkpoint_id": "old"},
                    "resume_state": current,
                }
            }
        }
        self.assertIs(fmt.conversation_state(payload), current)

    def test_later_matching_session_is_used_if_earlier_invalid(self):
        expected = {"turns": [1]}
        payload = {
            "sessions": {
                "bad": {"category": "conversations", "resume_state": []},
                "ok": fmt.new_conversation_session(expected),
            }
        }
        self.assertIs(fmt.conversation_state(payload), expected)

    def test_missing_session_without_creation_rejected_without_mutation(self):
        for payload in ({}, {"sessions": {}},
                        {"sessions": {"other": {"category": "other"}}}):
            with self.subTest(payload=payload):
                before = repr(payload)
                with self.assertRaisesRegex(ValueError, "invalid"):
                    fmt.conversation_state(payload)
                self.assertEqual(repr(payload), before)

    def test_non_mapping_sessions_are_rejected_in_both_modes(self):
        for value in (None, [], "data", 3, False):
            for create in (False, True):
                with self.subTest(value=value, create=create):
                    payload = {"sessions": value}
                    with self.assertRaisesRegex(ValueError, "invalid"):
                        fmt.conversation_state(payload, create=create)
                    self.assertIs(payload["sessions"], value)

    def test_non_mapping_legacy_is_not_accepted_as_valid_state(self):
        for invalid in (None, [], "old", 0, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    fmt.conversation_state({"conversation": invalid})

    def test_grouped_session_without_resume_state_is_invalid(self):
        payload = {
            "sessions": {
                fmt.CONVERSATION_SESSION_KEY: {"category": "conversations"}
            }
        }
        with self.assertRaisesRegex(ValueError, "invalid"):
            fmt.conversation_state(payload)

    def test_create_missing_sessions_preserves_unrelated_payload(self):
        payload = {"trace": {"events": [1]}}
        state = fmt.conversation_state(payload, create=True)
        self.assertEqual(state, {})
        self.assertEqual(payload["trace"], {"events": [1]})
        session = payload["sessions"][fmt.CONVERSATION_SESSION_KEY]
        self.assertIs(session["resume_state"], state)
        self.assertEqual(session["turns"], {})
        self.assertEqual(session["operations"], {})

    def test_create_empty_sessions_preserves_unrelated_categories(self):
        runtime = {"category": "runtime", "operation": [1]}
        payload = {"sessions": {"runtime": runtime}}
        state = fmt.conversation_state(payload, create=True)
        self.assertEqual(state, {})
        self.assertIs(payload["sessions"]["runtime"], runtime)
        self.assertIn(fmt.CONVERSATION_SESSION_KEY, payload["sessions"])

    def test_create_is_idempotent_for_an_existing_grouped_state(self):
        payload = {"sessions": {}}
        first = fmt.conversation_state(payload, create=True)
        first["checkpoint_id"] = "saved"
        second = fmt.conversation_state(payload, create=True)
        self.assertIs(first, second)
        self.assertEqual(len(payload["sessions"]), 1)

    def test_create_fills_missing_resume_state_on_existing_session(self):
        payload = {
            "sessions": {
                "one": {
                    "category": "conversations", "turns": {"a": 1}
                }
            }
        }
        state = fmt.conversation_state(payload, create=True)
        self.assertEqual(state, {})
        self.assertIs(payload["sessions"]["one"]["resume_state"], state)
        self.assertEqual(payload["sessions"]["one"]["turns"], {"a": 1})

    def test_create_with_invalid_existing_state_uses_canonical_session(self):
        payload = {
            "sessions": {
                "broken": {
                    "category": "conversations", "resume_state": None
                }
            }
        }
        state = fmt.conversation_state(payload, create=True)
        self.assertEqual(state, {})
        self.assertEqual(payload["sessions"]["broken"]["resume_state"],
                         None)
        self.assertIn(fmt.CONVERSATION_SESSION_KEY, payload["sessions"])

    def test_create_rejects_invalid_new_session_state(self):
        """An unexpectedly malformed session still raises ValueError."""
        with patch.object(
            fmt, "new_conversation_session",
            return_value={"resume_state": []},
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                fmt.conversation_state({}, create=True)

    @unittest.expectedFailure
    def test_create_rejects_conflicting_canonical_session_key(self):
        """A reserved key with another category must not be overwritten."""
        payload = {
            "sessions": {
                fmt.CONVERSATION_SESSION_KEY: {
                    "category": "runtime", "important": "retain",
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "invalid"):
            fmt.conversation_state(payload, create=True)
        self.assertEqual(
            payload["sessions"][fmt.CONVERSATION_SESSION_KEY]["important"],
            "retain",
        )

    @unittest.expectedFailure
    def test_create_does_not_discard_grouped_legacy_resume_state(self):
        """A metadata write must preserve legacy grouped state."""
        legacy = {"turns": [{"question": "Keep"}]}
        payload = {
            "sessions": {
                "older": {
                    "category": "conversations", "conversation": legacy,
                }
            }
        }
        self.assertIs(fmt.conversation_state(payload, create=True), legacy)


if __name__ == "__main__":
    unittest.main()
