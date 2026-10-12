"""Test the single canonical persisted conversation format.

Use plain mappings to validate reference semantics, strict read-only lookup,
non-mutating rejection, and creation of the reserved grouped session. The
retired top-level, grouped, and alternative-key formats must not be accepted.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from qachat import conversation_format as fmt


class TestConversationFormatConstants(unittest.TestCase):
    """Check the public vocabulary shared by the persistence components."""

    def test_public_exports_are_canonical(self) -> None:
        expected = {
            "CONVERSATION_SESSION_KEY",
            "CONVERSATION_SESSION_TITLE",
            "CONVERSATION_SESSION_CATEGORY",
            "CONVERSATION_RESUME_STATE_KEY",
            "new_conversation_session",
            "conversation_state",
        }
        self.assertEqual(set(fmt.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(fmt, name))
        self.assertFalse(hasattr(fmt, "LEGACY_CONVERSATION_KEY"))

    def test_session_identifiers_are_stable(self) -> None:
        self.assertEqual(fmt.CONVERSATION_SESSION_KEY, "05_conversations")
        self.assertEqual(fmt.CONVERSATION_SESSION_TITLE, "Conversation turns")
        self.assertEqual(fmt.CONVERSATION_SESSION_CATEGORY, "conversations")
        self.assertEqual(fmt.CONVERSATION_RESUME_STATE_KEY, "resume_state")


class TestNewConversationSession(unittest.TestCase):
    """Verify the structure and reference semantics of new sessions."""

    def test_default_is_an_independent_complete_session(self) -> None:
        first = fmt.new_conversation_session()
        second = fmt.new_conversation_session()
        self.assertEqual(first, {
            "title": fmt.CONVERSATION_SESSION_TITLE,
            "category": fmt.CONVERSATION_SESSION_CATEGORY,
            "turns": {},
            "operations": {},
            "resume_state": {},
        })
        self.assertIsNot(first, second)
        for name in ("turns", "operations", "resume_state"):
            self.assertIsNot(first[name], second[name])

    def test_supplied_state_is_retained_by_identity(self) -> None:
        saved = {"turns": [{"question": "Hi"}], "checkpoint_id": "c1"}
        session = fmt.new_conversation_session(saved)
        self.assertIs(session["resume_state"], saved)
        saved["checkpoint_id"] = "c2"
        self.assertEqual(session["resume_state"]["checkpoint_id"], "c2")

    def test_explicit_empty_state_is_retained_by_identity(self) -> None:
        saved: dict[str, object] = {}
        self.assertIs(fmt.new_conversation_session(saved)["resume_state"],
                      saved)

    def test_readable_groups_and_resume_turns_are_independent(self) -> None:
        session = fmt.new_conversation_session({"turns": [1]})
        session["turns"]["turn_0001"] = {"question": "Q"}
        self.assertEqual(session["operations"], {})
        self.assertEqual(session["resume_state"]["turns"], [1])


class TestConversationState(unittest.TestCase):
    """Check canonical lookup, creation, and rejection without data loss."""

    def _assert_invalid_unchanged(
        self, payload: object, *, create: bool
    ) -> None:
        before = deepcopy(payload)
        with self.assertRaisesRegex(ValueError, "invalid"):
            fmt.conversation_state(payload, create=create)
        self.assertEqual(payload, before)

    def test_valid_state_is_read_by_reference_without_mutation(self) -> None:
        saved = {"checkpoint_id": "retained"}
        payload = {"sessions": {
            "01_runtime": {"category": "runtime"},
            fmt.CONVERSATION_SESSION_KEY:
                fmt.new_conversation_session(saved),
        }, "trace_id": "t1"}
        before = deepcopy(payload)
        self.assertIs(fmt.conversation_state(payload), saved)
        self.assertIs(fmt.conversation_state(payload, create=True), saved)
        self.assertEqual(payload, before)

    def test_missing_session_rejected_when_creation_disabled(self) -> None:
        for payload in ({}, {"sessions": {}},
                        {"sessions": {"01_runtime": {"category": "runtime"}}}):
            with self.subTest(payload=payload):
                self._assert_invalid_unchanged(payload, create=False)

    def test_create_adds_complete_session_without_changing_other_data(
        self,
    ) -> None:
        for payload in ({"trace": {"events": [1]}},
                        {"sessions": {"runtime": {"category": "runtime"}}}):
            with self.subTest(payload=payload):
                old = deepcopy(payload)
                state = fmt.conversation_state(payload, create=True)
                self.assertEqual(state, {})
                session = payload["sessions"][fmt.CONVERSATION_SESSION_KEY]
                self.assertIs(session["resume_state"], state)
                self.assertEqual(session["turns"], {})
                self.assertEqual(session["operations"], {})
                self.assertEqual(session["title"],
                                 fmt.CONVERSATION_SESSION_TITLE)
                for key, value in old.items():
                    if key == "sessions":
                        for old_key, old_value in value.items():
                            self.assertEqual(payload["sessions"][old_key],
                                             old_value)
                    else:
                        self.assertEqual(payload[key], value)

    def test_create_is_idempotent(self) -> None:
        payload = {"sessions": {}}
        first = fmt.conversation_state(payload, create=True)
        first["checkpoint_id"] = "saved"
        second = fmt.conversation_state(payload, create=True)
        self.assertIs(first, second)
        self.assertEqual(len(payload["sessions"]), 1)

    def test_reject_non_mapping_documents(self) -> None:
        for invalid in (None, [], "text", 7, False):
            for create in (False, True):
                with self.subTest(invalid=invalid, create=create):
                    self._assert_invalid_unchanged(invalid, create=create)

    def test_reject_invalid_sessions_containers(self) -> None:
        for invalid in (None, [], "text", 7, False):
            for create in (False, True):
                with self.subTest(invalid=invalid, create=create):
                    self._assert_invalid_unchanged(
                        {"sessions": invalid}, create=create
                    )

    def test_reject_top_level_legacy_field_even_if_current_exists(
        self,
    ) -> None:
        for old_state in ({"checkpoint_id": "old"}, None, [], "text"):
            for create in (False, True):
                with self.subTest(old_state=old_state, create=create):
                    payload = {"conversation": old_state, "sessions": {
                        fmt.CONVERSATION_SESSION_KEY:
                            fmt.new_conversation_session({"turns": []}),
                    }}
                    self._assert_invalid_unchanged(payload, create=create)

    def test_reject_nested_legacy_trace_sessions(self) -> None:
        for create in (False, True):
            payload = {"sessions": {
                fmt.CONVERSATION_SESSION_KEY: fmt.new_conversation_session(),
            }, "trace": {"sessions": {"old": {"title": "Old"}}}}
            self._assert_invalid_unchanged(payload, create=create)

    def test_reject_missing_or_malformed_canonical_session(self) -> None:
        invalid = (
            None,
            [],
            "invalid",
            {"category": "runtime", "resume_state": {}},
            {"category": "conversations", "resume_state": {}},
            {"title": "Missing category", "resume_state": {}},
            {"title": 12, "category": "conversations", "turns": {},
             "operations": {}, "resume_state": {}},
            {"title": "T", "category": "conversations", "turns": [],
             "operations": {}, "resume_state": {}},
            {"title": "T", "category": "conversations", "turns": {},
             "operations": None, "resume_state": {}},
            {"title": "T", "category": "conversations", "turns": {},
             "operations": {}, "resume_state": []},
        )
        for value in invalid:
            for create in (False, True):
                with self.subTest(value=value, create=create):
                    self._assert_invalid_unchanged({"sessions": {
                        fmt.CONVERSATION_SESSION_KEY: value,
                    }}, create=create)

    def test_reject_grouped_legacy_field_in_reserved_session(self) -> None:
        for create in (False, True):
            session = fmt.new_conversation_session()
            session["conversation"] = {"turns": [1]}
            self._assert_invalid_unchanged({"sessions": {
                fmt.CONVERSATION_SESSION_KEY: session,
            }}, create=create)

    def test_reject_noncanonical_conversation_key(self) -> None:
        for create in (False, True):
            payload = {"sessions": {
                "custom": fmt.new_conversation_session({"turns": [1]}),
            }}
            self._assert_invalid_unchanged(payload, create=create)

    def test_reject_duplicate_conversation_category(self) -> None:
        for create in (False, True):
            payload = {"sessions": {
                fmt.CONVERSATION_SESSION_KEY: fmt.new_conversation_session(),
                "custom": fmt.new_conversation_session(),
            }}
            self._assert_invalid_unchanged(payload, create=create)

    def test_unrelated_trace_entries_do_not_block_canonical_session(
        self,
    ) -> None:
        payload = {"sessions": {
            "broken_unrelated": None,
            "01_runtime": {"category": "runtime", "operations": {}},
            fmt.CONVERSATION_SESSION_KEY: fmt.new_conversation_session(),
        }}
        self.assertEqual(fmt.conversation_state(payload), {})

    def test_reject_obsolete_chat_messages_in_resume_state(self) -> None:
        for create in (False, True):
            session = fmt.new_conversation_session({
                "turns": [], "chat_messages": [{"role": "user"}],
            })
            payload = {"sessions": {fmt.CONVERSATION_SESSION_KEY: session}}
            self._assert_invalid_unchanged(payload, create=create)

    def test_reject_bad_new_session_without_mutating_payload(self) -> None:
        with patch.object(
            fmt, "new_conversation_session",
            return_value={"resume_state": []},
        ):
            self._assert_invalid_unchanged({"trace_id": "retained"},
                                           create=True)

    def test_conflicting_reserved_key_preserves_other_sessions(self) -> None:
        runtime = {"category": "runtime", "operations": {"saved": 1}}
        payload = {"sessions": {
            "runtime": runtime,
            fmt.CONVERSATION_SESSION_KEY: {
                "category": "runtime", "important": "retain",
            },
        }}
        for create in (False, True):
            self._assert_invalid_unchanged(payload, create=create)
        self.assertIs(payload["sessions"]["runtime"], runtime)


if __name__ == "__main__":
    unittest.main()
