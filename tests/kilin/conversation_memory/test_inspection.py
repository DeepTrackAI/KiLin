"""Verify model-free project memory inventory and explicit exclusions.

Tests use active saved checkpoints without opening a GUI or loading models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import unittest

from kilin import rag
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import (
    FeedbackConflictError, FeedbackStore,
)
from kilin.conversation_memory.inspection import (
    inspect_memory_messages,
    read_memory_summaries,
)
from kilin.conversations import ConversationStore
from kilin.project_type import project_type
from tests.kilin._rag_support import _turn


class TestMemoryInspection(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.store = ConversationStore(self.root)
        self.conversation = self.store.create()
        self.turn = _turn(
            "Did we discuss optical tweezers?",
            "We considered an experiment, without deciding.",
            parameters={"intent": "writing"},
        )
        path = self.store.state_path(self.conversation)
        data = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(data)["turns"] = [rag._turn_to_data(self.turn)]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.config = project_type("writing")
        self.feedback = FeedbackStore(self.root)

    def _inspect(self, **kwargs):
        return inspect_memory_messages(self.root, self.config, **kwargs)

    def _summary_references(self, *, number=1, uuid=None):
        """Build the exact saved memory origin references for a test turn."""

        identity = uuid or self.turn.conversation_turn_uuid
        return [{
            "turn_uuid": identity, "role": role,
            "origins": [{
                "conversation_id": self.conversation.conversation_id,
                "turn_number": number,
            }],
        } for role in ("user", "assistant")]

    def test_dynamic_trace_summary_is_complete_and_refreshes(self):
        """Use an edited full trace summary without rewriting the file."""

        path = self.store.state_path(self.conversation)
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = "Scientific interpretation: " + "foldons " * 80
        turn = conversation_state(payload)["turns"][0]
        turn["parameters"]["representation"] = {
            "summary": summary[:200], "turn_id": 1,
        }
        payload["sessions"]["05_conversations"]["turns"] = {
            "turn_0001": {"stages": {"01_question": {"operations": {
                "02_conversation_turn_representation": {
                    "details": {"summary": summary},
                },
            }}}},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        references = self._summary_references()
        before = path.read_bytes()
        found = read_memory_summaries(self.store, references)
        self.assertEqual(found[0]["text"], self.turn.question)
        self.assertEqual(found[1]["text"], summary)
        self.assertEqual(path.read_bytes(), before)

        amended = "Updated answer summary: " + "new fact " * 50
        payload["sessions"]["05_conversations"]["turns"][
            "turn_0001"
        ]["stages"]["01_question"]["operations"][
            "02_conversation_turn_representation"
        ]["details"]["summary"] = amended
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(
            read_memory_summaries(self.store, references)[1]["text"],
            amended,
        )

        # An explicitly cleared summary must not resurrect an older copy.
        payload["sessions"]["05_conversations"]["turns"][
            "turn_0001"
        ]["stages"]["01_question"]["operations"][
            "02_conversation_turn_representation"
        ]["details"]["summary"] = ""
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(
            [item["role"] for item in read_memory_summaries(
                self.store, references
            )], ["user"],
        )

    def test_summary_fallback_branch_and_identity_validation(self):
        """Use compact summaries only when a trace operation is absent."""

        path = self.store.state_path(self.conversation)
        payload = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(payload)["turns"][0]["parameters"][
            "representation"
        ] = {"summary": "Older stored summary"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        inherited = self.store.create(
            branch_from=self.conversation.conversation_id, turn_count=1
        )
        references = self._summary_references()
        references[1]["origins"] = [{
            "conversation_id": inherited.conversation_id,
            "turn_number": 1,
        }]
        self.assertEqual(
            read_memory_summaries(self.store, references)[1]["text"],
            "Older stored summary",
        )
        self.assertEqual(read_memory_summaries(
            self.store, self._summary_references(uuid=uuid4().hex)
        ), [])
        self.assertEqual(read_memory_summaries(
            self.store, self._summary_references(number=2)
        ), [])
        invalid = self._summary_references()
        for malformed in (
            "invalid", [*invalid, {"role": "assistant"}],
            [{**invalid[0], "role": "system"}],
            [{**invalid[0], "origins": [{
                "conversation_id": "../../foreign", "turn_number": 1,
            }, {"conversation_id": "known", "turn_number": True}]}],
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    read_memory_summaries(self.store, malformed)

    def test_inventory_includes_messages_not_indexed_by_policy(self):
        page = self._inspect()
        self.assertEqual(page.total, 2)
        self.assertEqual(len(page.messages), 2)
        statuses = {m["role"]: m["status"] for m in page.messages}
        self.assertEqual(statuses, {
            "user": "eligible", "assistant": "policy",
        })
        self.assertEqual(page.revision, self.feedback.load().revision)
        self.assertTrue(all(
            m["conversation_id"] == self.conversation.conversation_id
            for m in page.messages
        ))
        self.assertFalse((self.root / "indexes").exists())

    def test_explicit_exclusion_stays_inspectable(self):
        uuid = self.turn.conversation_turn_uuid
        selected = self.feedback.update(
            self.conversation.conversation_id, uuid, "user",
            saved=True, reaction="liked",
        )
        current = self.feedback.update(
            self.conversation.conversation_id, uuid, "user",
            saved=None, reaction=None, excluded=True,
            expected_revision=selected.revision,
        )
        page = self._inspect()
        entry = next(m for m in page.messages if m["role"] == "user")
        self.assertEqual(entry["status"], "excluded")
        self.assertTrue(entry["saved"])
        self.assertEqual(entry["reaction"], "liked")
        self.assertEqual(page.revision, current.revision)
        with self.assertRaises(FeedbackConflictError):
            self.feedback.update(
                self.conversation.conversation_id, uuid, "user",
                saved=None, reaction=None, excluded=False,
                expected_revision=selected.revision,
            )
        self.assertEqual(self._inspect().messages[0]["status"], "excluded")

    def test_dislike_and_project_policy_have_separate_statuses(self):
        uuid = self.turn.conversation_turn_uuid
        self.feedback.update(
            self.conversation.conversation_id, uuid, "assistant",
            saved=True, reaction="disliked",
        )
        self.config = replace(
            self.config,
            conversation_memory=replace(
                self.config.conversation_memory, enabled=False,
            ),
        )
        page = self._inspect()
        status = {m["role"]: m["status"] for m in page.messages}
        self.assertEqual(status["assistant"], "disliked")
        self.assertEqual(status["user"], "policy")

    def test_paginated_list_covers_all_messages(self):
        first = self._inspect(offset=0, limit=1)
        second = self._inspect(offset=1, limit=1)
        self.assertEqual(first.total, 2)
        self.assertEqual(second.total, 2)
        self.assertNotEqual(first.messages[0]["role"],
                            second.messages[0]["role"])
        self.assertEqual(self._inspect(offset=2, limit=1).messages, ())
        for args in ({"offset": -1}, {"offset": 1.2},
                     {"limit": 0}, {"limit": 201}, {"limit": True}):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    self._inspect(**args)

    def test_archived_origins_are_no_longer_listed(self):
        self.store.archive(self.conversation.conversation_id)
        self.assertEqual(self._inspect().total, 0)

    def test_branch_occurrences_share_one_managed_message(self):
        self.store.create(
            branch_from=self.conversation.conversation_id,
            turn_count=1,
        )
        self.assertEqual(self._inspect().total, 2)


if __name__ == "__main__":
    unittest.main()
