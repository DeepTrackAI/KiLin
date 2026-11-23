"""Test project-owned feedback persistence, origins and file safety.

Use isolated filesystem fixtures and actual conversation/checkpoint records.
No inference models, embeddings, or external services are required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.conversation_format import conversation_state
from kilin.conversation_ownership import (
    ConversationFileLock,
    ConversationLockedError,
)
from kilin.conversations import ConversationStore
from kilin.conversation_memory.feedback import (
    FeedbackConflictError,
    FeedbackStore,
)
from kilin.conversation_memory.models import MessageFeedback
from kilin.rag import ConversationTurn, _turn_to_data


class TestFeedbackStore(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.conversations = ConversationStore(self.root)
        self.parent = self.conversations.create()
        self.one = ConversationTurn("2026-11-15T10:00:00+00:00",
                                    "Repeated question", "Same answer",
                                    (), {})
        self.two = ConversationTurn("2026-11-15T10:00:00+00:00",
                                    "Repeated question", "Same answer",
                                    (), {})
        self._set_turns(self.parent.conversation_id, [self.one, self.two])
        self.feedback = FeedbackStore(self.root)

    def _set_turns(self, conversation_id, turns):
        path = self.conversations.state_path(conversation_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(payload)["turns"] = [
            _turn_to_data(turn) for turn in turns
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _update(self, turn=None, *, role="assistant", saved=False,
                reaction="neutral", revision=None, conversation=None):
        return self.feedback.update(
            conversation or self.parent.conversation_id,
            (turn or self.one).conversation_turn_uuid,
            role, saved=saved, reaction=reaction,
            expected_revision=revision,
        )

    def test_reading_missing_feedback_creates_nothing(self):
        self.assertFalse((self.root / "memory").exists())
        snapshot = self.feedback.load()
        self.assertEqual(snapshot.get(
            self.one.conversation_turn_uuid, "assistant"
        ), MessageFeedback())
        self.assertEqual(len(snapshot.revision), 64)
        self.assertFalse((self.root / "memory").exists())

    def test_save_like_dislike_reset_and_separate_roles(self):
        question = self._update(role="user", saved=True)
        self.assertTrue(question.get(
            self.one.conversation_turn_uuid, "user"
        ).selected)
        liked = self._update(
            saved=True, reaction="liked", revision=question.revision
        )
        self.assertTrue(liked.get(
            self.one.conversation_turn_uuid, "assistant"
        ).selected)
        disliked = self._update(
            saved=True, reaction="disliked", revision=liked.revision
        )
        self.assertFalse(disliked.get(
            self.one.conversation_turn_uuid, "assistant"
        ).selected)
        self.assertTrue(disliked.get(
            self.one.conversation_turn_uuid, "user"
        ).selected)
        cleared = self.feedback.reset(
            self.parent.conversation_id, self.one.conversation_turn_uuid,
            "assistant", expected_revision=disliked.revision,
        )
        self.assertEqual(cleared.get(
            self.one.conversation_turn_uuid, "assistant"
        ), MessageFeedback())
        self.assertTrue(cleared.get(
            self.one.conversation_turn_uuid, "user"
        ).saved)
        reloaded = FeedbackStore(self.root).load()
        self.assertEqual(reloaded.entries, cleared.entries)

    def test_independently_generated_identical_text_has_own_rating(self):
        self.assertNotEqual(self.one.conversation_turn_uuid,
                            self.two.conversation_turn_uuid)
        result = self._update(reaction="disliked")
        self.assertEqual(result.get(
            self.two.conversation_turn_uuid, "assistant"
        ), MessageFeedback())
        self.assertEqual(result.get(
            self.one.conversation_turn_uuid, "assistant"
        ).reaction, "disliked")

    def test_branch_shares_inherited_turn_identity_and_feedback(self):
        branch = self.conversations.create(
            branch_from=self.parent.conversation_id, turn_count=1
        )
        parent_path = self.conversations.state_path(self.parent)
        branch_path = self.conversations.state_path(branch)
        self.assertEqual(
            conversation_state(json.loads(parent_path.read_text(
                encoding="utf-8",
            )))[
                "turns"
            ][0]["conversation_turn_uuid"],
            conversation_state(json.loads(branch_path.read_text(
                encoding="utf-8",
            )))[
                "turns"
            ][0]["conversation_turn_uuid"],
        )
        snapshot = self._update(
            conversation=branch.conversation_id, saved=True
        )
        self.assertTrue(snapshot.get(
            self.one.conversation_turn_uuid, "assistant"
        ).selected)
        self.conversations.archive(self.parent.conversation_id)
        self.assertFalse(self.conversations.contains_turn(
            self.parent.conversation_id, self.one.conversation_turn_uuid
        ))
        self.assertTrue(self.conversations.contains_turn(
            branch.conversation_id, self.one.conversation_turn_uuid
        ))
        with self.assertRaisesRegex(ValueError, "active project turn"):
            self._update(saved=False)
        changed = self._update(
            conversation=branch.conversation_id, reaction="liked"
        )
        self.assertEqual(changed.get(
            self.one.conversation_turn_uuid, "assistant"
        ).reaction, "liked")

    def test_stale_revision_does_not_write(self):
        before = self.feedback.load().revision
        first = self._update(saved=True, revision=before)
        content = self.feedback.path.read_bytes()
        with self.assertRaises(FeedbackConflictError):
            self._update(reaction="liked", revision=before)
        self.assertEqual(self.feedback.path.read_bytes(), content)
        second = self._update(
            turn=self.two, reaction="liked", revision=first.revision
        )
        self.assertTrue(second.get(
            self.one.conversation_turn_uuid, "assistant"
        ).saved)
        self.assertEqual(second.get(
            self.two.conversation_turn_uuid, "assistant"
        ).reaction, "liked")

    def test_exclusive_lock_prevents_interleaved_feedback_writes(self):
        self._update(saved=True)
        before = self.feedback.path.read_bytes()
        with ConversationFileLock(self.feedback.path):
            with self.assertRaises(ConversationLockedError):
                self._update(turn=self.two, reaction="liked")
        self.assertEqual(self.feedback.path.read_bytes(), before)
        self._update(turn=self.two, reaction="liked")
        snapshot = self.feedback.load()
        self.assertEqual(len(snapshot.entries), 2)

    def test_atomic_write_failure_preserves_previous_feedback(self):
        self._update(saved=True)
        before = self.feedback.path.read_bytes()
        with patch(
            "kilin.conversation_memory.feedback.atomic_write_text",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self._update(reaction="disliked")
        self.assertEqual(self.feedback.path.read_bytes(), before)

    def test_invalid_target_and_values_cannot_replace_feedback(self):
        self._update(saved=True)
        before = self.feedback.path.read_bytes()
        for identifier in ("missing", self.two.conversation_turn_uuid):
            with self.subTest(origin=identifier):
                with self.assertRaises(ValueError):
                    self.feedback.update(
                        "not-a-conversation", identifier,
                        "assistant", saved=False, reaction="liked",
                    )
        for values in (
            {"role": "system"}, {"saved": "true"},
            {"reaction": "excellent"}, {"revision": "invalid"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self._update(**values)
        self.assertEqual(self.feedback.path.read_bytes(), before)

    def test_corrupt_and_unknown_formats_fail_without_replacement(self):
        self._update(saved=True)
        for content in (
            "{invalid json", "[]", '{"format":"other","entries":{}}',
            '{"format":"conversation-memory-feedback","entries":',
        ):
            with self.subTest(content=content):
                self.feedback.path.write_text(content, encoding="utf-8")
                before = self.feedback.path.read_bytes()
                with self.assertRaises(ValueError):
                    self.feedback.load()
                with self.assertRaises(ValueError):
                    self._update(reaction="liked")
                self.assertEqual(self.feedback.path.read_bytes(), before)

    def test_file_symlink_is_rejected(self):
        victim = self.root / "victim.json"
        victim.write_text("unchanged", encoding="utf-8")
        directory = self.root / "memory"
        directory.mkdir()
        try:
            self.feedback.path.symlink_to(victim)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        with self.assertRaises(ValueError):
            self.feedback.load()
        with self.assertRaises(ValueError):
            self._update(saved=True)
        self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged")

    def test_rating_never_modifies_conversation_checkpoint(self):
        path = self.conversations.state_path(self.parent)
        original = path.read_bytes()
        self._update(saved=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "indexes").exists())


if __name__ == "__main__":
    unittest.main()
