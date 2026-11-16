"""Test disliked-answer exclusion across RAG answering and tool paths.

Use genuine project-owned feedback and saved turns with offline fake models.
The test confirms that rejecting an answer does not rewrite its checkpoint.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch

from kilin import rag
from kilin.answering import RouteDecision
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversations import ConversationStore
from kilin.project_type import project_type
from kilin.tools import ToolResult, ToolUse
from tests.kilin._rag_support import (
    _answer_patches,
    _evidence,
    _intent,
    _session,
    _turn,
    _type_config,
)


class TestRAGDislikedHistory(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.store = ConversationStore(self.root)
        info = self.store.create()
        self.conversation_id = info.conversation_id
        self.path = self.store.state_path(info)
        self.turns = [
            _turn("Good first question", "Reliable earlier answer",
                  (_evidence(start=10),)),
            _turn("REJECTED QUESTION", "REJECTED ANSWER",
                  (_evidence(start=99, text="REJECTED EVIDENCE"),),
                  {"related_turn_ids": [1]}),
            _turn("Good final question", "Final useful answer",
                  (_evidence(start=30),),
                  {"related_turn_ids": [2]}),
        ]
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        conversation_state(payload)["turns"] = [
            rag._turn_to_data(turn) for turn in self.turns
        ]
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self.session = _session(self.root)
        self.session.conversation_path = self.path
        self.session.turns = list(self.turns)
        self.feedback = FeedbackStore(self.root)
        self.feedback.update(
            self.conversation_id,
            self.turns[1].conversation_turn_uuid,
            "assistant", saved=False, reaction="disliked",
        )

    def test_feedback_is_independent_of_memory_enabled_default(self):
        config = project_type("code")
        self.assertFalse(config.conversation_memory.enabled)
        self.assertTrue(config.exclude_disliked_answers)
        self.assertEqual(
            self.session._excluded_history_turns(self.session.turns),
            frozenset((2,)),
        )

    def test_reopened_history_excludes_rejected_pair_not_checkpoint(self):
        original = self.path.read_bytes()
        messages, restored, _ = self.session._load_conversation()
        contents = "\n".join(message["content"] for message in messages)
        self.assertIn("Good first question", contents)
        self.assertIn("Good final question", contents)
        self.assertNotIn("REJECTED QUESTION", contents)
        self.assertNotIn("REJECTED ANSWER", contents)
        self.assertEqual(len(restored), 3)
        self.assertEqual(restored[1].answer, "REJECTED ANSWER")
        self.assertEqual(self.path.read_bytes(), original)

    def test_answer_filters_router_history_and_related_evidence(self):
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(2, 3),
            refers_back=True,
        )
        with _answer_patches(
            self.session, _intent(related=True), decision
        ) as mocks:
            self.session.ask("Continue", use_router=False)
        result = self.session.turns[-1]
        self.assertEqual(result.parameters["related_turn_ids"], [3])
        self.assertEqual(result.parameters["history_turn_ids"], [1, 3])
        self.assertEqual(result.parameters["evidence_from_turns"], [3])
        self.assertEqual([item.start_line for item in result.evidence],
                         [1, 30])
        history = self.session.chat_session.restore_history.call_args.args[0]
        rendered = "\n".join(message["content"] for message in history)
        self.assertNotIn("REJECTED ANSWER", rendered)
        self.assertNotIn("REJECTED QUESTION", rendered)
        self.assertIn("Final useful answer", rendered)
        self.assertNotIn("REJECTED EVIDENCE",
                         self.session.chat_session.reply.call_args.args[0])
        mocks.retrieve.assert_called_once()

    def test_reuse_falls_back_to_last_nonrejected_source(self):
        self.feedback.update(
            self.conversation_id,
            self.turns[2].conversation_turn_uuid,
            "assistant", saved=False, reaction="disliked",
        )
        with _answer_patches(
            self.session, _intent("reuse")
        ) as mocks:
            self.session.ask("Reuse", use_router=False)
        parameters = self.session.turns[-1].parameters
        self.assertEqual(parameters["evidence_from_turns"], [1])
        self.assertEqual(parameters["retrieval_mode"], "reuse")
        mocks.retrieve.assert_not_called()

    def test_related_chain_does_not_traverse_a_rejected_link(self):
        decision = RouteDecision(
            "x", False, related_turn_ids=(3,), refers_back=True
        )
        earlier = self.session._earlier_turns(
            decision, excluded_turn_ids=frozenset((2,))
        )
        self.assertEqual([turn.number for turn in earlier], [3])
        self.assertNotIn("REJECTED ANSWER", repr(earlier))
        self.assertNotIn("REJECTED EVIDENCE", repr(earlier))

    def test_tool_request_and_language_context_omit_rejected_turn(self):
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(1, 2, 3),
            refers_back=True,
        )
        use = ToolUse("project_inventory", object())
        result = ToolResult("ok", (), {"status": "ok"})
        contexts = []

        def runner(_use, request, context):
            contexts.append(request)
            context.ask_model("System", "Inspect", 12)
            return result

        with patch.object(rag, "run_tool", side_effect=runner), \
             patch.object(rag, "side_reply", return_value="done") as side:
            self.session._run_tool(
                use, "Continue", decision,
                trace_turn_id=None, report_phase=Mock(),
                excluded_turn_ids=frozenset((2,)),
            )
        self.assertEqual([item.number for item in contexts[0].earlier],
                         [3, 1])
        self.assertNotIn("REJECTED QUESTION", contexts[0].text)
        self.assertNotIn("REJECTED ANSWER", repr(contexts[0]))
        self.assertNotIn("REJECTED EVIDENCE", repr(contexts[0]))
        self.assertNotIn("REJECTED QUESTION",
                         repr(side.call_args.kwargs["language_context"]))

    def test_disabled_policy_restores_full_history(self):
        disabled = replace(
            project_type("code"), exclude_disliked_answers=False
        )
        with patch.object(
            self.session, "_behavior_config", return_value=disabled
        ):
            self.assertEqual(
                self.session._excluded_history_turns(self.session.turns),
                frozenset(),
            )
            messages, _, _ = self.session._load_conversation()
        contents = "\n".join(message["content"] for message in messages)
        self.assertIn("REJECTED QUESTION", contents)
        self.assertIn("REJECTED ANSWER", contents)

    def test_reset_feedback_restores_context_without_rewriting_history(self):
        self.feedback.reset(
            self.conversation_id,
            self.turns[1].conversation_turn_uuid, "assistant",
        )
        self.assertEqual(
            self.session._excluded_history_turns(self.session.turns),
            frozenset(),
        )
        messages, _, _ = self.session._load_conversation()
        self.assertIn("REJECTED ANSWER", repr(messages))

    def test_question_feedback_does_not_exclude_assistant_context(self):
        self.feedback.reset(
            self.conversation_id,
            self.turns[1].conversation_turn_uuid, "assistant",
        )
        self.feedback.update(
            self.conversation_id,
            self.turns[1].conversation_turn_uuid,
            "user", saved=False, reaction="disliked",
        )
        self.assertFalse(
            self.session._excluded_history_turns(self.session.turns)
        )

    def test_model_switch_does_not_carry_rejected_answer(self):
        original = self.session.chat_session
        original.messages = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "REJECTED QUESTION"},
            {"role": "assistant", "content": "REJECTED ANSWER"},
        ]
        replacement = MagicMock()
        self.session._new_chat_session = Mock(return_value=replacement)
        self.session.use_chat_model("other-model")
        messages = replacement.restore_history.call_args.args[0]
        contents = "\n".join(message["content"] for message in messages)
        self.assertNotIn("REJECTED ANSWER", contents)
        self.assertNotIn("REJECTED QUESTION", contents)
        self.assertIn("Good final question", contents)
        self.assertEqual(messages[0]["content"], "instructions")

    def test_bad_feedback_prevents_prompt_without_migration(self):
        self.feedback.path.write_text("{not valid json}", encoding="utf-8")
        original = self.feedback.path.read_bytes()
        with _answer_patches(self.session, _intent()) as mocks:
            with self.assertRaisesRegex(ValueError, "feedback is invalid"):
                self.session.ask("New question", use_router=False)
        self.assertEqual(self.feedback.path.read_bytes(), original)
        mocks.route.assert_not_called()
        mocks.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
