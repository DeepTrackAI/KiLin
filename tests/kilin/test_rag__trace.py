"""Test RAG trace integration with grouped conversation records.

Trace files use temporary directories and no live models.

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
import warnings
from kilin import rag
from kilin.conversation_format import CONVERSATION_SESSION_KEY
from kilin.project_type import Intent, RETRIEVAL_MODE_NONE
from kilin.tools import ToolUse
from kilin.trace_logging import (
    FAILURE_POLICY_RAISE,
    FAILURE_POLICY_WARN,
    FailurePolicy,
    TraceConfig,
    TraceLogger,
)

from tests.kilin._rag_support import (
    _answer_patches,
    _intent,
    _session,
)


class TestRAGTraceIntegration(unittest.TestCase):
    """Verify RAG turn outcomes in real grouped conversation-trace files."""

    def setUp(self) -> None:
        """Isolate conversation files without loading any model or index."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)

    def _bind_trace(
        self,
        *,
        owned: bool = False,
        failure_policy: FailurePolicy = FAILURE_POLICY_WARN,
    ) -> TraceLogger:
        """Connect a real logger to the RAG conversation's exact path."""
        trace = TraceLogger(
            TraceConfig(
                enabled=True,
                output_path=self.session.conversation_path,
                failure_policy=failure_policy,
            ),
            default_directory=self.root,
        )
        self.addCleanup(trace.close)
        self.session.trace = trace
        self.session._owns_trace = owned
        return trace

    def _payload(self, path: Path | None = None) -> dict[str, object]:
        """Read the single persisted trace and conversation document."""
        selected = path or self.session.conversation_path
        return json.loads(selected.read_text(encoding="utf-8"))

    def _conversation(self) -> dict[str, object]:
        """Return the readable conversation session from the active file."""
        return self._payload()["sessions"][CONVERSATION_SESSION_KEY]

    def _ask_with_real_save(
        self,
        question: str,
        intent: Intent,
        *,
        phase_callback: rag.AnswerPhaseCallback | None = None,
    ) -> str:
        """Run the real turn state machine and real checkpoint writer."""
        with _answer_patches(self.session, intent) as mocks:
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            return self.session.ask(
                question,
                use_router=False,
                phase_callback=phase_callback,
            )

    def test_full_trace_persists_metadata_and_checkpoint(
        self,
    ) -> None:
        """Keep complete diagnostics and resumable state in one file."""
        trace = self._bind_trace()
        answer = self._ask_with_real_save("Sensitive question", _intent())
        trace.close()
        self.assertEqual(answer, "model answer")
        payload = self._payload()
        timeline = self.session.turns[-1].timeline
        self.assertEqual(
            payload["event_count"], 13 + len(timeline.steps)
        )
        session = self._conversation()
        self.assertEqual(len(session["turns"]), 1)
        turn = session["turns"]["turn_0001"]
        self.assertEqual(turn["title"], "Turn 1: Sensitive question")
        self.assertEqual(
            turn["stages"]["03_retrieval"]["operations"][
                "02_retrieval_result"
            ]["details"]["evidence_count"],
            1,
        )
        checkpoint = turn["stages"]["05_generation"]["operations"][
            "01_state_checkpoint_committed"
        ]["details"]
        self.assertEqual(
            checkpoint["checkpoint_id"], session["resume_state"][
                "checkpoint_id"
            ]
        )
        self.assertEqual(checkpoint["turn_count"], 1)
        self.assertEqual(
            checkpoint["conversation_path"], str(trace.output_path)
        )
        self.assertEqual(
            session["resume_state"]["turns"][0]["question"],
            "Sensitive question",
        )
        question_ops = turn["stages"]["01_question"]["operations"]
        self.assertEqual(
            question_ops["01_turn"]["details"]["question"],
            "Sensitive question",
        )
        representation = next(
            value for key, value in question_ops.items()
            if key.endswith("_conversation_turn_representation")
        )
        self.assertEqual(
            representation["details"]["question"],
            "Sensitive question",
        )
        self.assertFalse(trace.output_path.with_suffix(".json.tmp").exists())

    def test_trace_keeps_question_and_related_turn_operations(
        self,
    ) -> None:
        """Keep exact question text and every phase in the same turn."""
        self._bind_trace()
        self._ask_with_real_save("Exact forensic question", _intent())
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(turn["title"], "Turn 1: Exact forensic question")
        question = turn["stages"]["01_question"]["operations"]
        self.assertEqual(
            question["01_turn"]["details"]["question"],
            "Exact forensic question",
        )
        self.assertEqual(question["01_turn"]["event"]["status"],
                         "completed")
        self.assertIn("03_retrieval", turn["stages"])
        self.assertIn("05_generation", turn["stages"])
        self.assertEqual(self._payload()["progress_checkpoint_count"], 0)

    def test_answer_without_retrieval_is_traced_and_saved(self) -> None:
        """Record explicit no-retrieval decisions and a valid checkpoint."""
        self._bind_trace()
        self._ask_with_real_save("Hello", _intent(RETRIEVAL_MODE_NONE))
        turn = self._conversation()["turns"]["turn_0001"]
        retrieval = turn["stages"]["03_retrieval"]["operations"]
        self.assertEqual(
            retrieval["02_retrieval_result"]["details"]["mode"],
            RETRIEVAL_MODE_NONE,
        )
        self.assertEqual(
            retrieval["02_retrieval_result"]["details"]["evidence_count"],
            0,
        )
        self.assertEqual(
            len(self._conversation()["resume_state"]["turns"]), 1
        )

    def test_generation_failure_is_logged_without_committing_a_turn(
        self,
    ) -> None:
        """Persist a failed turn span but no successful checkpoint event."""
        self._bind_trace()
        self.session.chat_session.reply.side_effect = RuntimeError(
            "model unavailable"
        )
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            self._ask_with_real_save("Failing question", _intent())
        self.assertEqual(self.session.turns, [])
        turn = self._conversation()["turns"]["turn_0001"]
        summary = turn["stages"]["01_question"]["operations"][
            "01_turn"
        ]
        self.assertEqual(summary["event"]["status"], "failed")
        self.assertEqual(summary["result"]["error_type"], "RuntimeError")
        self.assertIn("model unavailable", summary["result"]["traceback"])
        self.assertNotIn("05_generation", turn["stages"])
        self.assertEqual(self._conversation()["resume_state"], {})

    def test_retrieval_failure_is_logged_before_generation(self) -> None:
        """Report retrieval errors without falsely recording model output."""
        self._bind_trace()
        with _answer_patches(self.session, _intent()) as mocks:
            mocks.retrieve.side_effect = OSError("retrieval failed")
            with self.assertRaisesRegex(OSError, "retrieval failed"):
                self.session.ask("Search failure", use_router=False)
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(
            turn["stages"]["01_question"]["operations"][
                "01_turn"
            ]["event"]["status"],
            "failed",
        )
        self.assertNotIn("05_generation", turn["stages"])
        self.assertEqual(len(self.session.turns), 0)

    def test_two_turns_preserve_order_and_checkpoint_parent(self) -> None:
        """Maintain distinct trace turns with linked resumable checkpoints."""
        self._bind_trace()
        self._ask_with_real_save("First question", _intent())
        earlier = self.session._checkpoint_id
        self._ask_with_real_save("Second question", _intent())
        session = self._conversation()
        self.assertEqual(list(session["turns"]), [
            "turn_0001", "turn_0002"
        ])
        self.assertEqual(len(session["resume_state"]["turns"]), 2)
        checkpoint = session["turns"]["turn_0002"]["stages"][
            "05_generation"
        ]["operations"]["01_state_checkpoint_committed"]["details"]
        self.assertEqual(checkpoint["parent_checkpoint_id"], earlier)
        self.assertEqual(checkpoint["turn_count"], 2)
        self.assertNotEqual(checkpoint["checkpoint_id"], earlier)

    def test_reset_archives_previous_trace_without_retaining_old_turns(
        self,
    ) -> None:
        """Keep archived evidence intact while starting a fresh transcript."""
        self._bind_trace()
        self._ask_with_real_save("Before reset", _intent())
        prior_checkpoint = self.session._checkpoint_id
        archived = self.session.reset()
        self.assertIsNotNone(archived)
        self.assertTrue(archived.is_file())
        archived_session = self._payload(archived)["sessions"][
            CONVERSATION_SESSION_KEY
        ]
        self.assertEqual(len(archived_session["turns"]), 1)
        self.assertEqual(
            archived_session["resume_state"]["checkpoint_id"],
            prior_checkpoint,
        )
        active = self._conversation()
        self.assertEqual(active["turns"], {})
        self.assertEqual(active["resume_state"], {})
        self._ask_with_real_save("After reset", _intent())
        self.assertEqual(list(self._conversation()["turns"]), ["turn_0001"])
        self.assertEqual(self.session._checkpoint_id,
                         self._conversation()["resume_state"]["checkpoint_id"])

    def test_switch_conversation_keeps_trace_and_resume_paths_separate(
        self,
    ) -> None:
        """Close the owned old logger before switching to a new trace."""
        self._bind_trace(owned=True)
        self._ask_with_real_save("Old conversation", _intent())
        old_path = self.session.conversation_path
        new_path = old_path.with_name("conversation_second.json")
        next_trace = TraceLogger(
            TraceConfig(enabled=True, output_path=new_path),
            default_directory=self.root,
        )
        self.addCleanup(next_trace.close)
        restored = self.session.switch_conversation(
            new_path, trace=next_trace
        )
        self.assertEqual(restored, [])
        old_trace_events = self._payload(old_path)["sessions"][
            "01_trace"
        ]["operations"]
        self.assertIn("02_trace_closed", old_trace_events)
        self.assertEqual(
            self.session.conversation_path, new_path.resolve()
        )
        self._ask_with_real_save("New conversation", _intent())
        self.assertEqual(
            self._payload(old_path)["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["turns"][0]["question"],
            "Old conversation",
        )
        self.assertEqual(
            self._payload(new_path)["sessions"][CONVERSATION_SESSION_KEY][
                "resume_state"
            ]["turns"][0]["question"],
            "New conversation",
        )
        self.assertEqual(len(self._payload(old_path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["turns"]), 1)
        self.assertEqual(len(self._payload(new_path)["sessions"][
            CONVERSATION_SESSION_KEY
        ]["turns"]), 1)

    def test_conversation_write_failure_records_no_committed_checkpoint(
        self,
    ) -> None:
        """Report a failed save without claiming the checkpoint succeeded."""
        trace = self._bind_trace()
        with patch.object(
            trace, "update_conversation", side_effect=OSError("no space")
        ):
            with self.assertRaisesRegex(OSError, "no space"):
                self._ask_with_real_save("Save failure", _intent())
        self.assertEqual(self.session.turns, [])
        self.assertIsNone(self.session._checkpoint_id)
        self.session.chat_session.restore_history.assert_called()
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(
            turn["stages"]["01_question"]["operations"][
                "01_turn"
            ]["event"]["status"],
            "failed",
        )
        self.assertNotIn("05_generation", turn["stages"])
        self.assertEqual(self._conversation()["resume_state"], {})

    def test_failed_checkpoint_restores_committed_history_and_retries(
        self,
    ) -> None:
        """Exclude a failed turn from history, then link the next save."""
        trace = self._bind_trace()
        chat = self.session.chat_session

        def restore(messages: list[dict[str, str]]) -> None:
            """Make the model fake retain restored message history."""
            chat.messages = [dict(message) for message in messages]

        def reply(prompt: str, **options: object) -> str:
            """Simulate the real chat model adding generated messages."""
            chat.messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "model answer"},
            ])
            return "model answer"

        chat.restore_history.side_effect = restore
        chat.reply.side_effect = reply
        self._ask_with_real_save("First saved", _intent())
        first = self.session._checkpoint_id
        messages = [dict(message) for message in chat.messages]
        original_writer = trace._write_readable_trace

        def fail_pending_state() -> None:
            """Refuse only the attempted checkpoint write."""
            if trace._conversation.get("checkpoint_id") != first:
                raise OSError("checkpoint unavailable")
            original_writer()

        with patch.object(
            trace, "_write_readable_trace", side_effect=fail_pending_state
        ):
            with self.assertRaisesRegex(OSError, "checkpoint unavailable"):
                self._ask_with_real_save("Failed to save", _intent())
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(self.session._checkpoint_id, first)
        self.assertEqual(chat.messages, messages)
        self.assertEqual(trace._conversation["checkpoint_id"], first)
        saved = self._conversation()["resume_state"]
        self.assertEqual(saved["checkpoint_id"], first)
        self.assertEqual(len(saved["turns"]), 1)
        self.assertEqual(
            self._conversation()["turns"]["turn_0002"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "failed",
        )
        self._ask_with_real_save("Saved after retry", _intent())
        saved = self._conversation()["resume_state"]
        self.assertEqual(len(self.session.turns), 2)
        self.assertEqual(len(saved["turns"]), 2)
        self.assertEqual(
            saved["checkpoint"]["parent_checkpoint_id"], first
        )
        self.assertEqual(
            [turn["question"] for turn in saved["turns"]],
            ["First saved", "Saved after retry"],
        )

    def test_postcommit_trace_error_does_not_roll_back_checkpoint(
        self,
    ) -> None:
        """Keep a saved turn if its subsequent diagnostic event fails."""
        trace = self._bind_trace()
        original_emit = trace.emit

        def fail_event(event_type: str, **options: object) -> None:
            """Fail only the event emitted after checkpoint persistence."""
            if event_type == "state.checkpoint.committed":
                raise RuntimeError("diagnostic unavailable")
            original_emit(event_type, **options)

        with patch.object(trace, "emit", side_effect=fail_event):
            with self.assertRaisesRegex(
                RuntimeError, "diagnostic unavailable"
            ):
                self._ask_with_real_save("Saved despite trace", _intent())
        saved = self._conversation()["resume_state"]
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(self.session._checkpoint_id,
                         saved["checkpoint_id"])
        self.assertEqual(
            saved["turns"][0]["question"], "Saved despite trace"
        )
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "failed",
        )

    def test_checkpoint_event_write_warning_keeps_committed_answer(
        self,
    ) -> None:
        """Return the answer when optional post-save trace I/O fails."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_WARN)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Refuse the checkpoint event after its state is durable."""
            if trace._events[-1]["event_type"] == (
                "state.checkpoint.committed"
            ):
                raise OSError("diagnostic write failed")
            write()

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            answer = self._ask_with_real_save(
                "Saved despite warning", _intent()
            )
        self.assertEqual(answer, "model answer")
        self.assertEqual(len(self.session.turns), 1)
        self.assertTrue(any(
            "diagnostic write failed" in str(item.message)
            for item in captured
        ))
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )

    def test_checkpoint_event_write_strict_failure_keeps_state(
        self,
    ) -> None:
        """Raise for strict diagnostics without undoing the checkpoint."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_RAISE)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Refuse the checkpoint event but permit the state write."""
            if trace._events[-1]["event_type"] == (
                "state.checkpoint.committed"
            ):
                raise OSError("strict diagnostic failure")
            write()

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Could not write KíLín trace"
            ):
                self._ask_with_real_save(
                    "Saved despite strict failure", _intent()
                )
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )

    def test_intermediate_callback_failure_remains_a_failed_turn(
        self,
    ) -> None:
        """Propagate an observer error before the checkpoint is saved."""
        self._bind_trace()
        observed_phases: list[str] = []

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Interrupt the answer during evidence retrieval."""
            observed_phases.append(phase)
            if phase == rag.ANSWER_PHASE_RETRIEVING:
                raise RuntimeError("early notification failed")

        with self.assertRaisesRegex(
            RuntimeError, "early notification failed"
        ):
            self._ask_with_real_save(
                "Interrupted answer", _intent(),
                phase_callback=phase_callback,
            )
        self.assertIn(rag.ANSWER_PHASE_RETRIEVING, observed_phases)
        self.assertNotIn(rag.ANSWER_PHASE_COMPLETED, observed_phases)
        self.assertEqual(self.session.turns, [])
        self.assertEqual(self._conversation()["resume_state"], {})
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "failed",
        )

    def test_completed_callback_sees_committed_and_completed_turn(
        self,
    ) -> None:
        """Notify completion after saving and closing the successful span."""
        self._bind_trace()
        completed_state: list[tuple[str, int, object]] = []

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Inspect the stored trace at the completion notification."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                turn = self._conversation()["turns"]["turn_0001"]
                turn_status = (
                    turn["stages"]["01_question"]["operations"][
                        "01_turn"
                    ]["event"]["status"]
                )
                completed_state.append((
                    turn_status,
                    len(self._conversation()["resume_state"]["turns"]),
                    details["evidence_count"],
                ))

        answer = self._ask_with_real_save(
            "Completed notification", _intent(),
            phase_callback=phase_callback,
        )
        self.assertEqual(answer, "model answer")
        self.assertEqual(completed_state, [("completed", 1, 1)])

    def test_late_callback_failure_returns_committed_answer(self) -> None:
        """Record an observer error without marking the saved turn failed."""
        self._bind_trace()

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Fail only after the conversation checkpoint is committed."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("UI callback failed")

        answer = self._ask_with_real_save(
            "Callback failure", _intent(),
            phase_callback=phase_callback,
        )
        self.assertEqual(answer, "model answer")
        turn = self._conversation()["turns"]["turn_0001"]
        operations = turn["stages"]["01_question"]["operations"]
        self.assertEqual(
            operations["01_turn"]["event"]["status"], "completed"
        )
        saved = turn["stages"]["05_generation"]["operations"][
            "01_state_checkpoint_committed"
        ]
        self.assertEqual(saved["event"]["status"], "completed")
        self.assertEqual(
            len(self._conversation()["resume_state"]["turns"]), 1
        )
        self.assertEqual(len(self.session.turns), 1)
        notification = [
            operation for key, operation in operations.items()
            if key.endswith("turn_notification_failed")
        ]
        self.assertEqual(len(notification), 1)
        self.assertEqual(notification[0]["event"]["status"], "failed")
        self.assertEqual(
            notification[0]["details"]["phase"],
            rag.ANSWER_PHASE_COMPLETED,
        )
        self.assertEqual(
            notification[0]["details"]["error_type"], "RuntimeError"
        )
        self.assertEqual(
            notification[0]["details"]["message"], "UI callback failed"
        )
        self.assertIn(
            "UI callback failed", notification[0]["details"]["traceback"]
        )

        first_checkpoint = self.session._checkpoint_id
        self._ask_with_real_save("Second answer", _intent())
        self.assertEqual(len(self.session.turns), 2)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint"][
                "parent_checkpoint_id"
            ],
            first_checkpoint,
        )

    def test_late_callback_failure_without_tracing_returns_answer(
        self,
    ) -> None:
        """Allow a completed turn to succeed with tracing disabled."""
        phases: list[str] = []

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Record the phase and fail only on final notification."""
            phases.append(phase)
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("optional notification unavailable")

        answer = self._ask_with_real_save(
            "Untraced callback failure", _intent(),
            phase_callback=phase_callback,
        )
        self.assertEqual(answer, "model answer")
        self.assertEqual(phases[-1], rag.ANSWER_PHASE_COMPLETED)
        self.assertEqual(len(self.session.turns), 1)
        saved = self._conversation()["resume_state"]["turns"]
        self.assertEqual(saved[0]["question"], "Untraced callback failure")

    def test_tool_answer_survives_final_callback_failure(self) -> None:
        """Keep a tool-written answer when the final observer fails."""
        self._bind_trace()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )

        def phase_callback(
            phase: str, details: dict[str, object]
        ) -> None:
            """Fail only the post-checkpoint presentation notification."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("tool view disconnected")

        answer = self._ask_with_real_save(
            "Tool output", intent, phase_callback=phase_callback
        )
        self.assertEqual(answer, "tool answer")
        self.session.chat_session.reply.assert_not_called()
        self.assertEqual(
            self._conversation()["resume_state"]["turns"][0]["answer"],
            "tool answer",
        )
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertEqual(
            turn["stages"]["01_question"]["operations"][
                "01_turn"
            ]["event"]["status"],
            "completed",
        )

    def test_notification_trace_write_warning_keeps_saved_answer(
        self,
    ) -> None:
        """Ignore optional failed-event I/O under warn policy."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_WARN)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Simulate a diagnostic write failure after the checkpoint."""
            if trace._events[-1]["event_type"] == (
                "turn.notification.failed"
            ):
                raise OSError("trace disk unavailable")
            write()

        def fail_when_completed(
            phase: str, details: dict[str, object]
        ) -> None:
            """Raise an ordinary presentation exception after commit."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("observer failed")

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            answer = self._ask_with_real_save(
                "Warn on failed event", _intent(),
                phase_callback=fail_when_completed,
            )
        self.assertEqual(answer, "model answer")
        self.assertTrue(any(
            "trace disk unavailable" in str(item.message)
            for item in captured
        ))
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "completed",
        )

    def test_notification_trace_write_strict_policy_propagates(
        self,
    ) -> None:
        """Preserve strict trace errors after a committed RAG answer."""
        trace = self._bind_trace(failure_policy=FAILURE_POLICY_RAISE)
        write = trace._write_readable_trace

        def write_or_fail() -> None:
            """Refuse the notification-error diagnostic event only."""
            if trace._events[-1]["event_type"] == (
                "turn.notification.failed"
            ):
                raise OSError("trace disk unavailable")
            write()

        def fail_when_completed(
            phase: str, details: dict[str, object]
        ) -> None:
            """Raise after commit to trigger strict diagnostic writing."""
            if phase == rag.ANSWER_PHASE_COMPLETED:
                raise RuntimeError("observer failed")

        with patch.object(
            trace, "_write_readable_trace", side_effect=write_or_fail
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Could not write KíLín trace"
            ):
                self._ask_with_real_save(
                    "Strict failed event", _intent(),
                    phase_callback=fail_when_completed,
                )
        self.assertEqual(len(self.session.turns), 1)
        self.assertEqual(
            self._conversation()["resume_state"]["checkpoint_id"],
            self.session._checkpoint_id,
        )
        self.assertEqual(
            self._conversation()["turns"]["turn_0001"]["stages"][
                "01_question"
            ]["operations"]["01_turn"]["event"]["status"],
            "completed",
        )

    def test_tool_answer_commits_trace_without_chat_reply(self) -> None:
        """Persist tool-produced turns without invoking the chat model."""
        self._bind_trace()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )
        answer = self._ask_with_real_save("Inventory", intent)
        self.assertEqual(answer, "tool answer")
        self.session.chat_session.reply.assert_not_called()
        turn = self._conversation()["turns"]["turn_0001"]
        self.assertIn("05_generation", turn["stages"])
        self.assertEqual(
            self._conversation()["resume_state"]["turns"][0]["answer"],
            "tool answer",
        )


if __name__ == "__main__":
    unittest.main()
