"""Exercise incomplete model results across tools and conversation storage.

The coordinator must save an explicit failure notice, never a successful
empty search or thinking-only answer. All model calls are simulated.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from kilin import rag
from kilin._model_completion import (
    GenerationCancelled,
    IncompleteModelResponse,
)
from kilin.answering import RouteDecision
from kilin.project_type import RETRIEVAL_MODE_NONE
from kilin.tools import ToolResult, ToolUse
from tests.kilin._rag_support import (
    _answer_patches,
    _intent,
    _session,
    _turn,
)


class TestRAGCompletion(unittest.TestCase):
    """Do not confuse a completed request with a completed model answer."""

    def test_direct_truncation_is_saved_as_incomplete(self) -> None:
        """Keep usage and termination metadata without publishing a draft."""

        session = _session()
        session.chat_session.reply.return_value = "<think>draft"
        session.chat_session.last_finish_reason = "length"
        streamed = []
        with _answer_patches(session, _intent()):
            answer = session.ask(
                "Question", use_router=False, token_callback=streamed.append
            )
        turn = session.turns[-1]
        self.assertIn("could not be completed", answer)
        self.assertNotIn("<think>", answer)
        self.assertEqual(turn.evidence, ())
        self.assertEqual(turn.parameters["completion"]["status"], "incomplete")
        self.assertEqual(
            turn.parameters["completion"]["reason"], "token_limit"
        )
        self.assertEqual(turn.parameters["finish_reason"], "length")
        self.assertIn("usage", turn.parameters)
        self.assertIn("</think>", "".join(streamed))
        session.chat_session.reply.assert_called_once()

    def test_stopped_answer_saves_nothing(self) -> None:
        """Stop ends the answer, keeps no turn, and restores the history."""

        session = _session()
        before = [dict(message) for message in session.chat_session.messages]
        session.chat_session.reply.side_effect = GenerationCancelled(
            "The answer was stopped."
        )
        stop = Mock(return_value=True)
        with _answer_patches(session, _intent()):
            with self.assertRaises(GenerationCancelled):
                session.ask("Question", use_router=False, stop_requested=stop)
        self.assertEqual(session.turns, [])
        self.assertIs(session.chat_session.stop_requested, stop)
        session.chat_session.restore_history.assert_called_with(before)

    def test_failed_tool_does_not_publish_a_negative_result(self) -> None:
        """Stop at the first incomplete internal call without extra calls."""

        session = _session()
        session.chat_session.reply.return_value = "<think>classify"
        session.chat_session.last_finish_reason = "length"
        consumed = Mock()

        def run(use, request, context):
            """Attempt a classification before creating its result."""

            reply = context.ask_model("s", "p", 120)
            consumed(reply)
            return ToolResult("0 matches", (), {"name": use.name})

        with patch.object(rag, "run_tool", side_effect=run):
            result = session._run_tool(
                ToolUse("find_passages", object()), "Find passages",
                RouteDecision("topic_listing", True),
                trace_turn_id=None, report_phase=Mock(),
            )
        consumed.assert_not_called()
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertEqual(result.summary["finish_reason"], "length")
        self.assertNotIn("0 matches", result.answer)
        self.assertEqual(result.evidence, ())
        session.chat_session.reply.assert_called_once()

    def test_tool_writer_empty_final_is_incomplete(self) -> None:
        """Preserving thinking must not mask the absence of final content."""

        session = _session()
        session.chat_session.reply.return_value = "<think>done</think>"
        session.chat_session.last_finish_reason = "stop"

        def run(use, request, context):
            """Produce the user-visible tool reply."""

            return ToolResult(
                context.write("s", "p", 500), (), {"name": use.name},
                streamed=True,
            )

        with patch.object(rag, "run_tool", side_effect=run):
            result = session._run_tool(
                ToolUse("paper_overview", object()), "Overview",
                RouteDecision("paper_overview", True),
                trace_turn_id=None, report_phase=Mock(),
            )
        self.assertEqual(result.summary["reason"], "empty_answer")
        self.assertTrue(result.answer.strip())
        self.assertFalse(result.streamed)

    def test_incomplete_tool_status_reaches_saved_turn(self) -> None:
        """Save the tool's failure status with normal turn usage metadata."""

        session = _session()
        use = ToolUse("paper_overview", object())
        intent = _intent(RETRIEVAL_MODE_NONE, tool=use)
        with _answer_patches(session, intent) as mocks:
            mocks.run.return_value = ToolResult(
                "Incomplete operation", (), {
                    "name": "paper_overview", "status": "incomplete",
                    "reason": "empty_answer", "finish_reason": "stop",
                },
            )
            session.ask("Overview", use_router=False)
        self.assertEqual(
            session.turns[-1].parameters["completion"]["reason"],
            "empty_answer",
        )
        session.chat_session.reply.assert_not_called()

    def test_real_checkpoint_reopens_and_next_question_can_run(self) -> None:
        """Keep failed turns inspectable but out of restored model history."""

        with TemporaryDirectory() as directory:
            session = _session(Path(directory))
            session.conversation_path.parent.mkdir(parents=True)
            session.turns = [_turn("Old question", "<think>unfinished")]
            session.chat_session.reply.return_value = "<think>new draft"
            session.chat_session.last_finish_reason = "length"
            with _answer_patches(session, _intent()) as mocks:
                mocks.save.side_effect = lambda **kwargs: (
                    rag.RAGSession._save_conversation(session, **kwargs)
                )
                session.ask("Overview", use_router=False)
            messages, turns, checkpoint = session._load_conversation()
            self.assertEqual(len(turns), 2)
            self.assertTrue(checkpoint)
            self.assertEqual(len(messages), 1)
            self.assertTrue(all(item["content"].strip() for item in messages))
            session.chat_session.reply.return_value = "Completed answer."
            session.chat_session.last_finish_reason = "stop"
            with _answer_patches(session, _intent()):
                answer = session.ask("Compare", use_router=False)
            self.assertEqual(answer, "Completed answer.")
            for call in session.chat_session.restore_history.call_args_list:
                self.assertTrue(
                    all(m["content"].strip() for m in call.args[0])
                )

    def test_empty_backend_response_is_reported_without_a_crash(self) -> None:
        """Handle the typed failure raised by ChatSession for no text."""

        session = _session()
        session.chat_session.reply.side_effect = IncompleteModelResponse(
            "The model returned no text.", reason="empty_answer"
        )
        with _answer_patches(session, _intent()):
            answer = session.ask("Question", use_router=False)
        self.assertTrue(answer)
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(
            session.turns[0].parameters["completion"]["reason"], "empty_answer"
        )


if __name__ == "__main__":
    unittest.main()
