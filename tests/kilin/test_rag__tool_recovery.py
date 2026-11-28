"""Test bounded nonthinking recovery in the actual tool model adapter.

Keep the original request, user allowance and settings; count both attempts
and never feed truncated text to a parser or later model request.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, PropertyMock, patch

from kilin import rag
from kilin._model_completion import IncompleteModelResponse
from kilin.answering import RouteDecision
from kilin.project_type import RETRIEVAL_MODE_NONE
from kilin.tools import ToolResult, ToolUse, registered_tools
from kilin.tools._model_calls import ModelCallSettings
from kilin.tools._shared.model_results import partial_result
from tests.kilin.test_answering import _FakeChat
from tests.kilin._rag_support import _answer_patches, _intent, _session
from tests.kilin.test_rag__tool_thinking import _run


class _ScriptChat(_FakeChat):
    """Return deterministic text/finish pairs or raise a backend error."""

    def __init__(self, replies, *, thinking=True, ceiling=2048):
        """Initialize the scripted backend and original settings."""

        super().__init__()
        self.replies = iter(replies)
        self.thinking = thinking
        self.max_new_tokens = ceiling

    def reply(self, message, **options):
        """Record the request under the current state before responding."""

        reply = next(self.replies)
        if isinstance(reply, Exception):
            self.reply_error = reply
        else:
            self.response, self.last_finish_reason = reply
        return super().reply(message, **options)


class TestToolRecovery(unittest.TestCase):
    """Cover the trigger, budget, state, metadata and streaming contracts."""

    def test_retry_keeps_original_request_and_user_allowance(self):
        chat = _ScriptChat([
            ("<think>unfinished", "length"), ("Valid reply", "stop"),
        ])
        history = [dict(message) for message in chat.messages]
        result, _ = _run("reference_check", [128], chat=chat)
        first, second = chat.observations
        self.assertEqual(
            [first["thinking"], second["thinking"]], [True, False]
        )
        for call in (first, second):
            self.assertEqual(call["max_new_tokens"], 2048)
            self.assertEqual(call["temperature"], 0.6)
            self.assertEqual(call["message"], "prompt")
            self.assertEqual(call["history"],
                             [{"role": "system", "content": "system"}])
        self.assertEqual(chat.messages, history)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.max_new_tokens, 2048)
        calls = result.summary["internal_generation"]["calls"]
        self.assertEqual([call["attempt"] for call in calls], [1, 2])
        self.assertEqual([call["status"] for call in calls],
                         ["incomplete", "complete"])
        self.assertNotIn("unfinished", result.answer)
        self.assertNotEqual(result.summary.get("status"), "incomplete")

    def test_failed_retry_stops_after_two_attempts(self):
        chat = _ScriptChat([
            ("<think>unfinished", "length"), ("Still truncated", "length"),
        ])
        result, _ = _run("reference_check", [None], chat=chat)
        self.assertEqual(len(chat.observations), 2)
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertEqual(result.summary["reason"], "token_limit")
        self.assertNotIn("Still truncated", result.answer)
        self.assertTrue(chat.thinking)

    def test_other_failures_are_not_retried(self):
        cases = [
            ("find_passages", "<think>unfinished", "length"),
            ("reference_check", "<think>done</think>Partial", "length"),
            ("reference_check", "<think>unfinished", "stop"),
            ("reference_check", "<think>done</think>", "stop"),
        ]
        for tool, text, finish in cases:
            with self.subTest(tool=tool, text=text, finish=finish):
                chat = _ScriptChat([(text, finish)])
                result, _ = _run(tool, [None], chat=chat)
                self.assertEqual(len(chat.observations), 1)
                self.assertEqual(result.summary["status"], "incomplete")

    def test_forced_thinking_recovery_restores_nonthinking_session(self):
        chat = _ScriptChat([
            ("<think>unfinished", "length"), ("Recovered", "stop"),
            ("<think>finished</think>Next", "stop"),
        ], thinking=False, ceiling=256)
        with patch.object(
            type(registered_tools()["reference_check"]), "model_calls",
            new_callable=PropertyMock,
            return_value=ModelCallSettings(thinking="on"),
        ):
            result, _ = _run("reference_check", [128, None], chat=chat)
        self.assertEqual([call["thinking"] for call in chat.observations],
                         [True, False, True])
        self.assertEqual(
            [call["max_new_tokens"] for call in chat.observations],
            [256, 256, 256],
        )
        self.assertFalse(chat.thinking)
        self.assertEqual(result.summary["internal_generation"]["policy"],
                         {"thinking": "on"})

    def test_final_streaming_answer_is_not_retried(self):
        chat = _ScriptChat([("<think>unfinished", "length")])
        result, _ = _run("reference_check", [], chat=chat, write=True)
        self.assertEqual(len(chat.observations), 1)
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertNotIn("internal_generation", result.summary)
        self.assertTrue(chat.thinking)

    def test_each_new_request_restores_the_configured_policy(self):
        chat = _ScriptChat([
            ("<think>unfinished", "length"), ("Recovered", "stop"),
            ("<think>done</think>Next result", "stop"),
        ])
        result, _ = _run("reference_check", [None, None], chat=chat)
        self.assertEqual([c["thinking"] for c in chat.observations],
                         [True, False, True])
        calls = result.summary["internal_generation"]["calls"]
        self.assertEqual([c["attempt"] for c in calls], [1, 2, 1])
        self.assertNotIn("unfinished", result.answer)
        self.assertIn("done", result.answer)

    def test_backend_error_on_retry_restores_settings(self):
        chat = _ScriptChat([
            ("<think>unfinished", "length"), RuntimeError("backend failed"),
        ])
        history = [dict(message) for message in chat.messages]
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            _run("reference_check", [None], chat=chat)
        self.assertEqual(len(chat.observations), 2)
        self.assertEqual(chat.messages, history)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)

    def test_context_exposes_recovery_attempt_count(self):
        """Share adapter attempts with a tool without an extra model call."""

        session = _session()
        session.chat_session = _ScriptChat([
            ("<think>unfinished", "length"), ("Recovered", "stop"),
        ])

        def run(use, request, context):
            """Observe the adapter counter around a single request."""

            before = context.model_attempt_count()
            answer = context.ask_model("System", "Original request", None)
            after = context.model_attempt_count()
            return ToolResult(
                answer, (), {"name": use.name, "attempts": after - before}
            )

        with patch.object(rag, "run_tool", side_effect=run):
            result = session._run_tool(
                ToolUse("reference_check", object()),
                "Check it", RouteDecision("test", True),
                trace_turn_id=None, report_phase=Mock(),
            )
        self.assertEqual(result.summary["attempts"], 2)
        self.assertEqual(
            len(result.summary["internal_generation"]["calls"]), 2
        )
        self.assertEqual(len(session.chat_session.observations), 2)

    def test_real_chat_counts_both_attempts(self):
        session = _session()
        with patch.object(rag.ChatSession, "_load"):
            chat = rag.ChatSession(
                "test-model", backend="torch", thinking=True,
                max_new_tokens=2048, block_unrequested_cjk=False,
            )
        self.addCleanup(chat.close)
        chat._model = object()
        chat._tokenizer = Mock()
        chat._tokenizer.apply_chat_template.return_value = [1, 2, 3]
        chat._tokenizer.encode.return_value = [4]
        session.chat_session = chat
        replies = iter([("<think>unfinished", "length"), ("Answer", "stop")])

        def generate(*args, **kwargs):
            text, chat.last_finish_reason = next(replies)
            chat.output_token_total += 4
            chat._last_generation_output_token_count = 4
            return text

        def run(use, request, context):
            answer = context.ask_model("System", "Original request", None)
            return ToolResult(answer, (), {"name": use.name})

        use = ToolUse("reference_check", object())
        with (
            _answer_patches(session, _intent(RETRIEVAL_MODE_NONE, tool=use))
            as mocks,
            patch.object(rag, "run_tool", side_effect=run),
            patch.object(
                chat, "_torch_reply", side_effect=generate
            ) as backend,
        ):
            mocks.run.side_effect = lambda *args, **kwargs: (
                rag.RAGSession._run_tool(session, *args, **kwargs)
            )
            answer = session.ask("Check it", use_router=False)
        self.assertEqual(answer, "Answer")
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(chat.input_token_total, 6)
        self.assertEqual(chat.output_token_total, 8)
        usage = session.turns[-1].parameters["usage"]
        self.assertEqual(usage["input_tokens"], 6)
        self.assertEqual(usage["output_tokens"], 8)

    def test_partial_suffix_is_streamed_once(self):
        session = _session()
        session.chat_session = _ScriptChat([("Complete text", "stop")])
        streamed = []
        failure = {"item": "Reading part 2", "message": "Token limit",
                   "reason": "token_limit"}

        def run(use, request, context):
            answer = context.write("System", "Final", None)
            return partial_result(
                ToolResult(answer, (), {"name": use.name}, streamed=True),
                [failure],
            )

        with patch.object(rag, "run_tool", side_effect=run):
            result = session._run_tool(
                ToolUse("batch_review", object()), "Question",
                RouteDecision("document_reading", True),
                trace_turn_id=None, report_phase=Mock(),
                token_callback=streamed.append,
            )
        self.assertEqual("".join(streamed), result.answer)
        self.assertEqual(result.answer.count("Incomplete items"), 1)
        self.assertEqual(result.summary["status"], "partial")

    def test_partial_checkpoint_keeps_completed_content_and_warning(self):
        with TemporaryDirectory() as directory:
            session = _session(Path(directory))
            session.conversation_path.parent.mkdir(parents=True)
            use = ToolUse("reference_check", object())
            result = partial_result(ToolResult("Valid result", (), {
                "name": use.name,
            }), [{"item": "Claim 2", "message": "Token limit"}])
            with _answer_patches(
                session, _intent(RETRIEVAL_MODE_NONE, tool=use)
            ) as mocks:
                mocks.run.return_value = result
                mocks.save.side_effect = lambda **kwargs: (
                    rag.RAGSession._save_conversation(session, **kwargs)
                )
                session.ask("Check references", use_router=False)
            messages, turns, checkpoint = session._load_conversation()
            self.assertTrue(checkpoint)
            self.assertEqual(turns[0].parameters["completion"]["status"],
                             "partial")
            self.assertEqual(turns[0].answer, result.answer)
            self.assertTrue(any("Valid result" in m["content"]
                                for m in messages))
            self.assertTrue(any("Incomplete items" in m["content"]
                                for m in messages))


if __name__ == "__main__":
    unittest.main()
