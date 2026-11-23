"""Exercise tool-owned thinking policies through the real RAG side adapter.

Use a deterministic chat implementation, shipped tool policies and simulated
runners. The same adapter serves GUI and programmatic calls.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, PropertyMock, patch

from kilin import rag
from kilin.answering import RouteDecision
from kilin.tools import Tool, ToolResult, ToolUse
from kilin.tools._model_calls import load_model_call_settings
from tests.kilin.test_answering import _FakeChat
from tests.kilin._rag_support import _session


def _run(name, budgets, *, chat=None, write=False):
    """Run actual side adapters under a deterministic tool plan.

    Parameters
    ----------
    name: str
        Registered tool whose policy applies.
    budgets: list[int | None]
        Requested limits for internal calls.
    chat: _FakeChat | None, optional
        Chat implementation; defaults to a successful one.
    write: bool, optional
        Also generate a final answer with ``context.write``.

    Returns
    -------
    tuple[ToolResult, _FakeChat]
        Result metadata and observed model calls.

    """

    session = _session()
    session.chat_session = chat or _FakeChat(response="Result")
    if not hasattr(session.chat_session, "max_new_tokens"):
        session.chat_session.max_new_tokens = 8192

    def run(use, request, context):
        """Call the supplied model services as a real tool would."""

        for budget in budgets:
            context.ask_model("system", "prompt", budget)
        answer = context.write("system", "answer", 1024) if write else "Done"
        return ToolResult(answer, (), {"name": use.name, "model_calls": 7})

    with patch.object(rag, "run_tool", side_effect=run):
        result = session._run_tool(
            ToolUse(name, object()), "Question", RouteDecision("test", True),
            trace_turn_id=None, report_phase=Mock(),
        )
    return result, session.chat_session


class TestToolThinking(unittest.TestCase):
    """Check actual generation settings, metadata and failure isolation."""

    def test_classification_defaults_disable_internal_thinking(self) -> None:
        for tool in ("find_passages", "filter_results", "batch_classify"):
            with self.subTest(tool=tool):
                result, chat = _run(tool, [120, 1000])
                self.assertEqual(
                    [x["thinking"] for x in chat.observations], [False, False]
                )
                self.assertTrue(chat.thinking)
                self.assertEqual(result.summary["model_calls"], 7)
                self.assertEqual(len(chat.observations), 2)
                calls = result.summary["internal_generation"]["calls"]
                self.assertEqual(
                    [x["thinking"] for x in calls], [False, False]
                )
                self.assertEqual([x["status"] for x in calls],
                                 ["complete", "complete"])

    def test_mixed_analysis_and_final_write_use_separate_policy(self) -> None:
        result, chat = _run(
            "generate_paper_summary", [60, 500, 600], write=True
        )
        self.assertEqual(
            [x["thinking"] for x in chat.observations],
            [False, False, False, True],
        )
        self.assertEqual(
            [x["max_new_tokens"] for x in chat.observations],
            [60, 500, 600, 8192],
        )
        self.assertTrue(
            all(x["temperature"] == 0.6 for x in chat.observations)
        )
        self.assertEqual(
            len(result.summary["internal_generation"]["calls"]), 3
        )
        self.assertTrue(chat.thinking)

    def test_uncapped_reference_judgments_inherit(self) -> None:
        for tool in ("reference_check", "suggest_reference"):
            for original in (True, False):
                with self.subTest(tool=tool, original=original):
                    chat = _FakeChat(response="Result")
                    chat.thinking = original
                    _, chat = _run(tool, [None, 120], chat=chat)
                    self.assertIs(chat.observations[0]["thinking"], original)
                    self.assertEqual(
                        chat.observations[0]["max_new_tokens"], 8192
                    )
                    self.assertIs(chat.observations[1]["thinking"], original)
                    self.assertIs(chat.thinking, original)

    def test_show_source_keeps_narrative_thinking(self) -> None:
        _, chat = _run("show_source", [400])
        self.assertTrue(chat.observations[0]["thinking"])

    def test_session_ceiling_does_not_reclassify_a_long_request(self) -> None:
        chat = _FakeChat(response="Result")
        chat.max_new_tokens = 100
        result, chat = _run("generate_paper_summary", [600], chat=chat)
        self.assertFalse(chat.observations[0]["thinking"])
        self.assertEqual(chat.observations[0]["max_new_tokens"], 100)
        call = result.summary["internal_generation"]["calls"][0]
        self.assertEqual(call["requested_max_new_tokens"], 600)

    def test_json_can_reset_or_force_thinking(self) -> None:
        with TemporaryDirectory() as directory:
            for index, (policy, original, expected) in enumerate([
                ("inherit", True, True), ("on", False, True),
                ("off", True, False),
            ]):
                with self.subTest(policy=policy):
                    path = Path(directory) / f"defaults_{index}.json"
                    path.write_text(json.dumps({"MODEL_CALLS": {
                        "THINKING": policy,
                    }}), encoding="utf-8")
                    configured = load_model_call_settings(path)
                    chat = _FakeChat(response="Result")
                    chat.thinking = original
                    with patch.object(Tool, "model_calls", new_callable=(
                        PropertyMock
                    ), return_value=configured):
                        _, chat = _run("find_passages", [120], chat=chat)
                    self.assertIs(chat.observations[0]["thinking"], expected)
                    self.assertIs(chat.thinking, original)

    def test_incomplete_override_is_recorded_without_retry(self) -> None:
        chat = _FakeChat(response="<think>unfinished")
        chat.last_finish_reason = "length"
        result, chat = _run("find_passages", [120, 120], chat=chat)
        self.assertEqual(result.summary["status"], "incomplete")
        calls = result.summary["internal_generation"]["calls"]
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["thinking"])
        self.assertEqual(calls[0]["reason"], "token_limit")
        self.assertEqual(len(chat.observations), 1)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)
        self.assertNotIn("0 matches", result.answer)
        self.assertEqual(result.evidence, ())


if __name__ == "__main__":
    unittest.main()
