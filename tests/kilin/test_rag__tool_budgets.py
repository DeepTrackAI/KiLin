"""Propagate user-selected budgets through the shared tool coordinator.

Use all shipped policies and the real ChatSession/RAG adapters with backend
inference replaced by deterministic replies. No GUI or model is loaded.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from kilin import rag
from kilin.project_type import RETRIEVAL_MODE_NONE
from kilin.tools import TOOL_REGISTRY, ToolResult, ToolUse
from tests.kilin.test_rag import _answer_patches, _intent, _session
from tests.kilin.test_rag__tool_thinking import _run
from tests.kilin.test_answering import _FakeChat


class TestToolUserBudgets(unittest.TestCase):
    """Use the interface budget without changing the model or call count."""

    def test_all_tool_adapters_inherit_user_ceiling(self) -> None:
        off = {"find_passages", "filter_results", "batch_classify"}
        for name in TOOL_REGISTRY:
            for thinking in (True, False):
                for ceiling in (128, 2048, 16384):
                    with self.subTest(tool=name, thinking=thinking,
                                      ceiling=ceiling):
                        chat = _FakeChat(response="Result")
                        chat.thinking = thinking
                        chat.max_new_tokens = ceiling
                        result, chat = _run(name, [None], chat=chat)
                        call = chat.observations[0]
                        self.assertEqual(call["max_new_tokens"], ceiling)
                        self.assertIs(call["thinking"],
                                      thinking if name not in off else False)
                        self.assertEqual(call["temperature"], 0.6)
                        self.assertEqual(len(chat.observations), 1)
                        self.assertIs(chat.thinking, thinking)
                        self.assertEqual(
                            result.summary["internal_generation"]["policy"],
                            {"thinking": "off" if name in off else "inherit"},
                        )

    def test_per_question_interface_settings_reach_real_chat(self) -> None:
        """Exercise the same RAG.ask generation update used by the GUI.

        """

        session = _session()
        with patch.object(rag.ChatSession, "_load"):
            chat = rag.ChatSession(
                "test-model", backend="torch", thinking=True,
                max_new_tokens=16, block_unrequested_cjk=False,
            )
        self.addCleanup(chat.close)
        chat._model = object()
        chat._tokenizer = Mock()
        chat._tokenizer.apply_chat_template.return_value = [1, 2, 3]
        chat._tokenizer.encode.return_value = [4]
        session.chat_session = chat
        use = ToolUse("generate_paper_summary", object())
        intent = _intent(RETRIEVAL_MODE_NONE, tool=use)

        def run(_use, _request, context):
            """Exercise short metadata, analysis and final generation."""

            context.ask_model("System", "Metadata", 128)
            context.ask_model("System", "Analysis", None)
            answer = context.write("System", "Final", None)
            return ToolResult(answer, (), {"name": _use.name})

        for ceiling, thinking in ((2048, True), (16384, True),
                                  (4096, False), (64, False)):
            with self.subTest(ceiling=ceiling, thinking=thinking):
                with (
                    _answer_patches(session, intent) as mocks,
                    patch.object(rag, "run_tool", side_effect=run),
                    patch.object(chat, "_torch_reply",
                                 return_value="Complete answer") as generate,
                ):
                    mocks.run.side_effect = lambda *args, **kwargs: (
                        rag.RAGSession._run_tool(session, *args, **kwargs)
                    )
                    answer = session.ask(
                        "A new request", use_router=False,
                        generation_settings={
                            "max_new_tokens": ceiling,
                            "thinking": thinking, "temperature": 0.25,
                        },
                    )
                expected = ceiling if thinking else min(128, ceiling)
                self.assertEqual(
                    [call.args[1] for call in generate.call_args_list],
                    [expected, ceiling, ceiling],
                )
                self.assertEqual(answer, "Complete answer")
                self.assertEqual(chat.max_new_tokens, ceiling)
                self.assertIs(chat.thinking, thinking)
                self.assertEqual(chat.temperature, 0.25)
                self.assertEqual(
                    session.turns[-1].parameters["max_new_tokens"], ceiling
                )


if __name__ == "__main__":
    unittest.main()
