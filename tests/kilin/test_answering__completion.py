"""Check side-request completion and safe restoration of failed turns.

Use the same deterministic chat fixture as the answering tests. Preserve
router behavior while enforcing the stricter tool completion contract.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import patch

from kilin import answering
from kilin._model_completion import (
    IncompleteModelResponse,
    thinking_is_open,
)
from tests.kilin.test_chat__unit import _session
from tests.kilin.test_answering import _FakeChat, _turn


class TestSideReplyCompletion(unittest.TestCase):
    """Keep failed calls isolated from the ongoing conversation."""

    def test_failed_completion_restores_history_and_settings(self) -> None:
        """Capture metadata before restoring temporary session state."""

        for inherit in (False, True):
            with self.subTest(inherit=inherit):
                chat = _FakeChat(response="<think>unfinished")
                chat.last_finish_reason = "length"
                history = [dict(item) for item in chat.messages]
                streamed = []
                with self.assertRaises(IncompleteModelResponse) as failure:
                    answering.side_reply(
                        chat, "system", "prompt", max_new_tokens=120,
                        inherit_settings=inherit, preserve_thinking=True,
                        require_complete=True, token_callback=streamed.append,
                    )
                self.assertEqual(failure.exception.finish_reason, "length")
                self.assertEqual(chat.messages, history)
                self.assertTrue(chat.thinking)
                self.assertEqual(chat.temperature, 0.6)
                self.assertEqual(len(chat.observations), 1)
                self.assertEqual(streamed, ["<think>unfinished"])

    def test_complete_reply_keeps_thinking_preservation(self) -> None:
        """A successful user-facing tool reply remains unmodified."""

        chat = _FakeChat()
        chat.last_finish_reason = "stop"
        self.assertEqual(
            answering.side_reply(
                chat, "system", "prompt", max_new_tokens=500,
                inherit_settings=True, preserve_thinking=True,
                require_complete=True,
            ),
            chat.response,
        )

    def test_router_default_does_not_acquire_tool_failure_policy(self) -> None:
        """Retain existing router fallback behavior without stricter flags."""

        chat = _FakeChat(response="<think>unfinished")
        chat.last_finish_reason = "length"
        self.assertEqual(
            answering.side_reply(chat, "s", "q", max_new_tokens=64), ""
        )

    def test_history_omits_legacy_and_marked_incomplete_pairs(self) -> None:
        """Keep stored numbering but never restore an empty assistant."""

        turns = [
            _turn("Q1", "<think>unfinished"),
            _turn("Q2", "<think>complete</think>"),
            _turn("Q3", "Incomplete operation"),
            _turn("Q4", "Usable answer"),
        ]
        turns[2].parameters["completion"] = {"status": "incomplete"}
        history = answering.history_messages(turns, [1, 2, 3, 4])
        self.assertEqual(history, [
            {"role": "user", "content": "Q4"},
            {"role": "assistant", "content": "Usable answer"},
        ])
        self.assertEqual(len(turns), 4)
        self.assertEqual(turns[0].answer, "<think>unfinished")


class TestImplicitThinkingPrefix(unittest.TestCase):
    """Check template-opened reasoning without loading an actual model."""

    @staticmethod
    def _run(raw: str, *, thinking: bool, finish: str,
             prompt_open: bool) -> tuple[str | None, Exception | None,
                                         int, object]:
        """Exercise the actual chat and side-reply completion boundary.

        Parameters
        ----------
        raw: str
            Generated text, excluding the prompt's assistant prefix.
        thinking: bool
            Whether thinking is enabled for the request.
        finish: str
            Simulated model termination reason.
        prompt_open: bool
            Whether the rendered assistant prefix ends with ``<think>``.

        Returns
        -------
        tuple[str | None, Exception | None, int, object]
            Returned reply or failure, generation count and chat session.

        """

        chat = _session(
            thinking=thinking, max_new_tokens=100,
            block_unrequested_cjk=False,
        )
        history = [dict(message) for message in chat.messages]

        def template(*args, tokenize, **kwargs):
            """Represent either a Qwen3 or Qwen3.8 chat template."""

            if tokenize:
                return [4, 5, 6]
            return (
                "assistant\n<think>\n" if prompt_open
                else "assistant\n"
            )

        chat._tokenizer.apply_chat_template.side_effect = template
        generated = []

        def generate(*args, **kwargs):
            """Return fixed model output and simulated termination."""

            generated.append(raw)
            chat.last_finish_reason = finish
            return raw

        result = None
        error = None
        with patch.object(chat, "_torch_reply", side_effect=generate):
            try:
                result = answering.side_reply(
                    chat, "system", "prompt", max_new_tokens=None,
                    inherit_settings=True, preserve_thinking=True,
                    require_complete=True,
                )
            except IncompleteModelResponse as failure:
                error = failure

        assert chat.messages == history
        assert chat.thinking is thinking
        assert chat.temperature == 0.0
        return result, error, len(generated), chat

    def test_qwen38_implicit_opening_can_trigger_existing_retry(self):
        """Recover a thinking opener supplied by the prompt, not output."""

        result, error, count, chat = self._run(
            "Reasoning without close", thinking=True, finish="length",
            prompt_open=True,
        )
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(error.reason, "token_limit")
        self.assertEqual(error.raw_text, "<think>Reasoning without close")
        self.assertTrue(thinking_is_open(error.raw_text))
        self.assertEqual(count, 1)
        self.assertTrue(chat._last_prompt_opens_thinking)

    def test_qwen3_explicit_opening_is_unchanged(self):
        """An explicit opening continues to qualify for the retry."""

        _, error, count, chat = self._run(
            "<think>Reasoning without close", thinking=True,
            finish="length", prompt_open=False,
        )
        self.assertEqual(error.raw_text, "<think>Reasoning without close")
        self.assertEqual(count, 1)
        self.assertFalse(chat._last_prompt_opens_thinking)

    def test_closed_implicit_reasoning_is_not_mistaken_for_open(self):
        """Do not retry a token limit reached after closing reasoning."""

        _, error, count, chat = self._run(
            "Reasoning</think>Partial final", thinking=True,
            finish="length", prompt_open=True,
        )
        self.assertEqual(error.reason, "token_limit")
        self.assertEqual(error.raw_text, "Reasoning</think>Partial final")
        self.assertFalse(thinking_is_open(error.raw_text))
        self.assertEqual(count, 1)
        self.assertFalse(chat._last_prompt_opens_thinking)

    def test_non_exhausted_generation_does_not_inspect_template(self):
        """Limit extra template work to the observed failure condition."""

        result, error, count, chat = self._run(
            "Final", thinking=True, finish="stop", prompt_open=True,
        )
        self.assertEqual(result, "Final")
        self.assertIsNone(error)
        self.assertEqual(count, 1)
        self.assertFalse(chat._last_prompt_opens_thinking)
        self.assertEqual(
            chat._tokenizer.apply_chat_template.call_count, 1
        )

    def test_non_thinking_does_not_infer_an_implicit_opening(self):
        """A nonthinking token limit must not activate thinking recovery."""

        _, error, count, chat = self._run(
            "Partial final", thinking=False, finish="length",
            prompt_open=True,
        )
        self.assertEqual(error.reason, "token_limit")
        self.assertEqual(error.raw_text, "Partial final")
        self.assertFalse(thinking_is_open(error.raw_text))
        self.assertEqual(count, 1)
        self.assertFalse(chat._last_prompt_opens_thinking)

    def test_completed_implicit_reasoning_is_unchanged(self):
        """Keep successful model text and settings exactly as generated."""

        result, error, count, chat = self._run(
            "Reasoning</think>Final", thinking=True, finish="stop",
            prompt_open=True,
        )
        self.assertEqual(result, "Reasoning</think>Final")
        self.assertIsNone(error)
        self.assertEqual(count, 1)
        self.assertFalse(chat._last_prompt_opens_thinking)


if __name__ == "__main__":
    unittest.main()
