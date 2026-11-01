"""Regressions for completed, truncated and thinking-only model replies.

Validate termination independently of either inference backend. These tests
never load models and deliberately include parseable but truncated outputs.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin._model_completion import (
    IncompleteModelResponse,
    require_complete_reply,
    strip_thinking,
    thinking_is_open,
)


class TestModelCompletion(unittest.TestCase):
    """Distinguish valid negative results from incomplete generation."""

    def test_completed_replies_and_template_delimiters(self) -> None:
        """Retain explicit negatives and both supported thinking formats."""

        for text in (
            "NONE", "1, 2", "A final answer.",
            "<think>Reasoning</think>Answer",
            "Template-opened reasoning</think>Answer",
            "<think>A</think><think>B</think>Answer",
        ):
            with self.subTest(text=text):
                require_complete_reply(text, "stop")
                require_complete_reply(text)
                self.assertFalse(thinking_is_open(text))
                self.assertTrue(strip_thinking(text))

    def test_length_is_failure_even_when_output_is_parseable(self) -> None:
        """Do not accept valid-looking text terminated by a token cap."""

        for text in ("NONE", "1, 2", '{"result": true}', "Partial prose"):
            with self.subTest(text=text):
                with self.assertRaises(IncompleteModelResponse) as failure:
                    require_complete_reply(text, "length")
                error = failure.exception
                self.assertEqual(error.reason, "token_limit")
                self.assertEqual(error.finish_reason, "length")
                self.assertEqual(error.raw_text, text)
                self.assertEqual(error.to_dict()["status"], "incomplete")
                self.assertNotIn("raw_text", error.to_dict())

    def test_unclosed_thinking_never_becomes_an_answer(self) -> None:
        """Reject incomplete blocks even without backend metadata."""

        for text in ("<think>draft", "Answer<think>more"):
            with self.subTest(text=text):
                self.assertTrue(thinking_is_open(text))
                with self.assertRaises(IncompleteModelResponse) as failure:
                    require_complete_reply(text)
                self.assertEqual(
                    failure.exception.reason, "unfinished_thinking"
                )

    def test_no_final_text_is_not_a_success(self) -> None:
        """Reject empty replies and completed thinking without an answer."""

        for text in ("", "  ", "<think>draft</think>", "draft</think>"):
            with self.subTest(text=text):
                with self.assertRaises(IncompleteModelResponse) as failure:
                    require_complete_reply(text, "stop")
                self.assertEqual(failure.exception.reason, "empty_answer")


if __name__ == "__main__":
    unittest.main()
