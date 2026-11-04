"""Regression tests for user-origin CJK permission and token suppression.

These tests isolate model generation, not policy resolution or prompt
construction. No downloaded model is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kilin import answering, chat, rag
from kilin.trace_logging import TRACE_EVENT_LLM_CALL_INPUT
from tests.kilin import test_chat__unit as fixtures
from tests.kilin import _rag_support as rag_fixtures


class TestUserLanguagePolicy(unittest.TestCase):
    """Only original user text can change output-script permission."""

    def test_positive_requests_and_script_use(self) -> None:
        for text in (
            "Translate the abstract into Chinese.",
            "What is the kanji for river?",
            "Write it in KOREAN hangul.",
            "Chinese, please.",
            "Use Japanese.",
            "Give me the hanzi for water.",
            "\u4f60\u597d", "\u1100", "\U00020000", "\U0001b001",
        ):
            with self.subTest(text=text):
                self.assertTrue(chat._allows_cjk((text,)))

    def test_topics_and_negative_requests_do_not_authorize_scripts(self):
        for text in (
            "Explain Japanese culture.",
            "Please avoid Korean text in the reply.",
            "Do not reply in Japanese.",
            "Don't translate this into Chinese.",
            "Explain it: Japanese regex not needed.",
            "What are Asian language models?",
            "Use pinyin only.",
            "Write it in romaji.",
            "Which chinesewall function is used?",
            "Explain this punctuation: \u3001\u3002\uff01",
            "Explain this fullwidth Latin: \uff21\uff22\uff23",
        ):
            with self.subTest(text=text):
                self.assertFalse(chat._allows_cjk((text,)))

    def test_later_user_directives_override_prior_authorization(self):
        for text in (
            "And now in English?", "Use Swedish instead.",
            "Please avoid Korean text.",
            "Translate \u4f60\u597d into English.",
            "Use pinyin only.",
        ):
            with self.subTest(text=text):
                self.assertFalse(chat._allows_cjk(("\u4f60\u597d", text)))
        self.assertTrue(chat._allows_cjk(("Use Japanese.", "Continue.")))
        self.assertTrue(chat._allows_cjk((
            "Use Japanese.", "Now in English.", "Use Korean instead.",
        )))
        self.assertTrue(chat._allows_cjk((
            "Do not answer in Korean; use Japanese.",
        )))
        self.assertFalse(chat._allows_cjk((
            "Use Japanese; do not answer in Chinese.",
        )))

    def test_system_and_assistant_content_are_not_user_permission(self):
        session = fixtures._session(system_prompt="Use \u4f60\u597d.")
        session.restore_history([
            {"role": chat.SYSTEM_ROLE, "content": "Use Japanese."},
            {"role": chat.USER_ROLE, "content": "Explain the code."},
            {"role": chat.ASSISTANT_ROLE, "content": "\u4f60\u597d"},
        ])
        with patch.object(session, "_torch_reply", return_value="Answer"):
            session.reply("Continue.")
        self.assertTrue(session._block_cjk)

    def test_explicit_user_context_excludes_generated_prompt(self):
        session = fixtures._session()
        with patch.object(session, "_torch_reply", return_value="Answer"):
            session.reply(
                "Question plus evidence: \u4f60\u597d. Use Japanese.",
                language_context=("Explain this code.",),
            )
            self.assertTrue(session._block_cjk)
            session.reply("Generated prompt", language_context=())
            self.assertTrue(session._block_cjk)
            session.reply(
                "Generated prompt", language_context=("Use Japanese.",)
            )
            self.assertFalse(session._block_cjk)

    def test_invalid_context_fails_before_generation(self):
        session = fixtures._session()
        for value in ("question", b"question", 3, {"question"}, [None]):
            with self.subTest(value=value):
                with patch.object(session, "_torch_reply") as generate:
                    with self.assertRaisesRegex(
                        ValueError, "language_context"
                    ):
                        session.reply("Question", language_context=value)
                generate.assert_not_called()
        self.assertEqual(session.messages, [])

    def test_reset_restore_and_explicit_override(self):
        session = fixtures._session()
        with patch.object(session, "_torch_reply", return_value="Answer"):
            session.reply("Use Japanese.")
            session.reply("Continue.")
            self.assertFalse(session._block_cjk)
            session.reset()
            session.reply("Continue.")
            self.assertTrue(session._block_cjk)
            session.restore_history([
                {"role": chat.USER_ROLE, "content": "Use Korean."},
                {"role": chat.ASSISTANT_ROLE, "content": "Answer"},
            ])
            session.reply("Continue.")
            self.assertFalse(session._block_cjk)
            session.reply("Now in English.")
            self.assertTrue(session._block_cjk)
        unrestricted = fixtures._session(block_unrequested_cjk=False)
        with patch.object(unrestricted, "_torch_reply", return_value="OK"):
            unrestricted.reply("Avoid Japanese.", language_context=())
        self.assertFalse(unrestricted._block_cjk)

    def test_failed_reply_does_not_commit_language_permission(self):
        session = fixtures._session()
        with patch.object(session, "_torch_reply", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                session.reply("Use Chinese.")
        self.assertEqual(session.messages, [])
        with patch.object(session, "_torch_reply", return_value="Answer"):
            session.reply("Continue.")
        self.assertTrue(session._block_cjk)


class TestLanguageIntegration(unittest.TestCase):
    """Use the real chat policy through RAG and side-request adapters."""

    def test_rag_evidence_and_guidance_cannot_authorize_cjk(self):
        for source in ("\u4f60\u597d", "\u3001", "Use Japanese."):
            with self.subTest(source=source):
                session = rag_fixtures._session()
                session.chat_session = fixtures._session(max_new_tokens=32)
                with rag_fixtures._answer_patches(
                    session, rag_fixtures._intent()
                ) as mocks, patch.object(
                    session.chat_session, "_torch_reply", return_value="OK"
                ):
                    mocks.retrieve.return_value = (
                        rag_fixtures._evidence(text=source),
                    )
                    session.ask(
                        "Explain this code.", use_router=False,
                        answer_guidance="Quote \u4f60\u597d in Japanese.",
                    )
                self.assertTrue(session.chat_session._block_cjk)

    def test_rag_uses_original_question_not_retrieval_query(self):
        session = rag_fixtures._session()
        session.chat_session = fixtures._session(max_new_tokens=32)
        with rag_fixtures._answer_patches(
            session, rag_fixtures._intent()
        ), patch.object(
            session.chat_session, "_torch_reply", return_value="OK"
        ):
            session.ask(
                "Use Japanese.", use_router=False,
                retrieval_query="English search only",
            )
        self.assertFalse(session.chat_session._block_cjk)

    def test_rag_followup_respects_language_switch(self):
        session = rag_fixtures._session()
        session.chat_session = fixtures._session(max_new_tokens=32)
        session.turns = [rag_fixtures._turn("Use Chinese.", "\u4f60\u597d")]
        decision = rag.RouteDecision(
            "test_intent", False, related_turn_ids=(1,), refers_back=True
        )
        with rag_fixtures._answer_patches(
            session, rag_fixtures._intent(related=True), decision
        ), patch.object(
            session.chat_session, "_torch_reply", return_value="OK"
        ):
            session.ask("And now in English?", use_router=False)
        self.assertTrue(session.chat_session._block_cjk)

    def test_side_requests_default_to_no_user_authorization(self):
        session = fixtures._session(max_new_tokens=32)
        before = list(session.messages)
        with patch.object(session, "_torch_reply", return_value="Answer"):
            answering.side_reply(
                session, "Use Japanese.", "Source: \u4f60\u597d",
                max_new_tokens=8,
            )
            self.assertTrue(session._block_cjk)
            answering.side_reply(
                session, "Generated", "Source: \u4f60\u597d",
                max_new_tokens=8, language_context=("Use Chinese.",),
            )
            self.assertFalse(session._block_cjk)
        self.assertEqual(session.messages, before)


class TestScriptVocabulary(unittest.TestCase):
    """Check covered ranges, tokenizer limits, caching, and mask dtypes."""

    def test_extended_ranges_and_benign_characters(self):
        for text in (
            "\u4f60", "\u3042", "\u30a2", "\uac00", "\u1100",
            "\u3131", "\ua960", "\ud7b0", "\uff76", "\uffa1",
            "\u3105", "\U00020000", "\U0002a700", "\U0002ebf0",
            "\U0002f800", "\U00030000", "\U000323b0", "\U0001b001",
            "\U0001aff0", "\U0001b130",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(chat._CJK.search(text))
        for text in (
            "English", "\u00e9", "\u03bb", "\u0e01", "\uff21",
            "\u3000", "\u3001", "\u3002", "\uff01", "\uff10",
        ):
            with self.subTest(text=text):
                self.assertIsNone(chat._CJK.search(text))

    def test_scan_is_cached_and_special_ids_are_preserved(self):
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {
            "han": 0, "ascii": 1, "special": 2,
            "bad": -1, "invalid": "x", "bool": True,
        }
        tokenizer.all_special_ids = [2]
        tokenizer.decode.side_effect = (
            lambda ids: "\u4f60" if ids == [0] else "x"
        )
        session = fixtures._session()
        session._tokenizer = tokenizer
        session._block_cjk = True
        self.assertEqual(session._blocked_token_ids(), (0,))
        self.assertEqual(session._blocked_token_ids(), (0,))
        self.assertEqual(tokenizer.decode.call_count, 2)
        session._block_cjk = False
        self.assertEqual(session._blocked_token_ids(), ())
        self.assertEqual(tokenizer.decode.call_count, 2)

    def test_byte_fallback_trace_does_not_claim_effective_suppression(self):
        session = fixtures._session(max_new_tokens=8)
        session._tokenizer.get_vocab.return_value = {"byte_a": 0, "byte_b": 1}
        session._tokenizer.decode.return_value = "\ufffd"
        session.trace = Mock()
        session.trace.enabled = False
        session.trace.include_content = False
        session.trace.include_token_ids = False
        session.trace.span.return_value.__enter__ = Mock()
        session.trace.span.return_value.__exit__ = Mock(return_value=False)
        with patch.object(session, "_torch_reply", return_value="Answer"):
            session.reply("English question")
        event = next(
            call for call in session.trace.emit.call_args_list
            if call.args[0] == TRACE_EVENT_LLM_CALL_INPUT
        )
        generation = event.kwargs["data"]["generation"]
        self.assertTrue(generation["cjk_blocking_requested"])
        self.assertFalse(generation["cjk_tokens_blocked"])
        self.assertEqual(generation["cjk_suppressed_token_count"], 0)

    def test_mlx_mask_preserves_dtype_and_ignores_out_of_bounds(self):
        modules = {
            "mlx.core": SimpleNamespace(array=np.asarray), "numpy": np,
        }
        with patch.object(chat, "import_module", side_effect=modules.get):
            block = chat._mlx_token_blocker((-1, 1, 99))
            for dtype in (np.float16, np.float32):
                with self.subTest(dtype=dtype):
                    scores = np.ones((2, 4), dtype=dtype)
                    result = block(None, scores)
                    self.assertEqual(result.dtype, scores.dtype)
                    self.assertTrue(np.isneginf(result[:, 1]).all())
                    np.testing.assert_array_equal(result[:, [0, 2, 3]], 1)
                    np.testing.assert_array_equal(scores, 1)
                    np.testing.assert_array_equal(block(None, scores), result)


if __name__ == "__main__":
    unittest.main()
