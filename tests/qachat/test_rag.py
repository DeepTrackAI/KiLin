"""Tests for the assembled retrieval and chat session."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch, sentinel

from qachat.rag import RAGSession
from qachat.retrievals import RetrievalPlan


def _session() -> RAGSession:
    session = RAGSession.__new__(RAGSession)
    session.project = SimpleNamespace(
        profile=SimpleNamespace(
            retrieval_count=5, query_format="plain", kind="code"
        ),
    )
    session.indexes = SimpleNamespace(
        semantic=sentinel.semantic_index,
        lexical=sentinel.lexical_index,
        code_graph=SimpleNamespace(entities=[]),
    )
    session.embedding_session = sentinel.embedding_session
    session.chunks = []
    session._format_query = MagicMock(return_value="formatted query")
    return session


class TestHybridRetrieval(unittest.TestCase):
    """RAG delegates search/cache ownership to the retrieval layer."""

    @patch("qachat.rag.expand_retrieval_context", return_value=[])
    @patch("qachat.rag.plan_retrieval")
    @patch("qachat.rag.hybrid_search", return_value=[])
    @patch("qachat.rag.code_search", return_value=[])
    @patch("qachat.rag.lexical_search", return_value=[])
    @patch("qachat.rag.semantic_search", return_value=[])
    def test_search_uses_formatted_semantic_and_raw_lexical_query(
        self,
        search,
        _lexical,
        _code,
        _hybrid,
        _plan,
        _expand,
    ) -> None:
        session = _session()

        _plan.return_value = RetrievalPlan((), (), False, (), ())
        self.assertEqual(session._retrieve("question", 8), ())
        search.assert_called_once_with(
            sentinel.semantic_index,
            "formatted query",
            sentinel.embedding_session,
            count=8,
            trace_turn_id=None,
        )
        _lexical.assert_called_once_with(
            sentinel.lexical_index, "question", count=8
        )
        _code.assert_not_called()
        _hybrid.assert_called_once_with(
            semantic=[], lexical=[], code=[], count=8
        )

    @patch("qachat.rag.expand_retrieval_context", return_value=[])
    @patch("qachat.rag.plan_retrieval")
    @patch("qachat.rag.hybrid_search", return_value=[])
    @patch("qachat.rag.code_search", return_value=[])
    @patch("qachat.rag.lexical_search", return_value=[])
    @patch("qachat.rag.semantic_search", return_value=[])
    def test_status_reports_each_retrieval_stage(
        self,
        search,
        _lexical,
        _code,
        _hybrid,
        _plan,
        _expand,
    ) -> None:
        session = _session()

        _plan.return_value = RetrievalPlan((), (), False, (), ())
        messages = []
        session._retrieve("question", status_callback=messages.append)
        self.assertIn("Running lexical search", messages)
        self.assertIn("Combining and ranking search results", messages)
        self.assertIn("Expanding related context", messages)
        self.assertLess(
            messages.index("Running lexical search"),
            messages.index("Combining and ranking search results"),
        )
        self.assertEqual(search.call_args.kwargs["count"], 5)

    def test_close_releases_sessions_once(self) -> None:
        session = RAGSession.__new__(RAGSession)
        session._closed = False
        session.chat_session = MagicMock()
        session.embedding_session = MagicMock()

        session.close()

        session.close()
        self.assertTrue(session._closed)
        session.chat_session.close.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()


class TestAnswerGuidance(unittest.TestCase):
    """Optional guidance extends the prompt without changing the default."""

    def test_guidance_is_appended_only_when_given(self) -> None:
        from qachat.rag import _answer_prompt

        from qachat.rag import Evidence

        evidence = (Evidence(1, Path("a.md"), 1, 2, None, "Memory is."),)
        plain = _answer_prompt("What is memory?", evidence)
        guided = _answer_prompt("What is memory?", evidence, "Be brief.")

        self.assertNotIn("Be brief.", plain)
        self.assertTrue(guided.startswith(plain))
        self.assertTrue(guided.endswith(" Be brief."))
        self.assertEqual(
            _answer_prompt("What is memory?", evidence, None), plain
        )

    def test_message_without_evidence_is_sent_as_written(self) -> None:
        from qachat.rag import _answer_prompt

        # An instruction here was echoed back by the model.
        self.assertEqual(_answer_prompt("how are you?", ()), "how are you?")
        self.assertEqual(
            _answer_prompt("how are you?", (), "Reply briefly."),
            "how are you?\n\nReply briefly.",
        )
