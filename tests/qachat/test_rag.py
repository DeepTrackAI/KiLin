"""Tests for the assembled retrieval and chat session."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch, sentinel

from qachat.rag import RAGSession


def _session() -> RAGSession:
    session = RAGSession.__new__(RAGSession)
    session.project = SimpleNamespace(
        profile=SimpleNamespace(retrieval_count=5),
    )
    session.indexes = SimpleNamespace(
        semantic=sentinel.semantic_index,
        lexical=sentinel.lexical_index,
        code_graph=sentinel.code_graph,
    )
    session.embedding_session = sentinel.embedding_session
    session.chunks = []
    session._semantic_matrix_cache = None
    session._format_query = MagicMock(return_value="formatted query")
    return session


class TestSemanticMatrixCache(unittest.TestCase):

    @patch("qachat.rag.expand_retrieval_context", return_value=[])
    @patch("qachat.rag.plan_retrieval")
    @patch("qachat.rag.hybrid_search", return_value=[])
    @patch("qachat.rag.code_search", return_value=[])
    @patch("qachat.rag.lexical_search", return_value=[])
    @patch("qachat.rag.semantic_search", return_value=[])
    @patch("qachat.rag.semantic_matrix", return_value=sentinel.matrix)
    def test_numpy_search_reuses_matrix_after_loop(
        self,
        build_matrix,
        search,
        _lexical,
        _code,
        _hybrid,
        _plan,
        _expand,
    ) -> None:
        session = _session()

        session._retrieve("first", semantic_method="numpy")
        session._retrieve("middle", semantic_method="loop")
        session._retrieve("second", semantic_method="numpy")

        build_matrix.assert_called_once_with(sentinel.semantic_index)
        self.assertEqual(search.call_count, 3)
        first_call, middle_call, last_call = search.call_args_list
        self.assertEqual(first_call.kwargs["backend"], "numpy")
        self.assertEqual(middle_call.kwargs["backend"], "loop")
        self.assertEqual(last_call.kwargs["backend"], "numpy")
        self.assertIs(first_call.kwargs["matrix"], sentinel.matrix)
        self.assertIsNone(middle_call.kwargs["matrix"])
        self.assertIs(last_call.kwargs["matrix"], sentinel.matrix)

    @patch("qachat.rag.expand_retrieval_context", return_value=[])
    @patch("qachat.rag.plan_retrieval")
    @patch("qachat.rag.hybrid_search", return_value=[])
    @patch("qachat.rag.code_search", return_value=[])
    @patch("qachat.rag.lexical_search", return_value=[])
    @patch("qachat.rag.semantic_search", return_value=[])
    @patch("qachat.rag.semantic_matrix")
    def test_loop_search_does_not_build_matrix(
        self,
        build_matrix,
        search,
        _lexical,
        _code,
        _hybrid,
        _plan,
        _expand,
    ) -> None:
        session = _session()

        session._retrieve("question", semantic_method="loop")

        build_matrix.assert_not_called()
        self.assertEqual(search.call_args.kwargs["backend"], "loop")
        self.assertIsNone(search.call_args.kwargs["matrix"])

    def test_close_releases_matrix(self) -> None:
        session = RAGSession.__new__(RAGSession)
        session._closed = False
        session._semantic_matrix_cache = sentinel.matrix
        session.chat_session = MagicMock()
        session.embedding_session = MagicMock()

        session.close()

        self.assertIsNone(session._semantic_matrix_cache)
        session.chat_session.close.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()


class TestAnswerGuidance(unittest.TestCase):
    """Optional guidance extends the prompt without changing the default."""

    def test_guidance_is_appended_only_when_given(self) -> None:
        from qachat.rag import _answer_prompt

        plain = _answer_prompt("What is memory?", ())
        guided = _answer_prompt("What is memory?", (), "Be brief.")

        self.assertNotIn("Be brief.", plain)
        self.assertTrue(guided.startswith(plain))
        self.assertTrue(guided.endswith(" Be brief."))
        self.assertEqual(_answer_prompt("What is memory?", (), None), plain)
