"""Validate shared bounded-retrieval inputs independently of models.

Request validation must reject unsupported strategies, unsafe source paths,
and malformed card-kind filters before RAG begins a query.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.retrievals.evidence_budget import EvidenceLimits
from kilin.retrievals.requests import RetrievalRequest, RetrievalResult


class TestRetrievalRequest(unittest.TestCase):

    def setUp(self) -> None:
        self.limits = EvidenceLimits(3, 1, 8, 5000)

    def test_valid_request_is_immutable(self) -> None:
        request = RetrievalRequest(
            "test", self.limits,
            scope=(Path("documents/paper.pdf"),),
            card_kinds=("scientific_paper",),
        )
        self.assertEqual(request.limits.max_sources, 3)
        self.assertEqual(request.strategy, "source_first")
        with self.assertRaises(AttributeError):
            request.query = "changed"

    def test_invalid_queries_and_strategies(self) -> None:
        for query in ("", " \n", None, 5):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    RetrievalRequest(query, self.limits)
        for strategy in ("papers", "unknown", None, 5):
            with self.subTest(strategy=strategy):
                with self.assertRaises(ValueError):
                    RetrievalRequest("text", self.limits, strategy=strategy)
        with self.assertRaises(TypeError):
            RetrievalRequest("text", None)

    def test_invalid_source_paths(self) -> None:
        for path in (
            Path("/tmp/data.pdf"), Path("../secret"),
            Path("data/../other"), Path("C:\\secret.pdf"),
            Path("\\\\server\\share\\file.pdf"),
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    RetrievalRequest("text", self.limits, scope=(path,))
        with self.assertRaises(ValueError):
            RetrievalRequest("text", self.limits, scope=[Path("a.pdf")])

    def test_kind_filters_require_source_first(self) -> None:
        with self.assertRaises(ValueError):
            RetrievalRequest(
                "text", self.limits, strategy="hybrid",
                card_kinds=("scientific_paper",),
            )
        for kinds in (("",), (3,), ["scientific_paper"]):
            with self.subTest(kinds=kinds):
                with self.assertRaises(ValueError):
                    RetrievalRequest("text", self.limits, card_kinds=kinds)
        with self.assertRaises(ValueError):
            RetrievalRequest("text", self.limits, compare=1)

    def test_result_keeps_original_chunk_objects(self) -> None:
        result = RetrievalResult((), (Path("a.pdf"),), {"complete": False})
        self.assertEqual(result.passages, ())
        self.assertEqual(result.sources, (Path("a.pdf"),))


if __name__ == "__main__":
    unittest.main()
