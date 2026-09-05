"""Tests for transparent reciprocal-rank fusion."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.hybrid import reciprocal_rank_fusion
from qachat.retrieval import RetrievalMethod, RetrievalResult


def _result(
    identifier: str,
    method: RetrievalMethod,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            identifier=identifier,
            path=Path(f"{identifier}.py"),
            text=identifier,
            start_line=1,
            end_line=1,
            kind="function",
            symbol=identifier,
            section=None,
        ),
        score=score,
        method=method,
    )


class TestHybrid(unittest.TestCase):

    def test_fusion_deduplicates_and_explains_contributions(self) -> None:
        semantic = [
            _result("apple", "semantic", 0.99),
            _result("banana", "semantic", 0.75),
        ]
        lexical = [
            _result("banana", "lexical", 8.0),
            _result("apple", "lexical", 2.0),
        ]
        metadata = [_result("banana", "metadata", 5.0)]

        results = reciprocal_rank_fusion(
            [semantic, lexical, metadata],
            count=2,
            rank_constant=10,
        )

        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["banana", "apple"],
        )
        self.assertEqual(
            [item.method for item in results[0].contributions],
            ["semantic", "lexical", "metadata"],
        )
        self.assertEqual(
            [item.rank for item in results[0].contributions],
            [2, 1, 1],
        )
        self.assertTrue(all(item.reason for item in results[0].contributions))

    def test_fusion_uses_rank_instead_of_raw_method_scores(self) -> None:
        first = reciprocal_rank_fusion(
            [
                [
                    _result("a", "semantic", -1000.0),
                    _result("b", "semantic", 1_000_000.0),
                ]
            ],
            count=2,
        )
        second = reciprocal_rank_fusion(
            [
                [
                    _result("a", "semantic", 1_000_000.0),
                    _result("b", "semantic", -1000.0),
                ]
            ],
            count=2,
        )

        self.assertEqual(
            [result.chunk.identifier for result in first],
            ["a", "b"],
        )
        self.assertEqual(
            [result.score for result in first],
            [result.score for result in second],
        )

    def test_fusion_is_stable_and_ignores_duplicate_list_entries(
        self,
    ) -> None:
        results = reciprocal_rank_fusion(
            [
                [
                    _result("b", "semantic", 1.0),
                    _result("b", "semantic", 0.5),
                ],
                [_result("a", "lexical", 1.0)],
            ],
            count=5,
        )

        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["a", "b"],
        )
        self.assertEqual(len(results[1].contributions), 1)
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_fusion_rejects_invalid_settings_and_values(self) -> None:
        invalid_settings = (
            {"count": 0},
            {"count": True},
            {"rank_constant": 0},
            {"rank_constant": True},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    reciprocal_rank_fusion([], **settings)

        with self.assertRaises(ValueError):
            reciprocal_rank_fusion("results")
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([("result",)])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["result"]])


if __name__ == "__main__":
    unittest.main()
