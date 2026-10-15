"""Test reciprocal-rank fusion across heterogeneous retrieval channels.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from types import SimpleNamespace

from qachat.defaults import load_defaults_section
from qachat.retrievals.strategies import retrieval_hybrid as rh
from tests.qachat.retrievals._support import chunk


class TestHybridSearch(unittest.TestCase):
    """Verify reciprocal-rank scores and deterministic deduplication."""

    def setUp(self) -> None:
        """Prepare identifiable source records for ranking fixtures.

        """
        self.a = chunk("a")
        self.b = chunk("b")
        self.c = chunk("c")
        self.rank_constant = load_defaults_section("RETRIEVAL")[
            "RRF_RANK_CONSTANT"
        ]

    def test_empty_rankings(self) -> None:
        """Return no chunks if no retrieval channel contributes evidence.

        """
        self.assertEqual(rh.hybrid_search(), [])

    def test_single_channel_ignores_original_score(self) -> None:
        """Rank positions rather than incomparable raw channel scores.

        """
        items = [
            SimpleNamespace(chunk=self.a, score=-200),
            SimpleNamespace(chunk=self.b, score=999),
        ]
        result = rh.hybrid_search(semantic=items)
        self.assertEqual([item.chunk.identifier for item in result],
                         ["a", "b"])
        self.assertAlmostEqual(result[0].score, 1 / (self.rank_constant + 1))
        self.assertAlmostEqual(result[1].score, 1 / (self.rank_constant + 2))
        self.assertEqual(result[0].channels, ("semantic",))

    def test_multiple_channels_improve_score(self) -> None:
        """RRF sums independent rank contributions across channels.

        """
        item = SimpleNamespace(chunk=self.a)
        result = rh.hybrid_search(
            semantic=[item], lexical=[item], code=[item]
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].score, 3 / (self.rank_constant + 1))
        self.assertEqual(result[0].channels,
                         ("code", "lexical", "semantic"))

    def test_duplicate_within_channel_counts_only_once(self) -> None:
        """Repeated results do not inflate one channel's contribution.

        """
        rank = lambda c: SimpleNamespace(chunk=c)
        results = rh.hybrid_search(
            semantic=[rank(self.a), rank(self.a), rank(self.b)]
        )
        self.assertAlmostEqual(results[0].score, 1 / (self.rank_constant + 1))
        self.assertAlmostEqual(results[1].score, 1 / (self.rank_constant + 3))

    def test_equal_score_tie_breaks_on_identifier(self) -> None:
        """Equal scores must be reproducible independent of channel order.

        """
        ranked = [
            SimpleNamespace(chunk=self.b),
            SimpleNamespace(chunk=self.a),
        ]
        results = rh.hybrid_search(
            semantic=ranked,
            lexical=list(reversed(ranked)),
        )
        self.assertEqual(
            [result.chunk.identifier for result in results], ["a", "b"]
        )

    def test_count_and_rank_constant(self) -> None:
        """Apply user-supplied RRF smoothing and result ceiling.

        """
        ranked = [SimpleNamespace(chunk=c) for c in (self.a, self.b)]
        results = rh.hybrid_search(
            semantic=ranked, count=1, rank_constant=1
        )
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 0.5)

    def test_invalid_settings(self) -> None:
        """Require strict positive integers for both ranking parameters.

        """
        for name in ("count", "rank_constant"):
            for value in (0, -1, True, 1.5, "1"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        rh.hybrid_search(**{name: value})

    def test_result_fields_are_frozen(self) -> None:
        """Prevent mutation of a produced fused ranking record.

        """
        result = rh.hybrid_search(
            semantic=[SimpleNamespace(chunk=self.a)]
        )[0]
        with self.assertRaises(FrozenInstanceError):
            result.channels = ()


if __name__ == "__main__":
    unittest.main()
