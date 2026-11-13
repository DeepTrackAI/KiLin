"""Verify four-channel source fusion and original-passage selection.

All ranked inputs use deterministic chunks and vectors; no model is loaded.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.chunking import Chunk
from kilin.retrievals.source_cards.index import SourceCardMatch
from kilin.retrievals.strategies.retrieval_hybrid import HybridResult
from kilin.retrievals.strategies.retrieval_semantic import SemanticResult
from kilin.retrievals.strategies.retrieval_sources import (
    SourceResult,
    relevant_sources,
    select_source_passages,
    source_first_search,
)


def _chunk(identifier: str, path: str) -> Chunk:
    """Create a source chunk with stable provenance.

    Parameters
    ----------
    identifier: str
        Unique chunk identifier.
    path: str
        Source-relative path.

    Returns
    -------
    Chunk
        Deterministic chunk for ranking tests.

    """

    return Chunk(
        identifier, Path(path), identifier, 1, 2, "pdf_page", None,
        "Page 1",
    )


def _results(*pairs: tuple[str, str]) -> list[SemanticResult]:
    """Return chunks in a supplied ranked order.

    Parameters
    ----------
    pairs: tuple[str, str]
        Chunk IDs and source paths.

    Returns
    -------
    list[SemanticResult]
        Ranked results in the same order.

    """

    return [SemanticResult(1.0, _chunk(*pair)) for pair in pairs]


class TestSourceFusion(unittest.TestCase):

    def test_four_channels_contribute_independently(self):
        semantic = _results(
            ("a1", "a.pdf"), ("a2", "a.pdf"), ("b1", "b.pdf"),
        )
        lexical = _results(("b1", "b.pdf"), ("a1", "a.pdf"))
        code = _results(("c1", "c.py"))
        cards = (
            SourceCardMatch(0.8, Path("c.py"), "code_module"),
            SourceCardMatch(0.6, Path("b.pdf"), "scientific_paper"),
        )
        ranked = source_first_search(
            semantic=semantic, lexical=lexical, code=code, cards=cards,
        )
        self.assertEqual(
            [item.path for item in ranked],
            [Path("b.pdf"), Path("c.py"), Path("a.pdf")],
        )
        self.assertEqual(
            dict(ranked[0].channel_ranks),
            {"semantic": 2, "lexical": 1, "cards": 2},
        )
        self.assertEqual(
            dict(ranked[1].channel_ranks),
            {"cards": 1, "code": 1},
        )

    def test_repeated_passages_do_not_change_source_rank(self):
        ranks = source_first_search(
            semantic=_results(
                ("a1", "a.pdf"), ("a2", "a.pdf"),
                ("a3", "a.pdf"), ("b1", "b.pdf"),
            ),
        )
        self.assertEqual(
            [(item.path.name, dict(item.channel_ranks)["semantic"])
             for item in ranks],
            [("a.pdf", 1), ("b.pdf", 2)],
        )

    def test_duplicate_card_kinds_contribute_once(self):
        ranks = source_first_search(cards=(
            SourceCardMatch(0.9, Path("a.pdf"), "scientific_paper"),
            SourceCardMatch(0.8, Path("a.pdf"), "other_kind"),
            SourceCardMatch(0.7, Path("b.pdf"), "scientific_paper"),
        ))
        self.assertEqual(
            [dict(item.channel_ranks)["cards"] for item in ranks],
            [1, 2],
        )

    def test_card_only_source_is_discoverable(self):
        ranks = source_first_search(cards=(
            SourceCardMatch(0.9, Path("card.pdf"), "scientific_paper"),
        ))
        self.assertEqual(ranks[0].path, Path("card.pdf"))
        self.assertEqual(ranks[0].channels, ("cards",))

    def test_tie_breaks_by_path_not_insertion_order(self):
        ranked = source_first_search(
            semantic=_results(("z", "z.pdf")),
            lexical=_results(("a", "a.pdf")),
        )
        self.assertEqual(
            [item.path.name for item in ranked], ["a.pdf", "z.pdf"]
        )

    def test_empty_rankings_and_validation(self):
        self.assertEqual(source_first_search(), [])
        for bad in (0, -1, True, 1.5, "2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    source_first_search(count=bad)
                with self.assertRaises(ValueError):
                    source_first_search(rank_constant=bad)

    def test_maximum_source_count_is_enforced(self):
        ranked = source_first_search(
            semantic=_results(("a", "a.pdf"), ("b", "b.pdf")),
            count=1,
        )
        self.assertEqual([item.path.name for item in ranked], ["a.pdf"])


class TestOriginalPassages(unittest.TestCase):

    def setUp(self):
        self.ranked = [
            HybridResult(float(10 - index), result.chunk, ("semantic",))
            for index, result in enumerate(_results(
                ("a1", "a.pdf"), ("a2", "a.pdf"),
                ("a3", "a.pdf"), ("b1", "b.pdf"),
                ("c1", "c.pdf"), ("b2", "b.pdf"),
            ))
        ]

    def test_card_discovered_source_gets_original_evidence(self):
        results = select_source_passages(
            self.ranked, [Path("c.pdf"), Path("a.pdf")], count=3
        )
        self.assertEqual(
            [item.chunk.identifier for item in results],
            ["c1", "a1", "a2"],
        )

    def test_count_is_hard_limit_even_with_many_sources(self):
        results = select_source_passages(
            self.ranked,
            [Path("c.pdf"), Path("b.pdf"), Path("a.pdf")],
            count=2,
        )
        self.assertEqual(
            [item.chunk.identifier for item in results], ["c1", "b1"]
        )

    def test_duplicates_and_missing_sources_do_not_add_evidence(self):
        results = select_source_passages(
            self.ranked,
            [Path("missing.pdf"), Path("b.pdf"), Path("b.pdf")],
            count=5,
        )
        self.assertEqual(
            [item.chunk.identifier for item in results], ["b1", "b2"]
        )

    def test_invalid_count(self):
        for value in (0, -1, True, 0.5, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_source_passages(self.ranked, [], count=value)



class TestRelevantSources(unittest.TestCase):
    """Keep only the ranked sources a question is about."""

    def setUp(self) -> None:
        self.ranked = [
            SourceResult(Path(name), 1.0 - index / 10, ranks)
            for index, (name, ranks) in enumerate((
                ("a.pdf", (("cards", 2), ("semantic", 1))),
                ("b.pdf", (("cards", 3), ("semantic", 2))),
                ("c.pdf", (("cards", 1), ("semantic", 5))),
                ("d.pdf", (("cards", 4), ("semantic", 3))),
                ("e.pdf", (("cards", 5), ("semantic", 4))),
            ))
        ]
        # The best passages are a's and b's; d and e only further down.
        self.passages = [
            HybridResult(1.0, _chunk(identifier, path), ("semantic",))
            for identifier, path in (
                ("a1", "a.pdf"), ("b1", "b.pdf"), ("a2", "a.pdf"),
                ("b2", "b.pdf"), ("d1", "d.pdf"), ("e1", "e.pdf"),
                ("c1", "c.pdf"),
            )
        ]

    def _kept(self, **options: object) -> list[str]:
        return [
            item.path.name
            for item in relevant_sources(self.ranked, self.passages,
                                         **options)
        ]

    def test_sources_beyond_the_best_passages_are_left_out(self):
        # c's passages rank low, but its card ranks first.
        self.assertEqual(self._kept(depth=4), ["a.pdf", "b.pdf", "c.pdf"])
        self.assertEqual(
            self._kept(depth=5), ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
        )

    def test_named_sources_and_minimum_are_kept(self):
        self.assertEqual(
            self._kept(depth=1, keep={Path("e.pdf")}),
            ["a.pdf", "c.pdf", "e.pdf"],
        )
        ranked = [item for item in self.ranked if item.path.name != "c.pdf"]
        kept = relevant_sources(ranked, self.passages, depth=1, minimum=2)
        self.assertEqual([item.path.name for item in kept],
                         ["a.pdf", "b.pdf"])

    def test_invalid_depth_or_minimum(self):
        for options in ({"depth": 0}, {"depth": 2, "minimum": 0}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    relevant_sources(self.ranked, self.passages, **options)


if __name__ == "__main__":
    unittest.main()
