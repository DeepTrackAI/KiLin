"""Test code-aware entity matching and bounded graph traversal.

The tests exercise symbols, operator syntax, graph distance, and provenance.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from kilin.retrievals.code_graph import CodeEntity, CodeGraph
from kilin.retrievals.strategies import retrieval_code as rc
from tests.kilin.retrievals._support import chunk, graph


class TestCodeEntityResolution(unittest.TestCase):
    """Check identifier boundaries, symbols, and explicit query references."""

    def test_leaf_identifiers_and_camel_case(self) -> None:
        """Match complete identifiers without matching unrelated prefixes.

        """
        source_graph = graph(
            "pkg.loadModel", "pkg.load", "pkg.loader", "pkg.project",
        )
        result = rc.resolve_code_entities(
            source_graph, "How does loadModel run?"
        )
        self.assertEqual([item.name for item in result], ["pkg.loadModel"])
        self.assertEqual(
            rc.resolve_code_entities(source_graph, "projects folder"), []
        )

    def test_dotted_r_function_is_one_identifier(self) -> None:
        path = Path('r/main.R')
        source_graph = CodeGraph(
            entities=(
                CodeEntity('r.main.read.data', 'r_function', path),
                CodeEntity('r.main.read', 'r_function', path),
                CodeEntity('r.main.data', 'r_function', path),
            ),
            relations=(),
        )
        for query, names in (
            ('Explain read.data', {'r.main.read.data'}),
            ('Explain r.main.read.data', {'r.main.read.data'}),
            ('Explain read', {'r.main.read'}),
            ('Explain data', {'r.main.data'}),
            ('How to read data', {'r.main.read', 'r.main.data'}),
        ):
            with self.subTest(query=query):
                found = rc.resolve_code_entities(source_graph, query)
                self.assertEqual({item.name for item in found}, names)

    def test_longer_specific_identifier_preferred(self) -> None:
        """Prefer the longest recognized name when multiple names match.

        """
        source_graph = graph("pkg.read", "pkg.read_document")
        result = rc.resolve_code_entities(
            source_graph, "How does read_document call read?"
        )
        self.assertEqual([item.name for item in result],
                         ["pkg.read_document"])

    def test_qualified_and_ambiguous_leaf_names(self) -> None:
        """Return all matching graph entities when a leaf is ambiguous.

        """
        source_graph = graph("a.load", "b.load")
        result = rc.resolve_code_entities(source_graph, "load")
        self.assertEqual(
            {item.name for item in result}, {"a.load", "b.load"}
        )

    def test_empty_and_invalid_queries(self) -> None:
        """Reject whitespace-only text and non-string queries.

        """
        for query in ("", " \n", None, 0, []):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "query_text"):
                    rc.resolve_code_entities(graph("a.load"), query)

    def test_operator_symbols_and_reflected_dunders(self) -> None:
        """Literal operator symbols resolve corresponding Python methods.

        """
        source_graph = graph(
            "mod.Vector.__and__", "mod.Vector.__rand__",
            "mod.Vector.__add__", "mod.Vector.__eq__",
        )
        matched = rc.resolve_code_entities(source_graph, "What does a & b do?")
        self.assertEqual(
            {item.name for item in matched},
            {"mod.Vector.__and__", "mod.Vector.__rand__"},
        )
        equal = rc.resolve_code_entities(source_graph, "a == b")
        self.assertEqual([item.name for item in equal],
                         ["mod.Vector.__eq__"])

    def test_operator_symbols_do_not_match_embedded_punctuation(self) -> None:
        """Numbers and hyphenated prose should not seed arithmetic dunders.

        """
        source_graph = graph("p.__sub__", "p.__truediv__")
        self.assertEqual(
            rc.resolve_code_entities(source_graph, "well-known 9/30"), []
        )
        self.assertEqual(
            rc.resolve_code_entities(source_graph, "and the other"), []
        )

    def test_deduplicate_word_and_operator_matches(self) -> None:
        """An explicitly named dunder and operator refer to one entity.

        """
        source_graph = graph("p.__and__")
        result = rc.resolve_code_entities(
            source_graph, "Explain __and__ for a & b"
        )
        self.assertEqual([item.name for item in result], ["p.__and__"])

    def test_all_configured_operator_symbols(self) -> None:
        """Every configured literal operator maps to its declared dunders.

        """
        for operator, names in rc._OPERATOR_DUNDERS.items():
            with self.subTest(operator=operator):
                entities = graph(*(f"module.{name}" for name in names))
                found = rc.resolve_code_entities(
                    entities, f"How does (a {operator} b) work?"
                )
                self.assertEqual(
                    {item.name for item in found},
                    {f"module.{name}" for name in names},
                )

    def test_no_seeds_when_graph_is_empty(self) -> None:
        """Unknown symbols must not fabricate graph entities.

        """
        self.assertEqual(
            rc.resolve_code_entities(graph(), "load_document"), []
        )


    def test_name_matching_rejects_empty_name_words(self) -> None:
        """Punctuation-only entity names do not match ordinary questions.

        """
        self.assertFalse(rc._contains_whole_words("some query", ""))



class TestCodeSearch(unittest.TestCase):
    """Verify graph distance and chunk provenance in code search."""

    def test_dotted_r_function_chunk_is_retrieved(self) -> None:
        """A dotted R symbol is retrieved without separate leaf matches.

        """

        path = Path('r/main.R')
        source_graph = CodeGraph(
            entities=(
                CodeEntity('r.main.read.data', 'r_function', path),
                CodeEntity('r.main.data', 'r_function', path),
            ),
            relations=(),
        )
        chunks = [
            chunk('dotted', path='r/main.R', kind='r_function',
                  symbol='read.data'),
            chunk('plain', path='r/main.R', kind='r_function',
                  symbol='data'),
        ]
        found = rc.code_search(
            source_graph, chunks, 'Explain read.data', max_distance=0
        )
        self.assertEqual([result.chunk.identifier for result in found],
                         ['dotted'])

    def setUp(self) -> None:
        """Create a directed chain of named code functions.

        """
        self.graph = graph(
            "pkg.load", "pkg.parse", "pkg.store", "pkg.unrelated",
            relations=(
                ("pkg.load", "calls", "pkg.parse"),
                ("pkg.parse", "calls", "pkg.store"),
            ),
        )
        self.chunks = [
            chunk("load", "x", symbol="load"),
            chunk("parse", "x", symbol="parse"),
            chunk("store", "x", symbol="store"),
            chunk("other", "x", symbol="unrelated"),
            chunk("no-symbol", "x", symbol=None),
            chunk("bad-path", "x", path="bad.py", symbol="load"),
        ]

    def test_seeds_and_direct_neighbors(self) -> None:
        """Return matched and adjacent functions with distance-based scores.

        """
        results = rc.code_search(self.graph, self.chunks, "load", count=5)
        self.assertEqual([r.chunk.identifier for r in results],
                         ["load", "parse"])
        self.assertEqual([r.distance for r in results], [0, 1])
        self.assertEqual([r.score for r in results], [1.0, 0.5])
        with self.assertRaises(FrozenInstanceError):
            results[0].score = 0

    def test_zero_and_two_hop_limits(self) -> None:
        """Control graph expansion depth without changing seed identity.

        """
        one = rc.code_search(
            self.graph, self.chunks, "load", max_distance=0
        )
        three = rc.code_search(
            self.graph, self.chunks, "load", max_distance=2
        )
        self.assertEqual([r.chunk.identifier for r in one], ["load"])
        self.assertEqual([r.chunk.identifier for r in three],
                         ["load", "parse", "store"])
        self.assertEqual([r.distance for r in three], [0, 1, 2])

    def test_reversed_edges_are_traversable(self) -> None:
        """Graph search treats relations as undirected retrieval hints.

        """
        result = rc.code_search(self.graph, self.chunks, "parse")
        self.assertEqual(
            {r.chunk.identifier for r in result},
            {"load", "parse", "store"},
        )

    def test_unmatched_query_and_invalid_settings(self) -> None:
        """Unknown names return nothing; invalid limits raise errors.

        """
        self.assertEqual(
            rc.code_search(self.graph, self.chunks, "unrecognized"), []
        )
        for name, values in (
            ("count", (0, -1, True, 1.5, "2")),
            ("max_distance", (-1, True, 1.5, "1")),
        ):
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        rc.code_search(
                            self.graph, self.chunks, "load",
                            **{name: value},
                        )
        with self.assertRaisesRegex(ValueError, "query_text"):
            rc.code_search(self.graph, self.chunks, " ")

    def test_chunk_symbol_suffix_matching(self) -> None:
        """Qualified entities can match their source symbols by suffix.

        """
        self.assertTrue(rc._symbols_match("pkg.load", "load"))
        self.assertTrue(rc._symbols_match("pkg.load", "pkg.load"))
        self.assertFalse(rc._symbols_match("pkg.load", "load_more"))

    def test_forward_operator_beats_reflected_operator(self) -> None:
        """Prefer forward dunder over reflected sibling at the same depth.

        """
        source_graph = graph("pkg.__and__", "pkg.__rand__")
        chunks = [
            chunk("a", "x", symbol="__rand__"),
            chunk("z", "x", symbol="__and__"),
        ]
        ranked = rc.code_search(
            source_graph, chunks, "what does a & b do?"
        )
        self.assertEqual([r.chunk.identifier for r in ranked], ["z", "a"])

    def test_graph_distances_include_unlisted_relation_endpoints(self) -> None:
        """Traversal can visit unresolved names mentioned in graph edges.

        """
        g = graph("pkg.alpha", relations=(("pkg.alpha", "calls",
                                           "external"),))
        self.assertEqual(
            rc._graph_distances(g, {"pkg.alpha"}, 1),
            {"pkg.alpha": 0, "external": 1},
        )

    def test_multiple_candidate_entities_choose_nearest(self) -> None:
        """A chunk with a shared leaf name uses the shortest graph distance.

        """
        g = graph(
            "pkg.run", "pkg.other.run", "pkg.next",
            relations=(("pkg.run", "calls", "pkg.other.run"),),
        )
        result = rc.code_search(
            g, [chunk("one", symbol="run")], "pkg.run"
        )
        self.assertEqual(result[0].distance, 0)


if __name__ == "__main__":
    unittest.main()
