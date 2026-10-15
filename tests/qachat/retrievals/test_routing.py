"""Test deterministic routing for named files and code relationships.

No model is required: routing operates on paths, names, graph edges, and
previously ranked chunks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from qachat.retrievals import routing
from qachat.retrievals.code_graph import CodeEntity, CodeGraph, CodeRelation
from tests.qachat.retrievals._support import chunk, ranked


class TestRetrievalPlanning(unittest.TestCase):
    """Verify explicit path/function recognition and relationship signals."""

    def setUp(self) -> None:
        """Create overlapping function names in different source files.

        """
        self.graph = CodeGraph(
            entities=(
                CodeEntity("pkg.loader", "source_file",
                           Path("pkg/loader.py")),
                CodeEntity("pkg.loader.read", "python_function",
                           Path("pkg/loader.py")),
                CodeEntity("pkg.loader.parse", "python_function",
                           Path("pkg/loader.py")),
                CodeEntity("other.read", "python_function",
                           Path("other/read.py")),
                CodeEntity("pkg.notes", "source_file",
                           Path("pkg/notes.py")),
            ),
            relations=(
                CodeRelation("pkg.loader.read", "calls",
                             "pkg.loader.parse"),
                CodeRelation("pkg.loader.read", "imports",
                             "pkg.notes"),
            ),
        )

    def test_no_explicit_match_gives_none_signal(self) -> None:
        """Unrecognized prose must not invent a file or function reference.

        """
        result = routing.plan_retrieval("How does it work?", self.graph)
        self.assertEqual(result.paths, ())
        self.assertEqual(result.functions, ())
        self.assertFalse(result.relationship)
        self.assertEqual(result.signals, ("none",))
        with self.assertRaises(FrozenInstanceError):
            result.signals = ()

    def test_full_path_precedes_basename(self) -> None:
        """A path named exactly takes precedence over ambiguous basenames.

        """
        result = routing.plan_retrieval(
            "Explain pkg/loader.py", self.graph
        )
        self.assertEqual(result.paths, (Path("pkg/loader.py"),))
        self.assertEqual(result.signals, ("exact file",))

    def test_backslash_and_case_insensitive_path(self) -> None:
        """Windows-style path separators and case-folding work uniformly.

        """
        result = routing.plan_retrieval(
            "Explain PKG\\LOADER.PY", self.graph
        )
        self.assertEqual(result.paths, (Path("pkg/loader.py"),))

    def test_basename_with_identifier_boundaries(self) -> None:
        """Names are recognized only when they form a bounded filename.

        """
        correct = routing.plan_retrieval("loader.py implementation",
                                         self.graph)
        wrong = routing.plan_retrieval("myloader.pyx implementation",
                                       self.graph)
        self.assertEqual(correct.paths, (Path("pkg/loader.py"),))
        self.assertEqual(wrong.paths, ())

    def test_named_function_and_relationship_expansion(self) -> None:
        """Relationship language expands named functions to direct neighbors.

        """
        result = routing.plan_retrieval(
            "Which function calls parse?", self.graph
        )
        self.assertEqual(result.functions, ("pkg.loader.parse",))
        self.assertTrue(result.relationship)
        self.assertEqual(result.related_functions, ("pkg.loader.read",))
        self.assertEqual(result.signals,
                         ("exact function", "relationship"))

    def test_name_inside_filename_does_not_match_function(self) -> None:
        """Do not misidentify functions merely because a path contains them.

        """
        result = routing.plan_retrieval("Inspect other/read.py", self.graph)
        self.assertEqual(result.paths, (Path("other/read.py"),))
        self.assertEqual(result.functions, ())

    def test_relationship_requires_a_named_function(self) -> None:
        """Relationship words alone do not trigger graph expansion.

        """
        result = routing.plan_retrieval("Who calls functions?", self.graph)
        self.assertFalse(result.relationship)
        self.assertEqual(result.related_functions, ())

    def test_invalid_questions(self) -> None:
        """Reject blank or non-string questions before routing.

        """
        for question in ("", "  ", 12, None, []):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "query_text"):
                    routing.plan_retrieval(question, self.graph)

    def test_direct_neighbors_skip_nonfunctions_and_seeds(self) -> None:
        """Only distinct function entities may be included as neighbors.

        """
        result = routing._direct_function_neighbors(
            ("pkg.loader.read",), self.graph
        )
        self.assertEqual(result, ("pkg.loader.parse",))

    def test_path_detection_orders_multiple_hits(self) -> None:
        """Multiple explicit files are returned in deterministic path order.

        """
        result = routing._explicit_paths(
            "pkg/notes.py and pkg/loader.py", self.graph
        )
        self.assertEqual(result,
                         (Path("pkg/loader.py"), Path("pkg/notes.py")))

    def test_function_kinds_are_explicitly_limited(self) -> None:
        """Named class or module records are not routed as functions.

        """
        self.assertEqual(
            routing._explicit_functions(
                "pkg.notes", self.graph, ()
            ),
            (),
        )


class TestRetrievalExpansion(unittest.TestCase):
    """Check ordered, nonduplicating evidence expansion."""

    def setUp(self) -> None:
        """Build chunks with a file and two related source functions.

        """
        self.graph = CodeGraph(
            entities=(
                CodeEntity("pkg.run", "python_function", Path("pkg.py")),
                CodeEntity("pkg.helper", "python_function", Path("pkg.py")),
            ),
            relations=(CodeRelation("pkg.run", "calls", "pkg.helper"),),
        )
        self.a = chunk("a", "head", path="pkg.py", start=2,
                       symbol="run")
        self.b = chunk("b", "tail", path="pkg.py", start=4,
                       symbol="helper")
        self.c = chunk("c", "other", path="other.py", start=1)
        self.chunks = [self.c, self.b, self.a]

    def test_initial_ranked_results_preserve_order(self) -> None:
        """Initial RRF evidence precedes expansion without re-ranking.

        """
        plan = routing.RetrievalPlan((), (), False, (), ("none",))
        result = routing.expand_retrieval_context(
            [ranked(self.c), ranked(self.a)], self.chunks,
            self.graph, plan,
        )
        self.assertEqual([c.identifier for c in result], ["c", "a"])

    def test_explicit_file_includes_all_source_order_chunks(self) -> None:
        """File expansion appends unranked chunks in source order.

        """
        plan = routing.RetrievalPlan(
            (Path("pkg.py"),), (), False, (), ("exact file",)
        )
        result = routing.expand_retrieval_context(
            [ranked(self.c)], self.chunks, self.graph, plan
        )
        self.assertEqual([c.identifier for c in result], ["c", "a", "b"])

    def test_named_function_and_related_function(self) -> None:
        """Guarantee named function and its direct graph neighbor.

        """
        plan = routing.RetrievalPlan(
            (), ("pkg.run",), True, ("pkg.helper",),
            ("exact function", "relationship"),
        )
        result = routing.expand_retrieval_context(
            [], self.chunks, self.graph, plan
        )
        self.assertEqual([c.identifier for c in result], ["a", "b"])

    def test_missing_function_and_missing_symbol(self) -> None:
        """Unindexed names and symbol-free chunks cannot match functions.

        """
        plan = routing.RetrievalPlan(
            (), ("pkg.absent",), False, (), ("exact function",)
        )
        result = routing.expand_retrieval_context(
            [], self.chunks, self.graph, plan
        )
        self.assertEqual(result, [])
        entity = self.graph.entities[0]
        self.assertFalse(routing._chunk_matches_function(self.c, entity))

    def test_deduplication_uses_chunk_identifiers(self) -> None:
        """Repeated evidence from several sources appears only once.

        """
        plan = routing.RetrievalPlan(
            (Path("pkg.py"),), ("pkg.run",), True,
            ("pkg.helper",), ("all",),
        )
        result = routing.expand_retrieval_context(
            [ranked(self.a), ranked(self.a)], self.chunks,
            self.graph, plan,
        )
        self.assertEqual([c.identifier for c in result], ["a", "b"])

    def test_source_order_ties_use_identifier(self) -> None:
        """Path, line, and identifier form a complete ordering key.

        """
        a = chunk("a", start=3)
        b = chunk("b", start=3)
        self.assertEqual(
            routing._source_order([b, a]), [a, b]
        )

    def test_identifier_matching_escapes_punctuation(self) -> None:
        """Literal function names cannot act as injected regex syntax.

        """
        self.assertTrue(routing._contains_identifier(
            "show pkg.name?", "pkg.name"
        ))
        self.assertFalse(routing._contains_identifier(
            "show pkgxname", "pkg.name"
        ))
        self.assertFalse(routing._contains_identifier(
            "my_reading", "read"
        ))

    def test_function_matching_qualified_and_leaf_symbols(self) -> None:
        """Match qualified, suffix, and simple function symbols.

        """
        entity = self.graph.entities[0]
        self.assertTrue(routing._chunk_matches_function(self.a, entity))
        qualified = chunk("q", path="pkg.py", symbol="pkg.run")
        self.assertTrue(routing._chunk_matches_function(qualified, entity))
        self.assertFalse(routing._chunk_matches_function(self.b, entity))
        self.assertFalse(routing._chunk_matches_function(
            chunk("bad", path="else.py", symbol="run"), entity
        ))

    def test_remove_recognized_paths(self) -> None:
        """Remove full and basename forms before detecting function names.

        """
        text = routing._remove_path_mentions(
            "pkg/read.py and read.py call read", (Path("pkg/read.py"),)
        )
        self.assertIn("call read", text)
        self.assertNotIn("read.py", text)


if __name__ == "__main__":
    unittest.main()
