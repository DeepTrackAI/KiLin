"""Test deterministic context expansion for named files and code relationships.

Context expansion uses paths, names, graph edges, and ranked chunks.
No model is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from kilin.retrievals import context_expansion
from kilin.retrievals.code_graph import CodeEntity, CodeGraph, CodeRelation
from tests.kilin.retrievals._support import chunk, ranked


class TestContextExpansionPlanning(unittest.TestCase):
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
        result = context_expansion.plan_context_expansion(
            "How does it work?", self.graph
        )
        self.assertEqual(result.paths, ())
        self.assertEqual(result.functions, ())
        self.assertFalse(result.relationship)
        self.assertEqual(result.signals, ("none",))
        with self.assertRaises(FrozenInstanceError):
            result.signals = ()

    def test_full_path_precedes_basename(self) -> None:
        """A path named exactly takes precedence over ambiguous basenames.

        """
        result = context_expansion.plan_context_expansion(
            "Explain pkg/loader.py", self.graph
        )
        self.assertEqual(result.paths, (Path("pkg/loader.py"),))
        self.assertEqual(result.signals, ("exact file",))

    def test_backslash_and_case_insensitive_path(self) -> None:
        """Windows-style path separators and case-folding work uniformly.

        """
        result = context_expansion.plan_context_expansion(
            "Explain PKG\\LOADER.PY", self.graph
        )
        self.assertEqual(result.paths, (Path("pkg/loader.py"),))

    def test_basename_with_identifier_boundaries(self) -> None:
        """Names are recognized only when they form a bounded filename.

        """
        correct = context_expansion.plan_context_expansion(
            "loader.py implementation", self.graph
        )
        wrong = context_expansion.plan_context_expansion(
            "myloader.pyx implementation", self.graph
        )
        self.assertEqual(correct.paths, (Path("pkg/loader.py"),))
        self.assertEqual(wrong.paths, ())

    def test_named_function_and_relationship_expansion(self) -> None:
        """Relationship language expands named functions to direct neighbors.

        """
        result = context_expansion.plan_context_expansion(
            "Which function calls parse()?", self.graph
        )
        self.assertEqual(result.functions, ("pkg.loader.parse",))
        self.assertTrue(result.relationship)
        self.assertEqual(result.related_functions, ("pkg.loader.read",))
        self.assertEqual(result.signals,
                         ("exact function", "relationship"))

    def test_name_inside_filename_does_not_match_function(self) -> None:
        """Do not misidentify functions merely because a path contains them.

        """
        result = context_expansion.plan_context_expansion(
            "Inspect other/read.py", self.graph
        )
        self.assertEqual(result.paths, (Path("other/read.py"),))
        self.assertEqual(result.functions, ())

    def test_relationship_requires_a_named_function(self) -> None:
        """Relationship words alone do not trigger graph expansion.

        """
        result = context_expansion.plan_context_expansion(
            "Who calls functions?", self.graph
        )
        self.assertFalse(result.relationship)
        self.assertEqual(result.related_functions, ())

    def test_invalid_questions(self) -> None:
        """Reject blank or non-string questions during context planning.

        """
        for question in ("", "  ", 12, None, []):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "query_text"):
                    context_expansion.plan_context_expansion(
                        question, self.graph
                    )

    def test_direct_neighbors_skip_nonfunctions_and_seeds(self) -> None:
        """Only distinct function entities may be included as neighbors.

        """
        result = context_expansion._direct_function_neighbors(
            ("pkg.loader.read",), self.graph
        )
        self.assertEqual(result, ("pkg.loader.parse",))

    def test_path_detection_orders_multiple_hits(self) -> None:
        """Multiple explicit files are returned in deterministic path order.

        """
        result = context_expansion._explicit_paths(
            "pkg/notes.py and pkg/loader.py", self.graph
        )
        self.assertEqual(result,
                         (Path("pkg/loader.py"), Path("pkg/notes.py")))

    def test_ambiguous_plain_word_needs_a_code_marker(self) -> None:
        """A plain word shared by several functions must be marked as code.

        """
        result = context_expansion.plan_context_expansion(
            "Please read this.", self.graph
        )
        self.assertEqual(result.functions, ())
        for question in ("How does `read` work?", "What does read() do?",
                         "Explain the read function."):
            with self.subTest(question=question):
                result = context_expansion.plan_context_expansion(
                    question, self.graph
                )
                self.assertEqual(result.functions,
                                 ("other.read", "pkg.loader.read"))
        result = context_expansion.plan_context_expansion(
            "Explain loader.read.", self.graph
        )
        self.assertEqual(result.functions, ("pkg.loader.read",))

    def test_unique_plain_function_name_matches(self) -> None:
        """One top-level function with a plain-word name still matches.

        """
        result = context_expansion.plan_context_expansion(
            "How does parse work?", self.graph
        )
        self.assertEqual(result.functions, ("pkg.loader.parse",))

    def test_method_with_plain_name_needs_a_code_marker(self) -> None:
        """Ordinary words do not match methods; marked names do.

        """
        graph = CodeGraph(
            entities=(
                CodeEntity("m.Client.reply", "python_method", Path("m.py")),
                CodeEntity("m.Batch.to", "python_method", Path("m.py")),
                CodeEntity("m.go", "python_function", Path("m.py")),
            ),
            relations=(),
        )
        for question in ("Explain how to send a reply.",
                         "When does the go signal start?"):
            with self.subTest(question=question):
                result = context_expansion.plan_context_expansion(
                    question, graph
                )
                self.assertEqual(result.functions, ())
        for question in ("Explain the reply method.", "What does reply() do?",
                         "Explain Client.reply."):
            with self.subTest(question=question):
                result = context_expansion.plan_context_expansion(
                    question, graph
                )
                self.assertEqual(result.functions, ("m.Client.reply",))

    def test_code_like_names_match_without_markers(self) -> None:
        """Underscores, digits, and inner capitals identify code names.

        """
        for name in ("parse_file", "_helper", "parse2", "parseFile",
                     "ParseFile"):
            with self.subTest(name=name):
                self.assertTrue(context_expansion._looks_like_code(name))
        for name in ("parse", "Parse", "to", "reply"):
            with self.subTest(name=name):
                self.assertFalse(context_expansion._looks_like_code(name))

    def test_singular_call_is_a_relationship_phrase(self) -> None:
        """'Which functions call X' asks for relationships like 'calls'.

        """
        result = context_expansion.plan_context_expansion(
            "Which functions call `parse`?", self.graph
        )
        self.assertTrue(result.relationship)
        self.assertEqual(result.related_functions, ("pkg.loader.read",))

    def test_named_function_without_relationship_adds_callees(self) -> None:
        """Explaining a function also brings the functions it calls.

        """
        result = context_expansion.plan_context_expansion(
            "Explain loader.read.", self.graph
        )
        self.assertFalse(result.relationship)
        self.assertEqual(result.related_functions, ("pkg.loader.parse",))
        self.assertEqual(result.signals, ("exact function", "callees"))

    def test_callees_prefer_shared_words_and_respect_limit(self) -> None:
        """Callees sharing words with the question come first.

        """
        graph = CodeGraph(
            entities=(
                CodeEntity("m.route_question", "python_function",
                           Path("m.py")),
                CodeEntity("m._parse_route", "python_function", Path("m.py")),
                CodeEntity("m.side_reply", "python_function", Path("m.py")),
                CodeEntity("m.unrelated", "python_function", Path("m.py")),
                CodeEntity("m.Decision", "python_class", Path("m.py")),
            ),
            relations=(
                CodeRelation("m.route_question", "calls", "m.unrelated"),
                CodeRelation("m.route_question", "calls", "m.side_reply"),
                CodeRelation("m.route_question", "calls", "m._parse_route"),
                CodeRelation("m.route_question", "calls", "m.Decision"),
                CodeRelation("m.other", "calls", "m.route_question"),
            ),
        )
        question = "How does route_question use the router reply?"
        for limit, expected in (
            (3, ("m._parse_route", "m.side_reply", "m.unrelated")),
            (1, ("m._parse_route",)),
            (0, ()),
        ):
            with self.subTest(limit=limit):
                self.assertEqual(
                    context_expansion._ranked_callees(
                        ("m.route_question",), graph, question, limit
                    ),
                    expected,
                )

    def test_related_test_code_needs_a_question_about_tests(self) -> None:
        """Test functions are related context only for questions on tests.

        """
        graph = CodeGraph(
            entities=(
                CodeEntity("pkg.core.fuse_ranks", "python_function",
                           Path("pkg/core.py")),
                CodeEntity("pkg.api.search", "python_function",
                           Path("pkg/api.py")),
                CodeEntity("tests.test_core.test_fuse", "python_function",
                           Path("tests/test_core.py")),
                CodeEntity("pkg.core_test.check", "python_function",
                           Path("pkg/core_test.py")),
            ),
            relations=(
                CodeRelation("pkg.api.search", "calls",
                             "pkg.core.fuse_ranks"),
                CodeRelation("tests.test_core.test_fuse", "calls",
                             "pkg.core.fuse_ranks"),
                CodeRelation("pkg.core_test.check", "calls",
                             "pkg.core.fuse_ranks"),
            ),
        )
        result = context_expansion.plan_context_expansion(
            "Which functions call fuse_ranks?", graph
        )
        self.assertEqual(result.related_functions, ("pkg.api.search",))
        result = context_expansion.plan_context_expansion(
            "Which tests call fuse_ranks?", graph
        )
        self.assertEqual(
            result.related_functions,
            ("pkg.api.search", "pkg.core_test.check",
             "tests.test_core.test_fuse"),
        )

    def test_callees_follow_call_steps_up_to_depth(self) -> None:
        """Callees of callees are reachable, nearer ones first on ties.

        """
        graph = CodeGraph(
            entities=(
                CodeEntity("m.split_text", "python_function", Path("m.py")),
                CodeEntity("m._split_unit", "python_function", Path("m.py")),
                CodeEntity("m._overlap_start", "python_function",
                           Path("m.py")),
                CodeEntity("m._cut", "python_function", Path("m.py")),
            ),
            relations=(
                CodeRelation("m.split_text", "calls", "m._split_unit"),
                CodeRelation("m._split_unit", "calls", "m._overlap_start"),
                CodeRelation("m._overlap_start", "calls", "m._cut"),
            ),
        )
        question = "How does split_text keep the overlap?"
        for depth, expected in (
            (1, ("m._split_unit",)),
            (2, ("m._split_unit", "m._overlap_start")),
            (3, ("m._split_unit", "m._overlap_start", "m._cut")),
            (0, ()),
        ):
            with self.subTest(depth=depth):
                self.assertEqual(
                    routing._ranked_callees(
                        ("m.split_text",), graph, question, 5, depth
                    ),
                    expected,
                )

    def test_named_class_adds_its_methods(self) -> None:
        """A small class adds all methods; a large one the relevant ones.

        """
        small = [f"m.Lock.{name}" for name in ("acquire", "release")]
        special = [f"m.Lock.{name}" for name in ("__init__", "__exit__")]
        large = [
            f"m.Session.{name}"
            for name in ("_prepare_retrieval", "_index_metadata", "ask",
                         "close", "reset", "_emit_progress")
        ]
        graph = CodeGraph(
            entities=(
                CodeEntity("m.Lock", "python_class", Path("m.py")),
                CodeEntity("m.Session", "python_class", Path("m.py")),
                CodeEntity("m.Chunk", "python_class", Path("m.py")),
                *(CodeEntity(name, "python_method", Path("m.py"))
                  for name in (*small, *special, *large, "m.Chunk.text")),
            ),
            relations=(),
        )
        result = routing.plan_retrieval(
            "How does `Lock` stop two writers?", graph
        )
        self.assertEqual(result.related_functions, tuple(sorted(small)))
        self.assertIn("exact class", result.signals)
        self.assertEqual(
            routing._class_methods(
                ("m.Session",), graph,
                "How does Session decide to reuse the retrieval indexes?", 2,
            ),
            ("m.Session._index_metadata", "m.Session._prepare_retrieval"),
        )
        result = routing.plan_retrieval("What is in a Chunk?", graph)
        self.assertEqual(result.related_functions, ())
        result = routing.plan_retrieval(
            "What is in the Chunk class?", graph
        )
        self.assertEqual(result.related_functions, ("m.Chunk.text",))

    def test_plural_words_match_singular_names(self) -> None:
        """Simple plurals share words with singular identifier parts.

        """
        words = routing._identifier_words("saved indexes and policies")
        self.assertEqual(routing._shared_words(words, "_index_metadata"), 1)
        self.assertEqual(routing._shared_words(words, "policy"), 1)
        self.assertEqual(routing._shared_words(words, "_save"), 0)

    def test_demote_test_results_keeps_order_within_groups(self) -> None:
        """Results from test code follow source results unless asked for.

        """
        results = [
            ranked(chunk("t1", path="tests/test_a.py")),
            ranked(chunk("s1", path="pkg/a.py")),
            ranked(chunk("t2", path="pkg/a_test.py")),
            ranked(chunk("s2", path="pkg/b.py")),
        ]
        demoted = routing.demote_test_results(results, "How does a work?")
        self.assertEqual([r.chunk.identifier for r in demoted],
                         ["s1", "s2", "t1", "t2"])
        unchanged = routing.demote_test_results(results, "Which tests?")
        self.assertEqual([r.chunk.identifier for r in unchanged],
                         ["t1", "s1", "t2", "s2"])

    def test_identifier_words_split_code_names(self) -> None:
        """Split snake_case and camelCase identifiers into words.

        """
        self.assertEqual(
            context_expansion._identifier_words("_parse_route sideReply"),
            ["parse", "route", "side", "reply"],
        )

    def test_function_kinds_are_explicitly_limited(self) -> None:
        """Named class or module records are not matched as functions.

        """
        self.assertEqual(
            context_expansion._explicit_functions(
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
        plan = context_expansion.ContextExpansionPlan(
            (), (), False, (), ("none",)
        )
        result = context_expansion.expand_retrieval_context(
            [ranked(self.c), ranked(self.a)], self.chunks,
            self.graph, plan,
        )
        self.assertEqual([c.identifier for c in result], ["c", "a"])

    def test_explicit_file_includes_all_source_order_chunks(self) -> None:
        """File expansion appends unranked chunks in source order.

        """
        plan = context_expansion.ContextExpansionPlan(
            (Path("pkg.py"),), (), False, (), ("exact file",)
        )
        result = context_expansion.expand_retrieval_context(
            [ranked(self.c)], self.chunks, self.graph, plan
        )
        self.assertEqual([c.identifier for c in result], ["c", "a", "b"])

    def test_named_function_and_related_function(self) -> None:
        """Guarantee named function and its direct graph neighbor.

        """
        plan = context_expansion.ContextExpansionPlan(
            (), ("pkg.run",), True, ("pkg.helper",),
            ("exact function", "relationship"),
        )
        result = context_expansion.expand_retrieval_context(
            [], self.chunks, self.graph, plan
        )
        self.assertEqual([c.identifier for c in result], ["a", "b"])

    def test_callee_plan_without_relationship_expands(self) -> None:
        """Planned callees are added even without relationship language.

        """
        plan = context_expansion.ContextExpansionPlan(
            (), ("pkg.run",), False, ("pkg.helper",),
            ("exact function", "callees"),
        )
        result = context_expansion.expand_retrieval_context(
            [], self.chunks, self.graph, plan
        )
        self.assertEqual([c.identifier for c in result], ["a", "b"])

    def test_long_related_function_keeps_relevant_chunks(self) -> None:
        """Only the call site and the most relevant chunks are kept.

        """
        parts = [
            chunk("p1", "def caller(): setup", path="pkg.py", start=10,
                  symbol="caller"),
            chunk("p2", "unrelated cleanup", path="pkg.py", start=20,
                  symbol="caller"),
            chunk("p3", "result = run(data)", path="pkg.py", start=30,
                  symbol="caller"),
            chunk("p4", "save the cache", path="pkg.py", start=40,
                  symbol="caller"),
        ]
        kept = routing._focused_chunks(parts, {"run"}, ("cache",), 2)
        self.assertEqual([c.identifier for c in kept], ["p3", "p4"])
        self.assertEqual(
            routing._focused_chunks(parts, {"run"}, (), 0), parts
        )
        self.assertEqual(
            routing._focused_chunks(parts[:2], {"run"}, (), 2), parts[:2]
        )

    def test_missing_function_and_missing_symbol(self) -> None:
        """Unindexed names and symbol-free chunks cannot match functions.

        """
        plan = context_expansion.ContextExpansionPlan(
            (), ("pkg.absent",), False, (), ("exact function",)
        )
        result = context_expansion.expand_retrieval_context(
            [], self.chunks, self.graph, plan
        )
        self.assertEqual(result, [])
        entity = self.graph.entities[0]
        self.assertFalse(
            context_expansion._chunk_matches_function(self.c, entity)
        )

    def test_deduplication_uses_chunk_identifiers(self) -> None:
        """Repeated evidence from several sources appears only once.

        """
        plan = context_expansion.ContextExpansionPlan(
            (Path("pkg.py"),), ("pkg.run",), True,
            ("pkg.helper",), ("all",),
        )
        result = context_expansion.expand_retrieval_context(
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
            context_expansion._source_order([b, a]), [a, b]
        )

    def test_identifier_matching_escapes_punctuation(self) -> None:
        """Literal function names cannot act as injected regex syntax.

        """
        self.assertTrue(context_expansion._contains_identifier(
            "show pkg.name?", "pkg.name"
        ))
        self.assertFalse(context_expansion._contains_identifier(
            "show pkgxname", "pkg.name"
        ))
        self.assertFalse(context_expansion._contains_identifier(
            "my_reading", "read"
        ))

    def test_function_matching_qualified_and_leaf_symbols(self) -> None:
        """Match qualified, suffix, and simple function symbols.

        """
        entity = self.graph.entities[0]
        self.assertTrue(
            context_expansion._chunk_matches_function(self.a, entity)
        )
        qualified = chunk("q", path="pkg.py", symbol="pkg.run")
        self.assertTrue(
            context_expansion._chunk_matches_function(qualified, entity)
        )
        self.assertFalse(
            context_expansion._chunk_matches_function(self.b, entity)
        )
        self.assertFalse(context_expansion._chunk_matches_function(
            chunk("bad", path="else.py", symbol="run"), entity
        ))

    def test_remove_recognized_paths(self) -> None:
        """Remove full and basename forms before detecting function names.

        """
        text = context_expansion._remove_path_mentions(
            "pkg/read.py and read.py call read", (Path("pkg/read.py"),)
        )
        self.assertIn("call read", text)
        self.assertNotIn("read.py", text)


class TestContextExpansionExports(unittest.TestCase):
    """Require the new public names without legacy routing aliases."""

    def test_public_names(self) -> None:
        """Export the new API through both package namespaces.

        """
        import kilin
        from kilin import retrievals

        for namespace in (kilin, retrievals):
            with self.subTest(namespace=namespace.__name__):
                self.assertIs(
                    namespace.ContextExpansionPlan,
                    context_expansion.ContextExpansionPlan,
                )
                self.assertIs(
                    namespace.plan_context_expansion,
                    context_expansion.plan_context_expansion,
                )
                self.assertIs(
                    namespace.expand_retrieval_context,
                    context_expansion.expand_retrieval_context,
                )
                self.assertFalse(hasattr(namespace, "RetrievalPlan"))
                self.assertFalse(hasattr(namespace, "plan_retrieval"))


if __name__ == "__main__":
    unittest.main()
