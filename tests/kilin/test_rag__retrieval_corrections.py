"""Regression tests for broad retrieval with accurate scoring and provenance.

Search, expansion, prompt rendering, and persistence helpers are real;
only query embeddings are deterministic substitutes for local model calls.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin import rag
from kilin.retrievals import context_expansion as expansion
from kilin.retrievals import build_lexical_index
from kilin.retrievals.code_graph import CodeEntity, CodeGraph, CodeRelation
from tests.kilin import _rag_support as fixtures
from tests.kilin.retrievals._support import chunk, ranked


def _retrieval_session() -> rag.RAGSession:
    """Create aligned chunks, a code graph, and deliberately nonunit vectors.

    Returns
    -------
    rag.RAGSession
        Offline session exercising real retrieval and expansion.

    """

    session = fixtures._session()
    session.chunks = [
        chunk("a", "def load_value(): return 1", path="pkg/a.py",
              symbol="load_value"),
        chunk("b", "def parse_config(): return 2", path="pkg/b.py",
              symbol="parse_config"),
        chunk("guide", "Use pkg.b.parse_config() for loading settings.",
              path="docs/guide.md", kind="markdown_section"),
    ]
    session.indexes = SimpleNamespace(
        semantic=rag.SemanticIndex(
            tuple(session.chunks), ((1.0, 1.0), (10.0, 1.0), (1.0, 0.0))
        ),
        lexical=build_lexical_index(session.chunks),
        code_graph=CodeGraph(
            entities=(
                CodeEntity("pkg.a", "source_file", Path("pkg/a.py")),
                CodeEntity("pkg.b", "source_file", Path("pkg/b.py")),
                CodeEntity("pkg.a.load_value", "python_function",
                           Path("pkg/a.py")),
                CodeEntity("pkg.b.parse_config", "python_function",
                           Path("pkg/b.py")),
            ),
            relations=(),
        ),
    )
    session.embedding_session = Mock()
    session.embedding_session.embed_query.return_value = [1.0, 0.0]
    return session


class TestReusedSemanticScores(unittest.TestCase):
    """Expansion uses the same cosine scores, without extra model calls."""

    def test_nonunit_scores_keep_cosine_order(self):
        session = _retrieval_session()
        results = rag.semantic_search(
            session.indexes.semantic, "formatted", session.embedding_session,
            count=len(session.chunks), trace_turn_id="turn-1",
        )
        score = session._passage_relevance(results)
        self.assertEqual(results[0].chunk.identifier, "guide")
        self.assertGreater(score(session.chunks[2]), score(session.chunks[1]))
        self.assertAlmostEqual(score(session.chunks[1]), 10 / 101 ** 0.5)
        for result in results:
            self.assertEqual(score(result.chunk), result.score)
        self.assertEqual(score(chunk("missing")), float("-inf"))
        session.embedding_session.embed_query.assert_called_once_with(
            "formatted", trace_turn_id="turn-1"
        )

    def test_expansion_and_named_scope_embed_once(self):
        for scope in ((), (Path("pkg/a.py"),)):
            with self.subTest(scope=scope):
                session = _retrieval_session()
                evidence = session._retrieve(
                    "Explain loading settings", 3,
                    trace_turn_id="turn-7", scope=scope,
                )
                self.assertTrue(evidence)
                session.embedding_session.embed_query.assert_called_once_with(
                    "Explain loading settings", trace_turn_id="turn-7"
                )

    def test_each_new_query_gets_its_own_embedding(self):
        session = _retrieval_session()
        session._retrieve("Explain loading", 3)
        session._retrieve("Explain parsing", 3)
        self.assertEqual(session.embedding_session.embed_query.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in
             session.embedding_session.embed_query.call_args_list],
            ["Explain loading", "Explain parsing"],
        )

    def test_invalid_query_vectors_fail_before_expansion(self):
        for vector in ([1.0], [0.0, 0.0], [float("nan"), 1.0]):
            with self.subTest(vector=vector):
                session = _retrieval_session()
                session.embedding_session.embed_query.return_value = vector
                with patch.object(rag, "expand_retrieval_context") as expand:
                    with self.assertRaises(ValueError):
                        session._retrieve("Explain loading", 3)
                expand.assert_not_called()
                self.assertEqual(
                    session.embedding_session.embed_query.call_count, 1
                )

    def test_full_ranking_reuses_complete_semantic_results(self):
        session = _retrieval_session()
        results = rag.semantic_search(
            session.indexes.semantic, "question", session.embedding_session,
            count=len(session.chunks),
        )
        reused = session._rank_everything("question", semantic_results=results)
        session.embedding_session.embed_query.assert_called_once()
        separate = session._rank_everything("question")
        self.assertEqual(reused, separate)
        self.assertEqual(session.embedding_session.embed_query.call_count, 2)

    def test_fusion_retains_original_semantic_depth(self):
        session = _retrieval_session()
        with patch.object(
            rag, "hybrid_search", wraps=rag.hybrid_search
        ) as fuse:
            session._retrieve("Explain the process", 1)
        semantic = fuse.call_args_list[0].kwargs["semantic"]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0].chunk.identifier, "guide")


class TestFocusedAndAlternativeEvidence(unittest.TestCase):
    """Broader matches remain available without displacing explicit targets."""

    def test_named_file_is_first_but_cross_file_expansion_remains(self):
        session = _retrieval_session()
        evidence = session._retrieve(
            "Explain pkg/a.py", 3, scope=(Path("pkg/a.py"),)
        )
        self.assertEqual(evidence[0].path, Path("pkg/a.py"))
        self.assertEqual(evidence[0].context_role, "direct")
        other = next(
            item for item in evidence if item.path == Path("pkg/b.py")
        )
        self.assertEqual(other.context_role, "related")
        self.assertEqual(other.text, session.chunks[1].text)
        self.assertEqual([item.citation for item in evidence],
                         list(range(1, len(evidence) + 1)))
        prompt = rag._answer_prompt("Explain pkg/a.py", evidence)
        self.assertIn("[context: direct]", prompt)
        self.assertIn("[context: related]", prompt)
        self.assertIn("not proof of identity", prompt)

    def test_named_function_precedes_higher_ranked_unrelated_text(self):
        session = _retrieval_session()
        evidence = session._retrieve("Explain load_value()", 3)
        self.assertEqual(evidence[0].symbol, "load_value")
        self.assertEqual(evidence[0].context_role, "direct")
        self.assertIn(Path("docs/guide.md"), [item.path for item in evidence])

    def test_fallback_remains_available_but_is_labeled_alternative(self):
        session = _retrieval_session()
        guide = replace(
            session.chunks[2], text="Use other.parse_config() for settings."
        )
        session.chunks[2] = guide
        session.indexes.semantic = replace(
            session.indexes.semantic, chunks=tuple(session.chunks)
        )
        session.indexes.lexical = build_lexical_index(session.chunks)
        evidence = session._retrieve("Explain settings", 3)
        candidate = next(
            item for item in evidence if item.symbol == "parse_config"
        )
        self.assertEqual(candidate.context_role, "alternative")
        self.assertEqual(candidate.path, Path("pkg/b.py"))
        self.assertEqual(evidence[-1], candidate)
        prompt = rag._answer_prompt("Explain settings", evidence)
        self.assertIn("[context: alternative]", prompt)
        self.assertIn("pkg/b.py", prompt)

    def test_prior_targets_become_related_but_alternatives_stay_cautious(self):
        for old, expected in (("direct", "related"),
                              ("alternative", "alternative"), (None, None)):
            with self.subTest(old=old):
                item = replace(fixtures._evidence(), context_role=old)
                turn = fixtures._turn(evidence=(item,))
                added = rag._add_related_evidence((), [turn])
                reused = rag._merge_evidence([(item,)])
                self.assertEqual(added[0].context_role, expected)
                self.assertEqual(reused[0].context_role, expected)
                self.assertEqual(turn.evidence[0].context_role, old)

    def test_new_provenance_roundtrips_and_old_records_still_load(self):
        for role in (None, "direct", "related", "alternative"):
            with self.subTest(role=role):
                evidence = replace(fixtures._evidence(), context_role=role)
                turn = fixtures._turn(evidence=(evidence,))
                saved = rag._turn_to_data(turn)
                self.assertEqual(rag._turn_from_data(saved), turn)
                del saved["evidence"][0]["context_role"]
                restored = rag._turn_from_data(saved)
                self.assertIsNone(restored.evidence[0].context_role)
                self.assertEqual(restored.evidence[0].text, evidence.text)
        saved["evidence"][0]["context_role"] = "claimed_exact"
        with self.assertRaisesRegex(ValueError, "context_role"):
            rag._turn_from_data(saved)


class TestDocumentationBridgePriority(unittest.TestCase):
    """Distinguish resolvable references from plausible leaf-name fallbacks."""

    def setUp(self):
        self.graph = CodeGraph((
            CodeEntity("pkg.real.parse_config", "python_function",
                       Path("pkg/real.py")),
            CodeEntity("pkg.real.read_config", "python_function",
                       Path("pkg/real.py")),
        ), ())

    def _plan(self, text: str):
        """Plan expansion from one retrieved documentation passage."""
        return expansion.plan_context_expansion(
            "Explain settings", self.graph,
            [ranked(chunk("g", text, path="guide.md",
                          kind="markdown_section"))],
        )

    def test_unresolved_qualifier_retains_lower_confidence_candidate(self):
        plan = self._plan("Use other.parse_config().")
        self.assertEqual(plan.related_functions, ("pkg.real.parse_config",))
        self.assertEqual(
            plan.alternative_functions, ("pkg.real.parse_config",)
        )
        self.assertIn("alternative documentation names", plan.signals)
        self.assertNotIn("documentation names", plan.signals)

    def test_exact_suffix_and_unique_leaf_are_supported(self):
        for reference in ("pkg.real.parse_config", "real.parse_config",
                          "parse_config"):
            with self.subTest(reference=reference):
                plan = self._plan(f"Use {reference}().")
                self.assertEqual(plan.related_functions,
                                 ("pkg.real.parse_config",))
                self.assertEqual(plan.alternative_functions, ())

    def test_supported_match_takes_budget_before_repeated_fallback(self):
        with patch.object(expansion, "_DOCUMENTATION_NAME_EXPANSION_MAX", 1):
            plan = self._plan(
                "other.parse_config() " * 10 + "pkg.real.read_config()"
            )
        self.assertEqual(plan.related_functions, ("pkg.real.read_config",))
        self.assertEqual(plan.alternative_functions, ())

    def test_supported_occurrence_promotes_a_fallback(self):
        plan = self._plan("other.parse_config() then pkg.real.parse_config()")
        self.assertEqual(plan.related_functions, ("pkg.real.parse_config",))
        self.assertEqual(plan.alternative_functions, ())

    def test_ambiguous_unqualified_reference_is_not_guessed(self):
        self.graph = replace(self.graph, entities=(
            *self.graph.entities,
            CodeEntity("another.parse_config", "python_function",
                       Path("another.py")),
        ))
        plan = self._plan("Use other.parse_config().")
        self.assertEqual(plan.related_functions, ())
        plan = self._plan("Use pkg.real.parse_config().")
        self.assertEqual(plan.related_functions, ("pkg.real.parse_config",))
        self.assertEqual(plan.alternative_functions, ())

    def test_alternatives_do_not_displace_supported_related_passages(self):
        a = chunk("a", path="pkg/real.py", symbol="parse_config")
        b = chunk("b", path="pkg/real.py", symbol="read_config")
        plan = self._plan("other.parse_config() and pkg.real.read_config()")
        with patch.object(expansion, "_RELATED_PASSAGES_MAX", 1):
            selected = expansion.expand_retrieval_context(
                [], [a, b], self.graph, plan,
                lambda item: 100.0 if item.identifier == "a" else 0.0,
            )
        self.assertEqual(selected, [b])

    def test_alternative_priority_without_scorer_and_unbounded_budget(self):
        a = chunk("a", path="pkg/real.py", symbol="parse_config")
        b = chunk("b", path="pkg/real.py", symbol="read_config")
        plan = replace(
            self._plan("other.parse_config() and pkg.real.read_config()"),
            related_functions=(
                "pkg.real.parse_config", "pkg.real.read_config"
            ),
        )
        with patch.object(expansion, "_RELATED_PASSAGES_MAX", 1):
            selected = expansion.expand_retrieval_context(
                [], [a, b], self.graph, plan
            )
        self.assertEqual(selected, [b])
        with patch.object(expansion, "_RELATED_PASSAGES_MAX", 0):
            unbounded = expansion.expand_retrieval_context(
                [], [a, b], self.graph, plan
            )
        self.assertEqual(unbounded, [b, a])

    def test_multichunk_neighbors_respect_per_function_not_shared_cap(self):
        graph = CodeGraph((
            CodeEntity("pkg.core.parse", "python_function", Path("core.py")),
            CodeEntity("pkg.core.read", "python_function", Path("core.py")),
        ), (CodeRelation("pkg.core.read", "calls", "pkg.core.parse"),))
        target = chunk("target", path="core.py", symbol="parse")
        neighbors = [
            chunk(str(i), "parse()", path="core.py", symbol="read", start=i)
            for i in range(10, 14)
        ]
        for wording in ("Which function calls parse?",
                        "Which function calls parse()?"):
            plan = expansion.plan_context_expansion(wording, graph)
            with patch.object(expansion, "_RELATED_CHUNKS_MAX", 2), \
                    patch.object(expansion, "_RELATED_PASSAGES_MAX", 1):
                selected = expansion.expand_retrieval_context(
                    [], [target, *neighbors], graph, plan
                )
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected[0], target)
            self.assertEqual([item.start_line for item in selected[1:]],
                             [10, 11])
            with patch.object(expansion, "_RELATED_CHUNKS_MAX", 0):
                unbounded = expansion.expand_retrieval_context(
                    [], [target, *neighbors], graph, plan
                )
            self.assertEqual(unbounded, [target, *neighbors])


if __name__ == "__main__":
    unittest.main()
