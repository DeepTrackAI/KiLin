"""Verify complete-document selection from indexed retrieval evidence.

Selection is deterministic and uses estimated evidence budgets, without
loading external source files or embedding models.

"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from qachat.retrievals import document_reading as dr
from qachat.retrievals.code_graph import CodeEntity, CodeGraph
from tests.qachat.retrievals._support import chunk, ranked


class TestDocumentPathMatching(unittest.TestCase):
    """Verify filename, stem, author, and dominant-file heuristics."""

    def setUp(self) -> None:
        """Create a varied collection of literature source paths.

        """
        self.paths = [
            Path("papers/smith2020.pdf"),
            Path("papers/lee25networks.pdf"),
            Path("code/train.py"),
            Path("papers/smith2024.pdf"),
        ]

    def test_token_estimation_rounds_up(self) -> None:
        """Estimate one token per four characters, rounded upward.

        """
        for value, expected in (("", 0), ("a", 1),
                                ("abcd", 1), ("abcde", 2)):
            self.assertEqual(dr.estimate_tokens(value), expected)

    def test_full_path_preferred_to_basename(self) -> None:
        """An explicit repository path takes precedence over stem matching.

        """
        found = dr.named_paths("Read code/train.py", self.paths)
        self.assertEqual(found, [Path("code/train.py")])

    def test_backslash_path_matches(self) -> None:
        """Normalize Windows-style path separators in user questions.

        """
        found = dr.named_paths("inspect CODE\\TRAIN.PY", self.paths)
        self.assertEqual(found, [Path("code/train.py")])

    def test_basename_and_stem_matching(self) -> None:
        """Match exact filenames and sufficiently long bounded stems.

        """
        self.assertEqual(
            dr.named_paths("smith2020.pdf", self.paths),
            [Path("papers/smith2020.pdf")],
        )
        self.assertEqual(
            dr.named_paths("smith2020 study", self.paths),
            [Path("papers/smith2020.pdf")],
        )
        self.assertEqual(
            dr.named_paths("smith20201", self.paths), [],
        )

    def test_optional_author_year_recognition(self) -> None:
        """Author-name heuristics are opt-in and require safe boundaries.

        """
        self.assertEqual(dr.named_paths("Smith's paper", self.paths), [])
        self.assertEqual(
            dr.named_paths("Smith's paper", self.paths, authors=True),
            [Path("papers/smith2020.pdf"), Path("papers/smith2024.pdf")],
        )
        self.assertEqual(
            dr.named_paths("lee25 paper", self.paths, authors=True),
            [Path("papers/lee25networks.pdf")],
        )

    def test_author_boundary_cases(self) -> None:
        """Do not match author prefixes inside unrelated words.

        """
        self.assertFalse(dr._starts_stem("smithsonian", {"smith"}))
        self.assertTrue(dr._starts_stem("smith_data", {"smith"}))
        self.assertTrue(dr._starts_stem("smith2025", {"smith"}))
        self.assertTrue(dr._starts_stem("smith", {"smith"}))
        self.assertFalse(dr._starts_stem("smith25", {"other"}))

    def test_named_paths_keeps_input_order(self) -> None:
        """Multiple file-name matches preserve the candidate path ordering.

        """
        paths = [Path("b/train.py"), Path("a/train.py")]
        self.assertEqual(dr.named_paths("train.py", paths), paths)

    def test_dominant_path_scores_and_restriction(self) -> None:
        """Sum fusion scores per file, respecting explicit candidate paths.

        """
        first = chunk("a", path="a.py")
        second = chunk("b", path="b.py")
        hits = [ranked(first, 0.3), ranked(second, 0.5),
                ranked(first, 0.4)]
        self.assertEqual(dr.dominant_path(hits), Path("a.py"))
        self.assertEqual(
            dr.dominant_path(hits, [Path("b.py")]), Path("b.py")
        )
        self.assertIsNone(dr.dominant_path([], None))
        self.assertIsNone(dr.dominant_path(hits, [Path("missing.py")]))


class TestReadDocument(unittest.TestCase):
    """Verify target priority, token budgets, and evidence ordering."""

    def setUp(self) -> None:
        """Create two files and a recognized function within one file.

        """
        self.a = chunk("a", "aaaabbbb", path="a.py", start=1,
                       symbol="run")
        self.b = chunk("b", "ccccdddd", path="a.py", start=5,
                       symbol="helper")
        self.c = chunk("c", "eeefff", path="b.py", start=1)
        self.chunks = [self.a, self.b, self.c]
        self.graph = CodeGraph(
            entities=(
                CodeEntity("a.run", "python_function", Path("a.py")),
                CodeEntity("a.helper", "python_function", Path("a.py")),
            ),
            relations=(),
        )

    def test_no_chunks_or_ranking(self) -> None:
        """No evidence gives no selected document.

        """
        self.assertIsNone(dr.read_document("question", [], self.graph, [],
                                           max_tokens=100))
        self.assertIsNone(dr.read_document("question", self.chunks,
                                           self.graph, [], max_tokens=100))

    def test_named_function_wins_over_dominant_file(self) -> None:
        """A named function receives priority over unrelated ranked files.

        """
        selection = dr.read_document(
            "Explain helper", self.chunks, self.graph,
            [ranked(self.c, 1.0)], max_tokens=100,
        )
        self.assertEqual(selection.functions, ("a.helper",))
        self.assertEqual(selection.path, Path("a.py"))
        self.assertEqual([c.identifier for c in selection.chunks], ["b"])
        self.assertEqual(selection.reason, dr.DOCUMENT_REASON_NAMED_FUNCTION)
        self.assertTrue(selection.complete)
        with self.assertRaises(FrozenInstanceError):
            selection.complete = False


    def test_named_function_without_matching_chunk_falls_back(self) -> None:
        """If a named graph function lacks chunks, choose a file instead.

        """
        missing = CodeGraph(
            (CodeEntity("lost", "python_function", Path("none.py")),),
            (),
        )
        selected = dr.read_document(
            "Explain lost", self.chunks, missing,
            [ranked(self.c, 1.0)], max_tokens=100,
        )
        self.assertEqual(selected.path, Path("b.py"))
        self.assertEqual(selected.reason, dr.DOCUMENT_REASON_SEARCH)


    def test_named_file_wins_over_search(self) -> None:
        """An explicit filename narrows the document selection.

        """
        selection = dr.read_document(
            "Explain a.py", self.chunks, self.graph,
            [ranked(self.c, 1.0)], max_tokens=100,
        )
        self.assertEqual(selection.path, Path("a.py"))
        self.assertEqual(selection.reason, dr.DOCUMENT_REASON_NAMED_FILE)
        self.assertEqual(selection.total_chunks, 2)
        self.assertEqual([c.identifier for c in selection.chunks], ["a", "b"])

    def test_dominant_search_result_selects_file(self) -> None:
        """Choose the highest-scoring file when none is explicitly named.

        """
        selection = dr.read_document(
            "Explain the model", self.chunks, self.graph,
            [ranked(self.c, 0.8), ranked(self.a, 0.1)],
            max_tokens=100,
        )
        self.assertEqual(selection.path, Path("b.py"))
        self.assertEqual(selection.reason, dr.DOCUMENT_REASON_SEARCH)

    def test_explicit_path_override_when_ranked_has_other_file(self) -> None:
        """Caller-selected path is respected independently of rankings.

        """
        selection = dr.read_document(
            "question", self.chunks, self.graph,
            [ranked(self.c, 1.0)], max_tokens=100,
            path=Path("a.py"),
        )
        self.assertEqual(selection.path, Path("a.py"))
        self.assertEqual(selection.reason, dr.DOCUMENT_REASON_NAMED_FILE)

    def test_explicit_path_without_ranked_results(self) -> None:
        """An explicitly selected file needs no search hit to be read.

        """
        selection = dr.read_document(
            "question", self.chunks, self.graph, [],
            max_tokens=100, path=Path("a.py"),
        )
        self.assertEqual([c.identifier for c in selection.chunks], ["a", "b"])

    def test_budget_keeps_relevance_and_source_order(self) -> None:
        """Select relevant chunks first but return them in source order.

        """
        chunks = [
            chunk("a", "a" * 8, path="a.py", start=1),
            chunk("b", "b" * 8, path="a.py", start=2),
            chunk("c", "c" * 8, path="a.py", start=3),
        ]
        selection = dr.read_document(
            "Inspect a.py", chunks, CodeGraph((), ()),
            [ranked(chunks[2], 0.8), ranked(chunks[0], 0.3)],
            max_tokens=4,
        )
        self.assertEqual(
            [c.identifier for c in selection.chunks], ["a", "c"]
        )
        self.assertEqual(selection.estimated_tokens, 4)
        self.assertEqual(selection.total_chunks, 3)
        self.assertFalse(selection.complete)

    def test_no_chunk_fits_budget(self) -> None:
        """Return an incomplete empty selection for an undersized budget.

        """
        selection = dr.read_document(
            "a.py", self.chunks, self.graph, [], max_tokens=0
        )
        self.assertEqual(selection.chunks, ())
        self.assertFalse(selection.complete)
        self.assertEqual(selection.estimated_tokens, 0)

    def test_top_limits_dominant_file_evidence(self) -> None:
        """Only the leading ranked records influence unnamed file choice.

        """
        selection = dr.read_document(
            "question", self.chunks, self.graph,
            [ranked(self.a, 0.1), ranked(self.c, 9.0)],
            max_tokens=100, top=1,
        )
        self.assertEqual(selection.path, Path("a.py"))

    @unittest.expectedFailure
    def test_target_selection_uses_identifier_deduplication(self) -> None:
        """Pending: duplicated chunk IDs should not double-count evidence.

        """
        target = [self.a, self.a]
        selection = dr._fit(
            target, [], 100, path=Path("a.py"), functions=(),
            reason=dr.DOCUMENT_REASON_NAMED_FILE,
        )
        self.assertEqual(selection.estimated_tokens,
                         dr.estimate_tokens(self.a.text))
        self.assertEqual(selection.total_chunks, 2)


    @unittest.expectedFailure
    def test_pending_named_functions_remain_in_one_file(self) -> None:
        """Pending: one DocumentSelection must reference a single source path.

        """
        chunks = [
            chunk("one", "hello", path="a.py", symbol="foo"),
            chunk("two", "hello", path="b.py", symbol="bar"),
        ]
        graph = CodeGraph(
            (
                CodeEntity("a.foo", "python_function", Path("a.py")),
                CodeEntity("b.bar", "python_function", Path("b.py")),
            ),
            (),
        )
        selected = dr.read_document(
            "Explain foo and bar", chunks, graph, [], max_tokens=100
        )
        self.assertTrue(all(value.path == selected.path
                            for value in selected.chunks))


    def test_document_reason_choices_are_canonical(self) -> None:
        """Keep the public selection reasons stable across the package.

        """
        self.assertEqual(
            dr.DOCUMENT_REASON_CHOICES,
            ("named_function", "named_file", "search"),
        )


if __name__ == "__main__":
    unittest.main()
