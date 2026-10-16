"""Exercise semantic-index construction and both numerical search paths.

The unit tests use controlled embeddings to assert exact ranking contracts.
Actual embedding sessions are tested in the retrieval integration tests.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import unittest
from unittest.mock import patch

import numpy as np

from kilin.retrievals.strategies import retrieval_semantic as rs
from tests.kilin.retrievals._support import FakeEmbeddings, chunk


class TestSemanticIndexConstruction(unittest.TestCase):
    """Verify batching, provenance, and construction invariants."""

    def setUp(self) -> None:
        """Create a fixed collection of source chunks.

        """
        self.chunks = [chunk(str(i), "x" * i) for i in range(1, 6)]

    def test_single_and_multiple_batches_match(self) -> None:
        """Batch boundaries must not change stored document ordering.

        """
        one = FakeEmbeddings()
        many = FakeEmbeddings()
        single_index = rs.build_semantic_index(self.chunks, one)
        batched_index = rs.build_semantic_index(
            self.chunks, many, batch_size=2
        )
        self.assertEqual(single_index, batched_index)
        self.assertEqual(one.calls, [[value.text for value in self.chunks]])
        self.assertEqual([len(batch) for batch in many.calls], [2, 2, 1])
        self.assertEqual((one.cleared, many.cleared), (1, 3))
        self.assertEqual(single_index.chunks, tuple(self.chunks))
        self.assertIsInstance(single_index.embeddings, tuple)
        self.assertTrue(
            all(
                isinstance(vector, tuple)
                for vector in single_index.embeddings
            )
        )

    def test_progress_callback_reports_completed_chunks(self) -> None:
        """Report successful batches, including the final partial batch.

        """
        progress = []
        rs.build_semantic_index(
            self.chunks,
            FakeEmbeddings(),
            batch_size=2,
            progress_callback=lambda done, total: progress.append(
                (done, total)
            ),
        )
        self.assertEqual(progress, [(2, 5), (4, 5), (5, 5)])

    def test_empty_input_is_rejected_before_embedding(self) -> None:
        """Reject an empty index without requesting model calculations.

        """
        session = FakeEmbeddings()
        with self.assertRaisesRegex(ValueError, "At least one chunk"):
            rs.build_semantic_index([], session)
        self.assertEqual(session.calls, [])

    def test_invalid_batch_sizes(self) -> None:
        """Reject zero, negative, Boolean, and nonintegral batch sizes.

        """
        for size in (0, -1, True, False, 1.5, "2", []):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    rs.build_semantic_index(
                        self.chunks, FakeEmbeddings(), batch_size=size
                    )

    def test_batch_larger_than_input(self) -> None:
        """Oversized batches must still be handled as one operation.

        """
        session = FakeEmbeddings()
        rs.build_semantic_index(self.chunks, session, batch_size=100)
        self.assertEqual([len(call) for call in session.calls], [5])
        self.assertEqual(session.cleared, 1)

    def test_invalid_callback_fails_before_embedding(self) -> None:
        """Progress callbacks must be callable or absent.

        """
        session = FakeEmbeddings()
        with self.assertRaisesRegex(ValueError, "progress_callback"):
            rs.build_semantic_index(
                self.chunks, session, progress_callback=23
            )
        self.assertEqual(session.calls, [])

    def test_wrong_document_vector_count(self) -> None:
        """Index creation rejects backends dropping or adding rows.

        """
        session = FakeEmbeddings()
        with patch.object(session, "embed_documents", return_value=[]):
            with self.assertRaisesRegex(ValueError, "count"):
                rs.build_semantic_index(self.chunks, session)

    def test_mismatched_and_empty_dimensions(self) -> None:
        """Reject inconsistent or zero-dimensional stored vectors.

        """
        for vectors in (
            [[1.0], [1.0, 2.0]],
            [[], []],
        ):
            with self.subTest(vectors=vectors):
                session = FakeEmbeddings()
                with patch.object(
                    session, "embed_documents", return_value=vectors
                ):
                    with self.assertRaisesRegex(ValueError, "dimension"):
                        rs.build_semantic_index(
                            self.chunks[:2], session
                        )

    def test_backend_error_is_not_suppressed(self) -> None:
        """Embedding failures must not produce a partial saved index.

        """
        session = FakeEmbeddings()
        with patch.object(
            session,
            "embed_documents",
            side_effect=RuntimeError("model failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                rs.build_semantic_index(self.chunks, session)
        self.assertEqual(session.cleared, 0)

    def test_callback_error_is_not_suppressed(self) -> None:
        """Forward callback failures without embedding subsequent batches.

        """
        session = FakeEmbeddings()

        def fail(done: int, total: int) -> None:
            """Raise after the first completed batch.

            Parameters
            ----------
            done: int
                Completed chunks.
            total: int
                Total chunks.

            """
            self.assertEqual((done, total), (2, 5))
            raise RuntimeError("callback failure")

        with self.assertRaisesRegex(RuntimeError, "callback failure"):
            rs.build_semantic_index(
                self.chunks,
                session,
                batch_size=2,
                progress_callback=fail,
            )
        self.assertEqual([len(batch) for batch in session.calls], [2])
        self.assertEqual(session.cleared, 1)

    def test_records_are_frozen(self) -> None:
        """Prevent reassignment of structural index and result fields.

        """
        index = rs.build_semantic_index(self.chunks, FakeEmbeddings())
        with self.assertRaises(FrozenInstanceError):
            index.chunks = ()
        result = rs.SemanticResult(1.0, self.chunks[0])
        with self.assertRaises(FrozenInstanceError):
            result.score = 0.0


class TestSemanticSearch(unittest.TestCase):
    """Verify numerical rankings, input checks, and caching contracts."""

    def setUp(self) -> None:
        """Construct known two-dimensional orthogonal document vectors.

        """
        self.chunks = [
            chunk("a", "east"),
            chunk("b", "north"),
            chunk("c", "west"),
            chunk("d", "south"),
        ]
        self.session = FakeEmbeddings(
            {
                "east": [1.0, 0.0],
                "north": [0.0, 1.0],
                "west": [-1.0, 0.0],
                "south": [0.0, -1.0],
                "query": [1.0, 0.0],
            }
        )
        self.index = rs.build_semantic_index(self.chunks, self.session)

    def test_loop_ranking_scores_and_tie_stability(self) -> None:
        """Cosine scores must match reference vectors and input tie order.

        """
        results = rs.semantic_search(
            self.index, "query", self.session, count=4
        )
        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["a", "b", "d", "c"],
        )
        for result, expected in zip(results, [1, 0, 0, -1]):
            self.assertAlmostEqual(result.score, expected)

    def test_numpy_matches_loop_scores(self) -> None:
        """Independent numerical implementations agree away from ties.

        """
        loop = rs.semantic_search(
            self.index, "query", self.session, count=4
        )
        vectorized = rs.semantic_search(
            self.index, "query", self.session, count=4,
            backend="numpy"
        )
        self.assertEqual(
            {item.chunk.identifier for item in loop},
            {item.chunk.identifier for item in vectorized},
        )
        for item in vectorized:
            reference = next(
                result.score for result in loop
                if result.chunk.identifier == item.chunk.identifier
            )
            self.assertAlmostEqual(item.score, reference, places=5)

    def test_precomputed_matrix_is_reused(self) -> None:
        """Avoid rebuilding the index matrix when the caller supplies it.

        """
        matrix = rs.semantic_matrix(self.index)
        self.assertEqual(matrix.shape, (4, 2))
        self.assertEqual(matrix.dtype, np.dtype("float32"))
        with patch.object(
            rs, "semantic_matrix", side_effect=AssertionError("rebuilt")
        ):
            results = rs.semantic_search(
                self.index, "query", self.session, backend="numpy",
                matrix=matrix, count=1,
            )
        self.assertEqual(results[0].chunk.identifier, "a")

    def test_numpy_rebuilds_matrix_when_needed(self) -> None:
        """Default NumPy search constructs its own index matrix.

        """
        with patch.object(
            rs, "semantic_matrix", wraps=rs.semantic_matrix
        ) as convert:
            rs.semantic_search(
                self.index, "query", self.session, backend="numpy"
            )
        convert.assert_called_once_with(self.index)

    def test_count_is_a_strict_positive_integer(self) -> None:
        """Invalid limits fail before querying the embedding session.

        """
        for size in (0, -1, True, False, 1.5, "4", None):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "count"):
                    rs.semantic_search(
                        self.index, "query", self.session, count=size
                    )
        self.assertEqual(self.session.queries, [])

    def test_count_larger_than_index_returns_all(self) -> None:
        """Neither backend duplicates records to fill excess results.

        """
        for backend in ("loop", "numpy"):
            with self.subTest(backend=backend):
                result = rs.semantic_search(
                    self.index, "query", self.session,
                    count=10, backend=backend,
                )
                self.assertEqual(len(result), 4)

    def test_invalid_search_backend(self) -> None:
        """Reject unsupported numerical search backends.

        """
        for name in ("fast", "torch", None, 0):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "backend"):
                    rs.semantic_search(
                        self.index, "query", self.session, backend=name
                    )

    def test_empty_and_inconsistent_index(self) -> None:
        """Reject missing chunks, vectors, or unequal index lengths.

        """
        invalid = (
            rs.SemanticIndex((), ()),
            rs.SemanticIndex(tuple(self.chunks), ()),
            rs.SemanticIndex((), self.index.embeddings),
            rs.SemanticIndex(tuple(self.chunks), self.index.embeddings[:1]),
        )
        for index in invalid:
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    rs.semantic_search(index, "query", self.session)

    def test_query_vector_dimension_validation(self) -> None:
        """Query dimensions must agree with the stored embedding dimension.

        """
        self.session.mapping["query"] = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            rs.semantic_search(self.index, "query", self.session)

    def test_trace_identifier_is_forwarded_when_present(self) -> None:
        """Forward an explicit trace turn ID, preserving query content.

        """
        rs.semantic_search(
            self.index, "query", self.session, trace_turn_id="turn-9"
        )
        self.assertEqual(self.session.queries, [("query", "turn-9")])

    def test_numpy_and_loop_reject_zero_vectors(self) -> None:
        """A zero query or document vector is unsuitable for cosine ranking.

        """
        for backend in ("loop", "numpy"):
            self.session.mapping["query"] = [0.0, 0.0]
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "zero"):
                    rs.semantic_search(
                        self.index, "query", self.session, backend=backend
                    )
        self.session.mapping["query"] = [1.0, 0.0]
        bad = rs.SemanticIndex(
            self.index.chunks,
            ((0.0, 0.0), *self.index.embeddings[1:]),
        )
        for backend in ("loop", "numpy"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "zero"):
                    rs.semantic_search(
                        bad, "query", self.session, backend=backend
                    )

    def test_numpy_incompatible_matrix_shapes(self) -> None:
        """A dimension mismatch must fail rather than mislabel results.

        """
        for matrix in (
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 2, 1), dtype=np.float32),
        ):
            with self.subTest(shape=matrix.shape):
                with self.assertRaises((ValueError, IndexError)):
                    rs.semantic_search(
                        self.index, "query", self.session,
                        backend="numpy", matrix=matrix,
                    )

    def test_cosine_similarity_reference_values(self) -> None:
        """Check unit, inverse, orthogonal, and scale-invariant scores.

        """
        examples = (
            ((1, 0), (5, 0), 1.0),
            ((1, 0), (-5, 0), -1.0),
            ((1, 0), (0, 7), 0.0),
            ((3, 4), (6, 8), 1.0),
        )
        for first, second, expected in examples:
            with self.subTest(first=first, second=second):
                self.assertAlmostEqual(
                    rs._cosine_similarity(first, second), expected
                )

    def test_cosine_similarity_rejects_invalid_shapes(self) -> None:
        """Reject empty, mismatched, and zero-norm vectors.

        """
        for first, second in (
            ((), ()),
            ((1,), (1, 2)),
            ((0, 0), (1, 0)),
            ((1, 0), (0, 0)),
        ):
            with self.subTest(first=first, second=second):
                with self.assertRaises(ValueError):
                    rs._cosine_similarity(first, second)


    def test_reject_nonfinite_query_vectors(self) -> None:
        """Reject NaN query components before semantic ranking.

        """
        self.session.mapping["query"] = [float("nan"), 0.0]
        with self.assertRaises(ValueError):
            rs.semantic_search(
                self.index, "query", self.session, backend="loop"
            )


    def test_numpy_float32_precision_is_documented(self) -> None:
        """NumPy matrix storage explicitly uses float32 precision.

        """
        matrix = rs.semantic_matrix(self.index)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(math.isclose(float(matrix[0, 0]), 1.0))


class TestAdditionalSemanticVectorValidation(unittest.TestCase):
    """Check rejection of malformed embeddings from external adapters."""

    def setUp(self) -> None:
        """Construct a small corpus and a controlled embedding adapter.

        """
        self.chunks = [chunk("a", "east"), chunk("b", "north")]
        self.session = FakeEmbeddings({
            "east": [1.0, 0.0],
            "north": [0.0, 1.0],
            "query": [1.0, 0.0],
        })

    def test_bad_document_numeric_components(self) -> None:
        """Reject nonfinite, nonnumeric, or Boolean document values.

        """
        for value in (float("nan"), float("inf"),
                      -float("inf"), True, "1", None):
            with self.subTest(value=value):
                with patch.object(
                    self.session, "embed_documents",
                    return_value=[[1.0, value], [0.0, 1.0]],
                ):
                    with self.assertRaisesRegex(
                        ValueError, "finite numeric"
                    ):
                        rs.build_semantic_index(
                            self.chunks, self.session
                        )

    def test_zero_document_vector(self) -> None:
        """A zero backend vector is rejected during index construction.

        """
        with patch.object(
            self.session, "embed_documents",
            return_value=[[0.0, 0.0], [0.0, 1.0]],
        ):
            with self.assertRaisesRegex(ValueError, "zero vector"):
                rs.build_semantic_index(self.chunks, self.session)

    def test_nonsequence_document_vector(self) -> None:
        """An embedding adapter cannot return scalar vector records.

        """
        with patch.object(
            self.session, "embed_documents",
            return_value=[123, [0.0, 1.0]],
        ):
            with self.assertRaisesRegex(ValueError, "dimension"):
                rs.build_semantic_index(self.chunks, self.session)

    def test_reject_invalid_queries_in_both_scoring_backends(self) -> None:
        """Reject invalid queries for NumPy and loop scoring alike.

        """
        index = rs.build_semantic_index(self.chunks, self.session)
        for value in (float("nan"), float("inf"),
                      -float("inf"), True, "bad", None):
            self.session.mapping["query"] = [1.0, value]
            for backend in ("loop", "numpy"):
                with self.subTest(value=value, backend=backend):
                    with self.assertRaisesRegex(
                        ValueError, "finite numeric"
                    ):
                        rs.semantic_search(
                            index, "query", self.session,
                            backend=backend,
                        )

    def test_nonsequence_query_is_rejected(self) -> None:
        """Avoid numerical calls with scalar or absent query vectors.

        """
        index = rs.build_semantic_index(self.chunks, self.session)
        for vector in (None, 12, []):
            with self.subTest(vector=vector):
                with patch.object(
                    self.session, "embed_query", return_value=vector
                ):
                    with self.assertRaises(ValueError):
                        rs.semantic_search(index, "query", self.session)

    def test_valid_nonunit_vectors_preserve_cosine_results(self) -> None:
        """Vector validation must not require an L2 norm equal to one.

        """
        self.session.mapping["east"] = [3.0, 4.0]
        self.session.mapping["north"] = [0.0, 2.0]
        self.session.mapping["query"] = [3.0, 4.0]
        index = rs.build_semantic_index(self.chunks, self.session)
        for backend in ("loop", "numpy"):
            with self.subTest(backend=backend):
                results = rs.semantic_search(
                    index, "query", self.session, backend=backend
                )
                self.assertEqual(results[0].chunk.identifier, "a")
                self.assertAlmostEqual(results[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
