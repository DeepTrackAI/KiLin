"""Validate code-aware terms and reproducible sparse TF-IDF retrieval.

All tests use actual lexical index calculations and no embedding model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
import math
import unittest

from qachat.retrievals.strategies import retrieval_lexical as rl
from tests.qachat.retrievals._support import chunk


class TestLexicalTerms(unittest.TestCase):
    """Verify lexical extraction on prose and programming identifiers."""

    def test_plain_words_and_adjacent_bigrams(self) -> None:
        """Include lowercase unigrams followed by consecutive bigrams.

        """
        self.assertEqual(
            rl.lexical_terms("Alpha beta GAMMA"),
            ("alpha", "beta", "gamma", "alpha beta", "beta gamma"),
        )

    def test_camel_case_identifier_compounds(self) -> None:
        """Preserve full identifiers and expanded camel-case components.

        """
        terms = rl.lexical_terms("scanRepositoryNow")
        self.assertEqual(
            terms,
            (
                "scanrepositorynow", "scan", "repository", "now",
                "scanrepositorynow scan", "scan repository",
                "repository now",
            ),
        )

    def test_underscore_identifiers_and_short_components(self) -> None:
        """Drop one-character terms while retaining useful compounds.

        """
        terms = rl.lexical_terms("read_repository_files x a2 b")
        self.assertIn("read_repository_files", terms)
        self.assertIn("read", terms)
        self.assertIn("repository", terms)
        self.assertIn("files", terms)
        self.assertNotIn("x", terms)
        self.assertNotIn("b", terms)

    def test_empty_punctuation_and_non_ascii(self) -> None:
        """Unknown scripts contribute no terms under the ASCII policy.

        """
        self.assertEqual(rl.lexical_terms("... çä 漢字"), ())
        self.assertEqual(rl.lexical_terms(""), ())
        self.assertEqual(rl.lexical_terms("X"), ())

    def test_invalid_input_type(self) -> None:
        """Non-string input fails explicitly.

        """
        for value in (None, 12, ["text"], b"text"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "string"):
                    rl.lexical_terms(value)


class TestLexicalIndex(unittest.TestCase):
    """Check TF-IDF weights, ordering, and input validation."""

    def setUp(self) -> None:
        """Create independently searchable short source fragments.

        """
        self.chunks = [
            chunk("a", "optical trapping"),
            chunk("b", "deep learning"),
            chunk("c", "optical microscope"),
        ]
        self.index = rl.build_lexical_index(self.chunks)

    def test_index_shape_and_normalized_vectors(self) -> None:
        """Store one sparse unit vector per source chunk.

        """
        self.assertEqual(self.index.chunks, tuple(self.chunks))
        self.assertEqual(len(self.index.vectors), 3)
        self.assertTrue(self.index.inverse_document_frequency)
        for weights in self.index.vectors:
            self.assertAlmostEqual(
                math.sqrt(sum(x * x for x in weights.values())), 1.0
            )
        with self.assertRaises(FrozenInstanceError):
            self.index.chunks = ()

    def test_idf_prefers_rare_terms(self) -> None:
        """Terms in fewer documents have higher smoothed IDF.

        """
        weights = self.index.inverse_document_frequency
        self.assertGreater(weights["trapping"], weights["optical"])
        self.assertAlmostEqual(weights["optical"],
                               math.log(4 / 3) + 1.0)

    def test_retrieval_texts_can_differ_from_chunk_texts(self) -> None:
        """Optional prepared text changes vector content, not chunk identity.

        """
        index = rl.build_lexical_index(
            self.chunks, texts=["biology", "optics", "astronomy"]
        )
        self.assertEqual(index.chunks, tuple(self.chunks))
        self.assertIn("biology", index.inverse_document_frequency)
        self.assertNotIn("optical", index.inverse_document_frequency)

    def test_empty_and_mismatched_input(self) -> None:
        """Reject empty indexes and text/chunk count mismatches.

        """
        with self.assertRaisesRegex(ValueError, "At least one chunk"):
            rl.build_lexical_index([])
        with self.assertRaisesRegex(ValueError, "count"):
            rl.build_lexical_index(self.chunks, texts=["one"])

    def test_nonsearchable_texts(self) -> None:
        """Reject corpora containing no searchable ASCII-style terms.

        """
        with self.assertRaisesRegex(ValueError, "searchable"):
            rl.build_lexical_index([chunk("x", "...")])

    def test_unknown_terms_have_empty_tfidf_vectors(self) -> None:
        """Unknown query terms result in an empty sparse vector.

        """
        result = rl._tfidf_vector(Counter({"missing": 2}), {"known": 1.0})
        self.assertEqual(result, {})

    def test_logarithmic_term_frequency(self) -> None:
        """Repeated terms contribute logarithmic rather than raw TF.

        """
        vector = rl._tfidf_vector(
            Counter({"one": 4, "two": 1}),
            {"one": 2.0, "two": 1.0},
        )
        self.assertAlmostEqual(math.hypot(*vector.values()), 1.0)
        self.assertGreater(vector["one"], vector["two"])


class TestLexicalSearch(unittest.TestCase):
    """Ensure sparse search scoring, determinism, and safe validation."""

    def setUp(self) -> None:
        """Prepare an index containing a deliberate zero-score candidate.

        """
        self.chunks = [
            chunk("z", "particle tracking"),
            chunk("a", "optical microscopy"),
            chunk("m", "particle simulation"),
        ]
        self.index = rl.build_lexical_index(self.chunks)

    def test_relevant_terms_rank_first(self) -> None:
        """Return higher lexical cosine scores before unrelated chunks.

        """
        results = rl.lexical_search(self.index, "particle", count=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[-1].chunk.identifier, "a")
        self.assertEqual(results[-1].score, 0.0)
        self.assertGreater(results[0].score, 0)

    def test_zero_scores_sorted_by_identifier(self) -> None:
        """Unknown queries still return stable deterministic hits.

        """
        results = rl.lexical_search(self.index, "unknown", count=8)
        self.assertEqual(
            [result.chunk.identifier for result in results],
            ["a", "m", "z"],
        )
        self.assertTrue(all(result.score == 0 for result in results))

    def test_count_respected_and_result_immutable(self) -> None:
        """Never return more than count and freeze result fields.

        """
        results = rl.lexical_search(self.index, "particle", count=1)
        self.assertEqual(len(results), 1)
        with self.assertRaises(FrozenInstanceError):
            results[0].score = 0.0

    def test_invalid_count_and_query(self) -> None:
        """Reject nonpositive counts, Boolean counts, and blank queries.

        """
        for count in (0, -1, True, 1.5, "2"):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "count"):
                    rl.lexical_search(self.index, "particle", count=count)
        for query in (None, 3, "", "   "):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "query_text"):
                    rl.lexical_search(self.index, query)

    def test_empty_and_inconsistent_indexes(self) -> None:
        """Reject incomplete indexes before trying to rank them.

        """
        invalid = (
            rl.LexicalIndex((), (), {}),
            rl.LexicalIndex(self.index.chunks, (), {}),
            rl.LexicalIndex(self.index.chunks, self.index.vectors[:1], {}),
        )
        for index in invalid:
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    rl.lexical_search(index, "particle")

    def test_sparse_cosine_with_different_map_sizes(self) -> None:
        """Compute equal scores when either sparse vector is smaller.

        """
        small = {"a": 1.0}
        large = {"a": 0.5, "b": 0.5}
        self.assertEqual(rl._sparse_cosine_similarity(small, large), 0.5)
        self.assertEqual(rl._sparse_cosine_similarity(large, small), 0.5)


if __name__ == "__main__":
    unittest.main()
