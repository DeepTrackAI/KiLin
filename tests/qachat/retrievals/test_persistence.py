"""Test reproducible retrieval-index serialization and corruption checks.

Temporary JSON files exercise canonical payload integrity, schema changes,
structural validation, and embedding-model identity.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from qachat.retrievals import persistence as rp
from qachat.retrievals.code_graph import CodeEntity, CodeGraph, CodeRelation
from qachat.retrievals.strategies.retrieval_semantic import SemanticIndex
from qachat.retrievals.strategies.retrieval_lexical import (
    LexicalIndex,
    build_lexical_index,
)
from tests.qachat.retrievals._support import chunk


class _PersistentIndexFixtures(unittest.TestCase):
    """Provide deterministic source indexes and temporary JSON helpers."""

    def setUp(self) -> None:
        """Construct related indexes over two identifiable source chunks.

        """
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.target = Path(self.temporary.name) / "nested" / "index.json"
        self.chunks = [
            chunk("first", "optical traps", symbol="trap"),
            chunk("second", "deep learning", symbol="learn"),
        ]
        self.semantic = SemanticIndex(
            tuple(self.chunks), ((1.0, 0.0), (0.0, 1.0))
        )
        self.lexical = build_lexical_index(self.chunks)
        self.graph = CodeGraph(
            (CodeEntity("src.example", "source_file",
                        Path("src/example.py")),),
            (CodeRelation("src.example", "references", "external"),),
        )
        self.indexes = rp.prepare_persistent_indexes(
            self.semantic, self.lexical, self.graph
        )

    def _save_payload(self) -> dict[str, object]:
        """Save one valid index and return the parsed JSON payload.

        Returns
        -------
        dict[str, object]
            Serialized index fields for mutation tests.

        """
        rp.save_indexes(self.indexes, self.target)
        return json.loads(self.target.read_text(encoding="utf-8"))

    def _write_payload(
        self,
        payload: dict[str, object],
        *,
        reseal: bool = True,
    ) -> None:
        """Write a mutated payload, optionally updating its checksum.

        Parameters
        ----------
        payload: dict[str, object]
            Modified serialized index contents.
        reseal: bool, optional
            Recalculate the checksum when possible so structural tests reach
            the record validator. Set to False to test integrity detection.
            Defaults to True.

        """

        if reseal and "payload_sha256" in payload:
            try:
                payload["payload_sha256"] = rp._payload_checksum(payload)
            except ValueError:
                # Deliberately invalid JSON numbers cannot be resealed.
                pass
        self.target.write_text(json.dumps(payload), encoding="utf-8")


class TestPersistentIndexes(_PersistentIndexFixtures):
    """Validate index alignment, hashing, and durable file roundtrips."""

    def test_roundtrip_preserves_every_stored_field(self) -> None:
        """Reload exactly the stored chunks, vectors, weights, and graph.

        """
        saved = rp.save_indexes(self.indexes, self.target)
        self.assertEqual(saved, self.target.resolve())
        loaded = rp.load_indexes(saved)
        self.assertEqual(loaded, self.indexes)
        self.assertEqual(loaded.fingerprint,
                         rp._repository_fingerprint(
                             self.chunks, self.graph
                         ))
        with self.assertRaises(FrozenInstanceError):
            loaded.fingerprint = "changed"

    def test_json_contains_expected_index_records(self) -> None:
        """Persist both retrieval modalities without a format version.

        """
        payload = self._save_payload()
        self.assertNotIn("format_version", payload)
        self.assertEqual(len(payload["semantic_embeddings"]), 2)
        self.assertEqual(len(payload["lexical_vectors"]), 2)
        self.assertEqual(payload["chunks"][0]["path"],
                         "src/example.py")

    def test_source_content_and_graph_change_invalidates(self) -> None:
        """A changed chunk, order, or relation must invalidate the fingerprint.

        """
        self.assertTrue(rp.indexes_are_current(
            self.indexes, self.chunks, self.graph
        ))
        self.assertFalse(rp.indexes_are_current(
            self.indexes, list(reversed(self.chunks)), self.graph
        ))
        self.assertFalse(rp.indexes_are_current(
            self.indexes, self.chunks[:1], self.graph
        ))
        changed_graph = CodeGraph(self.graph.entities, ())
        self.assertFalse(rp.indexes_are_current(
            self.indexes, self.chunks, changed_graph
        ))

    def test_semantic_lexical_chunks_must_match(self) -> None:
        """Reject indexes whose chunk identity or order disagrees.

        """
        lexical = LexicalIndex(
            tuple(reversed(self.chunks)),
            self.lexical.vectors,
            self.lexical.inverse_document_frequency,
        )
        with self.assertRaisesRegex(ValueError, "share chunks"):
            rp.prepare_persistent_indexes(
                self.semantic, lexical, self.graph
            )

    def test_missing_model_identity_field_is_rejected(self) -> None:
        """Require explicit model provenance even for standalone indexes.

        """
        payload = self._save_payload()
        del payload["embedding_model_identity"]
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "fields"):
            rp.load_indexes(self.target)

    def test_missing_keys_and_malformed_json(self) -> None:
        """Reject missing structure, invalid JSON, and wrong graph fields.

        """
        original = self._save_payload()
        for key in ("chunks", "fingerprint", "code_graph",
                    "semantic_embeddings", "lexical_vectors",
                    "inverse_document_frequency"):
            with self.subTest(key=key):
                payload = dict(original)
                payload.pop(key)
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)
        self.target.write_text("{ invalid json", encoding="utf-8")
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)

    def test_chunk_data_mutation_detected_by_fingerprint(self) -> None:
        """Changing source text without recomputing the hash fails loading.

        """
        payload = self._save_payload()
        payload["chunks"][0]["text"] = "tampered text"
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            rp.load_indexes(self.target)

    def test_graph_mutation_detected_by_fingerprint(self) -> None:
        """Changes to code entities and relations fail fingerprint checks.

        """
        payload = self._save_payload()
        payload["code_graph"]["relations"][0]["target"] = "mutated"
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            rp.load_indexes(self.target)

    def test_graph_entity_container_validation(self) -> None:
        """Malformed graph entity and relation containers are rejected.

        """
        for key in ("entities", "relations"):
            with self.subTest(key=key):
                payload = self._save_payload()
                payload["code_graph"][key] = {"invalid": 1}
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)

    def test_missing_graph_record_fields(self) -> None:
        """Reject incomplete entity and relation records.

        """
        for collection, field in (
            ("entities", "name"),
            ("relations", "target"),
        ):
            with self.subTest(collection=collection):
                payload = self._save_payload()
                del payload["code_graph"][collection][0][field]
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)

    def test_missing_chunk_provenance_field(self) -> None:
        """Do not accept an incomplete serialized Chunk record.

        """
        payload = self._save_payload()
        del payload["chunks"][0]["start_line"]
        self._write_payload(payload)
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)

    def test_invalid_scalar_conversions(self) -> None:
        """Invalid scalar numeric values must not be silently interpreted.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"][0][0] = "not-a-number"
        self._write_payload(payload)
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)

    def test_invalid_json_container_types(self) -> None:
        """Reject non-index scalar JSON roots through explicit validation.

        """
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_text("null", encoding="utf-8")
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)

    def test_atomic_replacement_uses_temporary_sibling(self) -> None:
        """Write through a sibling temporary file before replacement.

        """
        self._save_payload()
        self.assertFalse(self.target.with_suffix(".json.tmp").exists())
        original = self.target.read_bytes()
        rp.save_indexes(self.indexes, self.target)
        self.assertEqual(self.target.read_bytes(), original)

    def test_error_from_unwritable_target_propagates(self) -> None:
        """File-system errors must not be suppressed as index corruption.

        """
        with patch.object(
            Path, "write_text", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                rp.save_indexes(self.indexes, self.target)

    def test_internal_converters_roundtrip(self) -> None:
        """Private converters retain paths, nullable fields, and relations.

        """
        value = self.chunks[0]
        self.assertEqual(rp._chunk_from_data(rp._chunk_to_data(value)),
                         value)
        self.assertEqual(rp._code_graph_from_data(
            rp._code_graph_to_data(self.graph)), self.graph)

    def test_reject_inconsistent_stored_vector_count(self) -> None:
        """Reject corrupt saved vector counts during index loading.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"] = payload["semantic_embeddings"][:1]
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "vector"):
            rp.load_indexes(self.target)

    def test_reject_nonfinite_stored_vectors(self) -> None:
        """Reject nonfinite semantic components during loading.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"][0][0] = float("nan")
        self._write_payload(payload)
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)

    def test_detect_modified_numeric_embeddings(self) -> None:
        """Changing vectors without updating the checksum is detected.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"][0] = [0.0, 1.0]
        self._write_payload(payload, reseal=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            rp.load_indexes(self.target)


class TestPreparedIndexValidation(unittest.TestCase):
    """Reject malformed in-memory indexes at the persistence boundary."""

    def setUp(self) -> None:
        """Construct aligned semantic and sparse lexical records.

        """
        self.chunks = (chunk("a", "physics diffusion"),
                       chunk("b", "optical tweezers"))
        self.semantic = SemanticIndex(
            self.chunks, ((3.0, 4.0), (0.0, 2.0))
        )
        self.lexical = build_lexical_index(list(self.chunks))
        self.graph = CodeGraph((), ())

    def _prepare(
        self,
        *,
        semantic: SemanticIndex | None = None,
        lexical: LexicalIndex | None = None,
    ) -> rp.PersistentIndexes:
        """Prepare an index while selectively replacing its inputs.

        Parameters
        ----------
        semantic: SemanticIndex | None, optional
            Replacement semantic records.
        lexical: LexicalIndex | None, optional
            Replacement lexical records.

        Returns
        -------
        rp.PersistentIndexes
            Validated combined records.

        """
        return rp.prepare_persistent_indexes(
            self.semantic if semantic is None else semantic,
            self.lexical if lexical is None else lexical,
            self.graph,
        )

    def test_valid_nonunit_semantic_vectors_are_allowed(self) -> None:
        """Cosine search must not require pre-normalized semantic vectors.

        """
        indexes = self._prepare()
        self.assertEqual(indexes.semantic.embeddings[0], (3.0, 4.0))

    def test_empty_index_is_rejected(self) -> None:
        """Persistent retrieval data must include at least one chunk.

        """
        with self.assertRaisesRegex(ValueError, "at least one chunk"):
            rp.prepare_persistent_indexes(
                SemanticIndex((), ()),
                LexicalIndex((), (), {}),
                self.graph,
            )

    def test_duplicate_chunk_identifier_is_rejected(self) -> None:
        """Distinct records may not share the same chunk identifier.

        """
        repeated = replace(self.chunks[1], identifier="a")
        chunks = (self.chunks[0], repeated)
        semantic = replace(self.semantic, chunks=chunks)
        lexical = replace(self.lexical, chunks=chunks)
        with self.assertRaisesRegex(ValueError, "identifiers.*unique"):
            self._prepare(semantic=semantic, lexical=lexical)

    def test_empty_or_wrong_type_chunk_identifier(self) -> None:
        """Identifiers used for deduplication must be nonempty strings.

        """
        for identifier in ("", None, 7):
            with self.subTest(identifier=identifier):
                first = replace(self.chunks[0], identifier=identifier)
                chunks = (first, self.chunks[1])
                with self.assertRaisesRegex(ValueError, "identifiers"):
                    self._prepare(
                        semantic=replace(self.semantic, chunks=chunks),
                        lexical=replace(self.lexical, chunks=chunks),
                    )

    def test_semantic_row_count_validation(self) -> None:
        """Reject missing and surplus semantic vectors at preparation.

        """
        for vectors in (self.semantic.embeddings[:1],
                        (*self.semantic.embeddings, (1.0, 0.0))):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(ValueError, "vector count"):
                    self._prepare(
                        semantic=replace(self.semantic, embeddings=vectors)
                    )

    def test_invalid_semantic_vector_dimensions(self) -> None:
        """Reject ragged, empty, and nonsequence semantic rows.

        """
        cases = (
            ((), (1.0, 2.0)),
            ((1.0,), (2.0, 3.0)),
            ((1.0, 2.0), None),
        )
        for vectors in cases:
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(ValueError, "dimension"):
                    self._prepare(semantic=replace(
                        self.semantic, embeddings=vectors
                    ))

    def test_semantic_numeric_validation(self) -> None:
        """Reject finite-type violations and zero vectors.

        """
        for value in (float("nan"), float("inf"),
                      -float("inf"), True, "1", None, object()):
            with self.subTest(value=value):
                vectors = ((1.0, value), (0.0, 1.0))
                with self.assertRaisesRegex(ValueError, "finite numeric"):
                    self._prepare(semantic=replace(
                        self.semantic, embeddings=vectors
                    ))
        with self.assertRaisesRegex(ValueError, "zero vector"):
            self._prepare(semantic=replace(
                self.semantic, embeddings=((0.0, 0.0), (1.0, 0.0))
            ))

    def test_nonrepresentable_semantic_component(self) -> None:
        """Reject integer magnitudes that overflow floating-point scoring.

        """
        vectors = ((10 ** 400, 1.0), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self._prepare(semantic=replace(
                self.semantic, embeddings=vectors
            ))

    def test_semantic_container_type_validation(self) -> None:
        """Index vector collections must be ordered sequences.

        """
        for vectors in (None, {0: (1.0, 0.0)}, "vectors"):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(ValueError, "vector count"):
                    self._prepare(semantic=replace(
                        self.semantic, embeddings=vectors
                    ))

    def test_lexical_row_count_validation(self) -> None:
        """Sparse records must represent precisely the same chunks.

        """
        for vectors in (self.lexical.vectors[:1],
                        (*self.lexical.vectors, {})):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(
                    ValueError, "Lexical vector count"
                ):
                    self._prepare(lexical=replace(
                        self.lexical, vectors=vectors
                    ))

    def test_lexical_container_validation(self) -> None:
        """Reject nonmapping TF-IDF statistics or sparse vectors.

        """
        with self.assertRaisesRegex(ValueError, "Lexical IDF"):
            self._prepare(lexical=replace(
                self.lexical, inverse_document_frequency=[]
            ))
        for vectors in (None, ("not a mapping", {})):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(ValueError, "Lexical vector"):
                    self._prepare(lexical=replace(
                        self.lexical, vectors=vectors
                    ))

    def test_invalid_idf_vocabulary(self) -> None:
        """Reject invalid IDF keys or non-finite lexical statistics.

        """
        for term in ("", 8):
            vocabulary = dict(self.lexical.inverse_document_frequency)
            vocabulary[term] = 1.0
            with self.subTest(term=term):
                with self.assertRaisesRegex(ValueError, "IDF terms"):
                    self._prepare(lexical=replace(
                        self.lexical,
                        inverse_document_frequency=vocabulary,
                    ))
        for value in (-1.0, float("nan"), float("inf"),
                      True, "1", None):
            vocabulary = dict(self.lexical.inverse_document_frequency)
            term = next(iter(vocabulary))
            vocabulary[term] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "IDF"):
                    self._prepare(lexical=replace(
                        self.lexical,
                        inverse_document_frequency=vocabulary,
                    ))

    def test_oversized_idf_number(self) -> None:
        """Reject IDF values that overflow Python float conversion.

        """
        vocabulary = dict(self.lexical.inverse_document_frequency)
        vocabulary[next(iter(vocabulary))] = 10 ** 400
        with self.assertRaisesRegex(ValueError, "IDF"):
            self._prepare(lexical=replace(
                self.lexical, inverse_document_frequency=vocabulary
            ))

    def test_invalid_lexical_vector_weights(self) -> None:
        """Reject invalid sparse TF-IDF weights and unknown vocabulary.

        """
        for value in (-0.1, float("nan"), float("inf"),
                      False, "1", None):
            vectors = [dict(vector) for vector in self.lexical.vectors]
            term = next(iter(vectors[0]))
            vectors[0][term] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Lexical vector"):
                    self._prepare(lexical=replace(
                        self.lexical, vectors=tuple(vectors)
                    ))

    def test_invalid_sparse_term_key_or_unknown_term(self) -> None:
        """Sparse TF-IDF terms must use nonempty vocabulary keys.

        """
        for term, message in (("", "Lexical terms"),
                              (3, "Lexical terms"),
                              ("unknown", "absent")):
            vectors = [dict(vector) for vector in self.lexical.vectors]
            vectors[0][term] = 1.0
            with self.subTest(term=term):
                with self.assertRaisesRegex(ValueError, message):
                    self._prepare(lexical=replace(
                        self.lexical, vectors=tuple(vectors)
                    ))

    def test_empty_sparse_vector_is_valid(self) -> None:
        """A source chunk may legitimately have no lexical terms.

        """
        lexical = replace(
            self.lexical, vectors=({}, self.lexical.vectors[1])
        )
        self.assertEqual(self._prepare(lexical=lexical).lexical, lexical)

    def test_zero_sparse_weights_are_permitted(self) -> None:
        """Finite zero sparse weights are not a data-integrity error.

        """
        vocabulary = dict(self.lexical.inverse_document_frequency)
        term = next(iter(vocabulary))
        vocabulary[term] = 0.0
        vectors = [dict(vector) for vector in self.lexical.vectors]
        vectors[0][term] = 0.0
        indexes = self._prepare(lexical=replace(
            self.lexical,
            vectors=tuple(vectors),
            inverse_document_frequency=vocabulary,
        ))
        self.assertEqual(indexes.lexical.vectors[0][term], 0.0)


class TestLoadedIndexValidation(_PersistentIndexFixtures):
    """Check all numeric and row invariants after JSON deserialization."""

    def test_saved_semantic_cardinality_errors(self) -> None:
        """Both absent and extra saved rows must fail at load time.

        """
        original = self._save_payload()
        for count in (0, 1, 3):
            payload = dict(original)
            payload["semantic_embeddings"] = (
                original["semantic_embeddings"][:count]
                if count < 3
                else [*original["semantic_embeddings"], [1.0, 0.0]]
            )
            with self.subTest(count=count):
                self._write_payload(payload)
                with self.assertRaisesRegex(ValueError, "vector count"):
                    rp.load_indexes(self.target)

    def test_saved_semantic_dimensions_and_zero_vectors(self) -> None:
        """Reject empty, ragged, and zero-norm stored embeddings.

        """
        for vectors, message in (
            ([[], [1.0, 0.0]], "dimension"),
            ([[1.0], [0.0, 1.0]], "dimension"),
            ([[0.0, 0.0], [0.0, 1.0]], "zero vector"),
        ):
            payload = self._save_payload()
            payload["semantic_embeddings"] = vectors
            with self.subTest(vectors=vectors):
                self._write_payload(payload)
                with self.assertRaisesRegex(ValueError, message):
                    rp.load_indexes(self.target)

    def test_saved_semantic_numeric_types_and_finiteness(self) -> None:
        """Do not coerce text, Boolean, or nonfinite stored components.

        """
        for value in ("1", True, None, [], {}, float("nan"),
                      float("inf"), -float("inf")):
            payload = self._save_payload()
            payload["semantic_embeddings"][0][0] = value
            with self.subTest(value=value):
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)

    def test_saved_large_integer_overflow(self) -> None:
        """Reject JSON numbers that cannot convert to finite float.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"][0][0] = 10 ** 400
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "too large"):
            rp.load_indexes(self.target)

    def test_saved_lexical_cardinality(self) -> None:
        """Missing and extra lexical rows must fail at load time.

        """
        for count in (1, 3):
            payload = self._save_payload()
            vectors = payload["lexical_vectors"]
            payload["lexical_vectors"] = (
                vectors[:1] if count == 1 else [*vectors, {}]
            )
            with self.subTest(count=count):
                self._write_payload(payload)
                with self.assertRaisesRegex(
                    ValueError, "Lexical vector count"
                ):
                    rp.load_indexes(self.target)

    def test_saved_lexical_weight_corruption(self) -> None:
        """Reject nonfinite, negative, and nonnumeric term weights.

        """
        for value in (float("nan"), float("inf"), -1.0,
                      True, "1", None):
            payload = self._save_payload()
            term = next(iter(payload["lexical_vectors"][0]))
            payload["lexical_vectors"][0][term] = value
            with self.subTest(value=value):
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)

    def test_saved_idf_corruption(self) -> None:
        """Reject invalid IDF values before the lexical search can run.

        """
        for value in (float("nan"), float("inf"), -1.0,
                      False, "1", None):
            payload = self._save_payload()
            term = next(iter(payload["inverse_document_frequency"]))
            payload["inverse_document_frequency"][term] = value
            with self.subTest(value=value):
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)

    def test_malformed_saved_lexical_mapping_containers(self) -> None:
        """Reject non-dictionary sparse vectors and IDF statistics.

        """
        for key in ("lexical_vectors", "inverse_document_frequency"):
            for value in (None, [], 12, "weights"):
                with self.subTest(key=key, value=value):
                    payload = self._save_payload()
                    if key == "lexical_vectors":
                        payload[key][0] = value
                    else:
                        payload[key] = value
                    self._write_payload(payload)
                    with self.assertRaises(ValueError):
                        rp.load_indexes(self.target)

    def test_saved_vocabulary_mismatch(self) -> None:
        """Do not accept document terms that cannot be queried.

        """
        payload = self._save_payload()
        payload["lexical_vectors"][0]["unindexed"] = 1.0
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "absent"):
            rp.load_indexes(self.target)

    def test_saved_duplicate_identifier_with_valid_fingerprint(self) -> None:
        """Reject duplicates even when the source fingerprint is updated.

        """
        payload = self._save_payload()
        payload["chunks"][1]["identifier"] = "first"
        chunks = [rp._chunk_from_data(item) for item in payload["chunks"]]
        payload["fingerprint"] = rp._repository_fingerprint(
            chunks, self.graph
        )
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "unique"):
            rp.load_indexes(self.target)

    def test_saved_empty_identifier_with_valid_fingerprint(self) -> None:
        """Reject blank identifiers independently of fingerprint checks.

        """
        payload = self._save_payload()
        payload["chunks"][0]["identifier"] = ""
        chunks = [rp._chunk_from_data(item) for item in payload["chunks"]]
        payload["fingerprint"] = rp._repository_fingerprint(
            chunks, self.graph
        )
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "identifiers"):
            rp.load_indexes(self.target)

    def test_loaded_nonunit_vectors_and_empty_sparse_rows(self) -> None:
        """Valid nonunit and empty sparse records survive persistence.

        """
        original = self.indexes
        semantic = replace(
            original.semantic, embeddings=((3.0, 4.0), (0.0, 2.0))
        )
        lexical = replace(
            original.lexical,
            vectors=({}, original.lexical.vectors[1]),
        )
        indexes = rp.prepare_persistent_indexes(
            semantic, lexical, self.graph
        )
        rp.save_indexes(indexes, self.target)
        self.assertEqual(rp.load_indexes(self.target), indexes)

    def test_save_rejects_mutated_lexical_weight(self) -> None:
        """Validation before save protects against nested mutable data.

        """
        term = next(iter(self.indexes.lexical.vectors[0]))
        self.indexes.lexical.vectors[0][term] = float("nan")
        with self.assertRaisesRegex(ValueError, "Lexical vector"):
            rp.save_indexes(self.indexes, self.target)
        self.assertFalse(self.target.exists())

    def test_save_rejects_manually_forged_semantic_index(self) -> None:
        """Do not serialize invalid directly assembled index bundles.

        """
        indexes = replace(
            self.indexes,
            semantic=replace(
                self.semantic, embeddings=((1.0, 0.0),)
            ),
        )
        with self.assertRaisesRegex(ValueError, "vector count"):
            rp.save_indexes(indexes, self.target)
        self.assertFalse(self.target.exists())

    def test_numpy_semantic_components_serialize_as_json_numbers(self) -> None:
        """Valid NumPy embedding scalars remain persistable as JSON floats.

        """
        vectors = (
            (np.float32(1.0), np.float32(0.0)),
            (np.float64(0.0), np.float64(1.0)),
        )
        semantic = replace(self.semantic, embeddings=vectors)
        indexes = rp.prepare_persistent_indexes(
            semantic, self.lexical, self.graph
        )
        rp.save_indexes(indexes, self.target)
        payload = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(payload["semantic_embeddings"],
                         [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(rp.load_indexes(self.target), indexes)

    def test_saved_content_has_integrity_fields(self) -> None:
        """Write the checksum and provenance alongside retrieval records.

        """
        payload = self._save_payload()
        self.assertEqual(
            set(payload),
            {
                "fingerprint", "chunks",
                "semantic_embeddings", "lexical_vectors",
                "inverse_document_frequency", "code_graph",
                "embedding_model_identity", "payload_sha256",
            },
        )


class TestPayloadIntegrity(_PersistentIndexFixtures):
    """Check canonical checksums, model provenance, and integrity."""

    def test_saved_checksum_matches_canonical_payload(self) -> None:
        """Bind all numeric and provenance fields into the integrity digest.

        """

        payload = self._save_payload()
        self.assertEqual(len(payload["payload_sha256"]), 64)
        self.assertEqual(
            payload["payload_sha256"], rp._payload_checksum(payload)
        )

    def test_canonical_checksum_ignores_json_key_and_whitespace_order(
        self,
    ) -> None:
        """Pretty printing and JSON object order cannot break a saved index.

        """

        payload = self._save_payload()
        reversed_order = dict(reversed(list(payload.items())))
        self.target.write_text(
            json.dumps(reversed_order, indent=7), encoding="utf-8"
        )
        self.assertEqual(rp.load_indexes(self.target), self.indexes)

    def test_checksum_covers_each_independent_payload_component(self) -> None:
        """Reject modifications even when source fingerprint is unchanged.

        """

        def change_vector(value: dict[str, object]) -> None:
            """Alter one finite semantic component.

            """

            value["semantic_embeddings"][0][0] = 0.5

        def change_lexical(value: dict[str, object]) -> None:
            """Alter one valid lexical weight.

            """

            term = next(iter(value["lexical_vectors"][0]))
            value["lexical_vectors"][0][term] = 0.4

        def change_idf(value: dict[str, object]) -> None:
            """Alter one valid inverse frequency.

            """

            term = next(iter(value["inverse_document_frequency"]))
            value["inverse_document_frequency"][term] = 1.5

        def change_graph(value: dict[str, object]) -> None:
            """Alter one relationship.

            """

            value["code_graph"]["relations"][0]["target"] = "changed"

        def change_model(value: dict[str, object]) -> None:
            """Change the claimed model provenance.

            """

            value["embedding_model_identity"] = "different-model"

        def change_source(value: dict[str, object]) -> None:
            """Change the repository identity metadata.

            """

            value["fingerprint"] = "different-fingerprint"

        for operation in (
            change_vector, change_lexical, change_idf, change_graph,
            change_model, change_source,
        ):
            with self.subTest(field=operation.__name__):
                payload = self._save_payload()
                operation(payload)
                self._write_payload(payload, reseal=False)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    rp.load_indexes(self.target)

    def test_resealed_payload_is_not_authenticated(self) -> None:
        """A checksum is not a defense against a writer who recomputes it.

        """

        payload = self._save_payload()
        payload["semantic_embeddings"][0] = [3.0, 4.0]
        self._write_payload(payload)
        loaded = rp.load_indexes(self.target)
        self.assertEqual(loaded.semantic.embeddings[0], (3.0, 4.0))

    def test_missing_and_invalid_checksum_are_rejected(self) -> None:
        """Require complete payload fields and a valid checksum.

        """

        for checksum in (None, "", False, "x" * 64):
            with self.subTest(checksum=checksum):
                payload = self._save_payload()
                payload["payload_sha256"] = checksum
                self._write_payload(payload, reseal=False)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)
        payload = self._save_payload()
        del payload["payload_sha256"]
        self._write_payload(payload, reseal=False)
        with self.assertRaisesRegex(ValueError, "fields"):
            rp.load_indexes(self.target)

    def test_unknown_payload_field_rejected(self) -> None:
        """Keep the index format explicit and reject undeclared fields.

        """

        payload = self._save_payload()
        payload["unrecognized"] = True
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "fields"):
            rp.load_indexes(self.target)

    def test_missing_integrity_fields_requires_rebuild(self) -> None:
        """Reject old files without checksums or model provenance.

        """

        payload = self._save_payload()
        payload.pop("payload_sha256")
        payload.pop("embedding_model_identity")
        self._write_payload(payload, reseal=False)
        with self.assertRaisesRegex(ValueError, "fields"):
            rp.load_indexes(self.target)

    def test_saved_embedding_model_identity_roundtrips(self) -> None:
        """Bind an explicit model identity to the saved semantic vectors.

        """

        indexes = rp.prepare_persistent_indexes(
            self.semantic, self.lexical, self.graph,
            embedding_model_identity="local-sha256:deadbeef",
        )
        rp.save_indexes(indexes, self.target)
        loaded = rp.load_indexes(self.target)
        self.assertEqual(loaded, indexes)
        self.assertFalse(rp.indexes_are_current(
            loaded, self.chunks, self.graph,
            embedding_model_identity="other-model",
        ))
        self.assertTrue(rp.indexes_are_current(
            loaded, self.chunks, self.graph,
            embedding_model_identity="local-sha256:deadbeef",
        ))

    def test_invalid_model_identity_is_rejected(self) -> None:
        """Forbid invalid provenance in memory or in persisted indexes.

        """

        for value in ("", "  ", 12, False, {}, []):
            with self.subTest(identity=value):
                with self.assertRaisesRegex(ValueError, "identity"):
                    rp.prepare_persistent_indexes(
                        self.semantic, self.lexical, self.graph,
                        embedding_model_identity=value,
                    )
                payload = self._save_payload()
                payload["embedding_model_identity"] = value
                self._write_payload(payload)
                with self.assertRaises(ValueError):
                    rp.load_indexes(self.target)

    def test_save_rejects_forged_source_fingerprint(self) -> None:
        """Do not create bundles claiming another source-revision digest.

        """

        forged = replace(self.indexes, fingerprint="bad")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            rp.save_indexes(forged, self.target)
        self.assertFalse(self.target.exists())

    def test_save_rejects_identity_mutated_after_preparation(self) -> None:
        """Reject manually constructed indexes with invalid identity fields.

        """

        forged = replace(self.indexes, embedding_model_identity="")
        with self.assertRaisesRegex(ValueError, "identity"):
            rp.save_indexes(forged, self.target)
        self.assertFalse(self.target.exists())

    def test_checksum_rejects_nonfinite_json_values(self) -> None:
        """Canonical index payloads cannot contain nonstandard JSON numbers.

        """

        with self.assertRaises(ValueError):
            rp._payload_checksum({"weight": float("nan")})

    def test_stored_number_rejects_nonfinite_values(self) -> None:
        """The low-level number converter rejects infinity independently.

        """

        with self.assertRaisesRegex(ValueError, "non-finite"):
            rp._stored_number(float("inf"))


if __name__ == "__main__":
    unittest.main()
