"""Test reproducible retrieval-index serialization and corruption checks.

Temporary JSON files exercise the real persistence implementation. Explicit
expected failures document integrity gaps without changing production code.

"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat.retrievals import persistence as rp
from qachat.retrievals.code_graph import CodeEntity, CodeGraph, CodeRelation
from qachat.retrievals.strategies.retrieval_semantic import SemanticIndex
from qachat.retrievals.strategies.retrieval_lexical import (
    LexicalIndex,
    build_lexical_index,
)
from tests.qachat.retrievals._support import chunk


class TestPersistentIndexes(unittest.TestCase):
    """Validate index alignment, hashing, and durable file roundtrips."""

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

    def _write_payload(self, payload: dict[str, object]) -> None:
        """Replace a saved index with a mutated JSON payload.

        Parameters
        ----------
        payload: dict[str, object]
            Modified serialized index contents.

        """
        self.target.write_text(json.dumps(payload), encoding="utf-8")

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

    def test_json_has_declared_format_version(self) -> None:
        """Serialize one stable supported format version.

        """
        payload = self._save_payload()
        self.assertEqual(payload["format_version"], 1)
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

    def test_unsupported_format_version(self) -> None:
        """Do not interpret an unsupported future persistence schema.

        """
        payload = self._save_payload()
        payload["format_version"] = 99
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "format version"):
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

    @unittest.expectedFailure
    def test_pending_reject_inconsistent_stored_vector_count(self) -> None:
        """Pending: corrupt saved vector counts must not pass index loading.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"] = payload["semantic_embeddings"][:1]
        self._write_payload(payload)
        with self.assertRaisesRegex(ValueError, "vector"):
            rp.load_indexes(self.target)

    @unittest.expectedFailure
    def test_pending_reject_nonfinite_stored_vectors(self) -> None:
        """Pending: corrupted nonfinite vectors must fail during loading.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"][0][0] = float("nan")
        self._write_payload(payload)
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)

    @unittest.expectedFailure
    def test_pending_detect_modified_numeric_embeddings(self) -> None:
        """Pending: changing vectors alone must invalidate their integrity.

        """
        payload = self._save_payload()
        payload["semantic_embeddings"][0] = [0.0, 1.0]
        self._write_payload(payload)
        with self.assertRaises(ValueError):
            rp.load_indexes(self.target)


if __name__ == "__main__":
    unittest.main()
