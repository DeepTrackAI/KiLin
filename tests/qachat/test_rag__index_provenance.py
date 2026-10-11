"""Check RAG index reuse and invalidation with real persistence.

The RAG indexing coordinator is exercised with controlled source scanning,
source chunks, and an in-memory embedding adapter. It saves and reloads real
index/metadata files while keeping network access and model downloads out of
the tests. Each new embedding adapter represents a freshly loaded model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qachat.rag import RAGSession
from qachat.retrievals.code_graph import CodeGraph
from qachat.retrievals import model_identity
from qachat.retrievals.model_identity import embedding_model_identity
from qachat.retrievals.persistence import load_indexes
from qachat.trace_logging import NullTraceLogger
from tests.qachat.retrievals._support import FakeEmbeddings, chunk


class TestRAGIndexProvenance(unittest.TestCase):
    """Exercise actual RAG reuse decisions without optional model libraries."""

    def setUp(self) -> None:
        """Create one source file, one mock model, and an index workspace.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        environment = patch.dict(
            os.environ, {"XDG_CACHE_HOME": str(root / "xdg")},
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.models = root / "embedding-model"
        self.models.mkdir()
        self.weights = self.models / "model.safetensors"
        self.weights.write_bytes(b"AAA")
        self.index_directory = root / "indexes"
        self.source = root / "sources"
        self.source.mkdir()
        self.source_chunk = chunk(
            "first", "physics diffusion", path="src/example.py"
        )
        self.graph = CodeGraph((), ())
        self.selected = [Path("src/example.py")]
        self.manifest = {"selected_files": ["src/example.py"]}

    def _embedding(self) -> FakeEmbeddings:
        """Create a fresh adapter representing the model on local disk.

        Returns
        -------
        FakeEmbeddings
            Synthetic vectors and resolved local model provenance.

        """

        embedding = FakeEmbeddings()
        embedding.model_id = str(self.models)
        embedding.model_source = str(self.models)
        return embedding

    def _coordinator(
        self,
        chunks: list | None = None,
    ) -> RAGSession:
        """Make a lightweight RAG coordinator with real file persistence.

        Parameters
        ----------
        chunks: list | None, optional
            Replacement source chunks. Defaults to the fixture chunk.

        Returns
        -------
        RAGSession
            An uninitialized coordinator with indexing dependencies supplied.

        Notes
        -----
        Production session construction loads chat models. Calling its
        indexing method on a lightweight instance isolates reuse decisions.

        """

        session = object.__new__(RAGSession)
        session.project = SimpleNamespace(
            source=self.source,
            indexes=self.index_directory,
            profile=SimpleNamespace(
                extensions=(".py",), max_file_bytes=100000,
            ),
        )
        session.trace = NullTraceLogger()
        session._emit_progress = Mock()
        session._emit_count_progress = Mock()
        session._index_metadata = Mock(return_value={"project_kind": "code"})
        session._prepare_source = Mock(
            return_value=(
                [self.source_chunk] if chunks is None else chunks,
                self.graph,
            )
        )
        return session

    def _prepare(
        self,
        coordinator: RAGSession,
        embedding: FakeEmbeddings,
    ) -> tuple:
        """Index or reuse using deterministic scan and manifest callbacks.

        Parameters
        ----------
        coordinator: RAGSession
            Prepared retrieval coordinator.
        embedding: FakeEmbeddings
            Adapter representing the active embedding model.

        Returns
        -------
        tuple
            Chunks, indexes, status, and change report from real orchestration.

        """

        report = {
            "repository": str(self.source),
            "complete": True,
            "selected_files": self.selected,
            "entries": [],
        }

        def compare(saved: dict | None, current: dict) -> dict:
            """Describe files changed relative to the saved manifest.

            Parameters
            ----------
            saved: dict | None
                Previously indexed source manifest.
            current: dict
                Active source manifest.

            Returns
            -------
            dict
                Flags and source paths expected by the RAG coordinator.

            """

            previous = set() if saved is None else set(
                saved["selected_files"]
            )
            selected = set(current["selected_files"])
            return {
                "matches": saved is not None and saved == current,
                "added": sorted(selected - previous),
                "modified": [],
                "removed": sorted(previous - selected),
            }

        with patch("qachat.rag.scan_repository", return_value=report), \
            patch(
                "qachat.rag.build_repository_manifest",
                return_value=self.manifest,
            ), patch(
                "qachat.rag.compare_repository_manifests",
                side_effect=compare,
            ):
            return coordinator._prepare_retrieval(embedding)

    def test_build_then_reuse_same_model_and_unchanged_sources(self) -> None:
        """Identical model and source identities permit disk reuse.

        """

        first = self._embedding()
        _, original, status, _ = self._prepare(self._coordinator(), first)
        self.assertEqual(status, "rebuilt")
        self.assertEqual(first.calls, [["physics diffusion"]])
        self.assertTrue(original.embedding_model_identity)

        second = self._embedding()
        _, loaded, status, changes = self._prepare(
            self._coordinator(), second
        )
        self.assertEqual(status, "reused")
        self.assertEqual(second.calls, [])
        self.assertFalse(changes["configuration_changed"])
        self.assertEqual(loaded, original)

    def test_new_session_uses_cached_model_file_hashes(self) -> None:
        """A fresh RAG session reuses disk hashes, not just memory cache.

        """

        self._prepare(self._coordinator(), self._embedding())
        with patch.object(
            model_identity, "_file_digest", wraps=model_identity._file_digest,
        ) as digest:
            second = self._embedding()
            _, _, status, _ = self._prepare(self._coordinator(), second)
        self.assertEqual(status, "reused")
        self.assertEqual(second.calls, [])
        digest.assert_not_called()

    def test_local_weight_replacement_forces_complete_rebuild(self) -> None:
        """Replacing weights at the same path invalidates every old vector.

        """

        self._prepare(self._coordinator(), self._embedding())
        old = load_indexes(self.index_directory / "retrieval.json")
        stat = self.weights.stat()
        self.weights.write_bytes(b"BBB")
        os.utime(self.weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = self._embedding()
        _, rebuilt, status, changes = self._prepare(
            self._coordinator(), second
        )
        self.assertEqual(status, "rebuilt")
        self.assertEqual(second.calls, [["physics diffusion"]])
        self.assertTrue(changes["configuration_changed"])
        self.assertNotEqual(
            rebuilt.embedding_model_identity, old.embedding_model_identity
        )
        self.assertEqual(rebuilt.fingerprint, old.fingerprint)

    def test_bad_payload_checksum_rebuilds_instead_of_reusing(self) -> None:
        """Both reuse and incremental paths must reject corrupted vectors.

        """

        self._prepare(self._coordinator(), self._embedding())
        target = self.index_directory / "retrieval.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["semantic_embeddings"][0][0] = 9.0
        target.write_text(json.dumps(payload), encoding="utf-8")
        second = self._embedding()
        _, _, status, _ = self._prepare(self._coordinator(), second)
        self.assertEqual(status, "rebuilt")
        self.assertEqual(second.calls, [["physics diffusion"]])

    def test_index_missing_integrity_fields_is_rebuilt(self) -> None:
        """Incomplete bundles do not recycle previously saved vectors.

        """

        self._prepare(self._coordinator(), self._embedding())
        target = self.index_directory / "retrieval.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        del payload["payload_sha256"]
        del payload["embedding_model_identity"]
        target.write_text(json.dumps(payload), encoding="utf-8")
        second = self._embedding()
        _, rebuilt, status, _ = self._prepare(
            self._coordinator(), second
        )
        self.assertEqual(status, "rebuilt")
        self.assertEqual(second.calls, [["physics diffusion"]])
        self.assertTrue(rebuilt.embedding_model_identity)
        restored = json.loads(target.read_text(encoding="utf-8"))
        self.assertIn("payload_sha256", restored)
        self.assertIn("embedding_model_identity", restored)
        self.assertNotIn("format_version", restored)

    def test_index_model_identity_mismatch_prevents_reuse(self) -> None:
        """Reject a validly checksummed bundle from another model.

        """

        from qachat.retrievals import persistence

        self._prepare(self._coordinator(), self._embedding())
        target = self.index_directory / "retrieval.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["embedding_model_identity"] = "another-model"
        payload["payload_sha256"] = persistence._payload_checksum(payload)
        target.write_text(json.dumps(payload), encoding="utf-8")
        second = self._embedding()
        _, _, status, _ = self._prepare(self._coordinator(), second)
        self.assertEqual(status, "rebuilt")
        self.assertEqual(second.calls, [["physics diffusion"]])

    def test_one_loaded_model_identity_is_calculated_only_once(self) -> None:
        """Source refresh must not reread large local weights repeatedly.

        """

        coordinator = self._coordinator()
        embedding = self._embedding()
        with patch(
            "qachat.rag.embedding_model_identity",
            wraps=embedding_model_identity,
        ) as digest:
            self._prepare(coordinator, embedding)
            self._prepare(coordinator, embedding)
        digest.assert_called_once()

    def test_incremental_build_reuses_unchanged_model_vectors(self) -> None:
        """Valid metadata retains source-level incremental vector reuse.

        """

        self._prepare(self._coordinator(), self._embedding())
        new_chunk = chunk("second", "chemistry", path="src/new.py")
        self.selected.append(Path("src/new.py"))
        self.manifest = {"selected_files": [
            "src/example.py", "src/new.py",
        ]}
        second = self._embedding()
        _, result, status, changes = self._prepare(
            self._coordinator([self.source_chunk, new_chunk]), second
        )
        self.assertEqual(status, "rebuilt")
        self.assertFalse(changes["configuration_changed"])
        self.assertEqual(second.calls, [["chemistry"]])
        self.assertEqual(len(result.semantic.embeddings), 2)

    def test_real_index_metadata_tracks_embedding_settings(self) -> None:
        """Project metadata records compatibility settings without versions.

        """

        coordinator = self._coordinator()
        coordinator.backend = "torch"
        coordinator.project.profile = SimpleNamespace(
            kind="code",
            extensions=(".py",),
            max_file_bytes=100000,
            target_tokens=500,
            max_tokens=700,
            overlap_tokens=80,
            query_format="plain",
            embedding_model=lambda backend: str(self.models),
        )
        metadata = RAGSession._index_metadata(coordinator)
        self.assertNotIn("format_version", metadata)
        self.assertEqual(metadata["embedding_model"], str(self.models))

    def test_missing_resolved_model_source_fails_closed(self) -> None:
        """Never trust a source-only identity when model paths are unknown.

        """

        embedding = self._embedding()
        del embedding.model_source
        with self.assertRaisesRegex(ValueError, "source"):
            self._prepare(self._coordinator(), embedding)


if __name__ == "__main__":
    unittest.main()
