"""Verify RAG integration with the active source-card pipeline.

Synthetic document passages and a fake generation session exercise card
reuse, invalidation and error recovery without loading a real model.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from kilin import rag
from kilin._model_completion import IncompleteModelResponse
from kilin.chunking import Chunk
from kilin.retrievals.document_headers import DocumentInfo
from tests.kilin import _rag_support as fixtures

REPLY = json.dumps({
    "title": "Model title",
    "authors": "A. Smith, B. Lee",
    "year": "2020",
    "abstract": "We study sleep [1].",
    "aim": "Why sleep declines [1].",
    "methods": "Actigraphy in a cohort [2, 9].",
    "data": "",
    "findings": "Sleep declines with age [3].",
    "limitations": 5,
})


def _page(identifier: str, text: str, page: int, path: str = "a.pdf",
          start: int = 1) -> Chunk:
    """Return one PDF passage on a page."""
    return Chunk(identifier, Path(path), text, start, start + 3,
                 "pdf_page", None, f"Page {page}")


def _paper(path: str = "a.pdf") -> list[Chunk]:
    """Return three passages of one paper, out of reading order."""
    return [
        _page(f"{path}-3", "Results show sleep declines.", 5, path),
        _page(f"{path}-1", "Title and abstract.", 1, path),
        _page(f"{path}-2", "Methods: we used actigraphy.", 2, path),
    ]


class TestSessionCards(unittest.TestCase):

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.session = fixtures._session(Path(folder.name))
        # This fixture has no real answering model; refresh still releases
        # and restores its mocked conversation during card generation.
        self.session._new_chat_session = Mock(return_value=MagicMock())
        self.session.project.profile = SimpleNamespace(
            **{**vars(fixtures._profile()), "type_id": "literature"}
        )
        self.session.project.indexes.mkdir(parents=True)
        self.embedding = SimpleNamespace(
            _retrieval_model_identity="embed:1",
            embed_documents=Mock(
                side_effect=lambda texts: [[1.0, 0.0] for _ in texts]
            ),
        )
        self.chunks = _paper("a.pdf") + _paper("b.pdf")
        self._hashes({"a.pdf": "ha", "b.pdf": "hb"})
        info = patch.object(rag, "document_info",
                            return_value=DocumentInfo("PDF title", None))
        info.start()
        self.addCleanup(info.stop)

    def _hashes(self, hashes):
        from hashlib import sha256
        self.session._current_source_manifest = {
            "files": {name: {"sha256": sha256(value.encode()).hexdigest()}
                      for name, value in hashes.items()}
        }

    def _prepare(self, reply=REPLY, *, failure=None, dimension=None):
        from kilin.retrievals.source_cards.indexing import (
            reconcile_source_cards,
        )
        data = json.loads(reply) if reply.startswith("{") else {}
        if data.get("methods"):
            data["methods"] = "Actigraphy in a cohort [2]."
        data.update({"applicable": True, "reason": "", "venue": "",
                     "identifiers": "", "limitations": ""})
        response = json.dumps(data) if data.get("abstract") else reply
        model = Mock(side_effect=failure) if failure else Mock(
            return_value=response
        )

        class FakeGenerator:
            def __init__(self, settings, **_kwargs):
                self.settings = settings

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def generate(self, system, prompt):
                return model(system, prompt)

        def coordinator(*args, **kwargs):
            return reconcile_source_cards(
                *args, **kwargs, generator_factory=FakeGenerator
            )

        with patch.object(rag, "reconcile_source_cards",
                          side_effect=coordinator):
            if dimension:
                from kilin.retrievals import (
                    PersistentIndexes, SemanticIndex, LexicalIndex,
                )
                from kilin.retrievals.code_graph import CodeGraph
                indexes = PersistentIndexes(
                    "fp", SemanticIndex((), ((1.0,) * dimension,)),
                    LexicalIndex((), (), {}), CodeGraph((), ()),
                )
            else:
                indexes = SimpleNamespace(semantic=SimpleNamespace(
                    embeddings=[]
                ))
            cards, vectors = self.session._prepare_source_cards(
                self.chunks, indexes, self.embedding
            )
        return cards, vectors, model

    def test_cards_are_written_embedded_and_reused(self):
        cards, vectors, model = self._prepare()
        self.assertEqual(model.call_count, 2)
        self.assertEqual(len(cards.records), 2)
        card = cards.get(Path("a.pdf"), "scientific_paper").result.card
        self.assertEqual(dict(card.metadata)["title"], "Model title")
        self.assertEqual(len(vectors.records[0].embeddings), 3)
        self.assertTrue(
            (self.session.project.indexes / "source_cards.json").is_file()
        )
        self.embedding.embed_documents.reset_mock()
        again, again_vectors, model = self._prepare()
        model.assert_not_called()
        self.embedding.embed_documents.assert_not_called()
        self.assertEqual(again, cards)
        self.assertEqual(again_vectors, vectors)

    def test_changed_file_writes_only_its_card(self):
        self._prepare()
        self._hashes({"a.pdf": "ha", "b.pdf": "changed"})
        _, _, model = self._prepare()
        self.assertEqual(model.call_count, 1)

    def test_new_embedding_model_reembeds_without_rewriting(self):
        self._prepare()
        self.embedding._retrieval_model_identity = "embed:2"
        self.embedding.embed_documents.reset_mock()
        _, _, model = self._prepare()
        model.assert_not_called()
        self.embedding.embed_documents.assert_called_once()

    def test_failed_cards_are_not_embedded_or_retried_by_one_model(self):
        from kilin.retrievals.source_cards import CARD_STATUS_FAILED
        cards, vectors, model = self._prepare(
            failure=IncompleteModelResponse("cut", reason="token_limit")
        )
        self.assertEqual(model.call_count, 2)
        self.assertTrue(all(
            entry.result.status == CARD_STATUS_FAILED
            for entry in cards.records
        ))
        self.assertFalse(vectors.records)
        self.embedding.embed_documents.assert_not_called()
        _, _, model = self._prepare()
        model.assert_not_called()
        from kilin.model_settings import SourceCardGenerationSettings
        from kilin.retrievals.source_cards import indexing
        with patch.object(indexing, "read_source_card_generation_settings",
                          side_effect=lambda root, backend: (
                              SourceCardGenerationSettings(
                                  "Qwen/Qwen3-4B", False, 0.0, 5000, 1200
                              )
                          )):
            _, vectors, model = self._prepare()
        self.assertEqual(model.call_count, 2)
        self.assertEqual(len(vectors.records), 2)

    def test_invalid_vector_cache_reembeds_without_rewriting(self):
        self._prepare()
        file = self.session.project.indexes / "source_card_vectors.json"
        data = json.loads(file.read_text(encoding="utf-8"))
        data["records"][0]["embeddings"][0][0] = "NaN"
        file.write_text(json.dumps(data), encoding="utf-8")
        self.embedding.embed_documents.reset_mock()
        cards, vectors, model = self._prepare()
        model.assert_not_called()
        self.assertEqual(len(cards.records), 2)
        self.assertEqual(len(vectors.records), 2)
        self.embedding.embed_documents.assert_called_once()

    def test_wrong_vector_dimension_reembeds_without_card_generation(self):
        self._prepare()
        self.embedding.embed_documents.side_effect = (
            lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        )
        self.embedding.embed_documents.reset_mock()
        cards, vectors, model = self._prepare(dimension=3)
        model.assert_not_called()
        self.assertEqual(len(vectors.records[0].embeddings[0]), 3)
        self.assertEqual(len(cards.records), 2)
        self.embedding.embed_documents.assert_called_once()

    def test_changed_source_failed_generation_is_persisted(self):
        from kilin.retrievals.source_cards import CARD_STATUS_FAILED
        from kilin.retrievals.source_cards.persistence import (
            load_source_cards,
        )
        self._prepare()
        self._hashes({"a.pdf": "new", "b.pdf": "hb"})
        cards, vectors, model = self._prepare("{}")
        self.assertEqual(model.call_count, 1)
        self.assertEqual(
            cards.get(Path("a.pdf"), "scientific_paper").result.status,
            CARD_STATUS_FAILED,
        )
        self.assertEqual(len(vectors.records), 1)
        stored = load_source_cards(
            self.session.project.indexes / "source_cards.json"
        )
        self.assertEqual(stored, cards)
        _, _, model = self._prepare()
        model.assert_not_called()

    def test_code_projects_and_errors_give_no_cards(self):
        self.session.project.profile = fixtures._profile()
        self.assertFalse(self._prepare()[0].records)
        self.session.project.profile = SimpleNamespace(
            **{**vars(fixtures._profile()), "type_id": "literature"}
        )
        self.embedding.embed_documents.side_effect = RuntimeError("closed")
        cards, vectors, _ = self._prepare()
        self.assertEqual(len(cards.records), 2)
        self.assertFalse(vectors.records)
        from kilin.retrievals.source_cards.persistence import (
            load_source_cards,
        )
        self.assertEqual(len(load_source_cards(
            self.session.project.indexes / "source_cards.json"
        ).records), 2)


if __name__ == "__main__":
    unittest.main()
