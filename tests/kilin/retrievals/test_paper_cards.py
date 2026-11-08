"""Verify paper cards: reading, checking, saving, and reuse at indexing.

The answering and embedding models are deterministic substitutes; card
files are written to temporary folders.

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
from kilin.retrievals import paper_cards as pc
from kilin.retrievals.papers import DocumentInfo
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


class TestCardPassages(unittest.TestCase):

    def test_reading_order_and_opening_passages(self):
        passages = pc.card_passages(_paper())
        self.assertEqual([chunk.identifier for chunk in passages],
                         ["a.pdf-1", "a.pdf-2", "a.pdf-3"])

    def test_budget_keeps_the_first_passage(self):
        long = [_page("p1", "word " * 400, 1), _page("p2", "word " * 400, 2)]
        passages = pc.card_passages(long, budget=10)
        self.assertEqual([chunk.identifier for chunk in passages], ["p1"])
        self.assertEqual(pc.card_passages([]), [])

    def test_field_passages_are_found_beyond_the_opening(self):
        chunks = [_page(f"p{n}", f"Introduction text {n}.", n)
                  for n in range(1, 8)]
        chunks.append(_page("lim", "Limitations: future work remains.", 9))
        identifiers = [c.identifier for c in pc.card_passages(chunks)]
        self.assertIn("lim", identifiers)
        self.assertEqual(identifiers[:3], ["p1", "p2", "p3"])

    def test_prompt_numbers_passages_with_pages(self):
        system, prompt = pc.card_prompt("a.pdf", pc.card_passages(_paper()))
        self.assertIn("JSON", system)
        self.assertIn("[2] (page 2) Methods: we used actigraphy.", prompt)
        for field in pc.CARD_FIELDS:
            self.assertIn(f'"{field}"', prompt)


class TestParseCard(unittest.TestCase):

    def setUp(self):
        self.passages = pc.card_passages(_paper())

    def _parse(self, reply, title=None):
        return pc.parse_card(reply, self.passages, path=Path("a.pdf"),
                             sha256="h", title=title, model="m")

    def test_fields_pages_and_citation_checks(self):
        card = self._parse("Here it is:\n```json\n" + REPLY + "\n```")
        self.assertTrue(card.complete)
        self.assertEqual(card.title, "Model title")
        self.assertEqual(card.year, "2020")
        self.assertEqual(card.field("methods"), "Actigraphy in a cohort.")
        # [9] was not shown to the model, so only page 2 is recorded.
        self.assertEqual(card.field_pages("methods"), (2,))
        self.assertEqual(card.field_pages("abstract"), (1,))
        self.assertEqual(card.field("data"), "")
        self.assertEqual(card.field("limitations"), "")
        self.assertEqual(card.field_pages("data"), ())

    def test_known_title_takes_precedence(self):
        self.assertEqual(self._parse(REPLY, "PDF title").title, "PDF title")

    def test_unusable_replies_give_a_title_card(self):
        for reply in (
            "", "no json here", "[1, 2]", "{broken", "{}",
            '{"title": "A study", "authors": "A. Smith"}',
            '{"methods": "Unsupported claim [99]."}',
            '{"methods": "Uncited claim."}',
        ):
            with self.subTest(reply=reply):
                card = self._parse(reply, None)
                self.assertFalse(card.complete)
                self.assertEqual(card.title, "a")
                self.assertEqual(card.fields, ())

    def test_uncited_fields_are_not_used_for_card_search(self):
        reply = json.dumps({
            "abstract": "Supported abstract [1].",
            "methods": "Unsupported method.",
            "findings": "Invalid passage [8].",
        })
        card = self._parse(reply)
        self.assertTrue(card.complete)
        self.assertEqual(card.abstract, "Supported abstract.")
        self.assertEqual(card.field("methods"), "")
        self.assertEqual(card.field("findings"), "")

    def test_enormous_card_citations_are_ignored_safely(self):
        reply = json.dumps({
            "methods": "Unsupported [" + "9" * 12000 + "].",
        })
        self.assertFalse(self._parse(reply).complete)

    def test_non_pdf_sources_can_have_cited_content(self):
        passage = Chunk(
            "md", Path("notes.md"), "Relevant description.",
            1, 2, "markdown_section", None, "Introduction"
        )
        card = pc.parse_card(
            '{"aim": "Description [1]."}', [passage],
            path=Path("notes.md"), sha256="h", title=None, model="m",
        )
        self.assertTrue(card.complete)
        self.assertEqual(card.field("aim"), "Description.")
        self.assertEqual(card.field_pages("aim"), ())

    def test_card_texts_group_fields(self):
        card = self._parse(REPLY)
        texts = pc.card_texts(card)
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(text.startswith("Model title") for text in texts))
        self.assertIn("Why sleep declines.", texts[0])
        self.assertIn("Actigraphy", texts[1])
        self.assertEqual(pc.card_texts(self._parse("")), ("a",))


class TestSavedCards(unittest.TestCase):

    def test_round_trip_and_format_check(self):
        card = pc.parse_card(REPLY, pc.card_passages(_paper()),
                             path=Path("d/a.pdf"), sha256="h", title=None,
                             model="m")
        with TemporaryDirectory() as folder:
            file = Path(folder) / "indexes" / "paper_cards.json"
            embeddings = ((1.0, 0.0),) * len(pc.card_texts(card))
            pc.save_cards(file, {card.path: card},
                          {card.path: embeddings}, "embed:1")
            cards, vectors, identity = pc.load_cards(file)
            self.assertEqual(cards, {card.path: card})
            self.assertEqual(vectors, {card.path: embeddings})
            self.assertEqual(identity, "embed:1")
            data = json.loads(file.read_text(encoding="utf-8"))
            data["format"] = "old"
            file.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(pc.load_cards(file), ({}, {}, None))
            self.assertEqual(pc.load_cards(Path(folder) / "none"),
                             ({}, {}, None))

    def test_reject_corrupt_vectors_on_write(self):
        card = pc.parse_card(
            REPLY, pc.card_passages(_paper()),
            path=Path("a.pdf"), sha256="h", title=None, model="m",
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cards.json"
            for invalid in (
                ((float("nan"), 0.0),) * 3,
                ((0.0, 0.0),) * 3,
                ((1.0, 0.0),),
                ((1.0, 0.0), (1.0,), (1.0, 0.0)),
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        pc.save_cards(
                            path, {card.path: card},
                            {card.path: invalid}, "embed:1",
                        )

    def test_corrupt_complete_cards_are_regenerated(self):
        card = pc.parse_card(
            REPLY, pc.card_passages(_paper()),
            path=Path("a.pdf"), sha256="h", title=None, model="m",
        )
        with TemporaryDirectory() as folder:
            file = Path(folder) / "cards.json"
            embeddings = ((1.0, 0.0),) * len(pc.card_texts(card))
            pc.save_cards(
                file, {card.path: card}, {card.path: embeddings},
                "embed:1",
            )
            data = json.loads(file.read_text(encoding="utf-8"))
            data["cards"]["a.pdf"]["abstract"] = ""
            data["cards"]["a.pdf"]["fields"] = {}
            file.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(pc.load_cards(file)[0], {})

    def test_invalid_loaded_vectors_keep_valid_card_text(self):
        card = pc.parse_card(
            REPLY, pc.card_passages(_paper()),
            path=Path("a.pdf"), sha256="h", title=None, model="m",
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cards.json"
            embeddings = ((1.0, 0.0),) * len(pc.card_texts(card))
            pc.save_cards(
                path, {card.path: card}, {card.path: embeddings},
                "embed:1",
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            data["cards"]["a.pdf"]["vectors"][1][0] = "NaN"
            path.write_text(json.dumps(data), encoding="utf-8")
            cards, vectors, _ = pc.load_cards(path)
            self.assertEqual(cards[card.path], card)
            self.assertEqual(vectors[card.path], ())


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
