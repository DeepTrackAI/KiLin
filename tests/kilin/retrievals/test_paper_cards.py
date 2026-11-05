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
from unittest.mock import Mock, patch

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
        for reply in ("", "no json here", "[1, 2]", "{broken"):
            with self.subTest(reply=reply):
                card = self._parse(reply, None)
                self.assertFalse(card.complete)
                self.assertEqual(card.title, "a")
                self.assertEqual(card.fields, ())

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
            pc.save_cards(file, {card.path: card},
                          {card.path: ((1.0, 0.0),)}, "embed:1")
            cards, vectors, identity = pc.load_cards(file)
            self.assertEqual(cards, {card.path: card})
            self.assertEqual(vectors, {card.path: ((1.0, 0.0),)})
            self.assertEqual(identity, "embed:1")
            data = json.loads(file.read_text(encoding="utf-8"))
            data["format"] = "old"
            file.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(pc.load_cards(file), ({}, {}, None))
            self.assertEqual(pc.load_cards(Path(folder) / "none"),
                             ({}, {}, None))


class TestSessionCards(unittest.TestCase):

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.session = fixtures._session(Path(folder.name))
        self.session.project.profile = SimpleNamespace(
            **{**vars(fixtures._profile()), "type_id": "literature"}
        )
        self.session.chat_session.model_id = "chat-a"
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
        self.session._current_source_manifest = {
            "files": {name: {"sha256": value}
                      for name, value in hashes.items()}
        }

    def _prepare(self, reply=REPLY):
        with patch.object(rag, "side_reply", return_value=reply) as model:
            cards, vectors = self.session._prepare_paper_cards(
                self.chunks, self.session.chat_session, self.embedding
            )
        return cards, vectors, model

    def test_cards_are_written_embedded_and_reused(self):
        cards, vectors, model = self._prepare()
        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.call_args.kwargs["require_complete"], True)
        self.assertEqual(set(cards), {Path("a.pdf"), Path("b.pdf")})
        self.assertEqual(cards[Path("a.pdf")].title, "PDF title")
        self.assertEqual(len(vectors[Path("a.pdf")]), 3)
        self.assertTrue(
            (self.session.project.indexes / "paper_cards.json").is_file()
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
        with patch.object(
            rag, "side_reply",
            side_effect=IncompleteModelResponse("cut", reason="token_limit"),
        ):
            cards, vectors = self.session._prepare_paper_cards(
                self.chunks, self.session.chat_session, self.embedding
            )
        self.assertFalse(cards[Path("a.pdf")].complete)
        self.assertEqual(vectors[Path("a.pdf")], ())
        self.embedding.embed_documents.assert_not_called()
        _, _, model = self._prepare()
        model.assert_not_called()
        self.session.chat_session.model_id = "chat-b"
        _, vectors, model = self._prepare()
        self.assertEqual(model.call_count, 2)
        self.assertEqual(len(vectors[Path("b.pdf")]), 3)

    def test_code_projects_and_errors_give_no_cards(self):
        self.session.project.profile = fixtures._profile()
        self.assertEqual(self._prepare()[:2], ({}, {}))
        self.session.project.profile = SimpleNamespace(
            **{**vars(fixtures._profile()), "type_id": "literature"}
        )
        self.embedding.embed_documents.side_effect = RuntimeError("closed")
        self.assertEqual(self._prepare()[:2], ({}, {}))


if __name__ == "__main__":
    unittest.main()
