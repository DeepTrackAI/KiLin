"""Preserve paper-card regression cases against the active card system.

The original standalone paper-card implementation has been retired.
These checks exercise the corresponding scientific-paper and persistence
contracts without requiring old cache formats or real model inference.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.chunking import Chunk
from kilin.retrievals.source_cards import (
    CARD_STATUS_FAILED,
    CARD_STATUS_VALID,
    SourceDescriptor,
)
from kilin.retrievals.source_cards.index import (
    SourceCardIndex,
    SourceCardVectorIndex,
)
from kilin.retrievals.source_cards.kinds.scientific_paper import (
    CARD_FIELDS,
    card_passages,
    card_prompt,
    card_texts,
    parse_card,
)
from kilin.retrievals.source_cards.persistence import (
    load_source_cards,
    load_source_card_vectors,
    save_source_cards,
    save_source_card_vectors,
)
from tests.kilin.retrievals.source_cards.kinds.test_scientific_paper import (
    DIGEST,
    PATH,
    _chunks,
    _reply,
    _source,
)
from tests.kilin.retrievals.source_cards.test_index import (
    _fixture,
    _vector,
)


def _parse(reply, passages=None, *, source=None):
    """Parse an example response using real chunk-level provenance.

    Parameters
    ----------
    reply: str
        Simulated generation output.
    passages: Sequence[Chunk] | None, optional
        Excerpts the model received.
    source: SourceDescriptor | None, optional
        Source identity corresponding to the excerpts.

    Returns
    -------
    SourceCardResult
        Deterministically validated scientific-paper outcome.

    """

    selected = _chunks() if passages is None else passages
    descriptor = _source(selected) if source is None else source
    return parse_card(
        reply, selected, source=descriptor, source_sha256=DIGEST,
        model_id="test-model", generation_settings="test-settings",
    )


class TestCardPassages(unittest.TestCase):

    def test_reading_order_and_opening_passages(self):
        pieces = _chunks()
        selected = card_passages(
            pieces[::-1], PATH.name, max_input_tokens=9000,
        )
        self.assertEqual(
            [item.identifier for item in selected[:3]],
            ["first", "second", "third"],
        )

    def test_budget_keeps_the_first_passage(self):
        pieces = (
            Chunk("first", PATH, "word " * 600, 1, 5,
                  "pdf_page", None, "Page 1"),
            Chunk("second", PATH, "word " * 600, 1, 5,
                  "pdf_page", None, "Page 2"),
        )
        selected = card_passages(
            pieces, PATH.name, max_input_tokens=650,
        )
        self.assertEqual(selected[0].identifier, "first")
        self.assertLess(len(selected[0].text), len(pieces[0].text))
        self.assertEqual(card_passages(
            [], PATH.name, max_input_tokens=650,
        ), [])

    def test_field_passages_are_found_beyond_the_opening(self):
        pieces = tuple(
            Chunk(f"p{n}", PATH, f"Introduction {n}", 1, 2,
                  "pdf_page", None, f"Page {n}")
            for n in range(1, 8)
        ) + (Chunk(
            "lim", PATH, "Limitations and future work remain.",
            1, 2, "pdf_page", None, "Page 9",
        ),)
        selected = card_passages(
            pieces, PATH.name, max_input_tokens=9000,
        )
        self.assertEqual(
            [item.identifier for item in selected[:3]],
            ["p1", "p2", "p3"],
        )
        self.assertIn("lim", [item.identifier for item in selected])

    def test_prompt_numbers_passages_with_pages(self):
        system, prompt = card_prompt(PATH.name, _chunks())
        self.assertIn("scientific paper", system)
        self.assertIn("[2]", prompt)
        self.assertIn("page 2", prompt)
        for name in CARD_FIELDS:
            self.assertIn(f'"{name}"', prompt)


class TestParseCard(unittest.TestCase):

    def test_fields_pages_and_citation_checks(self):
        result = _parse("```json\n" + _reply() + "\n```")
        self.assertEqual(result.status, CARD_STATUS_VALID)
        card = result.card
        values = {item.field: item for item in card.statements}
        self.assertEqual(values["methods"].chunk_ids, ("second",))
        self.assertEqual(
            values["abstract"].chunk_ids, ("first", "third"),
        )
        self.assertEqual(dict(card.metadata)["year"], "2026")

    def test_known_title_takes_precedence(self):
        result = _parse(_reply(title=""))
        self.assertEqual(
            dict(result.card.metadata)["title"], _source().title,
        )

    def test_unusable_replies_fail_validation(self):
        for response in (
            "", "no json", "[1, 2]", "{broken", "{}",
            '{"title": "Only metadata"}',
            _reply(abstract="", aim="", methods="", findings="",
                   limitations="", data=""),
        ):
            with self.subTest(response=response[:50]):
                self.assertEqual(_parse(response).status,
                                 CARD_STATUS_FAILED)

    def test_uncited_fields_are_not_used_for_card_search(self):
        result = _parse(_reply(
            methods="Unsupported method", findings="Bad [90]",
        ))
        self.assertEqual(result.status, CARD_STATUS_VALID)
        fields = {item.field for item in result.card.statements}
        self.assertNotIn("methods", fields)
        self.assertNotIn("findings", fields)
        self.assertIn("abstract", fields)

    def test_enormous_card_citations_are_ignored_safely(self):
        citation = "9" * 12000
        result = _parse(_reply(
            abstract="", aim="", methods=f"Unsupported [{citation}]",
            findings="", data="", limitations="",
        ))
        self.assertEqual(result.status, CARD_STATUS_FAILED)

    def test_non_pdf_sources_can_have_cited_content(self):
        path = Path("notes.md")
        passage = Chunk(
            "md", path, "Evidence for the authors' stated aim.",
            1, 2, "markdown_section", None, "Introduction",
        )
        descriptor = SourceDescriptor(path, "Notes", ".md", ("md",))
        result = _parse(
            _reply(abstract="", aim="The authors explained their aim [1].",
                   methods="", findings="", limitations="", data=""),
            [passage], source=descriptor,
        )
        self.assertEqual(result.status, CARD_STATUS_VALID)
        self.assertEqual(result.card.statements[0].chunk_ids, ("md",))

    def test_card_texts_group_fields(self):
        card = _parse(_reply()).card
        groups = card_texts(card)
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(item.startswith("Optical trapping study")
                            for item in groups))
        self.assertIn("fluorescence", groups[1])
        self.assertIn("quality", groups[2])


class TestSavedCards(unittest.TestCase):

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        record, _, _ = _fixture()
        self.cards = SourceCardIndex((record,))
        self.vectors = SourceCardVectorIndex(
            "embedding:test", (_vector(record),),
        )
        self.card_file = self.directory / "source_cards.json"
        self.vector_file = self.directory / "source_card_vectors.json"

    def _save(self):
        save_source_cards(self.cards, self.card_file)
        save_source_card_vectors(
            self.vectors, self.cards, self.vector_file,
        )

    def test_round_trip_and_format_check(self):
        self._save()
        self.assertEqual(load_source_cards(self.card_file), self.cards)
        self.assertEqual(
            load_source_card_vectors(self.vector_file, self.cards),
            self.vectors,
        )
        payload = json.loads(self.card_file.read_text())
        payload["format"] = "obsolete"
        self.card_file.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_source_cards(self.card_file)

    def test_reject_corrupt_vectors_on_write(self):
        from dataclasses import replace
        record = self.vectors.records[0]
        for values in (
            ((float("nan"), 0.0),),
            ((0.0, 0.0),),
            ((1.0, 0.0), (1.0,)),
        ):
            with self.subTest(values=values):
                invalid = replace(record, embeddings=values)
                with self.assertRaises(ValueError):
                    save_source_card_vectors(
                        SourceCardVectorIndex(
                            "embedding:test", (invalid,)
                        ), self.cards, self.vector_file,
                    )

    def test_corrupt_complete_cards_are_regenerated(self):
        self._save()
        data = json.loads(self.card_file.read_text())
        data["records"][0]["card"]["statements"] = []
        self.card_file.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            load_source_cards(self.card_file)

    def test_invalid_loaded_vectors_keep_valid_card_text(self):
        self._save()
        data = json.loads(self.vector_file.read_text())
        data["records"][0]["embeddings"][0][0] = "NaN"
        self.vector_file.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            load_source_card_vectors(self.vector_file, self.cards)
        self.assertEqual(load_source_cards(self.card_file), self.cards)


if __name__ == "__main__":
    unittest.main()
