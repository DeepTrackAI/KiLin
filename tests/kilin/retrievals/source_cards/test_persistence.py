"""Verify independent source-card files, integrity, and interrupted updates.

Temporary JSON files and deterministic vectors exercise persistence without
starting the answering model or changing ordinary retrieval indexes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from unittest.mock import patch

from kilin.retrievals.source_cards import (
    CARD_STATUS_FAILED,
    CARD_STATUS_NOT_APPLICABLE,
    SourceCardResult,
)
from kilin.retrievals.source_cards.index import (
    IndexedSourceCard,
    SourceCardIndex,
    SourceCardVectorIndex,
    plan_source_card_vectors,
)
from kilin.retrievals.source_cards.persistence import (
    load_source_card_vectors,
    load_source_cards,
    save_source_card_vectors,
    save_source_cards,
)
from kilin.retrievals.source_cards import persistence as store
from tests.kilin.retrievals.source_cards.test_index import (
    HASH, _fixture, _vector,
)


class TestSourceCardPersistence(unittest.TestCase):

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name) / "indexes"
        self.card_file = root / "source_cards.json"
        self.vector_file = root / "source_card_vectors.json"
        self.record, _, _ = _fixture()
        path = self.record.result.path
        self.excluded = IndexedSourceCard(
            SourceCardResult(path, "second", CARD_STATUS_NOT_APPLICABLE,
                             reason="certificate, not a paper"),
            HASH, 1, 1, "model:a", "temp:0",
        )
        self.cards = SourceCardIndex((self.record, self.excluded))
        self.vector = _vector(self.record)
        self.vectors = SourceCardVectorIndex(
            "embedding:a", (self.vector,)
        )

    def _payload(self, file):
        return json.loads(file.read_text(encoding="utf-8"))

    def _mutate(self, file, change, *, reseal=True):
        data = self._payload(file)
        change(data)
        if reseal:
            data["payload_sha256"] = store._checksum({
                key: value for key, value in data.items()
                if key != "payload_sha256"
            })
        file.write_text(json.dumps(data), encoding="utf-8")

    def test_roundtrip_preserves_status_and_independent_vectors(self):
        self.assertEqual(load_source_cards(self.card_file),
                         SourceCardIndex(()))
        self.assertIsNone(load_source_card_vectors(
            self.vector_file, self.cards
        ))
        self.assertEqual(save_source_cards(self.cards, self.card_file),
                         self.card_file.resolve())
        self.assertEqual(load_source_cards(self.card_file), self.cards)
        save_source_card_vectors(
            self.vectors, self.cards, self.vector_file
        )
        self.assertEqual(load_source_card_vectors(
            self.vector_file, self.cards
        ), self.vectors)
        self.assertNotIn("embeddings", self._payload(self.card_file))
        self.assertEqual(self._payload(self.vector_file)["format"],
                         "source-card-vectors-v1")

    def test_invalid_cache_entries_and_failed_attempts_are_persisted(self):
        failed = replace(self.excluded, result=SourceCardResult(
            self.excluded.result.path, "second", CARD_STATUS_FAILED,
            reason="model unavailable",
        ))
        cards = SourceCardIndex((self.record, failed))
        save_source_cards(cards, self.card_file)
        self.assertEqual(load_source_cards(self.card_file), cards)
        with self.assertRaisesRegex(ValueError, "valid"):
            save_source_card_vectors(
                SourceCardVectorIndex("embedding:a", (
                    replace(self.vector, kind="second"),
                )), cards, self.vector_file,
            )

    def test_stale_vector_group_is_ignored_after_card_update(self):
        other, _, _ = _fixture("b.pdf", content="Evidence B")
        # Different source, valid card and distinct key, but same model.
        cards = SourceCardIndex((self.record, other))
        other_vector = _vector(
            other, texts=("search b",), values=((0.0, 1.0),)
        )
        vectors = SourceCardVectorIndex(
            "embedding:a", (self.vector, other_vector)
        )
        save_source_cards(cards, self.card_file)
        save_source_card_vectors(vectors, cards, self.vector_file)
        changed = replace(self.record.result.card, statements=(
            replace(self.record.result.card.statements[0], text="Updated"),
        ))
        new_record = replace(self.record, result=replace(
            self.record.result, card=changed
        ))
        new_cards = SourceCardIndex((new_record, other))
        save_source_cards(new_cards, self.card_file)
        # Simulate a crash before rewriting the vectors file.
        retained = load_source_card_vectors(
            self.vector_file, load_source_cards(self.card_file)
        )
        self.assertEqual(retained.records, (other_vector,))
        plan = plan_source_card_vectors(
            new_cards, retained,
            {new_record.key: ("search a",), other.key: ("search b",)},
            embedding_model_identity="embedding:a",
        )
        self.assertEqual(plan.pending, (self.record.key,))
        self.assertEqual(plan.reused, (other_vector,))
        self.assertEqual(load_source_cards(self.card_file), new_cards)

    def test_card_file_checksum_and_format_reject_corruption(self):
        save_source_cards(self.cards, self.card_file)
        for change, reseal in (
            (lambda p: p["records"][0].update(source_sha256="0" * 64),
             False),
            (lambda p: p.update(format="old-paper-format"), True),
            (lambda p: p["records"][0].update(source_sha256="bad"), True),
            (lambda p: p["records"][0].update(path="../escape"), True),
            (lambda p: p["records"][0].update(schema_version=True), True),
            (lambda p: p.update(unexpected="field"), True),
        ):
            with self.subTest(change=change):
                save_source_cards(self.cards, self.card_file)
                self._mutate(self.card_file, change, reseal=reseal)
                with self.assertRaises(ValueError):
                    load_source_cards(self.card_file)
        self.card_file.write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_source_cards(self.card_file)

    def test_vector_file_rejects_bad_numbers_dimensions_and_checksum(self):
        save_source_card_vectors(
            self.vectors, self.cards, self.vector_file
        )
        for vector in ([0.0, 0.0], [float("nan"), 1.0],
                       [True, 0.0], [10 ** 1000, 0]):
            with self.subTest(vector=vector):
                save_source_card_vectors(
                    self.vectors, self.cards, self.vector_file
                )
                self._mutate(
                    self.vector_file,
                    lambda p: p["records"][0]["embeddings"].__setitem__(
                        0, vector
                    ),
                    reseal=not any(isinstance(v, float) and v != v
                                   for v in vector),
                )
                with self.assertRaises(ValueError):
                    load_source_card_vectors(self.vector_file, self.cards)
        save_source_card_vectors(
            self.vectors, self.cards, self.vector_file
        )
        self._mutate(self.vector_file, lambda p: p["records"][0].update(
            text_sha256="a" * 64), reseal=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_source_card_vectors(self.vector_file, self.cards)

    def test_numpy_embedding_components_are_stored_as_json_numbers(self):
        values = ((np.float32(1.0), np.float32(0.0)),)
        vector = replace(self.vector, embeddings=values)
        wrapped = SourceCardVectorIndex("embedding:a", (vector,))
        save_source_card_vectors(wrapped, self.cards, self.vector_file)
        payload = self._payload(self.vector_file)
        self.assertEqual(payload["records"][0]["embeddings"],
                         [[1.0, 0.0]])
        self.assertEqual(load_source_card_vectors(
            self.vector_file, self.cards
        ).records[0].embeddings, ((1.0, 0.0),))

    def test_duplicate_json_keys_and_unknown_format_rejected(self):
        self.card_file.parent.mkdir(parents=True, exist_ok=True)
        self.card_file.write_text('{"format":"a","format":"b"}',
                                  encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_source_cards(self.card_file)
        # Old paper-card data is not used as a migration fallback.
        self.card_file.unlink()
        (self.card_file.parent / "paper_cards.json").write_text(
            "{}", encoding="utf-8"
        )
        self.assertEqual(load_source_cards(self.card_file),
                         SourceCardIndex(()))

    def test_atomic_vector_write_failure_preserves_previous_vectors(self):
        save_source_card_vectors(
            self.vectors, self.cards, self.vector_file
        )
        before = self.vector_file.read_bytes()
        with patch.object(store, "atomic_write_text",
                          side_effect=OSError("disk is full")):
            with self.assertRaises(OSError):
                save_source_card_vectors(
                    SourceCardVectorIndex("embedding:a", ()),
                    self.cards, self.vector_file,
                )
        self.assertEqual(self.vector_file.read_bytes(), before)

    def test_atomic_write_failure_preserves_previous_cards(self):
        save_source_cards(self.cards, self.card_file)
        before = self.card_file.read_bytes()
        with patch.object(store, "atomic_write_text",
                          side_effect=OSError("disk is full")):
            with self.assertRaises(OSError):
                save_source_cards(
                    SourceCardIndex((self.record,)), self.card_file
                )
        self.assertEqual(self.card_file.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
