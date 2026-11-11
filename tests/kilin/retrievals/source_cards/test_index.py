"""Verify independent card indexing, invalidation, and source-level search.

Deterministic card kinds and vectors exercise cache planning without models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

from kilin.chunking import Chunk
from kilin.retrievals.source_cards import (
    CARD_STATUS_FAILED,
    CARD_STATUS_NOT_APPLICABLE,
    CARD_STATUS_VALID,
    CardStatement,
    SourceCard,
    SourceCardRegistry,
    SourceCardResult,
    SourceDescriptor,
)
from kilin.retrievals.source_cards.index import (
    IndexedSourceCard,
    SourceCardIndex,
    SourceCardVector,
    SourceCardVectorIndex,
    card_digest,
    plan_source_card_updates,
    plan_source_card_vectors,
    prepare_source_card_index,
    prepare_source_card_vectors,
    search_source_cards,
    search_text_digest,
)


HASH = sha256(b"original").hexdigest()
NEW_HASH = sha256(b"changed").hexdigest()


class TestKind:
    identifier = "example"
    schema_version = 1
    validation_version = 1

    def precheck(self, source):
        return bool(source.chunk_ids)

    def validate(self, card):
        if card.statements[0].field != "finding":
            raise ValueError("Expected the finding field.")


def _fixture(name="a.pdf", *, content="Evidence A"):
    """Create a complete source card with an original provenance chunk."""
    path = Path(name)
    source = SourceDescriptor(path, path.stem, path.suffix, ("c1",))
    chunk = Chunk("c1", path, content, 1, 1, "pdf_page", None, "Page 1")
    card = SourceCard(
        path, "example", 1, 1, HASH, (),
        (CardStatement("finding", content, ("c1",)),),
        model_id="model:a", generation_settings="temp:0",
    )
    record = IndexedSourceCard(
        SourceCardResult(path, "example", CARD_STATUS_VALID, card),
        HASH, 1, 1, "model:a", "temp:0",
    )
    return record, source, chunk


def _vector(record, texts=("search a",), values=((1.0, 0.0),)):
    """Create a card-bound vector group from precomputed embeddings."""
    return SourceCardVector(
        record.result.path, record.result.kind,
        card_digest(record.result.card), search_text_digest(texts), values,
    )


class TestCardIndex(unittest.TestCase):

    def setUp(self):
        self.record, self.source, self.chunk = _fixture()
        self.registry = SourceCardRegistry((TestKind(),))
        self.key = self.record.key

    def test_admit_current_cards_with_exact_original_provenance(self):
        index = prepare_source_card_index(
            (self.record,), {self.source.path: self.source},
            (self.chunk,), {self.source.path: HASH}, self.registry,
        )
        self.assertEqual(index.get(*self.key), self.record)
        self.assertIsNone(index.get(Path("absent.pdf"), "example"))
        with self.assertRaisesRegex(ValueError, "stale"):
            prepare_source_card_index(
                (self.record,), {self.source.path: self.source},
                (self.chunk,), {self.source.path: NEW_HASH}, self.registry,
            )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            prepare_source_card_index(
                (self.record,), {self.source.path: self.source},
                (), {self.source.path: HASH}, self.registry,
            )

    def test_records_have_unique_composite_keys(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            SourceCardIndex((self.record, self.record))
        other = IndexedSourceCard(
            SourceCardResult(
                self.source.path, "other", CARD_STATUS_NOT_APPLICABLE,
                reason="not suitable",
            ), HASH, 1, 1,
        )
        self.assertEqual(len(SourceCardIndex((self.record, other)).records),
                         2)

    def test_incremental_content_plan_and_model_independence(self):
        index = SourceCardIndex((self.record,))
        args = (
            index, {self.source.path: self.source},
            {self.source.path: HASH}, {self.source.path: "example"},
            (self.chunk,), self.registry,
        )
        plan = plan_source_card_updates(
            *args, generation_model_id="another-model",
        )
        self.assertEqual(plan.reused, (self.record,))
        self.assertEqual(plan.pending, ())
        modified = list(args)
        modified[2] = {self.source.path: NEW_HASH}
        self.assertEqual(
            plan_source_card_updates(*modified).pending, (self.key,)
        )
        modified = list(args)
        modified[4] = ()
        self.assertEqual(
            plan_source_card_updates(*modified).pending, (self.key,)
        )
        self.registry.get("example").validation_version = 2
        self.assertEqual(plan_source_card_updates(*args).pending,
                         (self.key,))

    def test_excluded_cached_and_failed_retried_on_model_change(self):
        path = self.source.path
        excluded = IndexedSourceCard(
            SourceCardResult(
                path, "example", CARD_STATUS_NOT_APPLICABLE,
                reason="not a scientific document",
            ), HASH, 1, 1, "model:a", "temp:0",
        )
        failed = replace(excluded, result=SourceCardResult(
            path, "example", CARD_STATUS_FAILED, reason="timeout"
        ))
        args = (
            {path: self.source}, {path: HASH}, {path: "example"},
            (self.chunk,), self.registry,
        )
        self.assertEqual(
            plan_source_card_updates(SourceCardIndex((excluded,)), *args,
                generation_model_id="model:b").reused, (excluded,)
        )
        self.assertEqual(
            plan_source_card_updates(SourceCardIndex((failed,)), *args,
                generation_model_id="model:b").pending, (self.key,)
        )
        self.assertEqual(
            plan_source_card_updates(SourceCardIndex((failed,)), *args,
                generation_model_id="model:a",
                generation_settings="temp:0").reused, (failed,)
        )
        self.assertEqual(
            plan_source_card_updates(SourceCardIndex((failed,)), *args,
                generation_model_id="model:a",
                generation_settings="temp:0", retry_failed=True
            ).pending, (self.key,)
        )

    def test_vector_model_and_text_invalidation(self):
        cards = SourceCardIndex((self.record,))
        texts = {self.key: ("search a",)}
        item = _vector(self.record)
        old = prepare_source_card_vectors(
            (item,), cards, embedding_model_identity="embedding:a",
        )
        plan = plan_source_card_vectors(
            cards, old, texts, embedding_model_identity="embedding:a"
        )
        self.assertEqual(plan.reused, (item,))
        self.assertEqual(plan.pending, ())
        self.assertEqual(plan_source_card_vectors(
            cards, old, texts, embedding_model_identity="embedding:b"
        ).pending, (self.key,))
        self.assertEqual(plan_source_card_vectors(
            cards, old, {self.key: ("changed",)},
            embedding_model_identity="embedding:a"
        ).pending, (self.key,))
        changed = replace(self.record.result.card, statements=(
            CardStatement("finding", "updated", ("c1",)),
        ))
        new = SourceCardIndex((replace(
            self.record, result=replace(self.record.result, card=changed)
        ),))
        self.assertEqual(plan_source_card_vectors(
            new, old, texts, embedding_model_identity="embedding:a"
        ).pending, (self.key,))
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_source_card_vectors(
                (item,), new, embedding_model_identity="embedding:a"
            )

    def test_vector_integrity_and_card_search(self):
        cards = SourceCardIndex((self.record,))
        good = _vector(self.record, values=((1.0, 0.0), (0.5, 0.5)))
        for bad in (((0.0, 0.0),), ((float("nan"), 0.0),),
                    ((1.0, 0.0), (1.0,)), ((True, 0.0),),
                    ((10 ** 1000, 0),)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SourceCardVectorIndex("embedding:a", (
                        replace(good, embeddings=bad),
                    ))
        index = prepare_source_card_vectors(
            (good,), cards, embedding_model_identity="embedding:a",
        )
        self.assertEqual(
            search_source_cards(index, (1.0, 0.0), count=1)[0].path,
            self.source.path,
        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            search_source_cards(index, (1.0,))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            search_source_cards(index, (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            search_source_cards(index, (float("inf"), 0.0))

    def test_only_missing_sources_are_pending(self):
        other, source_b, chunk_b = _fixture(
            "b.pdf", content="Another source"
        )
        selected = {
            self.source.path: "example", source_b.path: "example"
        }
        plan = plan_source_card_updates(
            SourceCardIndex((self.record,)),
            {self.source.path: self.source, source_b.path: source_b},
            {self.source.path: HASH, source_b.path: HASH},
            selected, (self.chunk, chunk_b), self.registry,
        )
        self.assertEqual(plan.reused, (self.record,))
        self.assertEqual(plan.pending, (other.key,))

    def test_missing_cards_and_zero_vector_index(self):
        self.assertEqual(plan_source_card_vectors(
            SourceCardIndex(()), None, {},
            embedding_model_identity="embedding:a"
        ).pending, ())
        self.assertEqual(search_source_cards(
            SourceCardVectorIndex("embedding:a", ()), (1.0, 0.0)
        ), ())
        with self.assertRaises(ValueError):
            plan_source_card_vectors(
                SourceCardIndex((self.record,)), None, {},
                embedding_model_identity="embedding:a"
            )


if __name__ == "__main__":
    unittest.main()
