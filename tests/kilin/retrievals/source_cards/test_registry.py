"""Verify trusted source-card kind registration and validation behavior.

Small deterministic dummy kinds demonstrate an extensible contract without
introducing a scientific-paper generator in this PR.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from kilin.chunking import Chunk
from kilin.retrievals.source_cards import (
    SOURCE_CARD_KINDS,
    CardStatement,
    SourceCard,
    SourceCardRegistry,
    SourceDescriptor,
)


class DummyKind:

    identifier = "dummy"
    schema_version = 2
    validation_version = 3

    def precheck(self, source):
        return bool(source.chunk_ids)

    def validate(self, card):
        if card.statements[0].field != "purpose":
            raise ValueError("Expected a purpose field.")


def _example():
    """Construct a card, descriptor and an available original chunk."""
    path = Path("notes.txt")
    card = SourceCard(
        path, "dummy", 2, 3, "a" * 64, (),
        (CardStatement("purpose", "Records a purpose.", ("id",)),),
    )
    source = SourceDescriptor(path, "Notes", ".txt", ("id",))
    chunk = Chunk("id", path, "Records a purpose.", 1, 1,
                  "text_paragraph", None, None)
    return card, source, (chunk,)


class TestRegistry(unittest.TestCase):

    def test_empty_shared_registry_is_inert_in_this_pr(self):
        self.assertEqual(SOURCE_CARD_KINDS.identifiers(), ())

    def test_register_get_and_validate(self):
        registry = SourceCardRegistry((DummyKind(),))
        self.assertEqual(registry.identifiers(), ("dummy",))
        self.assertEqual(registry.get("dummy").schema_version, 2)
        card, source, chunks = _example()
        self.assertTrue(registry.precheck("dummy", source))
        registry.validate(card, source, chunks)

    def test_precheck_is_a_conservative_gate_not_a_classifier(self):
        registry = SourceCardRegistry((DummyKind(),))
        _, source, _ = _example()
        self.assertFalse(registry.precheck(
            "dummy", replace(source, chunk_ids=())
        ))

    def test_invalid_or_duplicate_registration(self):
        class Invalid:
            identifier = "illegal-name"
            schema_version = 1
            validation_version = 1
            precheck = lambda self, source: True
            validate = lambda self, card: None

        class InvalidVersion(DummyKind):
            schema_version = True

        class MissingMethod(DummyKind):
            validate = None

        for kind in (Invalid(), InvalidVersion(), MissingMethod()):
            with self.subTest(kind=type(kind).__name__):
                with self.assertRaises(ValueError):
                    SourceCardRegistry((kind,))
        registry = SourceCardRegistry((DummyKind(),))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            registry.register(DummyKind())

    def test_unknown_or_invalid_identifiers_cannot_load_modules(self):
        registry = SourceCardRegistry()
        with self.assertRaisesRegex(ValueError, "Unknown"):
            registry.get("not_registered")
        for value in ("arbitrary.module:Kind", "Invalid", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    registry.get(value)

    def test_kind_schema_and_validator_are_enforced(self):
        registry = SourceCardRegistry((DummyKind(),))
        card, source, chunks = _example()
        for changed in (
            replace(card, schema_version=1),
            replace(card, validation_version=2),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "version"):
                    registry.validate(changed, source, chunks)
        wrong_field = replace(card, statements=(
            CardStatement("finding", "Content.", ("id",)),
        ))
        with self.assertRaisesRegex(ValueError, "purpose"):
            registry.validate(wrong_field, source, chunks)

    def test_precheck_must_return_a_real_boolean(self):
        class BadResult(DummyKind):
            def precheck(self, source):
                return 1

        registry = SourceCardRegistry((BadResult(),))
        _, source, _ = _example()
        with self.assertRaisesRegex(ValueError, "boolean"):
            registry.precheck("dummy", source)

    def test_generic_provenance_validation_precedes_kind_validation(self):
        registry = SourceCardRegistry((DummyKind(),))
        card, source, _ = _example()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            registry.validate(card, source, ())


if __name__ == "__main__":
    unittest.main()
