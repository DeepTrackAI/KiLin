"""Verify generic source-card records and exact chunk provenance.

Tests use synthetic chunks and no file extraction or language model.

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
    SourceCardResult,
    SourceDescriptor,
    validate_source_card,
)


HASH = sha256(b"Original source bytes").hexdigest()


def _chunk(identifier="c1", path="notes.txt"):
    """Create a provenance-bearing source chunk."""
    return Chunk(identifier, Path(path), "Original evidence.", 1, 2,
                 "text_paragraph", None, None)


def _descriptor(path="notes.txt", identifiers=("c1",)):
    """Describe an existing source and its available chunk identifiers."""
    source = Path(path)
    return SourceDescriptor(source, source.stem, source.suffix, identifiers)


def _card(path="notes.txt", *, statements=None):
    """Create a generic, valid-by-construction card for one source."""
    if statements is None:
        statements = (
            CardStatement("finding", "Original evidence.", ("c1",)),
        )
    return SourceCard(
        Path(path), "example", 1, 1, HASH,
        (("title", "Notes"),), statements,
        model_id="model:1", generation_settings="settings:1",
    )


class TestGenericSourceCards(unittest.TestCase):

    def test_provenance_accepts_original_chunk(self):
        card = _card()
        validate_source_card(card, _descriptor(), (_chunk(),))
        self.assertEqual(card.statements[0].chunk_ids, ("c1",))
        self.assertEqual(card.metadata, (("title", "Notes"),))

    def test_deterministic_card_can_have_no_model_identity(self):
        card = replace(_card(), model_id=None,
                       generation_settings=None)
        validate_source_card(card, _descriptor(), (_chunk(),))

    def test_source_ownership_rejects_cross_source_citation(self):
        with self.assertRaisesRegex(ValueError, "different source"):
            validate_source_card(
                _card(), _descriptor(), (_chunk(path="other.txt"),)
            )

    def test_missing_or_unshown_chunk_reference_is_rejected(self):
        card = _card(statements=(
            CardStatement("finding", "A claim.", ("c2",)),
        ))
        for shown in ((), (_chunk(),)):
            with self.subTest(shown=shown):
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    validate_source_card(
                        card, _descriptor(identifiers=("c1", "c2")), shown
                    )

    def test_references_must_be_in_source_descriptor(self):
        card = _card()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_source_card(card, _descriptor(identifiers=()),
                                 (_chunk(),))

    def test_descriptor_and_card_source_must_match(self):
        with self.assertRaisesRegex(ValueError, "different sources"):
            validate_source_card(_card(), _descriptor("other.txt"),
                                 (_chunk(),))

    def test_invalid_paths_are_rejected(self):
        invalid = [
            Path("."), Path("../outside.txt"), Path("a/../outside.txt"),
            Path("/absolute.txt"), Path("C:/outside.txt"),
        ]
        # WindowsPath has already normalized a backslash separator, so
        # that spelling cannot be tested as an invalid Path on Windows.
        backslash = Path(r"a\bad.txt")
        if "\\" in str(backslash):
            invalid.append(backslash)
        for value in invalid:
            with self.subTest(path=value):
                with self.assertRaisesRegex(ValueError, "relative"):
                    validate_source_card(
                        replace(_card(), path=value),
                        _descriptor(), (_chunk(),)
                    )

    def test_valid_nested_path_is_supported(self):
        card = _card("folder/notes.txt")
        validate_source_card(
            card, _descriptor("folder/notes.txt"),
            (_chunk(path="folder/notes.txt"),)
        )

    def test_versions_and_source_fingerprint_are_validated(self):
        for change in (
            {"schema_version": 0}, {"validation_version": True},
            {"schema_version": 1.5}, {"source_sha256": "old"},
            {"source_sha256": HASH.upper()},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    validate_source_card(
                        replace(_card(), **change),
                        _descriptor(), (_chunk(),)
                    )

    def test_metadata_pairs_are_unique_and_immutable(self):
        for metadata in (
            (("title", "A"), ("title", "B")),
            (("Invalid Key", "bad"),),
            (("title", 123),),
            ("title",),
            [("title", "bad")],
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    validate_source_card(
                        replace(_card(), metadata=metadata),
                        _descriptor(), (_chunk(),)
                    )

    def test_statement_structure_is_checked(self):
        for statements in (
            (), (CardStatement("finding", "  ", ("c1",)),),
            (CardStatement("Bad Field", "Text.", ("c1",)),),
            (CardStatement("finding", "Text.", ()),),
            (CardStatement("finding", "Text.", ("c1", "c1")),),
        ):
            with self.subTest(statements=statements):
                with self.assertRaises(ValueError):
                    validate_source_card(
                        _card(statements=statements),
                        _descriptor(), (_chunk(),)
                    )

    def test_duplicate_provenance_ids_are_not_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_source_card(_card(), _descriptor(),
                                 (_chunk(), _chunk()))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_source_card(
                _card(), _descriptor(identifiers=("c1", "c1")),
                (_chunk(),)
            )

    def test_invalid_generation_metadata_is_rejected(self):
        for metadata in (
            {"model_id": ""},
            {"generation_settings": 4},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    validate_source_card(
                        replace(_card(), **metadata),
                        _descriptor(), (_chunk(),)
                    )


class TestSourceCardResult(unittest.TestCase):

    def test_three_distinct_results(self):
        card = _card()
        valid = SourceCardResult(card.path, card.kind, CARD_STATUS_VALID,
                                 card=card)
        excluded = SourceCardResult(card.path, card.kind,
                                    CARD_STATUS_NOT_APPLICABLE,
                                    reason="Not a supported research paper")
        failed = SourceCardResult(card.path, card.kind,
                                  CARD_STATUS_FAILED,
                                  reason="Generation unavailable")
        self.assertIs(valid.card, card)
        self.assertIsNone(excluded.card)
        self.assertIsNone(failed.card)

    def test_invalid_outcomes_are_rejected(self):
        card = _card()
        cases = (
            (CARD_STATUS_VALID, None, None),
            (CARD_STATUS_VALID, card, "invalid reason"),
            (CARD_STATUS_FAILED, card, "failure"),
            (CARD_STATUS_NOT_APPLICABLE, None, " "),
            ("unknown", None, "unsupported"),
        )
        for status, value, reason in cases:
            with self.subTest(status=status, reason=reason):
                with self.assertRaises(ValueError):
                    SourceCardResult(card.path, card.kind, status,
                                     value, reason)

    def test_result_card_must_match_kind_and_source(self):
        for path, kind in ((Path("other.txt"), "example"),
                           (Path("notes.txt"), "different")):
            with self.subTest(path=path, kind=kind):
                with self.assertRaises(ValueError):
                    SourceCardResult(path, kind, CARD_STATUS_VALID,
                                     card=_card())


if __name__ == "__main__":
    unittest.main()
