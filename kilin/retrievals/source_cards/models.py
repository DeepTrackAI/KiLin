"""Represent source cards independently of any particular source kind.

A card records structured, source-level retrieval metadata. Statements cite
existing retrieval chunks, rather than storing page numbers or creating a
second source-extraction format. Metadata fields are kind-specific, while
validation of source identity and chunk references belongs to the core.

Cards are not answer evidence: later retrieval stages must return the original
chunks before making supported claims. This module performs no model calls,
file I/O, or indexing, and does not replace the active paper-card pipeline.

Key Features
------------

- **Immutable records**

    Represent source descriptors, cited statements, and generation metadata
    without mutable dictionary fields.

- **Separate outcomes**

    Distinguish valid cards, ineligible sources, and generation failures.

- **Provenance checks**

    Verify statement references against source-owned, existing chunks.

Module Structure
----------------

- ``SourceDescriptor``

    File identity and available chunk identifiers, without a content-kind
    classification.

- ``CardStatement``

    Kind-specific field text with original chunk identifiers.

- ``SourceCard``

    A source-level summary with generation and validation metadata.

- ``SourceCardResult`` and ``SourceCardStatus``

    Result of attempting a source card, independent of card content.

- ``validate_source_card``

    Check generic card structure, provenance, and source ownership.

Examples
--------
>>> from pathlib import Path
>>> from hashlib import sha256
>>> from kilin.chunking import Chunk
>>> from kilin.retrievals.source_cards import (
...     SourceDescriptor, CardStatement, SourceCard, validate_source_card,
... )
>>> path = Path("notes.txt")
>>> chunk = Chunk("c1", path, "A finding.", 1, 1, "text", None, None)
>>> source = SourceDescriptor(path, "Notes", ".txt", ("c1",))
>>> card = SourceCard(
...     path, "example", 1, 1, sha256(b"source").hexdigest(),
...     (), (CardStatement("finding", "A finding.", ("c1",)),),
... )
>>> validate_source_card(card, source, (chunk,))

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import re
from typing import Final, Literal, TypeAlias

from kilin.chunking import Chunk

__all__ = [
    "CARD_STATUS_FAILED",
    "CARD_STATUS_NOT_APPLICABLE",
    "CARD_STATUS_VALID",
    "CardStatement",
    "SourceCard",
    "SourceCardResult",
    "SourceCardStatus",
    "SourceDescriptor",
    "validate_source_card",
]

SourceCardStatus: TypeAlias = Literal[
    "valid", "not_applicable", "failed"
]
CARD_STATUS_VALID: Final[SourceCardStatus] = "valid"
CARD_STATUS_NOT_APPLICABLE: Final[SourceCardStatus] = "not_applicable"
CARD_STATUS_FAILED: Final[SourceCardStatus] = "failed"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    """Describe an extracted source without assigning its content kind.

    Parameters
    ----------
    path: pathlib.Path
        Source path relative to the repository's source folder.
    title: str
        Known title or deterministic file-name fallback.
    suffix: str
        Lowercase file suffix, such as ``".pdf"``; not a card kind.
    chunk_ids: tuple[str, ...]
        Available chunk identifiers in extraction order; may be empty.

    Attributes
    ----------
    path: pathlib.Path
        Source path relative to the source folder.
    title: str
        Source title, not necessarily a scientific-paper title.
    suffix: str
        File suffix, which does not determine card eligibility.
    chunk_ids: tuple[str, ...]
        Identifiers of extracted chunks, possibly empty.

    """

    path: Path
    title: str
    suffix: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CardStatement:
    """One substantive card statement with source-chunk provenance.

    Parameters
    ----------
    field: str
        Kind-owned field identifier, such as ``"methods"``.
    text: str
        Generated or extracted statement, without citation markup.
    chunk_ids: tuple[str, ...]
        Supporting original chunk identifiers; must be nonempty.

    Attributes
    ----------
    field: str
        Kind-owned field identifier.
    text: str
        Substantive statement text.
    chunk_ids: tuple[str, ...]
        Identifiers that must resolve to the card's source.

    """

    field: str
    text: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceCard:
    """Generic retrieval summary for exactly one source and card kind.

    Parameters
    ----------
    path: pathlib.Path
        Original source path relative to the source folder.
    kind: str
        Identifier of the registered source-card kind.
    schema_version: int
        Positive version of the kind's structured-content schema.
    validation_version: int
        Positive version of its eligibility/validation contract.
    source_sha256: str
        SHA-256 of the original source's bytes when the card was created.
    metadata: tuple[tuple[str, str], ...]
        Immutable kind-specific metadata, without provenance claims.
    statements: tuple[CardStatement, ...]
        Substantive statements tied to original source chunks.
    model_id: str | None, optional
        Generation model identity, or ``None`` for deterministic cards.
        Defaults to ``None``.
    generation_settings: str | None, optional
        Fingerprint of generation settings, if applicable. Defaults to
        ``None``. The full settings live in project configuration.

    Attributes
    ----------
    path: pathlib.Path
        Original source path.
    kind: str
        Registered kind identifier.
    schema_version: int
        Card-kind schema version.
    validation_version: int
        Card-kind validation version.
    source_sha256: str
        Source-content fingerprint.
    metadata: tuple[tuple[str, str], ...]
        Kind-specific metadata fields.
    statements: tuple[CardStatement, ...]
        Statements with chunk-level provenance.
    model_id: str | None
        Optional model identity.
    generation_settings: str | None
        Optional generation settings fingerprint.

    Notes
    -----
    A frozen record does not prove its content or citations. Call
    ``validate_source_card`` with the actual chunks, then the registered
    kind's validator before adding a card to an index.

    """

    path: Path
    kind: str
    schema_version: int
    validation_version: int
    source_sha256: str
    metadata: tuple[tuple[str, str], ...]
    statements: tuple[CardStatement, ...]
    model_id: str | None = None
    generation_settings: str | None = None


@dataclass(frozen=True, slots=True)
class SourceCardResult:
    """Outcome of attempting one source-card kind for one source.

    Parameters
    ----------
    path: pathlib.Path
        Source path for the attempt.
    kind: str
        Candidate card kind.
    status: SourceCardStatus
        ``valid``, ``not_applicable``, or ``failed``.
    card: SourceCard | None, optional
        Valid card only when status is ``valid``. Defaults to ``None``.
    reason: str | None, optional
        Explanation for an exclusion or failure; required for those states.
        Defaults to ``None``.

    Attributes
    ----------
    path: pathlib.Path
        Original source path.
    kind: str
        Candidate kind identifier.
    status: SourceCardStatus
        Outcome, independent of the card's substantive content.
    card: SourceCard | None
        Completed card for a valid outcome.
    reason: str | None
        Exclusion or failure explanation.

    """

    path: Path
    kind: str
    status: SourceCardStatus
    card: SourceCard | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent outcome records before they can be cached.

        Raises
        ------
        ValueError
            If status, card presence, identity, or reason is inconsistent.

        """

        _check_path(self.path)
        _check_identifier(self.kind, "card kind")
        if self.status == CARD_STATUS_VALID:
            if (
                not isinstance(self.card, SourceCard)
                or self.card.path != self.path
                or self.card.kind != self.kind
                or self.reason is not None
            ):
                raise ValueError("Valid results need a matching card.")
        elif self.status in (
            CARD_STATUS_FAILED, CARD_STATUS_NOT_APPLICABLE
        ):
            if (
                self.card is not None
                or not isinstance(self.reason, str)
                or not self.reason.strip()
            ):
                raise ValueError(
                    "Excluded or failed results need a reason, not a card."
                )
        else:
            raise ValueError("Unknown source-card result status.")


def validate_source_card(
    card: SourceCard,
    source: SourceDescriptor,
    chunks: Sequence[Chunk],
) -> None:
    """Validate generic card fields and exact source-chunk provenance.

    Parameters
    ----------
    card: SourceCard
        Card to validate; its kind-specific content is checked separately.
    source: SourceDescriptor
        Descriptor of the source being summarized.
    chunks: Sequence[Chunk]
        Chunks permitted to support statements. Pass only chunks supplied
        to a generation model to reject references it never saw.

    Raises
    ------
    ValueError
        If source identity, versions, metadata, or any provenance link is
        invalid, missing, duplicated, or refers to another source.

    Notes
    -----
    Checking that a chunk was available does not establish that it supports
    the claim. Each card kind must validate its own content and eligibility.

    """

    if not isinstance(card, SourceCard):
        raise ValueError("Expected a SourceCard.")
    if not isinstance(source, SourceDescriptor):
        raise ValueError("Expected a SourceDescriptor.")
    _check_path(card.path)
    _check_path(source.path)
    if card.path != source.path:
        raise ValueError("Card and descriptor refer to different sources.")
    _check_identifier(card.kind, "card kind")
    for name, version in (
        ("schema_version", card.schema_version),
        ("validation_version", card.validation_version),
    ):
        if type(version) is not int or version < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        not isinstance(card.source_sha256, str)
        or not _SHA256.fullmatch(card.source_sha256)
    ):
        raise ValueError("Source SHA-256 must be a lowercase hex digest.")
    for value in (card.model_id, card.generation_settings):
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError("Generation metadata must be nonempty text.")
    _check_descriptor(source)
    if type(card.metadata) is not tuple:
        raise ValueError("Card metadata must be immutable pairs.")
    metadata_keys: set[str] = set()
    for item in card.metadata:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Card metadata must contain key/value pairs.")
        key, value = item
        _check_identifier(key, "metadata key")
        if key in metadata_keys:
            raise ValueError("Card metadata contains duplicate keys.")
        if not isinstance(value, str):
            raise ValueError("Card metadata values must be text.")
        metadata_keys.add(key)
    if type(card.statements) is not tuple or not card.statements:
        raise ValueError("Cards require substantive statements.")
    available: dict[str, Chunk] = {}
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise ValueError("Provenance requires source chunks.")
        if (
            not isinstance(chunk.identifier, str)
            or not chunk.identifier.strip()
            or chunk.identifier in available
        ):
            raise ValueError("Provenance chunk identifiers must be unique.")
        available[chunk.identifier] = chunk
    permitted = set(source.chunk_ids)
    for statement in card.statements:
        if not isinstance(statement, CardStatement):
            raise ValueError("Expected a CardStatement.")
        _check_identifier(statement.field, "statement field")
        if not isinstance(statement.text, str) or not statement.text.strip():
            raise ValueError("Card statements require nonempty text.")
        if (
            type(statement.chunk_ids) is not tuple
            or not statement.chunk_ids
            or any(
                not isinstance(item, str) or not item.strip()
                for item in statement.chunk_ids
            )
            or len(set(statement.chunk_ids)) != len(statement.chunk_ids)
        ):
            raise ValueError("Statement chunk IDs must be unique and set.")
        for identifier in statement.chunk_ids:
            if identifier not in permitted or identifier not in available:
                raise ValueError("Statement cites an unavailable chunk.")
            if available[identifier].path != card.path:
                raise ValueError("Statement cites a different source.")


def _check_path(path: Path) -> None:
    """Reject source paths that cannot identify a project-owned file.

    Parameters
    ----------
    path: pathlib.Path
        Candidate repository-relative source path.

    Raises
    ------
    ValueError
        If the path is empty, absolute, traverses upward, or is Windows-
        rooted even when validated on another operating system.

    """

    if not isinstance(path, Path):
        raise ValueError("Source paths must be pathlib.Path objects.")
    spelling = str(path)
    windows = PureWindowsPath(spelling)
    if (
        spelling in ("", ".")
        or path.is_absolute()
        or ".." in path.parts
        or any("\\" in part for part in path.parts)
        or windows.drive
        or windows.root
    ):
        raise ValueError("Source path must be relative and contained.")


def _check_identifier(value: str, label: str) -> None:
    """Validate a registered kind or field identifier.

    Parameters
    ----------
    value: str
        Identifier to check.
    label: str
        Name used in an invalid-value diagnostic.

    Raises
    ------
    ValueError
        If the identifier is not a lowercase snake-case name.

    """

    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {label} identifier.")


def _check_descriptor(source: SourceDescriptor) -> None:
    """Validate source metadata without inferring its semantic kind.

    Parameters
    ----------
    source: SourceDescriptor
        Source description to check.

    Raises
    ------
    ValueError
        If the title, suffix, or chunk identifier list is malformed.

    """

    if not isinstance(source.title, str) or not source.title.strip():
        raise ValueError("Source titles must be nonempty strings.")
    if source.suffix != source.path.suffix.lower():
        raise ValueError("Source suffix must match its file name.")
    if type(source.chunk_ids) is not tuple:
        raise ValueError("Source chunk identifiers must be immutable.")
    if any(
        not isinstance(item, str) or not item.strip()
        for item in source.chunk_ids
    ) or len(set(source.chunk_ids)) != len(source.chunk_ids):
        raise ValueError("Source chunk identifiers must be unique strings.")
