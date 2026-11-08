"""Register trusted source-card kinds without project-defined code imports.

A card kind owns deterministic preliminary eligibility and content validation.
The registry validates the common card structure and its exact chunk-level
provenance before delegating to the kind. It does not generate cards or
construct retrieval indexes; those capabilities are introduced in later PRs.

Project types will select registered identifiers. They must not load arbitrary
Python objects from configuration strings. No concrete kind is activated by
this contract-only package; scientific-paper registration follows in PR6.

Key Features
------------

- **Explicit registration**

    Accept known Python implementations, not import paths in project JSON.

- **Version checks**

    Reject cards written to another schema or validation contract.

- **Separate validation**

    Check core provenance before kind-specific schema checks.

Module Structure
----------------

- ``SourceCardKind``

    Protocol for a registered card-kind implementation.

- ``SourceCardRegistry``

    Mutable registry with explicit duplicate and unknown-kind validation.

- ``SOURCE_CARD_KINDS``

    Shared registry initially empty until real kinds are registered.

Examples
--------
>>> from kilin.retrievals.source_cards import SourceCardRegistry
>>> registry = SourceCardRegistry()
>>> registry.identifiers()
()

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final, Protocol

from kilin.chunking import Chunk
from kilin.retrievals.source_cards.models import (
    SourceCard,
    SourceDescriptor,
    _check_descriptor,
    _check_identifier,
    _check_path,
    validate_source_card,
)

__all__ = [
    "SOURCE_CARD_KINDS",
    "SourceCardKind",
    "SourceCardRegistry",
]


class SourceCardKind(Protocol):
    """Contract implemented by trusted source-card kinds.

    Attributes
    ----------
    identifier: str
        Unique snake-case kind ID, selected from project configuration.
    schema_version: int
        Positive version of the kind-specific content schema.
    validation_version: int
        Positive version of eligibility and content checks.

    Notes
    -----
    ``precheck`` is conservative: ``False`` rules out the source without a
    model call; ``True`` only means generation may proceed. Full eligibility
    may be decided during a later single generation/verification call.

    """

    identifier: str
    schema_version: int
    validation_version: int

    def precheck(self, source: SourceDescriptor) -> bool:
        """Return whether a source is a possible candidate for this kind.

        Parameters
        ----------
        source: SourceDescriptor
            Existing source identity and extraction information.

        Returns
        -------
        bool
            ``False`` only for definitely ineligible sources; ``True`` is
            not a guarantee of eligibility.

        """

        ...

    def validate(self, card: SourceCard) -> None:
        """Validate fields and scientific or other kind-specific content.

        Parameters
        ----------
        card: SourceCard
            Already core-validated card of this kind.

        Raises
        ------
        ValueError
            If fields or metadata do not satisfy this kind's schema.

        """

        ...


class SourceCardRegistry:
    """Hold explicit source-card kinds without importing configured modules.

    Parameters
    ----------
    kinds: Iterable[SourceCardKind], optional
        Trusted implementations initially registered. Defaults to ``()``.

    Attributes
    ----------
    _kinds: dict[str, SourceCardKind]
        Registered kinds keyed by identifier; private to the registry.

    """

    def __init__(self, kinds: Iterable[SourceCardKind] = ()) -> None:
        """Register initial implementations using the normal checks.

        Parameters
        ----------
        kinds: Iterable[SourceCardKind], optional
            Initial trusted implementations. Defaults to ``()``.

        """

        self._kinds: dict[str, SourceCardKind] = {}
        for kind in kinds:
            self.register(kind)

    def register(self, kind: SourceCardKind) -> None:
        """Add one trusted implementation with a unique kind identifier.

        Parameters
        ----------
        kind: SourceCardKind
            Object providing the agreed kind contract.

        Raises
        ------
        ValueError
            If the kind is malformed or its identifier already exists.

        """

        identifier = getattr(kind, "identifier", None)
        _check_identifier(identifier, "card kind")
        for name in ("schema_version", "validation_version"):
            version = getattr(kind, name, None)
            if type(version) is not int or version < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not callable(getattr(kind, "precheck", None)) or not callable(
            getattr(kind, "validate", None)
        ):
            raise ValueError("Card kinds need precheck and validate methods.")
        if identifier in self._kinds:
            raise ValueError(f"Duplicate source-card kind: {identifier}.")
        self._kinds[identifier] = kind

    def get(self, identifier: str) -> SourceCardKind:
        """Find one registered kind without importing project-defined code.

        Parameters
        ----------
        identifier: str
            Configured source-card kind ID.

        Returns
        -------
        SourceCardKind
            Registered trusted implementation.

        Raises
        ------
        ValueError
            If no registered implementation has that identifier.

        """

        _check_identifier(identifier, "card kind")
        if identifier not in self._kinds:
            raise ValueError(f"Unknown source-card kind: {identifier}.")
        return self._kinds[identifier]

    def identifiers(self) -> tuple[str, ...]:
        """List registered source-card kinds in insertion order.

        Returns
        -------
        tuple[str, ...]
            Kind IDs available to trusted callers or project-type validation.

        """

        return tuple(self._kinds)

    def precheck(self, identifier: str, source: SourceDescriptor) -> bool:
        """Evaluate a kind's conservative deterministic eligibility gate.

        Parameters
        ----------
        identifier: str
            Registered kind ID.
        source: SourceDescriptor
            Source to screen without additional model inference.

        Returns
        -------
        bool
            True if the source may qualify; False if definitely excluded.

        Raises
        ------
        ValueError
            If the kind or descriptor is invalid or gate result is not bool.

        """

        if not isinstance(source, SourceDescriptor):
            raise ValueError("Expected a SourceDescriptor.")
        _check_path(source.path)
        _check_descriptor(source)
        allowed = self.get(identifier).precheck(source)
        if type(allowed) is not bool:
            raise ValueError("Card-kind precheck must return a boolean.")
        return allowed

    def validate(
        self,
        card: SourceCard,
        source: SourceDescriptor,
        chunks: Sequence[Chunk],
    ) -> None:
        """Validate card provenance, active schema version, and kind fields.

        Parameters
        ----------
        card: SourceCard
            Generic source-card record to validate.
        source: SourceDescriptor
            Source to which this card must belong.
        chunks: Sequence[Chunk]
            Existing source chunks permitted as provenance.

        Raises
        ------
        ValueError
            If the card, registry version, or kind-specific fields fail.

        """

        validate_source_card(card, source, chunks)
        kind = self.get(card.kind)
        if (
            card.schema_version != kind.schema_version
            or card.validation_version != kind.validation_version
        ):
            raise ValueError("Card kind schema or validation version changed.")
        kind.validate(card)


SOURCE_CARD_KINDS: Final[SourceCardRegistry] = SourceCardRegistry()
