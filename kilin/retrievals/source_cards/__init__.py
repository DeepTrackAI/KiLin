"""Expose generic source-card contracts and the trusted kind registry.

Source cards are optional whole-source retrieval descriptions whose
substantive statements refer to original chunks. This package introduces
only contracts: the active scientific-paper pipeline, persistence, model
calls, and answer retrieval remain unchanged until later PRs.

Key Features
------------

- **Card records**

    Keep source identity, kind-specific statements, and provenance separate
    from source extraction and the ordinary chunk indexes.

- **Trusted kinds**

    Register Python implementations explicitly instead of importing code
    from project-type configuration.

Module Structure
----------------

- ``SourceDescriptor``, ``CardStatement``, and ``SourceCard``

    Immutable source and card data structures.

- ``SourceCardResult`` and ``SourceCardStatus``

    Independent outcomes for successful, ineligible, and failed attempts.

- ``SourceCardKind``, ``SourceCardRegistry``, and ``SOURCE_CARD_KINDS``

    Registration and validation contract for trusted card kinds.

- ``describe_sources`` and ``validate_source_card``

    Build source identities and check card provenance without model calls.

Examples
--------
>>> from pathlib import Path
>>> from kilin.retrievals.source_cards import describe_sources
>>> describe_sources((Path("notes.md"),), ())[Path("notes.md")].suffix
'.md'

"""

from .descriptors import describe_sources
from .models import (
    CARD_STATUS_FAILED,
    CARD_STATUS_NOT_APPLICABLE,
    CARD_STATUS_VALID,
    CardStatement,
    SourceCard,
    SourceCardResult,
    SourceCardStatus,
    SourceDescriptor,
    validate_source_card,
)
from .registry import (
    SOURCE_CARD_KINDS,
    SourceCardKind,
    SourceCardRegistry,
)

__all__ = [
    "CARD_STATUS_FAILED",
    "CARD_STATUS_NOT_APPLICABLE",
    "CARD_STATUS_VALID",
    "CardStatement",
    "SOURCE_CARD_KINDS",
    "SourceCard",
    "SourceCardKind",
    "SourceCardRegistry",
    "SourceCardResult",
    "SourceCardStatus",
    "SourceDescriptor",
    "describe_sources",
    "validate_source_card",
]
