"""Expose generic source-card contracts and the trusted kind registry.

Source cards are optional whole-source retrieval descriptions whose
substantive statements refer to original chunks. This package introduces
generic contracts and independently persisted card/vector indexes. The active
scientific-paper generation and answer retrieval paths remain unchanged.

Key Features
------------

- **Card records**

    Keep source identity, kind-specific statements, and provenance separate
    from source extraction and the ordinary chunk indexes.

- **Independent card-generation session**

    Expose an optional lazy model adapter configured by KíLín core settings.

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

- ``SourceCardGenerator``

    Load and release a dedicated model for a card-generation batch.

- ``SourceCardIndex`` and ``SourceCardVectorIndex``

    Independently maintained card outcomes and prepared search vectors.

- ``save_source_cards`` and ``save_source_card_vectors``

    Atomically persist content and embeddings in different files.

Examples
--------
>>> from pathlib import Path
>>> from kilin.retrievals.source_cards import describe_sources
>>> describe_sources((Path("notes.md"),), ())[Path("notes.md")].suffix
'.md'

"""

from .descriptors import describe_sources
from .generation import SourceCardGenerator
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
from .index import (
    CardKey,
    IndexedSourceCard,
    SourceCardIndex,
    SourceCardMatch,
    SourceCardUpdatePlan,
    SourceCardVector,
    SourceCardVectorIndex,
    SourceCardVectorPlan,
    card_digest,
    plan_source_card_updates,
    plan_source_card_vectors,
    prepare_source_card_index,
    prepare_source_card_vectors,
    search_source_cards,
    search_text_digest,
)
from .persistence import (
    load_source_card_vectors,
    load_source_cards,
    save_source_card_vectors,
    save_source_cards,
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
    "CardKey",
    "CardStatement",
    "IndexedSourceCard",
    "SOURCE_CARD_KINDS",
    "SourceCard",
    "SourceCardIndex",
    "SourceCardGenerator",
    "SourceCardMatch",
    "SourceCardUpdatePlan",
    "SourceCardVector",
    "SourceCardVectorIndex",
    "SourceCardVectorPlan",
    "SourceCardKind",
    "SourceCardRegistry",
    "SourceCardResult",
    "SourceCardStatus",
    "SourceDescriptor",
    "card_digest",
    "describe_sources",
    "load_source_card_vectors",
    "load_source_cards",
    "plan_source_card_updates",
    "plan_source_card_vectors",
    "prepare_source_card_index",
    "prepare_source_card_vectors",
    "save_source_card_vectors",
    "save_source_cards",
    "search_source_cards",
    "search_text_digest",
    "validate_source_card",
]
