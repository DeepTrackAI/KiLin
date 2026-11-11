"""Expose trusted, opt-in kinds of provenance-linked source cards.

The initial built-in kind summarizes scientific papers. Future kinds may
represent patents, proposals, manuals, policy documents or other sources,
with different generation instructions and schemas but the same core index.

Key Features
------------

- **Explicit paper support**

    The scientific-paper kind is registered in the trusted core registry.

- **Kind-owned generation**

    Select excerpts, build prompts and validate cards outside of the index.

Module Structure
----------------

- ``ScientificPaperKind``

    First registered source-card kind, with paper-specific validation.

- ``generate_card``

    Single-call scientific-paper eligibility and summary generation.

Examples
--------
>>> from kilin.retrievals.source_cards.kinds import ScientificPaperKind
>>> ScientificPaperKind.identifier
'scientific_paper'

"""

from .scientific_paper import (
    ScientificPaperKind,
    card_passages,
    card_prompt,
    generate_card,
    parse_card,
)

__all__ = [
    "ScientificPaperKind",
    "card_passages",
    "card_prompt",
    "generate_card",
    "parse_card",
]
