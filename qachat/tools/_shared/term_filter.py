"""Find the passages that mention the exact terms of a question.

Some questions need the model's judgement but are about a named thing:
"does this paper use XYZ to analyse the data?" Only passages that mention
XYZ can say so, so tools that read every passage offer, as their quick
version, to read just those passages. Finding them takes no model call
beyond naming the terms, and when no passage mentions a term the answer
is immediate.

``term_hits`` names the terms with the router model (as exact-term
listing does), keeps only terms that occur in the question, and matches
them flexibly ("XYZ2", "XYZ 2", "xyz-2"). Terms that only name the
documents ("the paper", "this study") are dropped, since they say
nothing about the content.

Matching is case-insensitive and whole-word, so it finds spellings of the
term but not synonyms or paraphrases; the quick version is therefore a
narrower read, not the same answer faster. When the router fails or names
no term, there are no hits and the caller's quick version is a sample.

Key Features
------------

- **Exact terms**

    Take terms from the question only, with the same router prompt and
    checks as exact-term listing.

- **Passage hits**

    Return the indices of the passages that mention a term, in reading
    order.

- **Quick-option label**

    Name the quick version and say how many items it covers.

Module Structure
----------------

- ``term_hits``

    Return the question's exact terms and the passages that mention them.

- ``term_label``

    Name the quick version and say what it covers.

Examples
--------
>>> from qachat.tools._shared.term_filter import term_hits, term_label
>>> terms, hits = term_hits(request.question, context, passages)
>>> label, note = term_label(terms, hits)

``request`` and ``context`` are the ``ToolRequest`` and ``ToolContext`` a
tool's ``run`` receives; ``passages`` are the chunks in scope. With the
term "XYZ" found in six passages, ``note`` is "6 passages mention “XYZ”".

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from qachat.chunking import Chunk
from qachat.tools._contract import ToolContext
from qachat.tools._shared.terms import (
    term_pattern,
    terms_prompt,
    validated_terms,
)

__all__ = ["term_hits", "term_label"]

# Words that refer to the documents themselves, not to their content.
_DOCUMENT_WORDS: Final[frozenset[str]] = frozenset(
    "the a an this that these those paper papers study studies document "
    "documents file files article articles project source sources".split()
)
# Longest router reply naming the terms; the same budget as exact-term
# listing without categories.
_TERMS_REPLY_TOKENS: Final[int] = 96
# Name of the quick version that reads only the passages with a term.
_QUICK_LABEL: Final[str] = "Exact matches only"


def term_hits(
    question: str, context: ToolContext, passages: Sequence[Chunk]
) -> tuple[tuple[str, ...], list[int]]:
    """Return the question's exact terms and the passages that mention them.

    Parameters
    ----------
    question: str
        The question.
    context: ToolContext
        QAChat's services (``ask_router``).
    passages: Sequence[Chunk]
        The passages in scope, in reading order.

    Returns
    -------
    tuple[tuple[str, ...], list[int]]
        The terms (empty when the question names none) and the indices of
        the passages that mention any of them, in reading order.

    """

    system, prompt = terms_prompt(question)
    try:
        reply = context.ask_router(system, prompt, _TERMS_REPLY_TOKENS)
    except Exception:  # Without terms, the quick version is a sample.
        return (), []

    # A "term" that only names the documents ("the paper") finds nothing.
    terms = tuple(
        term for term in validated_terms(reply, question)
        if not set(term.casefold().split()) <= _DOCUMENT_WORDS
    )
    if not terms:
        return (), []

    patterns = [
        term_pattern(term, case_sensitive=False, whole_words=True)
        for term in terms
    ]
    hits = [
        index
        for index, chunk in enumerate(passages)
        if any(pattern.search(chunk.text) for pattern in patterns)
    ]
    return terms, hits


def term_label(
    terms: Sequence[str], hits: Sequence[int], unit: str = "passage"
) -> tuple[str, str]:
    """Name the quick version and say what it covers.

    Parameters
    ----------
    terms: Sequence[str]
        The exact terms.
    hits: Sequence[int]
        Indices of the items that contain a term.
    unit: str, optional
        What an item is. Defaults to ``"passage"``.

    Returns
    -------
    tuple[str, str]
        A short label, such as "Exact matches only", and a note, such as
        "6 passages mention “XYZ”".

    """

    quoted = " or ".join(f"“{term}”" for term in terms)
    if not hits:
        note = f"No {unit} mentions {quoted}"
    elif len(hits) == 1:
        note = f"1 {unit} mentions {quoted}"
    else:
        note = f"{len(hits)} {unit}s mention {quoted}"

    return _QUICK_LABEL, note
