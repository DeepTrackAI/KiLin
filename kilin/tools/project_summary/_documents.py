"""Choose a project's documents and the excerpt read from each.

Private to ``project_summary``. A project report needs every document to
contribute, so it does not search: it reads a short excerpt of each one.
An excerpt holds what the document's source card says (title, aim,
methods, findings), its opening passages (title, abstract), and the
passages that best match what the template asks for, within a token
budget.

Key Features
------------

- **Every document**

    Documents are the project's indexed sources, numbered in path order;
    the number is the report's citation of the document.

- **Focused excerpts**

    Passages are chosen by the words of the template's sections and
    per-document conditions, so a "Datasets" section reads the passages
    about data.

Module Structure
----------------

- ``ProjectDocument``

    One document: its number, path, title, passages and card.

- ``project_documents``

    The project's documents, optionally limited to some paths.

- ``document_excerpt``

    The text read from one document.

Examples
--------
>>> from kilin.tools.project_summary._documents import query_words
>>> sorted(query_words(["Datasets used", "the public datasets"]))
['datasets', 'public', 'used']

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from kilin.chunking import Chunk
from kilin.retrievals.document_reading import estimate_tokens

__all__ = [
    "ProjectDocument",
    "document_excerpt",
    "project_documents",
    "query_words",
]

# Card fields read into an excerpt, in this order, when the card has them.
_CARD_FIELDS: Final[tuple[str, ...]] = (
    "title", "venue", "year", "aim", "methods", "findings", "summary",
)
# Opening passages always read (title, authors, abstract).
_OPENING_PASSAGES: Final[int] = 2
_WORD = re.compile(r"[^\W\d_]{4,}")
# Words of a template's instructions that say nothing about a topic.
_COMMON: Final[frozenset[str]] = frozenset({
    "this", "that", "with", "from", "their", "they", "what", "which",
    "when", "where", "each", "every", "document", "documents", "paper",
    "papers", "write", "name", "about", "into", "than", "them", "there",
    "these", "those", "have", "does", "only", "also", "more", "most",
})


@dataclass(frozen=True, slots=True)
class ProjectDocument:
    """One document of the project, as a project report reads it.

    Attributes
    ----------
    number: int
        Its number in the report, from 1, as cited (``[3]``).
    path: pathlib.Path
        Its source path, as indexed.
    title: str
        The card's title, or the file name.
    passages: tuple[Chunk, ...]
        Its indexed passages, in order.
    card: Mapping[str, str] | None
        Its source card's fields, or None without a card.

    """

    number: int
    path: Path
    title: str
    passages: tuple[Chunk, ...]
    card: Mapping[str, str] | None = None


def project_documents(
    chunks: Sequence[Chunk],
    card: Callable[[Path], Mapping[str, str] | None] | None,
    paths: Sequence[Path] | None = None,
) -> list[ProjectDocument]:
    """Return the project's documents, numbered in path order.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The project's indexed passages.
    card: Callable[[pathlib.Path], Mapping[str, str] | None] | None
        ``ToolContext.card``, or None when cards cannot be read.
    paths: Sequence[pathlib.Path] | None, optional
        Only these documents, when given. Defaults to all.

    Returns
    -------
    list[ProjectDocument]
        One per source path with passages, sorted by path, with its
        passages in their order and its card, if any.

    """

    by_path: dict[Path, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_path[chunk.path].append(chunk)
    wanted = set(paths) if paths is not None else None
    documents: list[ProjectDocument] = []
    for path in sorted(by_path, key=lambda item: item.as_posix().casefold()):
        if wanted is not None and path not in wanted:
            continue
        fields = card(path) if card is not None else None
        title = (fields or {}).get("title", "").strip() or path.name
        passages = tuple(sorted(
            by_path[path], key=lambda chunk: (chunk.start_line, chunk.end_line)
        ))
        documents.append(ProjectDocument(
            len(documents) + 1, path, title, passages, fields or None,
        ))
    return documents


def query_words(texts: Sequence[str]) -> set[str]:
    """Return the words a template asks about, for choosing passages.

    Parameters
    ----------
    texts: Sequence[str]
        Section names, instructions and per-document conditions.

    Returns
    -------
    set[str]
        Their words of four or more letters, in lower case, without words
        that name no topic ("document", "write").

    """

    return {
        word.casefold()
        for text in texts
        for word in _WORD.findall(text)
        if word.casefold() not in _COMMON
    }


def document_excerpt(
    document: ProjectDocument, words: set[str], tokens: int
) -> str:
    """Return the text read from one document.

    Parameters
    ----------
    document: ProjectDocument
        The document.
    words: set[str]
        What the template asks about (``query_words``).
    tokens: int
        Most tokens of the excerpt.

    Returns
    -------
    str
        The card's fields ("Aim: ..."), then the opening passages, then
        the passages sharing most of ``words``, in document order, within
        ``tokens``; passages are separated by blank lines. The first
        passage is always read; another that does not fit is left out, not
        cut.

    """

    parts: list[str] = []
    used = 0
    if document.card:
        lines = [
            f"{field.capitalize()}: {document.card[field].strip()}"
            for field in _CARD_FIELDS
            if document.card.get(field, "").strip()
        ]
        if lines:
            card_text = "From the document's card:\n" + "\n".join(lines)
            parts.append(card_text)
            used += estimate_tokens(card_text)

    def score(chunk: Chunk) -> int:
        """Return how many of the asked-about words a passage holds.

        Parameters
        ----------
        chunk: Chunk
            A passage.

        Returns
        -------
        int
            The number of distinct ``words`` in it.

        """

        return len(words & {w.casefold() for w in _WORD.findall(chunk.text)})

    opening = list(document.passages[:_OPENING_PASSAGES])
    rest = sorted(
        document.passages[_OPENING_PASSAGES:], key=score, reverse=True
    )
    chosen: list[Chunk] = []
    for chunk in [*opening, *(c for c in rest if score(c))]:
        cost = estimate_tokens(chunk.text)
        if used + cost > tokens and chosen:
            continue  # The first passage is always read.
        chosen.append(chunk)
        used += cost
    order = {id(chunk): index for index, chunk in
             enumerate(document.passages)}
    chosen.sort(key=lambda chunk: order[id(chunk)])
    parts.extend(chunk.text.strip() for chunk in chosen)
    return "\n\n".join(part for part in parts if part)
