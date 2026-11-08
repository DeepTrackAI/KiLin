"""Build, store, and describe paper cards.

A paper card is a short record of one document in a literature project:
its title, authors, year, and abstract, and one or two sentences each on
its aim, methods, data, main findings, and limitations, with the pages
those sentences come from. Cards let retrieval choose papers before it
chooses passages: a question about sleep and aging first finds the papers
about that topic, then the passages inside them.

A card is written once per document by the answering model, without
thinking, from the document's opening passages and the passages that best
match each field. It is saved in the project's index folder with the
SHA-256 of the file's bytes, so it is written again only when the file
changes or the card format does. Cards guide retrieval only: answers cite
the passages, never the card.

Key Features
------------

- **Focused reading**

    Pick a document's opening passages and its passages about methods,
    data, findings, and limitations within a token budget.

- **Checked fields**

    Keep only citations of passages the model was shown, and record the
    pages each field comes from.

- **Saved cards**

    Save cards with their search vectors and reuse them while the file,
    the card format, and the embedding model stay the same.

Module Structure
----------------

- ``PAPER_CARD_FORMAT``

    Version of the card prompt and layout, saved with every card.

- ``CARD_FIELDS``

    Summary fields of a card, in order.

- ``CARD_INPUT_TOKENS``

    Estimated tokens of passages one card is written from.

- ``PaperCard``

    One document's card.

- ``card_passages``

    Choose the passages a card is written from.

- ``card_prompt``

    Prompts that ask the model for a card.

- ``parse_card``

    Turn the model's reply into a checked card.

- ``card_texts``

    Texts of a card that are embedded for paper search.

- ``load_cards`` and ``save_cards``

    Read and write the saved cards of a project.

Examples
--------
>>> from pathlib import Path
>>> from kilin.chunking import Chunk
>>> from kilin.retrievals.paper_cards import parse_card
>>> passage = Chunk("c1", Path("a.pdf"), "We used MRI.", 1, 1,
...                 "pdf_page", None, "Page 4")
>>> card = parse_card(
...     '{"title": "A study", "methods": "MRI scans [1]."}',
...     [passage], path=Path("a.pdf"), sha256="abc", title=None,
...     model="m",
... )
>>> card.field("methods"), card.field_pages("methods")
('MRI scans.', (4,))

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.chunking import Chunk
from kilin.retrievals.document_reading import estimate_tokens
from kilin.retrievals.strategies.retrieval_semantic import (
    _validate_semantic_embeddings,
)
from kilin.retrievals.strategies.retrieval_lexical import (
    build_lexical_index,
    lexical_search,
)

__all__ = [
    "CARD_FIELDS",
    "CARD_INPUT_TOKENS",
    "PAPER_CARD_FORMAT",
    "PaperCard",
    "card_passages",
    "card_prompt",
    "card_texts",
    "load_cards",
    "parse_card",
    "save_cards",
]

# Saved with every card; a change writes every card again.
PAPER_CARD_FORMAT: Final[str] = "paper-card-v2"
CARD_FIELDS: Final[tuple[str, ...]] = (
    "aim",
    "methods",
    "data",
    "findings",
    "limitations",
)
# What each field asks, for the prompt.
_FIELD_QUESTIONS: Final[dict[str, str]] = {
    "aim": "the research question or aim",
    "methods": "the approach and methods",
    "data": (
        "the data, samples, participants, datasets, software, or tools, "
        "by name"
    ),
    "findings": "the main findings or results",
    "limitations": "the limitations or future work the authors state",
}
# Words that find each field's passages inside one document.
_FIELD_QUERIES: Final[dict[str, str]] = {
    "methods": (
        "methods method approach procedure analysis model algorithm "
        "pipeline we used performed implemented"
    ),
    "data": (
        "data dataset participants subjects samples cohort software code "
        "available repository collected recruited"
    ),
    "findings": (
        "results findings show showed found demonstrate significant "
        "performance improved"
    ),
    "limitations": (
        "limitations limitation future work however caveat cannot "
        "restricted"
    ),
}
# Opening passages always read: title, authors, abstract.
_OPENING_PASSAGES: Final[int] = 3
# Best-matching passages read for each field.
_PASSAGES_PER_FIELD: Final[int] = 2
# Estimated tokens of passages one card is written from.
CARD_INPUT_TOKENS: Final[int] = 5_000
# Longest text kept for one field or the abstract.
_FIELD_MAX_CHARACTERS: Final[int] = 700
_ABSTRACT_MAX_CHARACTERS: Final[int] = 1_500
_CITATION: Final[re.Pattern[str]] = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
_PAGE_SECTION: Final[re.Pattern[str]] = re.compile(r"^Page (\d+)$")


@dataclass(frozen=True, slots=True)
class PaperCard:
    """One document's card.

    Parameters
    ----------
    path: pathlib.Path
        Document path relative to the project source folder.
    sha256: str
        SHA-256 of the document's bytes when the card was written.
    title: str
        Document title.
    authors: str
        Authors as the model read them; empty when unknown.
    year: str
        Publication year as the model read it; empty when unknown.
    abstract: str
        Short abstract; empty when unknown.
    fields: tuple[tuple[str, str], ...]
        Text of each summary field (``CARD_FIELDS``), without citations.
    pages: tuple[tuple[str, tuple[int, ...]], ...]
        Pages each field and the abstract come from.
    model: str
        Model that wrote the card.
    complete: bool
        Whether the model wrote the card; ``False`` for a card with the
        title only, written again on the next indexing.

    Attributes
    ----------
    path: pathlib.Path
        Document path relative to the project source folder.
    sha256: str
        SHA-256 of the document's bytes when the card was written.
    title: str
        Document title.
    authors: str
        Authors as the model read them; empty when unknown.
    year: str
        Publication year as the model read it; empty when unknown.
    abstract: str
        Short abstract; empty when unknown.
    fields: tuple[tuple[str, str], ...]
        Text of each summary field (``CARD_FIELDS``), without citations.
    pages: tuple[tuple[str, tuple[int, ...]], ...]
        Pages each field and the abstract come from.
    model: str
        Model that wrote the card.
    complete: bool
        Whether the model wrote the card.

    """

    path: Path
    sha256: str
    title: str
    authors: str
    year: str
    abstract: str
    fields: tuple[tuple[str, str], ...]
    pages: tuple[tuple[str, tuple[int, ...]], ...]
    model: str
    complete: bool

    def field(self, name: str) -> str:
        """Return the text of one summary field.

        Parameters
        ----------
        name: str
            Field name from ``CARD_FIELDS``, or ``"abstract"``.

        Returns
        -------
        str
            The field's text; empty when the card has none.

        """

        if name == "abstract":
            return self.abstract
        return dict(self.fields).get(name, "")

    def field_pages(self, name: str) -> tuple[int, ...]:
        """Return the pages one field comes from.

        Parameters
        ----------
        name: str
            Field name from ``CARD_FIELDS``, or ``"abstract"``.

        Returns
        -------
        tuple[int, ...]
            Page numbers in increasing order; empty when unknown.

        """

        return dict(self.pages).get(name, ())


def card_passages(
    chunks: Sequence[Chunk], budget: int = CARD_INPUT_TOKENS
) -> list[Chunk]:
    """Choose the passages a card is written from.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Passages of one document.
    budget: int, optional
        Estimated tokens to read. Defaults to ``CARD_INPUT_TOKENS``.

    Returns
    -------
    list[Chunk]
        The opening passages, then the passages that best match each
        field's words (lexical search inside the document), while they fit
        the budget, in reading order. The first passage is always kept.

    """

    ordered = sorted(chunks, key=_reading_position)
    if not ordered:
        return []
    wanted = list(range(min(_OPENING_PASSAGES, len(ordered))))
    index = build_lexical_index(ordered)
    position = {chunk.identifier: number for number, chunk in
                enumerate(ordered)}
    for query in _FIELD_QUERIES.values():
        ranked = lexical_search(index, query, count=len(ordered))
        wanted.extend(
            position[result.chunk.identifier]
            for result in ranked[:_PASSAGES_PER_FIELD]
        )
    picked: list[int] = []
    used = 0
    for number in dict.fromkeys(wanted):
        size = estimate_tokens(ordered[number].text)
        if picked and used + size > budget:
            continue
        picked.append(number)
        used += size
    return [ordered[number] for number in sorted(picked)]


def card_prompt(name: str, passages: Sequence[Chunk]) -> tuple[str, str]:
    """Return the prompts that ask the model for a card.

    Parameters
    ----------
    name: str
        The document's file name.
    passages: Sequence[Chunk]
        Passages to write the card from, numbered from one.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt.

    """

    system = (
        "You write a factual index card for one scientific document from "
        "passages of it. Use only the passages. Never guess: leave a field "
        "empty when the passages do not state it. Reply with one JSON "
        "object and nothing else."
    )
    listed = "\n\n".join(
        f"[{number}] ({_where(chunk)}) {chunk.text.strip()}"
        for number, chunk in enumerate(passages, start=1)
    )
    fields = "\n".join(
        f'- "{field}": one or two sentences on {question}, citing passages '
        "like [3]"
        for field, question in _FIELD_QUESTIONS.items()
    )
    user = f"""Passages of {name}:

{listed}

Write a JSON object with these keys:
- "title": the document's title
- "authors": the authors, as written
- "year": the publication year
- "abstract": the abstract in at most three sentences, citing passages
{fields}

Every value is a string; use "" when the passages do not say."""
    return system, user


def parse_card(
    reply: str,
    passages: Sequence[Chunk],
    *,
    path: Path,
    sha256: str,
    title: str | None,
    model: str,
) -> PaperCard:
    """Turn the model's reply into a checked card.

    Parameters
    ----------
    reply: str
        The model's reply, expected to hold one JSON object.
    passages: Sequence[Chunk]
        The passages the card was written from, numbered from one.
    path: pathlib.Path
        Document path.
    sha256: str
        SHA-256 of the document's bytes.
    title: str | None
        Title known without the model (``DocumentInfo``); it takes
        precedence over the model's. ``None`` uses the model's title, then
        the file name.
    model: str
        Model that wrote the reply.

    Returns
    -------
    PaperCard
        A complete card when a substantive field cites at least one
        supplied passage; otherwise a title-only incomplete card.
        Unsupported summary fields are dropped. Valid citations supply
        page metadata when the source has page numbers.

    """

    data = _json_object(reply)
    if data is None:
        return _title_card(path, sha256, title, model)

    def text(
        key: str, limit: int, *, cited: bool = False
    ) -> tuple[str, tuple[int, ...]]:
        """Return one reply value without citations, and its cited pages.

        Parameters
        ----------
        key: str
            Key of the reply's JSON object.
        limit: int
            Most characters kept.
        cited: bool, optional
            Require a citation to a supplied passage. Defaults to False.

        Returns
        -------
        tuple[str, tuple[int, ...]]
            The validated value and its cited pages, if available.
            Unsupported substantive fields return empty text.

        """

        value = data.get(key, "")
        if not isinstance(value, str):
            return "", ()
        pages: set[int] = set()
        supported = False
        for match in _CITATION.finditer(value):
            for number in match.group(1).split(","):
                normalized = number.strip().lstrip("0") or "0"
                if len(normalized) > 15:
                    continue
                index = int(normalized) - 1
                if 0 <= index < len(passages):
                    supported = True
                    page = _page(passages[index])
                    if page is not None:
                        pages.add(page)
        clean = " ".join(_CITATION.sub("", value).split())[:limit]
        if cited and not supported:
            return "", ()
        return clean, tuple(sorted(pages))

    abstract, abstract_pages = text(
        "abstract", _ABSTRACT_MAX_CHARACTERS, cited=True
    )
    fields = []
    pages = [("abstract", abstract_pages)] if abstract_pages else []
    for field in CARD_FIELDS:
        value, cited_pages = text(
            field, _FIELD_MAX_CHARACTERS, cited=True
        )
        if value:
            fields.append((field, value))
        if cited_pages:
            pages.append((field, cited_pages))
    if not abstract and not fields:
        return _title_card(path, sha256, title, model)
    model_title, _ = text("title", _FIELD_MAX_CHARACTERS)
    authors, _ = text("authors", _FIELD_MAX_CHARACTERS)
    year, _ = text("year", 20)
    return PaperCard(
        path=path,
        sha256=sha256,
        title=title or model_title or path.stem,
        authors=authors,
        year=year,
        abstract=abstract,
        fields=tuple(fields),
        pages=tuple(pages),
        model=model,
        complete=True,
    )


def card_texts(card: PaperCard) -> tuple[str, ...]:
    """Return the texts of a card that are embedded for paper search.

    Parameters
    ----------
    card: PaperCard
        Document card.

    Returns
    -------
    tuple[str, ...]
        Three texts, each starting with the title: the abstract and aim;
        the methods and data; the findings and limitations. Texts with
        nothing beyond the title are left out, except the first.

    Notes
    -----
    Separate texts let a question about methods match the methods text,
    instead of one vector that blends every field.

    """

    groups = (
        ("abstract", "aim"),
        ("methods", "data"),
        ("findings", "limitations"),
    )
    texts = []
    for number, group in enumerate(groups):
        parts = [card.field(name) for name in group if card.field(name)]
        if parts or number == 0:
            texts.append("\n".join([card.title, *parts]))
    return tuple(texts)


def load_cards(
    file: Path,
) -> tuple[
    dict[Path, PaperCard],
    dict[Path, tuple[tuple[float, ...], ...]],
    str | None,
]:
    """Read the saved cards of a project.

    Parameters
    ----------
    file: pathlib.Path
        Saved cards (``paper_cards.json`` in the index folder).

    Returns
    -------
    tuple[dict, dict, str | None]
        Cards and their search vectors by document path, and the identity
        of the embedding model that made the vectors. A missing, unreadable,
        or older-format file gives no cards.

    """

    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("format") != PAPER_CARD_FORMAT:
            return {}, {}, None
        cards: dict[Path, PaperCard] = {}
        vectors: dict[Path, tuple[tuple[float, ...], ...]] = {}
        dimension: int | None = None
        for name, item in data["cards"].items():
            path = Path(name)
            card = PaperCard(
                path=path,
                sha256=str(item["sha256"]),
                title=str(item["title"]),
                authors=str(item["authors"]),
                year=str(item["year"]),
                abstract=str(item["abstract"]),
                fields=tuple(
                    (str(key), str(value))
                    for key, value in item["fields"].items()
                ),
                pages=tuple(
                    (str(key), tuple(int(page) for page in value))
                    for key, value in item["pages"].items()
                ),
                model=str(item["model"]),
                complete=item["complete"],
            )
            if type(card.complete) is not bool:
                continue
            if card.complete and not (
                card.abstract.strip()
                or any(value.strip() for _, value in card.fields)
            ):
                # A corrupt title-only success must be regenerated.
                continue
            if not card.complete and (card.abstract or card.fields):
                continue
            cards[path] = card
            try:
                candidate = tuple(
                    tuple(float(value) for value in vector)
                    for vector in item.get("vectors", [])
                )
                if candidate:
                    if not cards[path].complete:
                        raise ValueError("Incomplete card has vectors.")
                    _validate_card_vectors(cards[path], candidate)
                    if dimension is None:
                        dimension = len(candidate[0])
                    elif len(candidate[0]) != dimension:
                        raise ValueError("Card dimensions differ.")
                vectors[path] = candidate
            except (ValueError, TypeError, OverflowError):
                vectors[path] = ()
        identity = data.get("embedding_model_identity")
        return cards, vectors, identity if isinstance(identity, str) else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}, {}, None


def save_cards(
    file: Path,
    cards: Mapping[Path, PaperCard],
    vectors: Mapping[Path, Sequence[Sequence[float]]],
    embedding_model_identity: str,
) -> None:
    """Write the cards of a project with their search vectors.

    Parameters
    ----------
    file: pathlib.Path
        Destination (``paper_cards.json`` in the index folder).
    cards: Mapping[pathlib.Path, PaperCard]
        Cards by document path.
    vectors: Mapping[pathlib.Path, Sequence[Sequence[float]]]
        Search vectors of each card (``card_texts``), by document path.
    embedding_model_identity: str
        Identity of the embedding model that made the vectors.

    """

    dimension: int | None = None
    for path, card in cards.items():
        candidate = tuple(tuple(vector) for vector in vectors.get(path, ()))
        if candidate:
            if not card.complete:
                raise ValueError("Incomplete cards cannot have vectors.")
            _validate_card_vectors(card, candidate)
            if dimension is None:
                dimension = len(candidate[0])
            elif len(candidate[0]) != dimension:
                raise ValueError("Card vector dimensions do not match.")
    data = {
        "format": PAPER_CARD_FORMAT,
        "embedding_model_identity": embedding_model_identity,
        "cards": {
            path.as_posix(): {
                "sha256": card.sha256,
                "title": card.title,
                "authors": card.authors,
                "year": card.year,
                "abstract": card.abstract,
                "fields": dict(card.fields),
                "pages": {key: list(value) for key, value in card.pages},
                "model": card.model,
                "complete": card.complete,
                "vectors": [
                    list(vector) for vector in vectors.get(path, ())
                ],
            }
            for path, card in sorted(cards.items())
        },
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        file, json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    )


def _validate_card_vectors(
    card: PaperCard, vectors: tuple[tuple[float, ...], ...]
) -> None:
    """Validate a card's embeddings like ordinary semantic index vectors.

    Parameters
    ----------
    card: PaperCard
        Completed card whose texts were embedded.
    vectors: tuple[tuple[float, ...], ...]
        Candidate vectors for its indexed card texts.

    Raises
    ------
    ValueError
        If count, dimension, numeric values, or norms are invalid.

    """

    _validate_semantic_embeddings(vectors, len(card_texts(card)))


def _title_card(
    path: Path, sha256: str, title: str | None, model: str
) -> PaperCard:
    """Return an incomplete card that holds the title only.

    Parameters
    ----------
    path: pathlib.Path
        Document path.
    sha256: str
        SHA-256 of the document's bytes.
    title: str | None
        Known title, or ``None`` for the file name.
    model: str
        Model that was asked.

    Returns
    -------
    PaperCard
        Card without summary fields, marked incomplete.

    """

    return PaperCard(path, sha256, title or path.stem, "", "", "", (), (),
                     model, False)


def _json_object(reply: str) -> dict | None:
    """Return the JSON object in a reply, if there is one.

    Parameters
    ----------
    reply: str
        Model reply, possibly with text or a code fence around the object.

    Returns
    -------
    dict | None
        The object from the first ``{`` to the last ``}``, or ``None``.

    """

    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _page(chunk: Chunk) -> int | None:
    """Return the page a PDF passage comes from.

    Parameters
    ----------
    chunk: Chunk
        Document passage.

    Returns
    -------
    int | None
        One-based page, or ``None`` for passages without pages.

    """

    match = _PAGE_SECTION.match(chunk.section or "")
    return int(match.group(1)) if match else None


def _reading_position(chunk: Chunk) -> tuple[int, int]:
    """Return a passage's page and first line, for reading order.

    Parameters
    ----------
    chunk: Chunk
        Document passage.

    Returns
    -------
    tuple[int, int]
        Page (zero when unknown) and first line.

    """

    return (_page(chunk) or 0, chunk.start_line)


def _where(chunk: Chunk) -> str:
    """Describe where a passage is, for the prompt.

    Parameters
    ----------
    chunk: Chunk
        Document passage.

    Returns
    -------
    str
        ``page N`` for PDF passages, otherwise the section or ``text``.

    """

    page = _page(chunk)
    if page is not None:
        return f"page {page}"
    return chunk.section or "text"
