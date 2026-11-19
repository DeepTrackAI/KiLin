"""Write suggested questions about a project's own sources.

An empty chat shows a few questions a user can click and send at once. A
project type lists one fill-in question per intent in
``suggestion_templates``, such as "Give me a short summary of {file}.".
This module fills them with the project's real file names and with topics
and exact terms that a small model picks from the sources, so each question
shows what KíLín can do with these sources and can be answered. The
heading above them is written the same way from ``welcome_templates``, such
as "What would you like to understand about {theme}?", where a theme is
a broad subject of the whole collection, so the heading covers the
questions below it.

The work has two steps. On explicit generation (for example, a request
for assistant guidance), a small model reads a selection of the relevant
files and picks a pool of up to three themes, ten topics, and ten terms.
The pool is saved in the index folder. Each new chat fills questions
from that saved pool, with randomly chosen files, themes, topics, and
terms, without a model call. Without a saved pool, the empty-chat screen
still offers file-name questions and the configured fixed suggestions.

File names come from the index, and topics and themes must occur as
phrases in an indexed passage. A question whose placeholder has no value
is left out. Document-oriented projects prefer documents over code files.

Key Features
------------

- **Source-specific questions**

    Real file names, topics, and terms instead of "…" gaps to fill in.

- **New questions on each chat**

    A saved pool of words, filled at random without waiting for a model.

- **Checked phrases**

    Topics and themes must occur as whole phrases in a relevant passage;
    terms must occur in relevant files.

- **Safe fallback**

    Without sources there are no questions; without saved words only the
    questions that need just file names are written. The caller then shows
    the project's fixed ``suggestions`` when needed.

Module Structure
----------------

- ``SUGGESTION_WORDS_FILE``

    Name of the saved pool of topics and terms in the index folder.

- ``Suggestion``

    One question and the intent it is meant to reach.

- ``SuggestionWords``

    The themes, topics, and terms that the questions may name.

- ``suggestion_files``

    Source file names to use in questions, selected by project purpose.

- ``ask_for_words``

    Ask a model for the themes, topics, and terms of the sources.

- ``checked_words``

    Keep only the themes, topics, and terms printed in relevant sources.

- ``fill_templates``

    Fill each intent's fill-in question with the project's values.

- ``suggestion_words``

    Return the pool, optionally asking the model when no valid pool exists.

- ``question_suggestions``

    Write fresh questions from the pool, with files and words at random.

- ``welcome_prompts``

    Write fresh welcome headings from the pool in the same way.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> from kilin.question_suggestions import SuggestionWords, fill_templates
>>> [item] = fill_templates(
...     (("paper_overview", "Give me a short summary of {file}."),),
...     ("tiunn26braph2.pdf",),
...     SuggestionWords((), ()),
... )
>>> item.question
'Give me a short summary of tiunn26braph2.pdf.'

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.chunking import Chunk
from kilin.retrievals.source_cards.models import SourceCard

__all__ = [
    "SUGGESTION_WORDS_FILE",
    "Suggestion",
    "SuggestionWords",
    "ask_for_words",
    "checked_words",
    "fill_templates",
    "question_suggestions",
    "suggestion_files",
    "suggestion_words",
    "welcome_prompts",
]

SUGGESTION_WORDS_FILE: Final[str] = "suggestion_words.json"
_FORMAT: Final[str] = "suggestion-words-v3"
# Questions written from each template, so chats can show other ones.
_VARIANTS: Final[int] = 3

# How much of the sources the model reads: a few lines per file, for the
# first files (usually the user's own) and a random mix of the others.
_DESCRIBED_FILES: Final[int] = 12
_FIRST_DESCRIBED: Final[int] = 2
_DESCRIPTION_CHARACTERS: Final[int] = 300
_REPLY_TOKENS: Final[int] = 400
# Function and class names shown per code file.
_DEFINED_NAMES: Final[int] = 8
# How many topics and terms the pool keeps, and how long they may be.
_MAX_WORDS: Final[int] = 10
_MAX_THEMES: Final[int] = 3
_TOPIC_WORDS: Final[tuple[int, int]] = (1, 6)
_THEME_WORDS: Final[tuple[int, int]] = (1, 4)
# A theme names one subject, not two joined together.
_JOINED: Final[re.Pattern[str]] = re.compile(r"\band\b|&|,|/", re.IGNORECASE)
_TERM_CHARACTERS: Final[int] = 40
# Recognize document files before text-only support files in mixed projects.
_DOCUMENT_SUFFIXES: Final[frozenset[str]] = frozenset(
    (".pdf", ".doc", ".docx", ".odt", ".rtf", ".tex")
)
_JSON_OBJECT: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)
# Text that ends where a sentence or line starts, within a few characters.
_SENTENCE_LOOKBACK: Final[int] = 4
# Topic words this short keep their most frequent printed spelling.
_SHORT_WORD: Final[int] = 2
_SENTENCE_START: Final[re.Pattern[str]] = re.compile(r"(?:\A|[.!?:\n])\s*\Z")
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{([a-z0-9]+)\}")
# Shared generator for the random choices; tests pass their own.
_RANDOM: Final[random.Random] = random.Random()
# Card fields that describe a paper best, in order.
_CARD_FIELDS: Final[tuple[str, ...]] = ("aim", "abstract")

_SYSTEM: Final[str] = (
    "You pick the themes, topics, and exact terms of a list of documents "
    "for example questions. Reply with JSON only."
)
_PROMPT: Final[str] = (
    "Documents:\n{documents}\n\n"
    "Reply with one JSON object: "
    '{{"themes": [...], "topics": [...], "terms": [...]}}. '
    "themes: 1 to 3 broad subjects that the documents share as a whole, "
    'each one single subject of 1 to 4 words without "and", such as '
    '"Alzheimer\'s disease" or "brain network analysis". '
    "topics: up to 10 subjects the documents discuss, or tasks and "
    "analyses their code is used for, 1 to 5 words each, such as "
    '"variational autoencoders" or "functional connectivity analysis". '
    "terms: up to 10 exact words, names, or acronyms printed in the "
    'documents, such as "MRI" or "load_data". '
    "Use the documents' own words. Write the themes and topics in lower "
    "case, as inside a sentence, keeping names and acronyms as printed."
)


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One suggested question and the intent it is meant to reach.

    Attributes
    ----------
    intent: str
        Intent ID of the template the question was written from.
    question: str
        The question, ready to send.

    """

    intent: str
    question: str


@dataclass(frozen=True, slots=True)
class SuggestionWords:
    """The themes, topics, and terms that suggested questions may name.

    Attributes
    ----------
    topics: tuple[str, ...]
        Subjects of the sources, such as "variational autoencoders".
    terms: tuple[str, ...]
        Exact words printed in the sources, such as "MRI".
    themes: tuple[str, ...], optional
        Broad subjects of the whole collection, one subject each, such as
        "Alzheimer's disease", for welcome headings. Defaults
        to ``()``, none.

    """

    topics: tuple[str, ...]
    terms: tuple[str, ...]
    themes: tuple[str, ...] = ()


def suggestion_files(
    chunks: Sequence[Chunk], *, prefer_code: bool = True
) -> tuple[str, ...]:
    """Return source file names to use in questions, most telling first.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The project's indexed passages.
    prefer_code: bool, optional
        When True, prefer files with code symbols. When False, prefer
        document files, falling back to other non-code files, then code.
        Defaults to True for direct callers.

    Returns
    -------
    tuple[str, ...]
        File names ordered by project purpose. A file is named by its base
        name, or by its path when two selected files share a base name.

    """

    paths = _preferred_paths(chunks, prefer_code)
    names = Counter(path.name for path in paths)
    return tuple(
        path.name if names[path.name] == 1 else path.as_posix()
        for path in paths
    )


def ask_for_words(
    chunks: Sequence[Chunk],
    cards: Sequence[SourceCard],
    ask: Callable[[str, str, int], str],
    rng: random.Random | None = None,
    *,
    prefer_code: bool = True,
) -> SuggestionWords:
    """Ask a model for the themes, topics, and terms of the sources.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The project's indexed passages.
    cards: Sequence[SourceCard]
        The project's source cards; a card's aim or abstract describes its
        file better than the file's first lines.
    ask: Callable[[str, str, int], str]
        Sends a system instruction, a prompt, and a reply limit to a model
        and returns its reply.
    rng: random.Random | None, optional
        Picks the random mix of files the model reads. Defaults to
        ``None``, a shared generator.
    prefer_code: bool, optional
        Whether to describe the project's code or documents when both
        are indexed. Defaults to True.

    Returns
    -------
    SuggestionWords
        The themes, topics, and terms of the reply, unchecked; empty when
        the reply is not the requested JSON.

    """

    documents = "\n".join(
        f"- {name}: {text}"
        for name, text in _descriptions(
            chunks, cards, rng or _RANDOM, prefer_code=prefer_code
        ).items()
    )
    reply = ask(_SYSTEM, _PROMPT.format(documents=documents), _REPLY_TOKENS)
    match = _JSON_OBJECT.search(reply)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    # A small model may still write one "theme" instead of a list.
    themes = _texts(data.get("themes")) or _texts([data.get("theme")])
    return SuggestionWords(
        _texts(data.get("topics")), _texts(data.get("terms")), themes
    )


def checked_words(
    words: SuggestionWords,
    chunks: Sequence[Chunk],
    *,
    prefer_code: bool = True,
) -> SuggestionWords:
    """Keep only the themes, topics, and terms printed in the sources.

    Parameters
    ----------
    words: SuggestionWords
        Topics and terms a model suggested.
    chunks: Sequence[Chunk]
        The project's indexed passages.
    prefer_code: bool, optional
        Check terms in the type's relevant files, not an unrelated code
        file in a document project. Defaults to True.

    Returns
    -------
    SuggestionWords
        Topics of one to six words and themes of one to four words, each
        one subject not joined by "and", whose complete phrase occurs in
        a single indexed passage, and terms that occur as whole words;
        at most ten topics, ten terms, and three themes. A term that is only
        a file name is dropped. Each topic and theme word is spelled as the
        sources print it, so "alzheimer's disease" becomes "Alzheimer's
        disease".

    """

    paths = set(_preferred_paths(chunks, prefer_code))
    selected = [chunk for chunk in chunks if chunk.path in paths]
    printed = "\n".join(chunk.text for chunk in selected)
    text = printed.casefold()
    passages = tuple(chunk.text.casefold() for chunk in selected)
    files = {chunk.path.name.casefold() for chunk in selected}
    topics = tuple(
        _printed_spelling(topic, printed)
        for topic in words.topics
        if _is_printed_topic(topic, passages, _TOPIC_WORDS)
    )
    themes = tuple(
        _printed_spelling(theme, printed)
        for theme in words.themes
        if _is_printed_topic(theme, passages, _THEME_WORDS)
        and not _JOINED.search(theme)
    )
    terms = tuple(
        term
        for term in words.terms
        if len(term) <= _TERM_CHARACTERS
        and term.casefold() not in files
        and _printed(term.casefold(), text)
    )
    return SuggestionWords(
        topics[:_MAX_WORDS], terms[:_MAX_WORDS], themes[:_MAX_THEMES]
    )


def fill_templates(
    templates: Sequence[tuple[str, str]],
    files: Sequence[str],
    words: SuggestionWords,
    variants: int = 1,
) -> list[Suggestion]:
    """Fill each intent's fill-in question with the project's values.

    Parameters
    ----------
    templates: Sequence[tuple[str, str]]
        Intent ID and template pairs, such as
        ``ProjectType.suggestion_templates``.
    files: Sequence[str]
        File names, as from ``suggestion_files``.
    words: SuggestionWords
        Checked topics and terms.
    variants: int, optional
        Most questions written from each template. Defaults to ``1``.

    Returns
    -------
    list[Suggestion]
        Up to ``variants`` different questions per template whose
        placeholders all have a value, grouped in template order. Values
        are taken in turn, so the questions name different files, topics,
        and terms where there are enough; a template without placeholders
        gives one question.

    """

    values = {
        "file": files,
        "topic": words.topics,
        "term": words.terms,
        "theme": words.themes,
    }
    used = Counter[str]()
    found: dict[int, list[Suggestion]] = {}
    for _ in range(variants):
        for position, (intent, template) in enumerate(templates):
            names = set(_PLACEHOLDER.findall(template))
            if not all(_has_value(name, values) for name in names):
                continue
            question = _fill(template, names, values, used)
            written = found.setdefault(position, [])
            if all(item.question != question for item in written):
                written.append(Suggestion(intent, question))
    return [item for position in sorted(found) for item in found[position]]


def suggestion_words(
    indexes: Path,
    chunks: Sequence[Chunk],
    cards: Sequence[SourceCard],
    ask: Callable[[str, str, int], str] | None,
    model_id: str | None,
    rng: random.Random | None = None,
    *,
    prefer_code: bool = True,
) -> SuggestionWords:
    """Return the pool of topics and terms, asking the model only when the
    saved pool is out of date.

    Parameters
    ----------
    indexes: pathlib.Path
        The project's index folder, where the pool is saved as
        ``SUGGESTION_WORDS_FILE``.
    chunks: Sequence[Chunk]
        The project's indexed passages.
    cards: Sequence[SourceCard]
        The project's source cards.
    ask: Callable[[str, str, int], str] | None
        Model to pick topics and terms, called only when the saved pool is
        out of date; None for no model.
    model_id: str | None
        Identifies the model, so another model picks the pool again.
    rng: random.Random | None, optional
        Picks the random mix of files the model reads. Defaults to
        ``None``, a shared generator.
    prefer_code: bool, optional
        Prefer code or document sources in mixed collections. Defaults
        to True; included in the cache fingerprint.

    Returns
    -------
    SuggestionWords
        Checked topics and terms; empty without sources or a model.

    Notes
    -----
    A pool is saved only when the model answered, so after a failure the
    next call tries again. A reply without usable words is saved: asking
    again about the same sources would not help.

    """

    if not chunks:
        return SuggestionWords((), ())
    fingerprint = _fingerprint(chunks, model_id, prefer_code)
    path = indexes / SUGGESTION_WORDS_FILE
    saved = _read_saved(path, fingerprint)
    if saved is not None:
        return saved
    if ask is None:
        return SuggestionWords((), ())
    try:
        words = checked_words(
            ask_for_words(
                chunks, cards, ask, rng, prefer_code=prefer_code
            ),
            chunks,
            prefer_code=prefer_code,
        )
    except Exception:  # A failed model leaves file-name questions.
        return SuggestionWords((), ())
    _save(path, fingerprint, words)
    return words


def question_suggestions(
    templates: Sequence[tuple[str, str]],
    chunks: Sequence[Chunk],
    words: SuggestionWords,
    rng: random.Random | None = None,
    *,
    prefer_code: bool = True,
) -> list[Suggestion]:
    """Write fresh questions from the pool, with files and words at random.

    Parameters
    ----------
    templates: Sequence[tuple[str, str]]
        Intent ID and template pairs of the project's type.
    chunks: Sequence[Chunk]
        The project's indexed passages.
    words: SuggestionWords
        The pool, as from ``suggestion_words``.
    rng: random.Random | None, optional
        Picks the files, topics, and terms. Defaults to ``None``, a shared
        generator.
    prefer_code: bool, optional
        Whether to prefer code files over documents. Defaults to True.

    Returns
    -------
    list[Suggestion]
        Up to three questions per template that could be filled, naming
        different files, topics, and terms, chosen anew on every call;
        empty when the project has no sources or no templates. No model is
        called.

    """

    if not chunks or not templates:
        return []
    files, shuffled = _shuffled_values(
        chunks, words, rng or _RANDOM, prefer_code=prefer_code
    )
    return fill_templates(templates, files, shuffled, _VARIANTS)


def welcome_prompts(
    templates: Sequence[str],
    chunks: Sequence[Chunk],
    words: SuggestionWords,
    rng: random.Random | None = None,
    *,
    prefer_code: bool = True,
) -> list[str]:
    """Write fresh welcome headings from the pool, with words at random.

    Parameters
    ----------
    templates: Sequence[str]
        Fill-in headings of the project's type, as
        ``ProjectType.welcome_templates``.
    chunks: Sequence[Chunk]
        The project's indexed passages.
    words: SuggestionWords
        The pool, as from ``suggestion_words``.
    rng: random.Random | None, optional
        Picks the files, themes, topics, and terms. Defaults to ``None``,
        a shared generator.
    prefer_code: bool, optional
        Whether to prefer code files over documents. Defaults to True.

    Returns
    -------
    list[str]
        One heading per template that could be filled, naming files,
        themes, topics, and terms chosen anew on every call, so a new chat
        may show another theme; empty when the project has no sources or no
        templates, so the caller shows the type's fixed
        ``welcome_prompts``. No model is called.

    """

    if not chunks or not templates:
        return []
    files, shuffled = _shuffled_values(
        chunks, words, rng or _RANDOM, prefer_code=prefer_code
    )
    filled = fill_templates(
        tuple(("", template) for template in templates), files, shuffled
    )
    return [item.question for item in filled]


def _shuffled_values(
    chunks: Sequence[Chunk],
    words: SuggestionWords,
    rng: random.Random,
    *,
    prefer_code: bool = True,
) -> tuple[list[str], SuggestionWords]:
    """Return the file names and the pool's words in a random order.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The project's indexed passages.
    words: SuggestionWords
        The pool of themes, topics, and terms.
    rng: random.Random
        Picks the order.
    prefer_code: bool, optional
        Whether to prefer code files over documents. Defaults to True.

    Returns
    -------
    tuple[list[str], SuggestionWords]
        Shuffled file names, as from ``suggestion_files``, and shuffled
        themes, topics, and terms.

    """

    files = list(suggestion_files(chunks, prefer_code=prefer_code))
    topics = list(words.topics)
    terms = list(words.terms)
    themes = list(words.themes)
    for values in (files, topics, terms, themes):
        rng.shuffle(values)
    return files, SuggestionWords(tuple(topics), tuple(terms), tuple(themes))


def _descriptions(
    chunks: Sequence[Chunk],
    cards: Sequence[SourceCard],
    rng: random.Random,
    *,
    prefer_code: bool = True,
) -> dict[str, str]:
    """Return a short description of some of the source files.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The project's indexed passages.
    cards: Sequence[SourceCard]
        The project's source cards.
    rng: random.Random
        Picks the random mix of files.
    prefer_code: bool, optional
        Prefer code or document files when both are present. Defaults
        to True.

    Returns
    -------
    dict[str, str]
        For the first two files of ``_ranked_paths`` (usually the user's
        own) and a random mix of the others, twelve in all: file name to
        its card's title and aim, or else its first lines, on one line and
        shortened. A code file starts with the names it defines, so the
        model can pick them as terms.

    """

    by_card = {card.path: card for card in cards}
    first: dict[Path, str] = {}
    symbols: dict[Path, list[str]] = {}
    for chunk in chunks:
        first.setdefault(chunk.path, chunk.text)
        if chunk.symbol and chunk.symbol not in symbols.get(chunk.path, []):
            symbols.setdefault(chunk.path, []).append(chunk.symbol)
    ranked = _preferred_paths(chunks, prefer_code)
    others = ranked[_FIRST_DESCRIBED:]
    chosen = ranked[:_FIRST_DESCRIBED] + rng.sample(
        others, min(len(others), _DESCRIBED_FILES - _FIRST_DESCRIBED)
    )
    descriptions: dict[str, str] = {}
    for path in chosen:
        card = by_card.get(path)
        text = _card_text(card) if card is not None else ""
        text = " ".join((text or first[path]).split())
        if path in symbols:
            names = ", ".join(symbols[path][:_DEFINED_NAMES])
            text = f"defines {names}. {text}"
        descriptions[path.name] = text[:_DESCRIPTION_CHARACTERS]
    return descriptions


def _ranked_paths(chunks: Sequence[Chunk]) -> list[Path]:
    """Return the source files in the order questions should name them.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        The project's indexed passages.

    Returns
    -------
    list[pathlib.Path]
        Files with code symbols first (in a code project, code rather than
        its README), then files higher in the folder tree (usually the
        user's own documents, not a folder of cited papers), then files
        with more passages, then by path.

    """

    counts = Counter(chunk.path for chunk in chunks)
    code = {chunk.path for chunk in chunks if chunk.symbol}
    return sorted(
        counts,
        key=lambda path: (
            path not in code,
            len(path.parts),
            -counts[path],
            path.as_posix(),
        ),
    )


def _preferred_paths(
    chunks: Sequence[Chunk], prefer_code: bool
) -> list[Path]:
    """Select files relevant to a code or document-oriented project.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Indexed passages with paths and optional code symbols.
    prefer_code: bool
        True for code projects, False for document-oriented projects.

    Returns
    -------
    list[pathlib.Path]
        Ranked code files when available for code projects. For other
        projects, ranked document files, then other non-code files if
        there are no recognized documents, then all files as fallback.

    """

    paths = _ranked_paths(chunks)
    code = {chunk.path for chunk in chunks if chunk.symbol}
    if prefer_code and code:
        return [path for path in paths if path in code]
    if not prefer_code:
        documents = [
            path for path in paths
            if path.suffix.casefold() in _DOCUMENT_SUFFIXES
        ]
        other = [path for path in paths if path not in code]
        return documents or other or paths
    return paths


def _card_text(card: SourceCard) -> str:
    """Return a card's title and aim (or abstract), or an empty string.

    Parameters
    ----------
    card: SourceCard
        Source card of one file.

    Returns
    -------
    str
        Title followed by the first available description field.

    """

    metadata = dict(card.metadata)
    fields = {item.field: item.text for item in card.statements}
    summary = next(
        (fields[name] for name in _CARD_FIELDS if fields.get(name)), ""
    )
    return " ".join(
        part for part in (metadata.get("title", ""), summary) if part
    )


def _texts(value: object) -> tuple[str, ...]:
    """Return the distinct nonblank texts of a JSON list, stripped.

    Parameters
    ----------
    value: object
        A JSON value that should be a list of strings.

    Returns
    -------
    tuple[str, ...]
        Stripped texts without quotes, in order, without repeats; empty for
        anything that is not a list.

    """

    if not isinstance(value, list):
        return ()
    texts = (
        " ".join(item.strip().strip("\"'").split())
        for item in value
        if isinstance(item, str)
    )
    return tuple(dict.fromkeys(text for text in texts if text))


def _is_printed_topic(
    topic: str, passages: Sequence[str], length: tuple[int, int]
) -> bool:
    """Return whether a topic has a fitting length and is printed.

    Parameters
    ----------
    topic: str
        Topic or theme as the model wrote it.
    passages: Sequence[str]
        Lowercase text from each indexed passage.
    length: tuple[int, int]
        Fewest and most words allowed.

    Returns
    -------
    bool
        True when the word count is in range and the full phrase occurs
        in at least one passage, ignoring capitalization and whitespace.

    """

    words = topic.casefold().split()
    if not length[0] <= len(words) <= length[1]:
        return False
    phrase = r"\s+".join(re.escape(word) for word in words)
    pattern = re.compile(rf"(?<!\w){phrase}(?!\w)")
    return any(pattern.search(text) is not None for text in passages)


def _printed(word: str, text: str) -> bool:
    """Return whether a lowercase word appears in text as a whole word.

    Parameters
    ----------
    word: str
        Lowercase word or phrase.
    text: str
        Lowercase source text.

    Returns
    -------
    bool
        True when the word is printed and not only part of a longer word.

    """

    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def _printed_spelling(topic: str, printed: str) -> str:
    """Spell each word of a topic as the sources print it.

    Parameters
    ----------
    topic: str
        Topic as the model wrote it.
    printed: str
        Source text with its original capitals.

    Returns
    -------
    str
        The topic with each word in its most frequent printed form inside
        a sentence, such as "Alzheimer's" or "MRI", or in lower case when
        it is printed in lower case at all or only with a capital that
        starts a sentence. A word of one or two letters, such as the "T"
        of "T cell", always takes its most frequent form, since such
        words are also printed in lower case elsewhere ("t-test"). A word
        that is not printed is kept as written.

    """

    words = []
    for word in topic.split():
        forms = Counter(
            match.group(0)
            for match in re.finditer(
                rf"(?<!\w){re.escape(word)}(?!\w)", printed, re.IGNORECASE
            )
            if not _starts_sentence(printed, match.start())
        )
        short = len(word) <= _SHORT_WORD
        if not forms or (word.lower() in forms and not short):
            words.append(
                word.lower()
                if _printed(word.casefold(), printed.casefold())
                else word
            )
        else:
            words.append(forms.most_common(1)[0][0])
    return " ".join(words)


def _starts_sentence(text: str, position: int) -> bool:
    """Return whether a position starts a sentence or a line.

    Parameters
    ----------
    text: str
        Source text.
    position: int
        Index of a word's first character.

    Returns
    -------
    bool
        True at the start of the text, or after a sentence end, colon, or
        line break and optional spaces.

    """

    before = text[max(0, position - _SENTENCE_LOOKBACK) : position]
    return _SENTENCE_START.search(before) is not None


def _fill(
    template: str,
    names: set[str],
    values: dict[str, Sequence[str]],
    used: Counter[str],
) -> str:
    """Fill one template with the next unused value of each placeholder.

    Parameters
    ----------
    template: str
        Fill-in question.
    names: set[str]
        Its placeholder names; each has a value.
    values: dict[str, Sequence[str]]
        Available values by placeholder name.
    used: Counter[str]
        How many values of each name were taken; updated in place.

    Returns
    -------
    str
        The question. ``{file2}`` takes the file after ``{file}``, so the
        two always differ.

    """

    filled: dict[str, str] = {}
    for name in ("file", "topic", "term", "theme"):
        if name in names or (name == "file" and "file2" in names):
            options = values[name]
            filled[name] = options[used[name] % len(options)]
            used[name] += 1
    if "file2" in names:
        files = values["file"]
        filled["file2"] = files[used["file"] % len(files)]
        used["file"] += 1
    return _PLACEHOLDER.sub(lambda match: filled[match.group(1)], template)


def _has_value(name: str, values: dict[str, Sequence[str]]) -> bool:
    """Return whether a placeholder can be filled.

    Parameters
    ----------
    name: str
        Placeholder name, such as ``file``.
    values: dict[str, Sequence[str]]
        Available values by placeholder name.

    Returns
    -------
    bool
        True when there is a value; ``file2`` needs two files.

    """

    if name == "file2":
        return len(values["file"]) >= 2
    return bool(values.get(name))


def _fingerprint(
    chunks: Sequence[Chunk], model_id: str | None, prefer_code: bool
) -> str:
    """Return what the saved pool depends on, as one digest.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        Indexed passages; their identifiers change with their text.
    model_id: str | None
        Model that picks topics and terms.
    prefer_code: bool
        Source policy for a code or document-oriented project.

    Returns
    -------
    str
        SHA-256 hex digest.

    """

    data = {
        "chunks": sorted(chunk.identifier for chunk in chunks),
        "model": model_id,
        "prefer_code": prefer_code,
    }
    return sha256(json.dumps(data).encode("utf-8")).hexdigest()


def _read_saved(path: Path, fingerprint: str) -> SuggestionWords | None:
    """Return a saved pool that is still current, or None.

    Parameters
    ----------
    path: pathlib.Path
        Saved pool file.
    fingerprint: str
        Digest of what the pool depends on now.

    Returns
    -------
    SuggestionWords | None
        The saved pool; None when the file is missing, unreadable,
        malformed, or out of date.

    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["format"] != _FORMAT or data["fingerprint"] != fingerprint:
            return None
        return SuggestionWords(
            tuple(str(item) for item in data["topics"]),
            tuple(str(item) for item in data["terms"]),
            tuple(str(item) for item in data["themes"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save(path: Path, fingerprint: str, words: SuggestionWords) -> None:
    """Save a pool with what it depends on; a failure is ignored.

    Parameters
    ----------
    path: pathlib.Path
        Saved pool file.
    fingerprint: str
        Digest of what the pool depends on.
    words: SuggestionWords
        Topics and terms to save.

    """

    data = {
        "format": _FORMAT,
        "fingerprint": fingerprint,
        "themes": list(words.themes),
        "topics": list(words.topics),
        "terms": list(words.terms),
    }
    try:
        atomic_write_text(path, json.dumps(data, indent=2) + "\n")
    except OSError:
        pass  # Saving only avoids asking the model again.
