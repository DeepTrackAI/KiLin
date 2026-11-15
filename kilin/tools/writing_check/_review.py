"""Check the grammar and style of one paragraph with the model.

Private to ``writing_check``. Each paragraph is sent to the answering
model on its own, once for grammar and, when asked, once for style, with
what the model needs to judge it:

1. **The document**: its card (title, venue, aim, methods, findings), or
   when it has none a short brief written by one model call from its
   abstract, introduction and headings.
2. **The glossary** from the code checks: the authors' acronyms, preferred
   terms and spelling, which the model must keep.
3. **The section** and what it is for ("methods: what was done, usually
   in the past tense").
4. **The paragraph**, with the sentence before and the sentence after it.

The model answers with blocks of ORIGINAL, SUGGESTION and REASON. Every
finding is checked before it is kept: the original words must be in the
paragraph, and the suggestion must keep its numbers, negations,
citation placeholders, and the glossary's acronyms and terms. A finding
that fails is dropped, not reported, so invented errors do not reach the
user.

Key Features
------------

- **One paragraph per call**

    The prompt stays small whatever the document's length.

- **Checked suggestions**

    No suggestion changes a number, a negation, a citation or a term
    the authors chose.

Module Structure
----------------

- ``document_context``, ``brief_prompt`` and ``document_brief``

    What every call is told about the document.

- ``paragraph_prompt``, ``neighbours`` and ``parse_findings``

    One paragraph's prompt with its context, and the findings in the
    reply.

- ``trim_change``

    A change cut to the words it changes.

- ``locate``, ``rejection`` and ``meaning_change``

    The paragraph's own words for a quote, why a suggested change cannot
    be kept, and the changes of meaning no suggestion may make.

- ``Dropped``

    A suggested change that was not kept, and why.

- ``review_paragraph``

    Check one paragraph at one level, with the model.

Examples
--------
>>> from kilin.tools.writing_check._review import parse_findings
>>> parse_findings("ORIGINAL: the data shows\\nSUGGESTION: the data show\\n"
...                "REASON: plural")
[('the data shows', 'the data show', 'plural')]

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Final

from kilin._model_completion import IncompleteModelResponse, strip_thinking
from kilin.tools.writing_check._checks import (
    LEVEL_GRAMMAR,
    LEVEL_STYLE,
    Finding,
    Glossary,
    make_finding,
)
from kilin.tools.writing_check._document import Document

__all__ = [
    "CONTEXT_QUOTED",
    "Dropped",
    "brief_prompt",
    "document_brief",
    "document_context",
    "locate",
    "meaning_change",
    "neighbours",
    "paragraph_prompt",
    "parse_findings",
    "rejection",
    "review_paragraph",
    "trim_change",
]

# Why a suggestion quoting the context, not the paragraph, is dropped.
CONTEXT_QUOTED: Final[str] = "it quotes the context, not the paragraph"

# Card fields shown to the model, in order, each cut to a length.
_CARD_FIELDS: Final[tuple[str, ...]] = (
    "title", "venue", "aim", "methods", "findings",
)
_FIELD_CHARACTERS: Final[int] = 300
# Text from the document given to the brief call.
_BRIEF_CHARACTERS: Final[int] = 4000
# Style suggestions asked for, at most, per paragraph.
_STYLE_MAX: Final[int] = 5
# What each kind of section is for, by a word in its title.
_ROLES: Final[tuple[tuple[str, str], ...]] = (
    ("abstract", "abstract: the whole work in brief"),
    ("introduction", "introduction: background in the present tense, the "
     "aim of this work"),
    ("background", "background: established knowledge, present tense"),
    ("method", "methods: what was done, usually in the past tense"),
    ("material", "methods: what was done, usually in the past tense"),
    ("result", "results: what was found, past tense, without "
     "interpretation"),
    ("discussion", "discussion: interpretation and limitations"),
    ("conclusion", "conclusions: what the work shows"),
)
_BLOCK: Final[re.Pattern[str]] = re.compile(
    r"ORIGINAL:\s*(?P<original>.+?)\s*\n\s*SUGGESTION:\s*(?P<suggestion>.*?)"
    r"\s*\n\s*REASON:\s*(?P<reason>.+?)\s*(?=\n\s*ORIGINAL:|\Z)",
    re.DOTALL,
)
_NUMBER: Final[re.Pattern[str]] = re.compile(r"\d+(?:[.,]\d+)*")
_NEGATION: Final[re.Pattern[str]] = re.compile(
    r"\b(?:not|no|never|none|neither|nor|cannot|without)\b|n't\b",
    re.IGNORECASE,
)
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(
    r"\[cite\]|\[author\]|\[year\]|\(ref\)"
)
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")
# Hedges and modal verbs: a suggestion may not add, drop or swap them.
_HEDGE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:would|could|may|might|can|should|must|will|shall|virtually|"
    r"approximately|nearly|almost|roughly|likely|possibly|potentially|"
    r"probably|perhaps|presumably|apparently|largely|mostly|partly|"
    r"somewhat|relatively|typically|usually|often|generally)\b",
    re.IGNORECASE,
)
# Words of shared context kept on each side of a trimmed change.
_TRIM_CONTEXT: Final[int] = 2
# Characters a model may write differently when it quotes the text; each
# is replaced by one character, so positions are kept.
_SAME: Final[dict[int, str]] = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'", "“": '"', "”": '"', "„": '"',
    "″": '"', "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-",
    "\u00a0": " ",
})
# Ends a model may add to a quote: a full stop, a comma, an ellipsis.
_QUOTE_ENDS: Final[re.Pattern[str]] = re.compile(r"^(?:\.\.\.|…)\s*|"
                                                r"\s*(?:\.\.\.|…|[.,;:])$")


@dataclass(frozen=True, slots=True)
class Dropped:
    """A suggested change that was not kept.

    Attributes
    ----------
    original: str
        The words the model quoted.
    suggestion: str
        The words it suggested.
    why: str
        Why it was dropped (``rejection``).

    """

    original: str
    suggestion: str
    why: str


def document_context(
    card: Mapping[str, str] | None, brief: str, glossary: Glossary
) -> str:
    """Return what every paragraph call is told about the document.

    Parameters
    ----------
    card: Mapping[str, str] | None
        The document's card fields (``ToolContext.card``), or None.
    brief: str
        A brief of the document, used without a card.
    glossary: Glossary
        The code checks' glossary.

    Returns
    -------
    str
        Lines naming the document (card fields, each cut short, or the
        brief) and the glossary; ``""`` when there is nothing to say.

    """

    lines = []
    if card:
        for name in _CARD_FIELDS:
            value = " ".join(str(card.get(name, "")).split())
            if value:
                lines.append(f"{name.capitalize()}: "
                             f"{value[:_FIELD_CHARACTERS]}")
    elif brief:
        lines.append(f"About the document: {brief}")
    described = glossary.describe()
    if described:
        lines.append(described)
    return "\n".join(lines)


def brief_prompt(document: Document) -> tuple[str, str]:
    """Return the prompts for a brief of a document without a card.

    Parameters
    ----------
    document: Document
        The document.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for two or three sentences on the
        field, the kind of document and its aim, from its headings and
        first paragraphs.

    """

    headings = "; ".join(
        f"{section.number} {section.title}".strip()
        for section in document.sections
    )
    opening = ""
    for paragraph in document.paragraphs:
        if len(opening) >= _BRIEF_CHARACTERS:
            break
        opening += paragraph.text + "\n"
    system = "You describe documents briefly and factually."
    user = (
        f"Headings: {headings or 'none'}\n\nBeginning of the document:\n"
        f"{opening[:_BRIEF_CHARACTERS]}\n\nIn two or three sentences, "
        "say what field this document is in, what kind of document it is "
        "(for example a research article or a thesis chapter), and what "
        "it is about. Write only those sentences."
    )
    return system, user


def document_brief(
    document: Document, ask_model: Callable[[str, str, int | None], str]
) -> str:
    """Return a brief of a document, written by one model call.

    Parameters
    ----------
    document: Document
        The document.
    ask_model: Callable[[str, str, int | None], str]
        The answering model.

    Returns
    -------
    str
        The brief on one line, or ``""`` when the call fails: the checks
        then go on without it.

    """

    try:
        reply = ask_model(*brief_prompt(document), None)
    except Exception:  # Without a brief, the paragraphs are still checked.
        return ""
    return " ".join(strip_thinking(reply).split())


def paragraph_prompt(
    level: str,
    document: Document,
    index: int,
    context: str,
) -> tuple[str, str]:
    """Return the prompts that check one paragraph at one level.

    Parameters
    ----------
    level: str
        ``grammar`` or ``style``.
    document: Document
        The document.
    index: int
        The paragraph.
    context: str
        ``document_context``.

    Returns
    -------
    tuple[str, str]
        System and user prompt: the document, its section and role, the
        sentences around the paragraph, and the instructions for the
        level, asking for ORIGINAL, SUGGESTION and REASON blocks or NONE.

    """

    paragraph = document.paragraphs[index]
    section = ""
    if paragraph.section is not None:
        found = document.sections[paragraph.section]
        section = f"{found.number} {found.title}".strip()
        role = _role(found.title)
        if role:
            section += f" ({role})"
    before, after = neighbours(document, index)
    if level == LEVEL_GRAMMAR:
        system = (
            "You are a careful copy editor of scientific writing. You "
            "correct only clear errors; you never rewrite for taste."
        )
        task = (
            "Find the grammar errors in the paragraph: agreement, verb "
            "forms, wrong tense, articles, prepositions, punctuation, "
            "spelling, missing or extra words. Do not report style or "
            "wording choices, and never replace a correct word with "
            "another."
        )
    else:
        system = (
            "You are an experienced editor of scientific writing. You "
            "suggest few changes, each clearly improving the text."
        )
        task = (
            "Suggest at most "
            f"{_STYLE_MAX} changes, and only where a reader would "
            "stumble: a wordy phrase, an ambiguous reference (\"it\", "
            "\"this\"), an awkward or hard-to-follow construction. Give "
            "the most useful first; suggesting nothing is often right. "
            "Never replace a correct word with a synonym or to sound more "
            "formal, never add, remove or soften a claim, a hedge "
            "(\"virtually\", \"approximately\") or a modal verb "
            "(\"would\", \"could\", \"may\"), and do not report grammar "
            "errors."
        )
    parts = []
    if context:
        parts.append(f"The document:\n{context}")
    if section:
        parts.append(f"Section: {section}")
    if before or after:
        lines = ["Context, for reference only (do not check or quote it):"]
        if before:
            lines.append(f"Sentence before the paragraph: {before}")
        if after:
            lines.append(f"Sentence after the paragraph: {after}")
        parts.append("\n".join(lines))
    parts.append(
        f'Paragraph to check:\n"""\n{paragraph.text}\n"""'
    )
    parts.append(
        f"{task} Check only the paragraph between the triple quotes. "
        "Keep the authors' acronyms, terms and spelling, field "
        "terminology and names in the software, numbers, and the "
        "placeholders [cite], [author], [year] and (ref).\n\nFor each "
        "change, write exactly:\n"
        "ORIGINAL: <the shortest exact words from the paragraph that "
        "need the change, a few words, not the whole sentence>\n"
        "SUGGESTION: <those words changed>\n"
        "REASON: <a few words, on one line>\n\nWrite NONE if nothing "
        "needs changing."
    )
    return system, "\n\n".join(parts)


def neighbours(document: Document, index: int) -> tuple[str, str]:
    """Return the sentences around a paragraph, given as context.

    Parameters
    ----------
    document: Document
        The document.
    index: int
        The paragraph.

    Returns
    -------
    tuple[str, str]
        The last sentence of the paragraph before and the first sentence
        of the paragraph after, ``""`` where there is none.

    """

    before = (
        _SENTENCE_END.split(document.paragraphs[index - 1].text)[-1]
        if index > 0 else ""
    )
    after = (
        _SENTENCE_END.split(document.paragraphs[index + 1].text)[0]
        if index + 1 < len(document.paragraphs) else ""
    )
    return before, after


def trim_change(words: str, suggestion: str) -> tuple[str, str, int]:
    """Return a change cut to its changed words, with a little context.

    Parameters
    ----------
    words: str
        The paragraph's words the change replaces.
    suggestion: str
        The suggested words.

    Returns
    -------
    tuple[str, str, int]
        The original and suggested words without the words they share
        at the start and the end, keeping up to two of those words on
        each side for context, and how many characters were cut from the
        start of the original.

    """

    old, new = words.split(), suggestion.split()
    start = 0
    while start < min(len(old), len(new)) and old[start] == new[start]:
        start += 1
    end = 0
    while (
        end < min(len(old), len(new)) - start
        and old[-1 - end] == new[-1 - end]
    ):
        end += 1
    keep_start = max(0, start - _TRIM_CONTEXT)
    keep_end = max(0, end - _TRIM_CONTEXT)
    original = " ".join(old[keep_start:len(old) - keep_end])
    changed = " ".join(new[keep_start:len(new) - keep_end])
    cut = len(" ".join(old[:keep_start])) + (1 if keep_start else 0)
    return original, changed, cut


def _role(title: str) -> str:
    """Return what a section is for, from its title.

    Parameters
    ----------
    title: str
        The section's title.

    Returns
    -------
    str
        A short description, or ``""`` for a title of no known kind.

    """

    folded = title.casefold()
    for word, role in _ROLES:
        if word in folded:
            return role
    return ""


def parse_findings(reply: str) -> list[tuple[str, str, str]]:
    """Return the changes a model reply suggests.

    Parameters
    ----------
    reply: str
        The model's reply.

    Returns
    -------
    list[tuple[str, str, str]]
        Original words, suggestion and reason of each complete block, with
        surrounding quotation marks removed and the reason cut to its
        first line (a NONE the model adds after it is not part of it);
        empty for NONE.

    """

    found = []
    for match in _BLOCK.finditer(strip_thinking(reply)):
        reason = match["reason"].strip().split("\n", 1)[0]
        reason = re.sub(r"\s*\bNONE\s*$", "", reason)
        original, suggestion, reason = (
            _unquoted(text)
            for text in (match["original"], match["suggestion"], reason)
        )
        if original:
            found.append((original, suggestion, reason))
    return found


def _unquoted(text: str) -> str:
    """Return text on one line, without surrounding quotation marks.

    Parameters
    ----------
    text: str
        A field of a reply.

    Returns
    -------
    str
        The text with spaces collapsed and outer quotes removed.

    """

    text = " ".join(text.split())
    if len(text) >= 2 and text[0] in "\"“'" and text[-1] in "\"”'":
        text = text[1:-1].strip()
    return text


def _comparable(text: str) -> str:
    """Return text with quotes, dashes and case made alike.

    Parameters
    ----------
    text: str
        Text with single spaces.

    Returns
    -------
    str
        The same length of text: curly quotes straight, dashes as "-",
        letters in lower case where that keeps the length.

    """

    text = text.translate(_SAME)
    lowered = text.lower()
    return lowered if len(lowered) == len(text) else text


def locate(original: str, paragraph: str) -> tuple[str, int] | None:
    """Return the paragraph's own words for a quote, and where they are.

    Parameters
    ----------
    original: str
        The words the model quotes.
    paragraph: str
        The paragraph's readable text.

    Returns
    -------
    tuple[str, int] | None
        The words as the paragraph writes them and their position, found
        with spaces, quotation marks, dashes and case ignored, and without
        a full stop, comma or ellipsis the model added at either end; None
        when the words are not in the paragraph.

    """

    text = " ".join(paragraph.split())
    quote = " ".join(original.split())
    for candidate in (quote, _QUOTE_ENDS.sub("", quote).strip()):
        if not candidate:
            continue
        where = _comparable(text).find(_comparable(candidate))
        if where >= 0:
            return text[where:where + len(candidate)], where
    return None


def rejection(
    original: str, suggestion: str, paragraph: str, glossary: Glossary
) -> str:
    """Return why a suggested change cannot be kept, or ``""``.

    Parameters
    ----------
    original: str
        The words the model quotes.
    suggestion: str
        The words it suggests.
    paragraph: str
        The paragraph's readable text.
    glossary: Glossary
        The code checks' glossary.

    Returns
    -------
    str
        ``""`` when the change can be kept; otherwise the first reason it
        cannot: the words are not in the paragraph (``locate``), nothing
        changes, or the suggestion changes numbers, negations, citation
        placeholders, a hedge or modal verb ("virtually", "would"), a
        defined acronym or a preferred term, or brings back a form the
        document does not use. The suggestion is compared with the
        paragraph's own words.

    """

    found = locate(original, paragraph)
    if found is None:
        return "the words are not in the paragraph"
    original = found[0]
    if " ".join(suggestion.split()) == original:
        return "nothing changes"
    return meaning_change(original, suggestion, glossary)


def meaning_change(original: str, changed: str, glossary: Glossary) -> str:
    """Return how changed words alter the meaning, or ``""``.

    Parameters
    ----------
    original: str
        The words as the document writes them.
    changed: str
        The words as suggested.
    glossary: Glossary
        The code checks' glossary.

    Returns
    -------
    str
        ``""`` when nothing that carries meaning changes; otherwise the
        first change found: numbers, citation placeholders, negations,
        hedges or modal verbs, a defined acronym or preferred term
        removed, or a term form the document does not use.

    """

    suggestion = changed
    for pattern, what in (
        (_NUMBER, "numbers"), (_PLACEHOLDER, "citations"),
    ):
        if Counter(pattern.findall(original)) != Counter(
            pattern.findall(suggestion)
        ):
            return f"it changes {what}"
    if len(_NEGATION.findall(original)) != len(
        _NEGATION.findall(suggestion)
    ):
        return "it changes a negation"
    if Counter(word.casefold() for word in _HEDGE.findall(original)) != (
        Counter(word.casefold() for word in _HEDGE.findall(suggestion))
    ):
        return "it changes a hedge or modal verb"
    kept = [*glossary.acronyms, *set(glossary.terms.values())]
    for term in kept:
        pattern = rf"(?<![\w-]){re.escape(term)}(?![\w-])"
        if re.search(pattern, original, re.IGNORECASE) and not re.search(
            pattern, suggestion, re.IGNORECASE
        ):
            return f"it removes “{term}”"
    for form in glossary.terms:
        pattern = rf"(?<![\w-]){re.escape(form)}(?![\w-])"
        if re.search(pattern, suggestion, re.IGNORECASE):
            return f"it writes “{form}”, which the document does not use"
    return ""


def review_paragraph(
    level: str,
    document: Document,
    index: int,
    context: str,
    glossary: Glossary,
    ask_model: Callable[[str, str, int | None], str],
) -> tuple[list[Finding], list[Dropped], str]:
    """Check one paragraph at one level with the model.

    Parameters
    ----------
    level: str
        ``grammar`` or ``style``.
    document: Document
        The document.
    index: int
        The paragraph.
    context: str
        ``document_context``.
    glossary: Glossary
        The code checks' glossary.
    ask_model: Callable[[str, str, int | None], str]
        The answering model (``ToolContext.ask_model``).

    Returns
    -------
    tuple[list[Finding], list[Dropped], str]
        The kept findings (level ``grammar`` or ``style``, check
        ``model``, quoting the paragraph's own words, cut to the changed
        words by ``trim_change``), the suggestions dropped by
        ``rejection`` with the reason (``CONTEXT_QUOTED`` for words from
        the surrounding sentences), and why the call failed (``""`` when
        it did not). A failed call is reported, not fatal.

    """

    system, user = paragraph_prompt(level, document, index, context)
    try:
        reply = ask_model(system, user, None)
    except IncompleteModelResponse as error:
        return [], [], f"Not completed ({error.reason}): {error}"
    except Exception as error:  # A failed call is reported, not fatal.
        return [], [], f"Model error: {error}"
    text = document.paragraphs[index].text
    found = parse_findings(reply)
    if level == LEVEL_STYLE:
        found = found[:_STYLE_MAX]
    context_text = " ".join(neighbours(document, index))
    kept: list[Finding] = []
    dropped: list[Dropped] = []
    for original, suggestion, reason in found:
        why = rejection(original, suggestion, text, glossary)
        located = locate(original, text)
        if located is None and locate(original, context_text) is not None:
            why = CONTEXT_QUOTED
        if why or located is None:
            dropped.append(Dropped(original, suggestion, why))
            continue
        words, offset = located
        words, suggestion, cut = trim_change(words, suggestion)
        kept.append(make_finding(
            document.paragraphs, index, level, "model", words,
            suggestion, reason, offset + cut,
        ))
    return kept, dropped, ""
