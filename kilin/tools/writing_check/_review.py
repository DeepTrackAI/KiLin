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

- ``paragraph_prompt`` and ``parse_findings``

    One paragraph's prompt, and the findings in the reply.

- ``rejection``

    Why a suggested change cannot be kept, if it cannot.

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
    "brief_prompt",
    "document_brief",
    "document_context",
    "paragraph_prompt",
    "parse_findings",
    "rejection",
    "review_paragraph",
]

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
    before = (
        _SENTENCE_END.split(document.paragraphs[index - 1].text)[-1]
        if index > 0 else ""
    )
    after = (
        _SENTENCE_END.split(document.paragraphs[index + 1].text)[0]
        if index + 1 < len(document.paragraphs) else ""
    )
    if level == LEVEL_GRAMMAR:
        system = (
            "You are a careful copy editor of scientific writing. You "
            "correct only clear errors; you never rewrite for taste."
        )
        task = (
            "Find the grammar errors in the paragraph only: agreement, "
            "verb forms, wrong tense, articles, prepositions, punctuation, "
            "spelling, missing or extra words. Do not report style or "
            "wording choices."
        )
    else:
        system = (
            "You are an experienced editor of scientific writing. You "
            "suggest few changes, each clearly improving the text."
        )
        task = (
            "Suggest at most "
            f"{_STYLE_MAX} changes that clearly improve clarity or "
            "concision: wordy phrases, ambiguous references (\"it\", "
            "\"this\"), informal words, a sentence hard to follow. Give "
            "the most useful first. Do not report grammar errors, and do "
            "not change the meaning."
        )
    parts = []
    if context:
        parts.append(f"The document:\n{context}")
    if section:
        parts.append(f"Section: {section}")
    if before:
        parts.append(f"Sentence before the paragraph: {before}")
    parts.append(f'Paragraph:\n"""\n{paragraph.text}\n"""')
    if after:
        parts.append(f"Sentence after the paragraph: {after}")
    parts.append(
        f"{task} Keep the authors' acronyms, terms and spelling, field "
        "terminology, numbers, and the placeholders [cite], [author], "
        "[year] and (ref).\n\nFor each change, write exactly:\n"
        "ORIGINAL: <the shortest exact words from the paragraph that "
        "need the change>\nSUGGESTION: <those words changed>\n"
        "REASON: <a few words>\n\nWrite NONE if nothing needs changing."
    )
    return system, "\n\n".join(parts)


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
        surrounding quotation marks removed; empty for NONE.

    """

    found = []
    for match in _BLOCK.finditer(strip_thinking(reply)):
        original, suggestion, reason = (
            _unquoted(match[name])
            for name in ("original", "suggestion", "reason")
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
        cannot: the words are not in the paragraph, nothing changes, or
        the suggestion changes numbers, negations, citation placeholders,
        a defined acronym or a preferred term, or brings back a form the
        document does not use.

    """

    if " ".join(original.split()) not in " ".join(paragraph.split()):
        return "the words are not in the paragraph"
    if " ".join(suggestion.split()) == " ".join(original.split()):
        return "nothing changes"
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
) -> tuple[list[Finding], int, str]:
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
    tuple[list[Finding], int, str]
        The kept findings (level ``grammar`` or ``style``, check
        ``model``), how many were dropped by ``rejection``, and why the
        call failed (``""`` when it did not). A failed call is reported,
        not fatal.

    """

    system, user = paragraph_prompt(level, document, index, context)
    try:
        reply = ask_model(system, user, None)
    except IncompleteModelResponse as error:
        return [], 0, f"Not completed ({error.reason}): {error}"
    except Exception as error:  # A failed call is reported, not fatal.
        return [], 0, f"Model error: {error}"
    text = document.paragraphs[index].text
    found = parse_findings(reply)
    if level == LEVEL_STYLE:
        found = found[:_STYLE_MAX]
    kept, dropped = [], 0
    for original, suggestion, reason in found:
        if rejection(original, suggestion, text, glossary):
            dropped += 1
            continue
        squeezed = " ".join(text.split())
        offset = squeezed.find(" ".join(original.split()))
        kept.append(make_finding(
            document.paragraphs, index, level, "model", original,
            suggestion, reason, max(offset, 0),
        ))
    return kept, dropped, ""
